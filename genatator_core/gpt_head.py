from __future__ import annotations

"""Autoregressive T5Gemma head for nucleotide-level segmentation.

The head deliberately owns only the decoder half of T5Gemma.  Its encoder
input is a sequence of already-computed, per-nucleotide representations from a
GENATATOR backbone.  The caller is responsible for removing backbone special
tokens while expanding BPE states to nucleotides; ``nucleotide_mask`` then
removes padded/uncovered nucleotide positions before every decoder call.

The decoder vocabulary is deliberately categorical and contains exactly two
targets: exon and intron. Teacher forcing embeds the previous ground-truth
class id, generation feeds back the previous argmax class id, and loss is
cross-entropy over those two classes. The surrounding model adapter restores
the repository's legacy five-channel output shape for metrics/postprocessing;
UTR and CDS are never decoder targets.
"""

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # T5GemmaDecoder is intentionally an internal Transformers class: using it
    # directly avoids constructing (and then discarding) the text encoder.
    from transformers.models.t5gemma.configuration_t5gemma import (
        T5GemmaConfig,
        T5GemmaModuleConfig,
    )
    from transformers.models.t5gemma.modeling_t5gemma import T5GemmaDecoder

    _T5GEMMA_IMPORT_ERROR: Exception | None = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - version dependent
    T5GemmaConfig = None  # type: ignore[assignment,misc]
    T5GemmaModuleConfig = None  # type: ignore[assignment,misc]
    T5GemmaDecoder = None  # type: ignore[assignment,misc]
    _T5GEMMA_IMPORT_ERROR = exc


DEFAULT_GPT_CONTEXT_SIZE = 8192
DEFAULT_GPT_ENCODER_LOOKAHEAD = 8192
GPT_TARGET_CLASS_NAMES = ("exon", "intron")
GPT_TARGET_CLASS_COUNT = len(GPT_TARGET_CLASS_NAMES)
FULL_SEGMENTATION_EXON_INDEX = 1
FULL_SEGMENTATION_INTRON_INDEX = 2


@contextmanager
def _temporary_eval(module: nn.Module) -> Iterator[None]:
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


class T5GemmaSegmentationHead(nn.Module):
    """A shallow, decoder-only T5Gemma segmentation head.

    Parameters are intentionally small by default (two 256-wide layers).  All
    decoder layers use T5Gemma sliding self-attention, while cross-attention is
    restricted to the current decoder window plus ``encoder_lookahead`` future
    nucleotide states.  With both defaults this is an 8192-token decoder
    context and a 16384-token cross-attention span.

    ``decoder`` exists for dependency-injected unit tests.  Production callers
    should leave it as ``None`` so the official Transformers implementation is
    constructed from a fresh config without downloading weights.
    """

    def __init__(
        self,
        encoder_dim: int,
        num_labels: int = GPT_TARGET_CLASS_COUNT,
        *,
        decoder_hidden_size: int = 256,
        decoder_intermediate_size: int | None = None,
        num_decoder_layers: int = 2,
        num_attention_heads: int = 4,
        num_key_value_heads: int | None = None,
        context_size: int = DEFAULT_GPT_CONTEXT_SIZE,
        encoder_lookahead: int = DEFAULT_GPT_ENCODER_LOOKAHEAD,
        dropout_rate: float = 0.0,
        attention_dropout: float = 0.0,
        decoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        encoder_dim = int(encoder_dim)
        num_labels = int(num_labels)
        decoder_hidden_size = int(decoder_hidden_size)
        num_decoder_layers = int(num_decoder_layers)
        num_attention_heads = int(num_attention_heads)
        context_size = int(context_size)
        encoder_lookahead = int(encoder_lookahead)

        if encoder_dim <= 0 or decoder_hidden_size <= 0:
            raise RuntimeError("encoder_dim and decoder_hidden_size must be positive")
        if num_labels != GPT_TARGET_CLASS_COUNT:
            raise RuntimeError(
                "The segmentation GPT decoder has exactly two target classes "
                f"(exon/intron), got num_labels={num_labels}"
            )
        if not 2 <= num_decoder_layers <= 4:
            raise RuntimeError(
                "The segmentation GPT head must remain shallow: "
                f"num_decoder_layers must be in [2, 4], got {num_decoder_layers}"
            )
        if context_size <= 0:
            raise RuntimeError(f"context_size must be positive, got {context_size}")
        if encoder_lookahead < 0:
            raise RuntimeError(
                f"encoder_lookahead must be non-negative, got {encoder_lookahead}"
            )
        if decoder_hidden_size % num_attention_heads:
            raise RuntimeError(
                "decoder_hidden_size must be divisible by num_attention_heads: "
                f"{decoder_hidden_size} % {num_attention_heads} != 0"
            )

        self.encoder_dim = encoder_dim
        self.num_labels = num_labels
        self.decoder_hidden_size = decoder_hidden_size
        self.num_decoder_layers = num_decoder_layers
        self.context_size = context_size
        self.encoder_lookahead = encoder_lookahead
        self.cross_attention_span = context_size + encoder_lookahead

        if decoder is None:
            decoder = self._build_t5gemma_decoder(
                hidden_size=decoder_hidden_size,
                intermediate_size=int(
                    decoder_intermediate_size
                    if decoder_intermediate_size is not None
                    else decoder_hidden_size * 4
                ),
                num_layers=num_decoder_layers,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=int(
                    num_key_value_heads
                    if num_key_value_heads is not None
                    else num_attention_heads
                ),
                context_size=context_size,
                dropout_rate=float(dropout_rate),
                attention_dropout=float(attention_dropout),
            )
        else:
            actual_hidden = int(getattr(getattr(decoder, "config", None), "hidden_size", -1))
            if actual_hidden != decoder_hidden_size:
                raise RuntimeError(
                    "Injected decoder hidden size does not match decoder_hidden_size: "
                    f"{actual_hidden} != {decoder_hidden_size}"
                )
            if not hasattr(decoder, "embed_tokens"):
                raise RuntimeError("The T5Gemma decoder must expose embed_tokens for BOS")
        self.decoder = decoder

        self.encoder_projection: nn.Module
        if encoder_dim == decoder_hidden_size:
            self.encoder_projection = nn.Identity()
        else:
            self.encoder_projection = nn.Linear(encoder_dim, decoder_hidden_size)

        # BOS comes from T5Gemma's freshly initialized embedding table. Normal
        # targets use a separate categorical table: id 0 = exon, id 1 = intron.
        self.target_embedding = nn.Embedding(num_labels, decoder_hidden_size)
        self.classifier = nn.Linear(decoder_hidden_size, num_labels)
        self.bos_token_id = int(getattr(self.decoder.config, "bos_token_id", 2))

    @staticmethod
    def _build_t5gemma_decoder(
        *,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        context_size: int,
        dropout_rate: float,
        attention_dropout: float,
    ) -> nn.Module:
        if T5GemmaDecoder is None or T5GemmaConfig is None or T5GemmaModuleConfig is None:
            raise RuntimeError(
                "T5GemmaSegmentationHead requires transformers>=4.53.0; that is "
                "the first Transformers release containing T5Gemma."
            ) from _T5GEMMA_IMPORT_ERROR
        if intermediate_size <= 0:
            raise RuntimeError("decoder_intermediate_size must be positive")
        if num_key_value_heads <= 0 or num_attention_heads % num_key_value_heads:
            raise RuntimeError(
                "num_key_value_heads must be positive and divide num_attention_heads"
            )

        head_dim = hidden_size // num_attention_heads
        module_kwargs = dict(
            # Only PAD/EOS/BOS ids exist because exon/intron targets use the
            # separate target_embedding and arrive through inputs_embeds.
            vocab_size=3,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            query_pre_attn_scalar=head_dim,
            max_position_embeddings=context_size,
            sliding_window=context_size,
            # Alternating global layers would violate the requested fixed
            # decoder context.  Cross-attention remains global within the
            # explicit encoder window supplied below.
            layer_types=["sliding_attention"] * num_layers,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=1,
            bos_token_id=2,
            tie_word_embeddings=False,
        )
        # Use distinct objects.  Transformers 4.53 copied these internally,
        # while newer strict/dataclass configs retain the supplied instances.
        encoder_config = T5GemmaModuleConfig(**module_kwargs)
        decoder_config = T5GemmaModuleConfig(**module_kwargs)
        config = T5GemmaConfig(
            encoder=encoder_config,
            decoder=decoder_config,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            tie_word_embeddings=False,
        )
        # A direct additive cross-attention mask is used during generation.
        # Eager attention has a stable, explicit 4-D additive-mask contract
        # across the supported T5Gemma releases.
        config._attn_implementation = "eager"
        config.decoder._attn_implementation = "eager"
        return T5GemmaDecoder(config.decoder)

    def _bos_embedding(self, *, device: torch.device) -> torch.Tensor:
        token_id = torch.tensor([[self.bos_token_id]], dtype=torch.long, device=device)
        return self.decoder.embed_tokens(token_id)

    @staticmethod
    def _full_cross_attention_mask(
        query_length: int,
        key_length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        return {
            "full_attention": torch.zeros(
                (1, 1, int(query_length), int(key_length)),
                device=device,
                dtype=dtype,
            )
        }

    def _sliding_cross_attention_mask(
        self,
        *,
        target_position: int,
        encoder_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        # The first query remains aligned to encoder position zero until BOS
        # would leave the decoder window.  Thereafter both windows move one
        # nucleotide at a time.
        start = max(0, int(target_position) - self.context_size + 1)
        end = min(int(encoder_length), start + self.cross_attention_span)
        min_value = torch.finfo(dtype).min
        mask = torch.full(
            (1, 1, 1, int(encoder_length)),
            min_value,
            device=device,
            dtype=dtype,
        )
        mask[..., start:end] = 0
        return {"full_attention": mask}

    def _teacher_forced_single(
        self,
        projected_encoder: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        if projected_encoder.ndim != 3 or projected_encoder.shape[0] != 1:
            raise RuntimeError(
                "Teacher forcing requires one unpadded sample shaped [1, N, H]"
            )
        if target_ids.ndim != 2 or target_ids.shape != projected_encoder.shape[:2]:
            raise RuntimeError(
                "Teacher-forcing target ids must have shape [1, N]"
            )
        if target_ids.dtype != torch.long:
            raise RuntimeError("Teacher-forcing target ids must use torch.long")

        length = int(projected_encoder.shape[1])
        if length == 0:
            raise RuntimeError("The GPT head received an empty nucleotide sequence")
        logits_chunks: list[torch.Tensor] = []
        for start in range(0, length, self.context_size):
            end = min(length, start + self.context_size)
            target = target_ids[:, start:end]
            bos = self._bos_embedding(device=projected_encoder.device).to(
                dtype=projected_encoder.dtype
            )
            if target.shape[1] == 1:
                decoder_inputs = bos
            else:
                previous_targets = self.target_embedding(target[:, :-1]).to(bos.dtype)
                decoder_inputs = torch.cat((bos, previous_targets), dim=1)

            encoder_end = min(length, start + self.cross_attention_span)
            encoder_window = projected_encoder[:, start:encoder_end, :]
            query_length = int(decoder_inputs.shape[1])
            decoder_output = self.decoder(
                inputs_embeds=decoder_inputs,
                attention_mask=torch.ones(
                    (1, query_length),
                    dtype=torch.bool,
                    device=decoder_inputs.device,
                ),
                position_ids=torch.arange(
                    query_length,
                    dtype=torch.long,
                    device=decoder_inputs.device,
                ).unsqueeze(0),
                encoder_hidden_states=encoder_window,
                encoder_attention_mask=self._full_cross_attention_mask(
                    query_length,
                    int(encoder_window.shape[1]),
                    device=decoder_inputs.device,
                    dtype=decoder_inputs.dtype,
                ),
                use_cache=False,
            )
            chunk_logits = self.classifier(decoder_output.last_hidden_state)
            if chunk_logits.shape[:2] != target.shape[:2]:
                raise RuntimeError(
                    "T5Gemma decoder did not preserve teacher-forcing length: "
                    f"decoder={tuple(chunk_logits.shape)} target={tuple(target.shape)}"
                )
            logits_chunks.append(chunk_logits)
        return torch.cat(logits_chunks, dim=1)

    def _generate_single(self, projected_encoder: torch.Tensor) -> torch.Tensor:
        if projected_encoder.ndim != 3 or projected_encoder.shape[0] != 1:
            raise RuntimeError("Generation requires one unpadded sample shaped [1, N, H]")
        length = int(projected_encoder.shape[1])
        if length == 0:
            raise RuntimeError("The GPT head received an empty nucleotide sequence")

        previous_prediction: torch.Tensor | None = None
        past_key_values = None
        logits: list[torch.Tensor] = []
        for position in range(length):
            if previous_prediction is None:
                decoder_input = self._bos_embedding(device=projected_encoder.device)
            else:
                decoder_input = self.target_embedding(previous_prediction)
            decoder_input = decoder_input.to(dtype=projected_encoder.dtype)
            decoder_output = self.decoder(
                inputs_embeds=decoder_input,
                encoder_hidden_states=projected_encoder,
                encoder_attention_mask=self._sliding_cross_attention_mask(
                    target_position=position,
                    encoder_length=length,
                    device=decoder_input.device,
                    dtype=decoder_input.dtype,
                ),
                past_key_values=past_key_values,
                use_cache=True,
            )
            step_hidden = decoder_output.last_hidden_state[:, -1:, :]
            step_logits = self.classifier(step_hidden)
            logits.append(step_logits)
            previous_prediction = step_logits.argmax(dim=-1)
            past_key_values = decoder_output.past_key_values
        return torch.cat(logits, dim=1)

    def _validate_inputs(
        self,
        encoder_embeddings: torch.Tensor,
        nucleotide_mask: torch.Tensor,
        labels: torch.Tensor | None,
        labels_mask: torch.Tensor | None,
    ) -> None:
        if encoder_embeddings.ndim != 3:
            raise RuntimeError(
                "encoder_embeddings must have shape [batch, nucleotides, channels], "
                f"got {tuple(encoder_embeddings.shape)}"
            )
        if encoder_embeddings.shape[-1] != self.encoder_dim:
            raise RuntimeError(
                f"Expected encoder_dim={self.encoder_dim}, got {encoder_embeddings.shape[-1]}"
            )
        if nucleotide_mask is None or nucleotide_mask.shape != encoder_embeddings.shape[:2]:
            raise RuntimeError(
                "nucleotide_mask with shape [batch, nucleotides] is required to "
                "remove PAD, special, and uncovered positions"
            )
        if labels is not None:
            if labels.ndim != 3 or labels.shape[:2] != encoder_embeddings.shape[:2]:
                raise RuntimeError(
                    "Segmentation labels must have shape [batch, nucleotides, classes], "
                    f"got {tuple(labels.shape)}"
                )
            if labels.shape[-1] not in {GPT_TARGET_CLASS_COUNT, 5}:
                raise RuntimeError(
                    "GPT labels must contain either exon/intron only (2 channels) "
                    f"or the legacy five segmentation channels, got {labels.shape[-1]}"
                )
        if labels_mask is not None and labels_mask.shape != encoder_embeddings.shape[:2]:
            raise RuntimeError(
                "labels_mask must have shape [batch, nucleotides], got "
                f"{tuple(labels_mask.shape)}"
            )

    @staticmethod
    def _target_ids(labels: torch.Tensor) -> torch.Tensor:
        """Convert two-channel or legacy five-channel labels to exon/intron ids."""

        if labels.shape[-1] == 5:
            categorical = labels[
                ..., [FULL_SEGMENTATION_EXON_INDEX, FULL_SEGMENTATION_INTRON_INDEX]
            ]
        elif labels.shape[-1] == GPT_TARGET_CLASS_COUNT:
            categorical = labels
        else:  # guarded by _validate_inputs; keep this helper safe in isolation.
            raise RuntimeError(f"Unsupported GPT label width: {labels.shape[-1]}")
        rounded = categorical.round()
        if not bool(
            torch.allclose(
                categorical.float(), rounded.float(), atol=1e-6, rtol=0.0
            )
        ):
            raise RuntimeError("GPT exon/intron targets must be binary")
        if not bool(((rounded == 0) | (rounded == 1)).all()):
            raise RuntimeError("GPT exon/intron targets must contain only 0 or 1")
        if not bool((rounded.sum(dim=-1) == 1).all()):
            raise RuntimeError(
                "Every retained nucleotide must have exactly one exon/intron target"
            )
        return rounded.argmax(dim=-1).long()

    def forward(
        self,
        encoder_embeddings: torch.Tensor,
        *,
        nucleotide_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        labels_mask: torch.Tensor | None = None,
        pos_weight: torch.Tensor | None = None,
        autoregressive: bool | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Return loss and two categorical exon/intron logits per nucleotide.

        Training with labels uses chunked teacher forcing.  Evaluation defaults
        to autoregressive generation even when validation labels are present;
        in that case validation loss is computed from generated logits.  A
        label-free call is always autoregressive inference.
        """

        self._validate_inputs(encoder_embeddings, nucleotide_mask, labels, labels_mask)
        # Categorical cross-entropy has no BCE-style positive-class weights.
        # Keep this argument only so existing dataset/collator batches remain
        # compatible with the GPT wrapper.
        del pos_weight
        if autoregressive is None:
            autoregressive = not self.training or labels is None
        if not autoregressive and labels is None:
            raise RuntimeError("Teacher forcing requires segmentation labels")

        batch_size, output_length, _ = encoder_embeddings.shape
        full_logits = encoder_embeddings.new_zeros(
            (int(batch_size), int(output_length), self.num_labels)
        )
        loss_sum: torch.Tensor | None = None
        loss_element_count = 0

        for sample_index in range(int(batch_size)):
            sample_mask = nucleotide_mask[sample_index].bool()
            if not bool(sample_mask.any()):
                raise RuntimeError(f"Sample {sample_index} has no retained nucleotide positions")
            sample_encoder = encoder_embeddings[sample_index, sample_mask, :].unsqueeze(0)
            projected = self.encoder_projection(sample_encoder)

            sample_labels: torch.Tensor | None = None
            sample_target_ids: torch.Tensor | None = None
            if labels is not None:
                sample_labels = labels[sample_index, sample_mask, :].unsqueeze(0)
                sample_target_ids = self._target_ids(sample_labels)

            if autoregressive:
                # Validation and inference must not consume ground-truth history.
                with torch.no_grad():
                    sample_logits = self._generate_single(projected)
            else:
                assert sample_target_ids is not None
                sample_logits = self._teacher_forced_single(projected, sample_target_ids)

            full_logits[sample_index, sample_mask, :] = sample_logits[0].to(
                dtype=full_logits.dtype
            )

            if sample_target_ids is not None:
                if labels_mask is None:
                    sample_loss_mask = torch.ones(
                        sample_target_ids.shape,
                        dtype=torch.bool,
                        device=sample_target_ids.device,
                    )
                else:
                    sample_loss_mask = labels_mask[sample_index, sample_mask].bool().unsqueeze(0)
                if not bool(sample_loss_mask.any()):
                    raise RuntimeError(f"Sample {sample_index} has an empty label mask")
                sample_loss = F.cross_entropy(
                    sample_logits[sample_loss_mask].float(),
                    sample_target_ids[sample_loss_mask],
                    reduction="sum",
                )
                loss_sum = sample_loss if loss_sum is None else loss_sum + sample_loss
                loss_element_count += int(sample_loss_mask.sum().item())

        loss = None
        if labels is not None:
            if loss_sum is None or loss_element_count <= 0:
                raise RuntimeError("No valid elements were available for segmentation CE loss")
            loss = loss_sum / float(loss_element_count)
        return loss, full_logits

    @torch.no_grad()
    def generate(
        self,
        encoder_embeddings: torch.Tensor,
        *,
        nucleotide_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressively generate two exon/intron logits per real nucleotide."""

        self._validate_inputs(encoder_embeddings, nucleotide_mask, None, None)
        with _temporary_eval(self):
            _, logits = self.forward(
                encoder_embeddings,
                nucleotide_mask=nucleotide_mask,
                autoregressive=True,
            )
        return logits


__all__ = [
    "DEFAULT_GPT_CONTEXT_SIZE",
    "DEFAULT_GPT_ENCODER_LOOKAHEAD",
    "GPT_TARGET_CLASS_COUNT",
    "GPT_TARGET_CLASS_NAMES",
    "T5GemmaSegmentationHead",
]
