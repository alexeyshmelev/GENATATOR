from __future__ import annotations

"""Categorical T5Gemma head for nucleotide-level segmentation.

The head deliberately owns only the decoder half of T5Gemma.  Its encoder
input is a sequence of already-computed, per-nucleotide representations from a
GENATATOR backbone.  The caller is responsible for removing backbone special
tokens while expanding BPE states to nucleotides; ``nucleotide_mask`` then
removes padded/uncovered nucleotide positions before every decoder call.

The decoder deliberately models only two mutually exclusive internal tokens:
``intron`` and ``exon``.  Teacher forcing embeds the previous ground-truth
class id, while inference feeds back the offset-1 head's argmax.  The public
output is adapted back to GENATATOR's established five-track order so the
shared validation and post-processing code can remain unchanged; 5' UTR,
3' UTR, and CDS are unavailable from this decoder, and CDS is supplied by the
existing inference heuristic.

Optionally, one decoder state can predict several future tokens.  Each offset
has an independent linear output head, but all heads share the same freshly
initialized shallow T5Gemma decoder.  Only offset 1 participates in inference.
"""

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers.cache_utils import (
        DynamicCache,
        EncoderDecoderCache,
        SlidingWindowCache,
    )
    # T5GemmaDecoder is intentionally an internal Transformers class: using it
    # directly avoids constructing (and then discarding) the text encoder.
    from transformers.models.t5gemma.configuration_t5gemma import (
        T5GemmaConfig,
        T5GemmaModuleConfig,
    )
    from transformers.models.t5gemma.modeling_t5gemma import T5GemmaDecoder

    _T5GEMMA_IMPORT_ERROR: Exception | None = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - version dependent
    DynamicCache = None  # type: ignore[assignment,misc]
    EncoderDecoderCache = None  # type: ignore[assignment,misc]
    SlidingWindowCache = None  # type: ignore[assignment,misc]
    T5GemmaConfig = None  # type: ignore[assignment,misc]
    T5GemmaModuleConfig = None  # type: ignore[assignment,misc]
    T5GemmaDecoder = None  # type: ignore[assignment,misc]
    _T5GEMMA_IMPORT_ERROR = exc


DEFAULT_GPT_CONTEXT_SIZE = 8192
DEFAULT_GPT_ENCODER_LOOKAHEAD = 8192
SEGMENTATION_NUM_LABELS = 5
INTRON_CLASS_ID = 0
EXON_CLASS_ID = 1
UTR5_LABEL_INDEX = 0
EXON_LABEL_INDEX = 1
INTRON_LABEL_INDEX = 2
UTR3_LABEL_INDEX = 3
CDS_LABEL_INDEX = 4
UNAVAILABLE_TRACK_LOGIT = -1.0e4


class _RollingCacheIncompatible(RuntimeError):
    """Signals a cache-layout mismatch that permits a bounded safe fallback."""


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
    restricted to a moving encoder window of at most ``context_size`` plus
    ``encoder_lookahead`` nucleotide states.  The final full encoder window is
    anchored at the sequence end instead of shrinking from the left.  With both
    defaults this is an 8192-token decoder context and a 16384-token
    cross-attention span.

    ``decoder`` exists for dependency-injected unit tests.  Production callers
    should leave it as ``None`` so the official Transformers implementation is
    constructed from a fresh config without downloading weights.

    When ``add_encoder_to_decoder_input`` is enabled, every non-BOS decoder
    input is the sum of its categorical target/prediction embedding and the
    aligned projected encoder state.  Cross-attention is unchanged.
    """

    def __init__(
        self,
        encoder_dim: int,
        num_labels: int = 5,
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
        generation_threshold: float | None = None,
        multi_token_prediction: int = 1,
        add_encoder_to_decoder_input: bool = False,
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
        multi_token_prediction = int(multi_token_prediction)

        if encoder_dim <= 0 or decoder_hidden_size <= 0:
            raise RuntimeError("encoder_dim and decoder_hidden_size must be positive")
        if num_labels != SEGMENTATION_NUM_LABELS:
            raise RuntimeError(
                "The GPT compatibility interface requires exactly five segmentation "
                f"labels, got {num_labels}"
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
        if generation_threshold is not None and not 0.0 < float(generation_threshold) < 1.0:
            raise RuntimeError("generation_threshold must be strictly between 0 and 1")
        if multi_token_prediction <= 0:
            raise RuntimeError(
                "multi_token_prediction must be a positive integer, got "
                f"{multi_token_prediction}"
            )
        if not isinstance(add_encoder_to_decoder_input, bool):
            raise RuntimeError("add_encoder_to_decoder_input must be a bool")
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
        self.multi_token_prediction = multi_token_prediction
        self.add_encoder_to_decoder_input = add_encoder_to_decoder_input

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
        # Chunked teacher forcing still spans very long samples.  Checkpointing
        # decoder layers prevents all per-chunk layer activations from being
        # retained until the single outer backward pass.  It is active only in
        # training mode and does not alter validation or inference.
        if hasattr(self.decoder, "gradient_checkpointing_enable"):
            self.decoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        self.encoder_projection: nn.Module
        if encoder_dim == decoder_hidden_size:
            self.encoder_projection = nn.Identity()
        else:
            self.encoder_projection = nn.Linear(encoder_dim, decoder_hidden_size)

        # BOS comes from T5Gemma's own embedding table.  The two categorical
        # transcript-structure tokens use a separate learned embedding table.
        self.label_embedding = nn.Embedding(2, decoder_hidden_size)
        self.classifiers = nn.ModuleList(
            nn.Linear(decoder_hidden_size, 2)
            for _ in range(multi_token_prediction)
        )
        self.bos_token_id = int(getattr(self.decoder.config, "bos_token_id", 2))

    @property
    def classifier(self) -> nn.Linear:
        """Offset-1 classifier retained as a read-only compatibility alias."""

        return self.classifiers[0]

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
            # Only PAD/EOS/BOS ids exist because decoder_input_ids are never
            # used for labels.  Label vectors arrive through inputs_embeds.
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
        # SDPA is essential here: eager attention materializes a [heads, C, C+A]
        # score tensor per layer (about 1 GiB at C=A=8192 with four BF16 heads).
        # Teacher-forcing inputs are unpadded and receive mask mappings with
        # None values, allowing fused causal self-attention and unmasked cross-
        # attention without any dense 4-D masks.
        config._attn_implementation = "sdpa"
        config.decoder._attn_implementation = "sdpa"
        return T5GemmaDecoder(config.decoder)

    def _bos_embedding(self, *, device: torch.device) -> torch.Tensor:
        token_id = torch.tensor([[self.bos_token_id]], dtype=torch.long, device=device)
        return self.decoder.embed_tokens(token_id)

    @staticmethod
    def _teacher_self_attention_mask() -> dict[str, None]:
        # Each independent decoder chunk is unpadded and no longer than C.
        # With SDPA, None selects fused causal attention through the module's
        # ``is_causal`` flag without allocating a [C, C] mask.
        return {"sliding_attention": None}

    @staticmethod
    def _unmasked_cross_attention() -> dict[str, None]:
        # Encoder windows are physically sliced and unpadded.  Cross attention
        # is non-causal, so no mask at all is the exact desired operation.
        return {"full_attention": None}

    @staticmethod
    def _bounded_generation_self_attention_mask(
        *,
        filled_length: int,
        cache_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Mask only unused slots in the fixed-size sliding cache.

        This is at most ``[1, 1, 1, context_size]`` rather than a dense
        sequence-by-sequence mask.  Once the cache is full, every physical slot
        holds one of the most recent ``context_size`` decoder tokens.
        """

        cache_length = int(cache_length)
        filled_length = min(int(filled_length), cache_length)
        mask = torch.full(
            (1, 1, 1, cache_length),
            torch.finfo(dtype).min,
            device=device,
            dtype=dtype,
        )
        mask[..., :filled_length] = 0
        return {"sliding_attention": mask}

    def _encoder_window_bounds(
        self,
        target_position: int,
        encoder_length: int,
    ) -> tuple[int, int]:
        # The first query remains aligned to encoder position zero until BOS
        # would leave the decoder window.  Thereafter the encoder window moves
        # right only until its right edge reaches the sequence end.  The final
        # full window remains anchored there instead of shrinking from the left.
        moving_start = max(0, int(target_position) - self.context_size + 1)
        last_full_start = max(0, int(encoder_length) - self.cross_attention_span)
        start = min(moving_start, last_full_start)
        end = min(int(encoder_length), start + self.cross_attention_span)
        return start, end

    def _new_generation_cache(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ):
        if (
            SlidingWindowCache is None
            or DynamicCache is None
            or EncoderDecoderCache is None
        ):
            raise RuntimeError(
                "Bounded GPT generation requires the cache classes provided by "
                "transformers>=4.53.0"
            )
        self_cache = SlidingWindowCache(
            config=self.decoder.config,
            max_batch_size=1,
            max_cache_len=self.context_size,
            device=device,
            dtype=dtype,
        )
        return EncoderDecoderCache(
            self_attention_cache=self_cache,
            cross_attention_cache=DynamicCache(),
        )

    @staticmethod
    def _refresh_cross_attention_cache(past_key_values):
        """Keep bounded self history while invalidating a moved encoder slice."""

        if DynamicCache is None or EncoderDecoderCache is None:
            raise RuntimeError(
                "Bounded GPT generation requires transformers cache classes"
            )
        return EncoderDecoderCache(
            self_attention_cache=past_key_values.self_attention_cache,
            cross_attention_cache=DynamicCache(),
        )

    def _roll_cross_attention_cache(
        self,
        past_key_values,
        projected_encoder: torch.Tensor,
        previous_bounds: tuple[int, int],
        current_bounds: tuple[int, int],
    ):
        """Move cached cross K/V without re-projecting the overlapping window.

        Generation moves monotonically to the right until the full encoder
        window is anchored at the sequence end.  Therefore a moved physical
        slice is the old overlap followed by a newly visible right-edge suffix.
        Existing per-layer K/V are sliced to that overlap, and only the suffix
        is passed through each layer's cross-attention projections.  Once the
        final window is anchored, its bounds and cached K/V remain unchanged.

        ``DynamicCache.key_cache``/``value_cache`` are stable in the supported
        Transformers 4.53 implementation but remain semi-internal.  If a future
        release changes that representation, fall back to rebuilding the new
        *bounded* physical slice rather than risking incorrect K/V alignment.
        """

        old_start, old_end = previous_bounds
        new_start, new_end = current_bounds
        if new_start < old_start or new_end < old_end:
            return self._refresh_cross_attention_cache(past_key_values)

        cross_cache = past_key_values.cross_attention_cache
        overlap_start = max(old_start, new_start)
        overlap_end = min(old_end, new_end)
        overlap_length = max(0, overlap_end - overlap_start)
        expected_old_length = old_end - old_start
        append_start = overlap_end
        append_end = new_end

        try:
            key_cache = cross_cache.key_cache
            value_cache = cross_cache.value_cache
            if len(key_cache) < self.num_decoder_layers:
                raise _RollingCacheIncompatible(
                    "Cross-attention cache has missing layers"
                )

            new_keys: list[torch.Tensor] = []
            new_values: list[torch.Tensor] = []
            for layer_index in range(self.num_decoder_layers):
                cached_key = key_cache[layer_index]
                cached_value = value_cache[layer_index]
                if int(cached_key.shape[-2]) != expected_old_length:
                    raise _RollingCacheIncompatible(
                        "Cross-attention cache length does not match its encoder slice"
                    )
                local_start = overlap_start - old_start
                local_end = local_start + overlap_length
                retained_key = cached_key[..., local_start:local_end, :]
                retained_value = cached_value[..., local_start:local_end, :]

                if append_end > append_start:
                    new_encoder_states = projected_encoder[
                        :, append_start:append_end, :
                    ]
                    cross_attention = self.decoder.layers[layer_index].cross_attn
                    hidden_shape = (
                        *new_encoder_states.shape[:-1],
                        -1,
                        int(cross_attention.head_dim),
                    )
                    appended_key = cross_attention.k_proj(new_encoder_states)
                    appended_key = appended_key.view(hidden_shape).transpose(1, 2)
                    appended_value = cross_attention.v_proj(new_encoder_states)
                    appended_value = appended_value.view(hidden_shape).transpose(1, 2)
                    retained_key = torch.cat((retained_key, appended_key), dim=-2)
                    retained_value = torch.cat(
                        (retained_value, appended_value), dim=-2
                    )

                expected_new_length = new_end - new_start
                if int(retained_key.shape[-2]) != expected_new_length:
                    raise _RollingCacheIncompatible(
                        "Rolling cross-attention cache produced a misaligned window"
                    )
                new_keys.append(retained_key.contiguous())
                new_values.append(retained_value.contiguous())

            # Commit only after every layer has passed validation, so an
            # incompatibility cannot leave a partially updated cache behind.
            for layer_index, (new_key, new_value) in enumerate(
                zip(new_keys, new_values)
            ):
                key_cache[layer_index] = new_key
                value_cache[layer_index] = new_value
            if hasattr(cross_cache, "_seen_tokens"):
                cross_cache._seen_tokens = new_end - new_start
            return past_key_values
        except (
            AttributeError,
            IndexError,
            TypeError,
            ValueError,
            _RollingCacheIncompatible,
        ):
            return self._refresh_cross_attention_cache(past_key_values)

    def _teacher_forced_single(
        self,
        projected_encoder: torch.Tensor,
        target_classes: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if projected_encoder.ndim != 3 or projected_encoder.shape[0] != 1:
            raise RuntimeError(
                "Teacher forcing requires one unpadded sample shaped [1, N, H]"
            )
        if target_classes.ndim != 2 or target_classes.shape != projected_encoder.shape[:2]:
            raise RuntimeError(
                "Teacher-forcing classes must have shape [1, N] aligned with the encoder"
            )

        length = int(projected_encoder.shape[1])
        if length == 0:
            raise RuntimeError("The GPT head received an empty nucleotide sequence")
        logits_chunks: list[list[torch.Tensor]] = [
            [] for _ in range(self.multi_token_prediction)
        ]
        for start in range(0, length, self.context_size):
            end = min(length, start + self.context_size)
            target = target_classes[:, start:end]
            bos = self._bos_embedding(device=projected_encoder.device).to(
                dtype=projected_encoder.dtype
            )
            if target.shape[1] == 1:
                decoder_inputs = bos
            else:
                previous_targets = self.label_embedding(target[:, :-1])
                previous_targets = previous_targets.to(dtype=bos.dtype)
                if self.add_encoder_to_decoder_input:
                    previous_targets = (
                        previous_targets
                        + projected_encoder[:, start : end - 1, :]
                    )
                decoder_inputs = torch.cat((bos, previous_targets), dim=1)

            encoder_end = min(length, start + self.cross_attention_span)
            encoder_window = projected_encoder[:, start:encoder_end, :]
            query_length = int(decoder_inputs.shape[1])
            decoder_output = self.decoder(
                inputs_embeds=decoder_inputs,
                attention_mask=self._teacher_self_attention_mask(),
                position_ids=torch.arange(
                    query_length,
                    dtype=torch.long,
                    device=decoder_inputs.device,
                ).unsqueeze(0),
                encoder_hidden_states=encoder_window,
                encoder_attention_mask=self._unmasked_cross_attention(),
                use_cache=False,
            )
            for offset_index, classifier in enumerate(self.classifiers):
                chunk_logits = classifier(decoder_output.last_hidden_state)
                if chunk_logits.shape[:2] != target.shape:
                    raise RuntimeError(
                        "T5Gemma decoder did not preserve teacher-forcing length: "
                        f"decoder={tuple(chunk_logits.shape)} target={tuple(target.shape)}"
                    )
                logits_chunks[offset_index].append(chunk_logits)
        return tuple(torch.cat(chunks, dim=1) for chunks in logits_chunks)

    def _generate_single(self, projected_encoder: torch.Tensor) -> torch.Tensor:
        if projected_encoder.ndim != 3 or projected_encoder.shape[0] != 1:
            raise RuntimeError("Generation requires one unpadded sample shaped [1, N, H]")
        length = int(projected_encoder.shape[1])
        if length == 0:
            raise RuntimeError("The GPT head received an empty nucleotide sequence")

        previous_prediction: torch.Tensor | None = None
        past_key_values = self._new_generation_cache(
            device=projected_encoder.device,
            dtype=projected_encoder.dtype,
        )
        previous_encoder_bounds: tuple[int, int] | None = None
        logits: list[torch.Tensor] = []
        for position in range(length):
            if previous_prediction is None:
                decoder_input = self._bos_embedding(device=projected_encoder.device)
            else:
                decoder_input = self.label_embedding(previous_prediction)
            decoder_input = decoder_input.to(dtype=projected_encoder.dtype)
            if (
                self.add_encoder_to_decoder_input
                and previous_prediction is not None
            ):
                decoder_input = (
                    decoder_input
                    + projected_encoder[:, position - 1 : position, :]
                )
            encoder_bounds = self._encoder_window_bounds(position, length)
            if (
                previous_encoder_bounds is not None
                and encoder_bounds != previous_encoder_bounds
            ):
                past_key_values = self._roll_cross_attention_cache(
                    past_key_values,
                    projected_encoder,
                    previous_encoder_bounds,
                    encoder_bounds,
                )
            encoder_start, encoder_end = encoder_bounds
            encoder_window = projected_encoder[:, encoder_start:encoder_end, :]
            cache_position = torch.tensor(
                [position],
                dtype=torch.long,
                device=decoder_input.device,
            )
            decoder_output = self.decoder(
                inputs_embeds=decoder_input,
                attention_mask=self._bounded_generation_self_attention_mask(
                    filled_length=position + 1,
                    cache_length=self.context_size,
                    device=decoder_input.device,
                    dtype=decoder_input.dtype,
                ),
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position,
                encoder_hidden_states=encoder_window,
                encoder_attention_mask=self._unmasked_cross_attention(),
                past_key_values=past_key_values,
                use_cache=True,
            )
            step_hidden = decoder_output.last_hidden_state[:, -1:, :]
            # Auxiliary future-token heads are a training objective only.
            step_logits = self.classifier(step_hidden)
            logits.append(step_logits)
            previous_prediction = step_logits.argmax(dim=-1)
            past_key_values = decoder_output.past_key_values
            previous_encoder_bounds = encoder_bounds
        return torch.cat(logits, dim=1)

    @staticmethod
    def _categorical_targets(
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Convert five segmentation tracks to categorical ``[intron, exon]`` ids."""

        if labels.ndim != 3 or labels.shape[-1] != SEGMENTATION_NUM_LABELS:
            raise RuntimeError(
                "Categorical GPT targets require labels shaped [1, N, 5], got "
                f"{tuple(labels.shape)}"
            )
        if valid_mask.shape != labels.shape[:2]:
            raise RuntimeError(
                "Categorical GPT target mask must align with labels, got "
                f"{tuple(valid_mask.shape)} and {tuple(labels.shape)}"
            )
        class_scores = torch.stack(
            (labels[..., INTRON_LABEL_INDEX], labels[..., EXON_LABEL_INDEX]),
            dim=-1,
        )
        active = class_scores.ge(0.5).sum(dim=-1)
        malformed = valid_mask.bool() & active.ne(1)
        if bool(malformed.any()):
            count = int(malformed.sum().item())
            raise RuntimeError(
                "GPT exon/intron targets must contain exactly one active class at every "
                f"valid nucleotide; found {count} malformed positions"
            )
        return class_scores.argmax(dim=-1).long()

    @staticmethod
    def _five_track_compatibility_logits(
        categorical_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Map ``[intron, exon]`` logits to the shared five-track interface.

        UTR and CDS channels receive a large finite negative value, making them
        explicitly unavailable while remaining stable when forward and reverse-
        complement logits are averaged.  The exon/intron competition remains
        exactly the categorical head's competition.
        """

        if categorical_logits.ndim != 3 or categorical_logits.shape[-1] != 2:
            raise RuntimeError(
                "GPT compatibility mapping expects [batch, length, 2] logits, got "
                f"{tuple(categorical_logits.shape)}"
            )
        output = categorical_logits.new_full(
            (*categorical_logits.shape[:2], SEGMENTATION_NUM_LABELS),
            UNAVAILABLE_TRACK_LOGIT,
        )
        output[..., EXON_LABEL_INDEX] = categorical_logits[..., EXON_CLASS_ID]
        output[..., INTRON_LABEL_INDEX] = categorical_logits[..., INTRON_CLASS_ID]
        return output

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
            expected = (*encoder_embeddings.shape[:2], self.num_labels)
            if tuple(labels.shape) != expected:
                raise RuntimeError(f"Expected labels shape {expected}, got {tuple(labels.shape)}")
        if labels_mask is not None and labels_mask.shape != encoder_embeddings.shape[:2]:
            raise RuntimeError(
                "labels_mask must have shape [batch, nucleotides], got "
                f"{tuple(labels_mask.shape)}"
            )

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
        """Return ``(loss, logits)`` with the repository's segmentation shape.

        Every call carrying labels uses chunked teacher forcing, irrespective of
        ``module.training``.  Thus validation is the same single forward-pass
        objective as training rather than a nucleotide-by-nucleotide decode.
        Autoregressive decoding is reserved for label-free inference.
        """

        self._validate_inputs(encoder_embeddings, nucleotide_mask, labels, labels_mask)
        if autoregressive is None:
            autoregressive = labels is None
        if not autoregressive and labels is None:
            raise RuntimeError("Teacher forcing requires segmentation labels")
        if autoregressive and labels is not None:
            raise RuntimeError(
                "Autoregressive GPT decoding is inference-only and cannot consume labels; "
                "call generate() after removing ground-truth tensors"
            )

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
            sample_loss_mask: torch.Tensor | None = None
            if labels is not None:
                sample_labels = labels[sample_index, sample_mask, :].unsqueeze(0)
                if labels_mask is None:
                    sample_loss_mask = torch.ones(
                        sample_labels.shape[:2],
                        dtype=torch.bool,
                        device=sample_labels.device,
                    )
                else:
                    sample_loss_mask = labels_mask[
                        sample_index, sample_mask
                    ].bool().unsqueeze(0)
                if not bool(sample_loss_mask.any()):
                    raise RuntimeError(f"Sample {sample_index} has an empty label mask")

            if autoregressive:
                categorical_logits = self._generate_single(projected)
                offset_logits = (categorical_logits,)
            else:
                assert sample_labels is not None and sample_loss_mask is not None
                target_classes = self._categorical_targets(
                    sample_labels,
                    sample_loss_mask,
                )
                offset_logits = self._teacher_forced_single(projected, target_classes)
                categorical_logits = offset_logits[0]

            compatibility_logits = self._five_track_compatibility_logits(
                categorical_logits
            )
            full_logits[sample_index, sample_mask, :] = compatibility_logits[0].to(
                dtype=full_logits.dtype
            )

            if sample_labels is not None:
                assert sample_loss_mask is not None
                target_classes = self._categorical_targets(sample_labels, sample_loss_mask)
                sample_length = int(target_classes.shape[1])
                for offset_index, prediction_logits in enumerate(offset_logits):
                    shift = offset_index
                    valid_length = sample_length - shift
                    if valid_length <= 0:
                        # DDP still needs every configured auxiliary head to
                        # participate in the graph when K exceeds this sample's
                        # length.  This contributes an exact zero and produces
                        # zero-valued (rather than missing) gradients.
                        zero_loss = prediction_logits.sum() * 0.0
                        loss_sum = zero_loss if loss_sum is None else loss_sum + zero_loss
                        continue
                    shifted_mask = sample_loss_mask[:, shift:]
                    if not bool(shifted_mask.any()):
                        zero_loss = prediction_logits.sum() * 0.0
                        loss_sum = zero_loss if loss_sum is None else loss_sum + zero_loss
                        continue
                    shifted_targets = target_classes[:, shift:]
                    aligned_logits = prediction_logits[:, :valid_length, :]
                    sample_loss = F.cross_entropy(
                        aligned_logits[shifted_mask].float(),
                        shifted_targets[shifted_mask],
                        reduction="sum",
                    )
                    loss_sum = sample_loss if loss_sum is None else loss_sum + sample_loss
                    loss_element_count += int(shifted_mask.sum().item())

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
        """Autoregressively generate via offset 1 and return five-track logits."""

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
    "EXON_CLASS_ID",
    "EXON_LABEL_INDEX",
    "INTRON_CLASS_ID",
    "INTRON_LABEL_INDEX",
    "UNAVAILABLE_TRACK_LOGIT",
    "T5GemmaSegmentationHead",
]
