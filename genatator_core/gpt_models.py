from __future__ import annotations

"""Encoder adapters for the T5Gemma nucleotide segmentation head."""

from typing import Any, Dict

import torch
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput

from .amt_models import AMTTokenClassifier
from .backbones import HiddenStateBackbone
from .gpt_head import GPT_TARGET_CLASS_COUNT, T5GemmaSegmentationHead
from .legacy_rmt import RMTEncoderForLetterLevelTokenClassificationUNETsegmentedRepeater


def _framing_special_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Return framing/padding ids without treating an unknown nucleotide as PAD."""

    ids = {
        int(value)
        for name in ("pad_token_id", "cls_token_id", "sep_token_id", "bos_token_id", "eos_token_id")
        if (value := getattr(tokenizer, name, None)) is not None
    }
    if not ids:
        raise RuntimeError("The GPT segmentation encoder requires tokenizer special-token ids")
    return tuple(sorted(ids))


def _content_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    special_token_ids: tuple[int, ...],
) -> torch.Tensor:
    mask = (
        attention_mask.bool()
        if attention_mask is not None
        else torch.ones_like(input_ids, dtype=torch.bool)
    )
    for token_id in special_token_ids:
        mask = mask & input_ids.ne(int(token_id))
    return mask


def _resolve_bpe_content_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    labels_mask: torch.Tensor | None,
    special_token_ids: tuple[int, ...],
    *,
    context: str,
) -> torch.Tensor:
    """Enforce framing removal even when a dataset content mask is supplied.

    ``labels_mask`` is normally the most precise offset-derived content mask,
    but it must never be allowed to re-enable an attended special token or a
    padded position.  It may legitimately be a subset of the framing mask, so
    intersection is safer than requiring equality.
    """

    framing_mask = _content_mask(input_ids, attention_mask, special_token_ids)
    if labels_mask is None:
        return framing_mask
    if labels_mask.shape != input_ids.shape:
        raise RuntimeError(
            f"{context}: labels_mask must align with input_ids; "
            f"got {tuple(labels_mask.shape)} and {tuple(input_ids.shape)}"
        )
    return labels_mask.bool() & framing_mask


def expand_bpe_states_to_nucleotides(
    *,
    token_hidden: torch.Tensor,
    token_content_mask: torch.Tensor,
    embedding_repeater: torch.Tensor,
    letter_level_tokens: torch.Tensor,
    letter_level_attention_mask: torch.Tensor | None,
    letter_level_labels_mask: torch.Tensor | None,
    nucleotide_embedding: nn.Module,
    context: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build exact per-nucleotide cross-attention inputs for the GPT head.

    BPE special/PAD positions are first removed with ``token_content_mask``.
    ``embedding_repeater`` then expands retained BPE states to real nucleotide
    positions, where a learned nucleotide embedding is concatenated exactly as
    in the existing UNET path.  Padding and BPE-uncovered positions remain zero
    in the assembled tensor and false in the returned mask.
    """

    if token_hidden.ndim != 3:
        raise RuntimeError(
            f"{context}: token_hidden must be [B, T, H], got {tuple(token_hidden.shape)}"
        )
    if token_content_mask is None or token_content_mask.shape != token_hidden.shape[:2]:
        raise RuntimeError(
            f"{context}: token_content_mask must align with token_hidden"
        )
    if embedding_repeater is None or letter_level_tokens is None:
        raise RuntimeError(
            f"{context}: embedding_repeater and letter_level_tokens are required"
        )
    if letter_level_attention_mask is None:
        if letter_level_labels_mask is None:
            raise RuntimeError(
                f"{context}: letter_level_attention_mask is required without a label mask"
            )
        letter_level_attention_mask = letter_level_labels_mask

    batch_size = int(token_hidden.shape[0])
    if letter_level_tokens.ndim != 2:
        raise RuntimeError(
            f"{context}: letter_level_tokens must be [B, N], got "
            f"{tuple(letter_level_tokens.shape)}"
        )
    if embedding_repeater.shape != letter_level_tokens.shape:
        raise RuntimeError(
            f"{context}: embedding_repeater must match letter_level_tokens; "
            f"got {tuple(embedding_repeater.shape)} and {tuple(letter_level_tokens.shape)}"
        )
    if letter_level_attention_mask.shape != letter_level_tokens.shape:
        raise RuntimeError(
            f"{context}: letter_level_attention_mask must match letter_level_tokens; "
            f"got {tuple(letter_level_attention_mask.shape)} and "
            f"{tuple(letter_level_tokens.shape)}"
        )
    if int(letter_level_tokens.shape[0]) != batch_size:
        raise RuntimeError(
            f"{context}: nucleotide batch size {letter_level_tokens.shape[0]} "
            f"does not match token batch size {batch_size}"
        )
    nucleotide_length = int(letter_level_tokens.shape[1])
    hidden_size = int(token_hidden.shape[-1])
    output = token_hidden.new_zeros((batch_size, nucleotide_length, hidden_size * 2))
    nucleotide_mask = torch.zeros(
        (batch_size, nucleotide_length),
        dtype=torch.bool,
        device=token_hidden.device,
    )

    for sample_index in range(batch_size):
        content_hidden = token_hidden[
            sample_index, token_content_mask[sample_index].bool(), :
        ]
        if content_hidden.shape[0] == 0:
            raise RuntimeError(
                f"{context}: sample {sample_index} has no retained BPE content tokens"
            )
        repeater = embedding_repeater[sample_index].long()
        sample_mask = letter_level_attention_mask[sample_index].bool() & repeater.ge(0)
        compact_repeater = repeater[sample_mask]
        if compact_repeater.numel() == 0:
            raise RuntimeError(
                f"{context}: sample {sample_index} has no BPE-covered nucleotides"
            )
        if int(compact_repeater.max().item()) >= int(content_hidden.shape[0]):
            raise RuntimeError(
                f"{context}: repeater index {int(compact_repeater.max().item())} exceeds "
                f"{content_hidden.shape[0]} retained BPE states"
            )
        nucleotide_hidden = nucleotide_embedding(
            letter_level_tokens[sample_index, sample_mask]
        ).to(dtype=token_hidden.dtype)
        combined = torch.cat(
            (nucleotide_hidden, content_hidden[compact_repeater]), dim=-1
        )
        output[sample_index, sample_mask, :] = combined
        nucleotide_mask[sample_index, sample_mask] = True
    return output, nucleotide_mask


class _GPTSegmentationModelBase(nn.Module):
    gpt_head: T5GemmaSegmentationHead

    # Preserve the repository-wide five-track interface without pretending the
    # categorical decoder predicts UTR or CDS. Sigmoid(-80) is effectively zero
    # while remaining finite in fp32, fp16, and bf16 inference/RC averaging.
    UNAVAILABLE_SEGMENTATION_LOGIT = -80.0

    @classmethod
    def _restore_five_track_logits(cls, target_logits: torch.Tensor) -> torch.Tensor:
        if target_logits.ndim != 3 or target_logits.shape[-1] != GPT_TARGET_CLASS_COUNT:
            raise RuntimeError(
                "GPT head must return [batch, nucleotides, 2] exon/intron logits, "
                f"got {tuple(target_logits.shape)}"
            )
        unavailable = torch.full_like(
            target_logits[..., 0], cls.UNAVAILABLE_SEGMENTATION_LOGIT
        )
        # Add one shared per-position offset when necessary. This preserves the
        # exon-vs-intron argmax/softmax exactly while guaranteeing that neither
        # categorical class can fall below an unavailable legacy slot.
        minimum_target = target_logits.amin(dim=-1, keepdim=True)
        required_minimum = cls.UNAVAILABLE_SEGMENTATION_LOGIT + 1.0
        compatible_targets = target_logits + (
            required_minimum - minimum_target
        ).clamp_min(0.0)
        # Legacy order: 5UTR, exon, intron, 3UTR, CDS.
        return torch.stack(
            (
                unavailable,
                compatible_targets[..., 0],
                compatible_targets[..., 1],
                unavailable,
                unavailable,
            ),
            dim=-1,
        )

    def _decode(
        self,
        encoder_embeddings: torch.Tensor,
        nucleotide_mask: torch.Tensor,
        *,
        letter_level_labels: torch.Tensor | None,
        letter_level_labels_mask: torch.Tensor | None,
        pos_weight: torch.Tensor | None,
        autoregressive: bool | None,
    ) -> TokenClassifierOutput:
        loss, target_logits = self.gpt_head(
            encoder_embeddings,
            nucleotide_mask=nucleotide_mask,
            labels=letter_level_labels,
            labels_mask=letter_level_labels_mask,
            pos_weight=pos_weight,
            autoregressive=autoregressive,
        )
        logits = self._restore_five_track_logits(target_logits)
        return TokenClassifierOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(self, **kwargs) -> torch.Tensor:
        """Generate five-slot compatible logits from two decoder target classes."""

        kwargs.pop("letter_level_labels", None)
        kwargs.pop("pos_weight", None)
        was_training = self.training
        self.eval()
        try:
            output = self.forward(autoregressive=True, **kwargs)
        finally:
            self.train(was_training)
        return output.logits


class GenaModernGPTSegmentationModel(_GPTSegmentationModelBase):
    """Direct GENA/ModernGENA encoder followed by the generative head."""

    def __init__(
        self,
        backbone_path: str,
        backbone_kind: str,
        tokenizer: Any,
        *,
        nucleotide_vocab_size: int,
        num_labels: int = 5,
        trust_remote_code: bool = True,
        allow_unsafe_torch_load: bool = True,
        gpt_config: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_backbone = HiddenStateBackbone(
            backbone_path,
            backbone_kind,
            trust_remote_code=trust_remote_code,
            modernbert_num_labels=num_labels,
            allow_unsafe_torch_load=allow_unsafe_torch_load,
        )
        self.hidden_size = int(self.hidden_backbone.hidden_size)
        self.special_token_ids = _framing_special_token_ids(tokenizer)
        self.nucleotide_embedding = nn.Embedding(
            int(nucleotide_vocab_size), self.hidden_size
        )
        self.gpt_head = T5GemmaSegmentationHead(
            encoder_dim=self.hidden_size * 2,
            num_labels=GPT_TARGET_CLASS_COUNT,
            **dict(gpt_config or {}),
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        labels_mask=None,
        embedding_repeater=None,
        letter_level_tokens=None,
        letter_level_labels=None,
        letter_level_labels_mask=None,
        letter_level_attention_mask=None,
        pos_weight=None,
        autoregressive=None,
        **kwargs,
    ) -> TokenClassifierOutput:
        hidden = self.hidden_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).logits
        content_mask = _resolve_bpe_content_mask(
            input_ids,
            attention_mask,
            labels_mask,
            self.special_token_ids,
            context="GenaModernGPTSegmentationModel",
        )
        encoder_embeddings, nucleotide_mask = expand_bpe_states_to_nucleotides(
            token_hidden=hidden,
            token_content_mask=content_mask,
            embedding_repeater=embedding_repeater,
            letter_level_tokens=letter_level_tokens,
            letter_level_attention_mask=letter_level_attention_mask,
            letter_level_labels_mask=letter_level_labels_mask,
            nucleotide_embedding=self.nucleotide_embedding,
            context="GenaModernGPTSegmentationModel",
        )
        return self._decode(
            encoder_embeddings,
            nucleotide_mask,
            letter_level_labels=letter_level_labels,
            letter_level_labels_mask=letter_level_labels_mask,
            pos_weight=pos_weight,
            autoregressive=autoregressive,
        )


class RMTGPTSegmentationModel(_GPTSegmentationModelBase):
    """RMT recurrent encoder followed by the generative head."""

    def __init__(
        self,
        base_model: nn.Module,
        tokenizer: Any,
        *,
        nucleotide_vocab_size: int,
        num_labels: int = 5,
        rmt_config: Dict[str, Any],
        gpt_config: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.special_token_ids = _framing_special_token_ids(tokenizer)
        self.rmt_encoder = RMTEncoderForLetterLevelTokenClassificationUNETsegmentedRepeater(
            base_model,
            encoder_only=True,
            tokenizer=tokenizer,
            num_labels=int(num_labels),
            **dict(rmt_config),
        )
        self.hidden_size = int(self.rmt_encoder.hidden_size)
        self.nucleotide_embedding = nn.Embedding(
            int(nucleotide_vocab_size), self.hidden_size
        )
        self.gpt_head = T5GemmaSegmentationHead(
            encoder_dim=self.hidden_size * 2,
            num_labels=GPT_TARGET_CLASS_COUNT,
            **dict(gpt_config or {}),
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        labels_mask=None,
        embedding_repeater=None,
        letter_level_tokens=None,
        letter_level_labels=None,
        letter_level_labels_mask=None,
        letter_level_attention_mask=None,
        pos_weight=None,
        autoregressive=None,
        **kwargs,
    ) -> TokenClassifierOutput:
        original_content_mask = _resolve_bpe_content_mask(
            input_ids,
            attention_mask,
            labels_mask,
            self.special_token_ids,
            context="RMTGPTSegmentationModel",
        )
        token_hidden, recurrent_content_mask = self.rmt_encoder._encode_rmt_tokens(
            input_ids,
            labels=None,
            labels_mask=original_content_mask,
            token_type_ids=token_type_ids,
        )
        if recurrent_content_mask is None:
            raise RuntimeError("RMT GPT encoding did not return its content-token mask")
        encoder_embeddings, nucleotide_mask = expand_bpe_states_to_nucleotides(
            token_hidden=token_hidden,
            token_content_mask=recurrent_content_mask,
            embedding_repeater=embedding_repeater,
            letter_level_tokens=letter_level_tokens,
            letter_level_attention_mask=letter_level_attention_mask,
            letter_level_labels_mask=letter_level_labels_mask,
            nucleotide_embedding=self.nucleotide_embedding,
            context="RMTGPTSegmentationModel",
        )
        return self._decode(
            encoder_embeddings,
            nucleotide_mask,
            letter_level_labels=letter_level_labels,
            letter_level_labels_mask=letter_level_labels_mask,
            pos_weight=pos_weight,
            autoregressive=autoregressive,
        )


class AMTGPTSegmentationModel(_GPTSegmentationModelBase):
    """Associative-memory encoder followed by the generative head."""

    def __init__(
        self,
        backbone_path: str,
        backbone_kind: str,
        tokenizer: Any,
        *,
        nucleotide_vocab_size: int,
        num_labels: int = 5,
        trust_remote_code: bool = True,
        allow_unsafe_torch_load: bool = True,
        amt_config: Dict[str, Any],
        gpt_config: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.special_token_ids = _framing_special_token_ids(tokenizer)
        self.amt_encoder = AMTTokenClassifier(
            backbone_path=backbone_path,
            backbone_kind=backbone_kind,
            num_labels=int(num_labels),
            trust_remote_code=trust_remote_code,
            use_unet=False,
            encoder_only=True,
            allow_unsafe_torch_load=allow_unsafe_torch_load,
            **dict(amt_config),
        )
        self.hidden_size = int(self.amt_encoder.hidden_size)
        self.nucleotide_embedding = nn.Embedding(
            int(nucleotide_vocab_size), self.hidden_size
        )
        self.gpt_head = T5GemmaSegmentationHead(
            encoder_dim=self.hidden_size * 2,
            num_labels=GPT_TARGET_CLASS_COUNT,
            **dict(gpt_config or {}),
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        labels_mask=None,
        embedding_repeater=None,
        letter_level_tokens=None,
        letter_level_labels=None,
        letter_level_labels_mask=None,
        letter_level_attention_mask=None,
        pos_weight=None,
        autoregressive=None,
        **kwargs,
    ) -> TokenClassifierOutput:
        hidden = self.amt_encoder.encode_hidden(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        content_mask = _resolve_bpe_content_mask(
            input_ids,
            attention_mask,
            labels_mask,
            self.special_token_ids,
            context="AMTGPTSegmentationModel",
        )
        encoder_embeddings, nucleotide_mask = expand_bpe_states_to_nucleotides(
            token_hidden=hidden,
            token_content_mask=content_mask,
            embedding_repeater=embedding_repeater,
            letter_level_tokens=letter_level_tokens,
            letter_level_attention_mask=letter_level_attention_mask,
            letter_level_labels_mask=letter_level_labels_mask,
            nucleotide_embedding=self.nucleotide_embedding,
            context="AMTGPTSegmentationModel",
        )
        return self._decode(
            encoder_embeddings,
            nucleotide_mask,
            letter_level_labels=letter_level_labels,
            letter_level_labels_mask=letter_level_labels_mask,
            pos_weight=pos_weight,
            autoregressive=autoregressive,
        )


class CaduceusGPTSegmentationModel(_GPTSegmentationModelBase):
    """Caduceus nucleotide encoder followed by the generative head."""

    def __init__(
        self,
        caduceus_model: nn.Module,
        hidden_size: int,
        tokenizer: Any,
        *,
        nucleotide_vocab_size: int,
        num_labels: int = 5,
        gpt_config: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.caduceus_model = caduceus_model
        self.hidden_size = int(hidden_size)
        self.special_token_ids = _framing_special_token_ids(tokenizer)
        self.nucleotide_embedding = nn.Embedding(
            int(nucleotide_vocab_size), self.hidden_size
        )
        self.gpt_head = T5GemmaSegmentationHead(
            encoder_dim=self.hidden_size * 2,
            num_labels=GPT_TARGET_CLASS_COUNT,
            **dict(gpt_config or {}),
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        letter_level_labels=None,
        letter_level_labels_mask=None,
        pos_weight=None,
        autoregressive=None,
        **kwargs,
    ) -> TokenClassifierOutput:
        output = self.caduceus_model(input_ids=input_ids)
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            hidden = output[0]
        if hidden.shape[-1] != self.hidden_size:
            raise RuntimeError(
                f"Caduceus hidden width {hidden.shape[-1]} != {self.hidden_size}"
            )
        nucleotide_mask = _content_mask(
            input_ids, attention_mask, self.special_token_ids
        )
        if letter_level_labels_mask is not None and not bool(
            torch.equal(nucleotide_mask, letter_level_labels_mask.bool())
        ):
            raise RuntimeError(
                "Caduceus special-token removal disagrees with the nucleotide label mask"
            )
        nucleotide_hidden = self.nucleotide_embedding(input_ids).to(dtype=hidden.dtype)
        encoder_embeddings = torch.cat((nucleotide_hidden, hidden), dim=-1)
        return self._decode(
            encoder_embeddings,
            nucleotide_mask,
            letter_level_labels=letter_level_labels,
            letter_level_labels_mask=letter_level_labels_mask,
            pos_weight=pos_weight,
            autoregressive=autoregressive,
        )


__all__ = [
    "AMTGPTSegmentationModel",
    "CaduceusGPTSegmentationModel",
    "GenaModernGPTSegmentationModel",
    "RMTGPTSegmentationModel",
    "expand_bpe_states_to_nucleotides",
]
