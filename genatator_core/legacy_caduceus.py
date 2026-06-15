from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput

logger = logging.getLogger(__name__)


def infer_caduceus_hidden_size(config, backbone_path: str) -> int:
    """Infer the Caduceus output width before training.

    The original working code used fixed linear heads: PH token middle-loss heads
    were 256-wide, while PS transcript-type heads were 512-wide. This helper keeps
    that logic explicit and logs exactly what was inferred. No LazyLinear fallback is
    used, so shape mismatches fail loudly in forward().
    """
    d_model = None
    for name in ("d_model", "hidden_size", "n_embd", "dim"):
        value = getattr(config, name, None)
        if value is not None:
            d_model = int(value)
            break
    if d_model is None:
        raise RuntimeError(
            "Could not infer Caduceus d_model from config. Expected one of: "
            "d_model, hidden_size, n_embd, dim. Add model.hidden_size explicitly."
        )
    repo = str(backbone_path).lower()
    if "caduceus-ps" in repo or "_ps_" in repo or "-ps_" in repo:
        hidden = 2 * d_model
        reason = "PS/RC-equivariant checkpoint: output hidden state is 2 * d_model"
    elif "caduceus-ph" in repo or "_ph_" in repo or "-ph_" in repo:
        hidden = d_model
        reason = "PH checkpoint: output hidden state is d_model"
    else:
        # For custom local paths, use the explicit config flag if present.
        if bool(getattr(config, "bidirectional_weight_tie", True)) is False and bool(getattr(config, "rcps", False)):
            hidden = 2 * d_model
            reason = "local config has rcps=True and bidirectional_weight_tie=False"
        else:
            hidden = d_model
            reason = "local/custom Caduceus path without PS marker: using d_model"
    logger.info(
        "[caduceus.shape] backbone_path=%s d_model=%d inferred_hidden_size=%d reason=%s bidirectional_weight_tie=%s",
        backbone_path,
        d_model,
        hidden,
        reason,
        getattr(config, "bidirectional_weight_tie", None),
    )
    return int(hidden)


class CaduceusMiddleLossTokenClassifier(nn.Module):
    """Active Caduceus token-classification wrapper with middle loss only.

    This keeps the original GENATATOR Caduceus middle-loss idea: classify the final
    hidden state and one middle hidden state, then average the two BCE losses. The
    old implementation iterated over batch elements; this version uses tensor masks
    directly and supports batch size > 1.
    """

    def __init__(self, caduceus_model, num_labels: int, hidden_size: int):
        super().__init__()
        self.caduceus_model = caduceus_model
        self.num_labels = int(num_labels)
        self.hidden_size = int(hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.num_labels)
        self.fc_middle = nn.Linear(self.hidden_size, self.num_labels)
        self._logged_shape = False
        logger.info(
            "[CaduceusMiddleLossTokenClassifier] hidden_size=%d num_labels=%d class=%s",
            self.hidden_size,
            self.num_labels,
            type(caduceus_model).__name__,
        )

    def forward(self, input_ids=None, attention_mask=None, letter_level_labels=None, letter_level_labels_mask=None, pos_weight=None, **kwargs):
        out = self.caduceus_model(input_ids=input_ids, output_hidden_states=True)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        hidden_states = out.hidden_states if hasattr(out, "hidden_states") else None
        if hidden_states is None:
            raise RuntimeError("Caduceus model did not return hidden_states with output_hidden_states=True")
        middle = hidden_states[len(hidden_states) // 2]
        if hidden.shape[-1] != self.hidden_size or middle.shape[-1] != self.hidden_size:
            raise RuntimeError(
                f"Caduceus hidden-size mismatch: configured hidden_size={self.hidden_size}, "
                f"last={tuple(hidden.shape)}, middle={tuple(middle.shape)}. "
                "Fix infer_caduceus_hidden_size() or set the correct model.hidden_size."
            )
        if not self._logged_shape:
            logger.info(
                "[CaduceusMiddleLossTokenClassifier] detected last_hidden_state=%s middle_hidden_state=%s input_ids=%s",
                tuple(hidden.shape),
                tuple(middle.shape),
                tuple(input_ids.shape),
            )
            self._logged_shape = True
        logits = self.fc(hidden)
        middle_logits = self.fc_middle(middle)
        loss = None
        if letter_level_labels is not None:
            if letter_level_labels_mask is None:
                raise RuntimeError("Caduceus token loss requires letter_level_labels_mask")
            mask = letter_level_labels_mask.bool()
            if mask.sum() == 0:
                raise RuntimeError("Caduceus loss mask is empty.")
            weight = None
            if pos_weight is not None:
                weight = pos_weight[0, 0, :].to(logits.device).float() if pos_weight.dim() == 3 else pos_weight.to(logits.device).float()
            loss_fct = BCEWithLogitsLoss(pos_weight=weight)
            loss_last = loss_fct(logits[mask].float(), letter_level_labels[mask].float())
            loss_middle = loss_fct(middle_logits[mask].float(), letter_level_labels[mask].float())
            loss = 0.5 * (loss_last + loss_middle)
        return TokenClassifierOutput(loss=loss, logits=logits)


class CaduceusTranscriptTypeMiddleLossClassifier(nn.Module):
    """Caduceus transcript-type classifier with final + middle transcript-type loss."""

    def __init__(self, caduceus_model, hidden_size: int):
        super().__init__()
        self.caduceus_model = caduceus_model
        self.hidden_size = int(hidden_size)
        self.type_head = nn.Linear(self.hidden_size, 1)
        self.type_head_middle = nn.Linear(self.hidden_size, 1)
        self._logged_shape = False
        logger.info(
            "[CaduceusTranscriptTypeMiddleLossClassifier] hidden_size=%d class=%s",
            self.hidden_size,
            type(caduceus_model).__name__,
        )

    def forward(self, input_ids=None, attention_mask=None, transcript_type=None, **kwargs):
        out = self.caduceus_model(input_ids=input_ids, output_hidden_states=True)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        hidden_states = out.hidden_states if hasattr(out, "hidden_states") else None
        if hidden_states is None:
            raise RuntimeError("Caduceus model did not return hidden_states with output_hidden_states=True")
        middle = hidden_states[len(hidden_states) // 2]
        if hidden.shape[-1] != self.hidden_size or middle.shape[-1] != self.hidden_size:
            raise RuntimeError(
                f"Caduceus hidden-size mismatch: configured hidden_size={self.hidden_size}, "
                f"last={tuple(hidden.shape)}, middle={tuple(middle.shape)}."
            )
        if not self._logged_shape:
            logger.info(
                "[CaduceusTranscriptTypeMiddleLossClassifier] detected last_hidden_state=%s middle_hidden_state=%s input_ids=%s",
                tuple(hidden.shape),
                tuple(middle.shape),
                tuple(input_ids.shape),
            )
            self._logged_shape = True
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        last_idx = (attention_mask.long() * torch.arange(attention_mask.shape[1], device=attention_mask.device)).max(dim=1).values
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        logits = self.type_head(hidden[batch_idx, last_idx])
        logits_middle = self.type_head_middle(middle[batch_idx, last_idx])
        loss = None
        if transcript_type is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = 0.5 * (loss_fct(logits, transcript_type.float()) + loss_fct(logits_middle, transcript_type.float()))
        return SequenceClassifierOutput(loss=loss, logits=logits)
