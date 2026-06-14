from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput

logger = logging.getLogger(__name__)


class CaduceusMiddleLossTokenClassifier(nn.Module):
    """Only active Caduceus token-classification wrapper.

    It preserves the middle-loss logic from the provided Caduceus classes while replacing
    per-sample loops with normal tensor masking. The classifier dimensions are inferred by
    LazyLinear on the first forward pass and logged immediately.
    """

    def __init__(self, caduceus_model, num_labels: int):
        super().__init__()
        self.caduceus_model = caduceus_model
        self.num_labels = int(num_labels)
        self.fc = nn.LazyLinear(self.num_labels)
        self.fc_middle = nn.LazyLinear(self.num_labels)
        self._logged_shape = False
        logger.info("[CaduceusMiddleLossTokenClassifier] num_labels=%d; hidden width will be detected on first forward", self.num_labels)

    def forward(self, input_ids=None, attention_mask=None, letter_level_labels=None, letter_level_labels_mask=None, pos_weight=None, **kwargs):
        out = self.caduceus_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = out.last_hidden_state
        hidden_states = out.hidden_states
        middle = hidden_states[len(hidden_states) // 2]
        if not self._logged_shape:
            logger.info("[CaduceusMiddleLossTokenClassifier] detected last_hidden_state shape=%s middle_hidden_state shape=%s", tuple(hidden.shape), tuple(middle.shape))
            self._logged_shape = True
        logits = self.fc(hidden)
        middle_logits = self.fc_middle(middle)
        loss = None
        if letter_level_labels is not None:
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
    """Caduceus transcript-type classifier with middle + final transcript-type loss."""

    def __init__(self, caduceus_model):
        super().__init__()
        self.caduceus_model = caduceus_model
        self.type_head = nn.LazyLinear(1)
        self.type_head_middle = nn.LazyLinear(1)
        self._logged_shape = False
        logger.info("[CaduceusTranscriptTypeMiddleLossClassifier] hidden width will be detected on first forward")

    def forward(self, input_ids=None, attention_mask=None, transcript_type=None, **kwargs):
        out = self.caduceus_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = out.last_hidden_state
        middle = out.hidden_states[len(out.hidden_states) // 2]
        if not self._logged_shape:
            logger.info("[CaduceusTranscriptTypeMiddleLossClassifier] detected last_hidden_state shape=%s middle_hidden_state shape=%s", tuple(hidden.shape), tuple(middle.shape))
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
