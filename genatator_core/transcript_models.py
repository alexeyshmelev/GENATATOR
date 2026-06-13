from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers import ModernBertForTokenClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class ModernBertForTranscriptType(nn.Module):
    """ModernGENA transcript-type classifier using ModernBertForTokenClassification.

    Class choice requested in the revision: `ModernBertForTokenClassification`.
    The model emits one token logit per position and the classifier uses the last non-pad
    token logit as the interval-level mRNA/lnc_RNA score.
    """
    def __init__(self, backbone_path: str, trust_remote_code: bool = True):
        super().__init__()
        self.model = ModernBertForTokenClassification.from_pretrained(backbone_path, num_labels=1, trust_remote_code=trust_remote_code)

    def forward(self, input_ids=None, attention_mask=None, transcript_type=None, **kwargs):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        token_logits = out.logits.squeeze(-1)
        last_idx = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        logits = token_logits[torch.arange(input_ids.shape[0], device=input_ids.device), last_idx].view(-1, 1)
        loss = None
        if transcript_type is not None:
            loss = BCEWithLogitsLoss()(logits, transcript_type.float())
        return SequenceClassifierOutput(loss=loss, logits=logits)
