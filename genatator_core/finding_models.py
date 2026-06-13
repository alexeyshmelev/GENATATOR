from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers import ModernBertForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput


class ModernBertForGenatatorFinding(nn.Module):
    """ModernGENA fine-tuning wrapper for edge/region models.

    Class choice requested in the revision:
    `transformers.ModernBertForTokenClassification` is loaded as the real backbone+token head.

    Intentional change from the older provided `AnnotationModel`, which used `ModernBertModel`
    followed by a custom classifier: the HF token-classification head is now part of the
    backbone class itself. The loss remains a masked multilabel BCE, matching the edge/region
    multilabel targets.
    """

    def __init__(self, backbone_path: str, num_labels: int, dropout: float = 0.0, trust_remote_code: bool = True):
        super().__init__()
        self.model = ModernBertForTokenClassification.from_pretrained(
            backbone_path,
            num_labels=num_labels,
            classifier_dropout=dropout,
            trust_remote_code=trust_remote_code,
        )
        self.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, labels_mask=None, **kwargs):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        loss = None
        if labels is not None:
            if labels_mask is None:
                labels_mask = attention_mask.bool()
            loss = BCEWithLogitsLoss()(logits[labels_mask], labels[labels_mask].float())
        return TokenClassifierOutput(loss=loss, logits=logits)


class AutoBackboneTokenClassifier(nn.Module):
    """Simple token classifier for non-ModernBERT BPE backbones.

    Used for GENA/ModernGENA without RMT/ARMT when a plain token-level model is requested.
    It loads only the backbone from HF/local path and creates the finetuning head locally.
    """
    def __init__(self, backbone, num_labels: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, labels_mask=None, **kwargs):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        logits = self.classifier(out.last_hidden_state)
        loss = None
        if labels is not None:
            if labels_mask is None:
                labels_mask = attention_mask.bool()
            loss = BCEWithLogitsLoss()(logits[labels_mask], labels[labels_mask].float())
        return TokenClassifierOutput(loss=loss, logits=logits)
