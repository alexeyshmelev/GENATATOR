from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput

from .backbones import HiddenStateBackbone
from .unet import UNET1DSegmentationHead

logger = logging.getLogger(__name__)


class PlainTokenClassifier(nn.Module):
    """Token-level classifier for GENA/ModernGENA without RMT/AMT/UNET."""

    def __init__(self, backbone_path: str, backbone_kind: str, num_labels: int, trust_remote_code: bool = True, allow_unsafe_torch_load: bool = True):
        super().__init__()
        self.hidden_backbone = HiddenStateBackbone(backbone_path, backbone_kind, trust_remote_code=trust_remote_code, modernbert_num_labels=num_labels, allow_unsafe_torch_load=allow_unsafe_torch_load)
        self.hidden_size = self.hidden_backbone.hidden_size
        self.num_labels = int(num_labels)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        logger.info("[PlainTokenClassifier] backbone=%s hidden=%d labels=%d", backbone_kind, self.hidden_size, self.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, labels_mask=None, **kwargs):
        hidden = self.hidden_backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
        logits = self.classifier(hidden)
        loss = None
        if labels is not None:
            mask = labels_mask.bool() if labels_mask is not None else attention_mask.bool()
            if mask.sum() == 0:
                raise RuntimeError("PlainTokenClassifier loss mask is empty")
            loss = BCEWithLogitsLoss()(logits[mask].float(), labels[mask].float())
        return TokenClassifierOutput(loss=loss, logits=logits)


class TokenClassifierWithUNet(nn.Module):
    """GENA/ModernGENA token backbone plus nucleotide-resolution UNET head.

    This is the non-RMT/non-AMT UNET variant. BPE hidden states are expanded to
    nucleotides through `embedding_repeater`, concatenated with nucleotide token embeddings,
    and processed by the same 1D UNET head used by the RMT repeater path.
    Batch size is intentionally restricted to 1.
    """

    def __init__(self, backbone_path: str, backbone_kind: str, num_labels: int, trust_remote_code: bool = True, nucleotide_vocab_size: int = 1000, unet_cycles: int = 1, unet_channels=None, allow_unsafe_torch_load: bool = True):
        super().__init__()
        self.hidden_backbone = HiddenStateBackbone(backbone_path, backbone_kind, trust_remote_code=trust_remote_code, modernbert_num_labels=num_labels, allow_unsafe_torch_load=allow_unsafe_torch_load)
        self.hidden_size = self.hidden_backbone.hidden_size
        self.num_labels = int(num_labels)
        self.unet_cycles = int(unet_cycles)
        if self.unet_cycles < 1:
            raise RuntimeError("unet_cycles must be >= 1")
        self.nucleotide_embedding = nn.Embedding(int(nucleotide_vocab_size), self.hidden_size)
        self.unet_input_dim = self.hidden_size * 2
        self.unet = UNET1DSegmentationHead(self.unet_input_dim, self.unet_input_dim, output_channels_list=unet_channels)
        self.activation_fn = nn.SiLU()
        self.fc = nn.Linear(self.unet_input_dim, self.num_labels)
        logger.info("[TokenClassifierWithUNet] backbone=%s hidden=%d unet_input_dim=%d labels=%d cycles=%d", backbone_kind, self.hidden_size, self.unet_input_dim, self.num_labels, self.unet_cycles)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, labels_mask=None, embedding_repeater=None, letter_level_tokens=None, letter_level_labels=None, letter_level_labels_mask=None, pos_weight=None, **kwargs):
        if input_ids.shape[0] != 1:
            raise RuntimeError("TokenClassifierWithUNet requires batch size 1")
        if embedding_repeater is None or letter_level_tokens is None or letter_level_labels_mask is None:
            raise RuntimeError("UNET model requires embedding_repeater, letter_level_tokens, and letter_level_labels_mask")
        hidden = self.hidden_backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
        valid_token_mask = labels_mask[0].bool() if labels_mask is not None else attention_mask[0].bool()
        token_hidden = hidden[0, valid_token_mask, :].unsqueeze(0)
        raw_lmask = letter_level_labels_mask[0].bool()
        repeater_full = embedding_repeater[0].long()
        lmask = raw_lmask & (repeater_full >= 0)
        dropped = int((raw_lmask & (repeater_full < 0)).sum().item())
        if dropped:
            logger.info("[TokenClassifierWithUNet] dropped %d nucleotide positions not covered by retained BPE tokens", dropped)
        repeater = repeater_full[lmask]
        if repeater.numel() == 0:
            raise RuntimeError("UNET repeater is empty after removing uncovered BPE positions")
        if repeater.max().item() >= token_hidden.shape[1]:
            raise RuntimeError(f"UNET repeater max {repeater.max().item()} incompatible with token length {token_hidden.shape[1]}")
        nt_emb = self.nucleotide_embedding(letter_level_tokens[0][lmask].unsqueeze(0))
        x = torch.cat((nt_emb, token_hidden[:, repeater, :]), dim=-1)
        target = letter_level_labels[0][lmask].unsqueeze(0) if letter_level_labels is not None else None
        weight = pos_weight[0, 0, :].to(x.device).float() if pos_weight is not None else None
        loss_fct = BCEWithLogitsLoss(pos_weight=weight)
        loss = 0.0
        logits = None
        for _ in range(self.unet_cycles):
            z = self.activation_fn(self.unet(x.transpose(1, 2))).transpose(1, 2)
            logits = self.fc(z)
            if target is not None:
                loss = loss + loss_fct(logits.float(), target.float())
            x = x + z
        full_logits = logits.new_zeros((1, letter_level_tokens.shape[1], self.num_labels))
        full_logits[:, lmask, :] = logits
        return TokenClassifierOutput(loss=(loss / self.unet_cycles if target is not None else None), logits=full_logits)


class TranscriptTypeClassifier(nn.Module):
    """GENA/ModernGENA transcript-type classifier without RMT/AMT/UNET."""

    def __init__(self, backbone_path: str, backbone_kind: str, trust_remote_code: bool = True, allow_unsafe_torch_load: bool = True):
        super().__init__()
        self.hidden_backbone = HiddenStateBackbone(backbone_path, backbone_kind, trust_remote_code=trust_remote_code, modernbert_num_labels=1, allow_unsafe_torch_load=allow_unsafe_torch_load)
        self.hidden_size = self.hidden_backbone.hidden_size
        self.classifier = nn.Linear(self.hidden_size, 1)
        logger.info("[TranscriptTypeClassifier] backbone=%s hidden=%d labels=1", backbone_kind, self.hidden_size)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, transcript_type=None, **kwargs):
        hidden = self.hidden_backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
        idx = (attention_mask.long() * torch.arange(attention_mask.shape[1], device=attention_mask.device)).max(dim=1).values
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]
        logits = self.classifier(pooled)
        loss = BCEWithLogitsLoss()(logits, transcript_type.float()) if transcript_type is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)
