from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput

from .backbones import HiddenStateBackbone
from .unet import DEFAULT_UNET_CHUNK_SIZE, UNET1DSegmentationHead, run_samplewise_chunked_unet

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
    The backbone is batched normally.  Its outputs are expanded and passed
    through UNET one sample at a time, in exact-length nucleotide chunks.
    """

    def __init__(self, backbone_path: str, backbone_kind: str, num_labels: int, trust_remote_code: bool = True, nucleotide_vocab_size: int = 1000, unet_cycles: int = 1, unet_channels=None, unet_chunk_size: int = DEFAULT_UNET_CHUNK_SIZE, allow_unsafe_torch_load: bool = True):
        super().__init__()
        self.hidden_backbone = HiddenStateBackbone(backbone_path, backbone_kind, trust_remote_code=trust_remote_code, modernbert_num_labels=num_labels, allow_unsafe_torch_load=allow_unsafe_torch_load)
        self.hidden_size = self.hidden_backbone.hidden_size
        self.num_labels = int(num_labels)
        self.unet_cycles = int(unet_cycles)
        if self.unet_cycles < 1:
            raise RuntimeError("unet_cycles must be >= 1")
        self.unet_chunk_size = int(unet_chunk_size)
        if self.unet_chunk_size <= 0:
            raise RuntimeError("unet_chunk_size must be positive")
        self.nucleotide_embedding = nn.Embedding(int(nucleotide_vocab_size), self.hidden_size)
        self.unet_input_dim = self.hidden_size * 2
        self.unet = UNET1DSegmentationHead(self.unet_input_dim, self.unet_input_dim, output_channels_list=unet_channels)
        self.activation_fn = nn.SiLU()
        self.fc = nn.Linear(self.unet_input_dim, self.num_labels)
        logger.info("[TokenClassifierWithUNet] backbone=%s hidden=%d unet_input_dim=%d labels=%d cycles=%d chunk=%d", backbone_kind, self.hidden_size, self.unet_input_dim, self.num_labels, self.unet_cycles, self.unet_chunk_size)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, labels_mask=None, embedding_repeater=None, letter_level_tokens=None, letter_level_labels=None, letter_level_labels_mask=None, letter_level_attention_mask=None, pos_weight=None, **kwargs):
        if labels_mask is None:
            raise RuntimeError("UNET model requires labels_mask to identify retained BPE content tokens")
        hidden = self.hidden_backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
        loss, logits = run_samplewise_chunked_unet(
            token_hidden=hidden,
            token_content_mask=labels_mask,
            embedding_repeater=embedding_repeater,
            letter_level_tokens=letter_level_tokens,
            letter_level_attention_mask=letter_level_attention_mask,
            letter_level_labels=letter_level_labels,
            letter_level_labels_mask=letter_level_labels_mask,
            pos_weight=pos_weight,
            nucleotide_embedding=self.nucleotide_embedding,
            unet=self.unet,
            activation_fn=self.activation_fn,
            classifier=self.fc,
            cycles=self.unet_cycles,
            chunk_size=self.unet_chunk_size,
            context="TokenClassifierWithUNet",
        )
        return TokenClassifierOutput(loss=loss, logits=logits)


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
