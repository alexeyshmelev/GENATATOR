from __future__ import annotations

import importlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import TokenClassifierOutput

from .backbones import HiddenStateBackbone
from .config import local_or_remote
from .unet import UNET1DSegmentationHead
from .torch_compat import allow_transformers_torch_load_on_legacy_torch

logger = logging.getLogger(__name__)


class _BackboneHiddenForMemory(nn.Module):
    def __init__(self, hidden_backbone: HiddenStateBackbone):
        super().__init__()
        self.hidden_backbone = hidden_backbone
        self.config = hidden_backbone.config

    def forward(self, *args, **kwargs):
        kwargs.pop("use_cache", None)
        kwargs.pop("past_key_values", None)
        return self.hidden_backbone(*args, **kwargs)


class AMTTokenClassifier(nn.Module):
    """AMT memory wrapper for GENA/ModernGENA only.

    Active class choice: the remote AMT implementation is expected to expose
    `AssociativeMemoryCell` and `AssociativeRecurrentWrapper`. The parameter is
    named `amt_repo_id` throughout this repository. No parameters are frozen.
    """

    def __init__(self, backbone_path: str, backbone_kind: str, num_labels: int, trust_remote_code: bool = True, amt_repo_id: str = "irodkin/armt-neox-tiny", use_unet: bool = False, nucleotide_vocab_size: int = 1000, unet_cycles: int = 1, unet_channels=None, allow_unsafe_torch_load: bool = True, **amt_kwargs):
        super().__init__()
        if backbone_kind not in {"gena", "moderngena"}:
            raise RuntimeError(f"AMT is allowed only for GENA/ModernGENA, got backbone_kind={backbone_kind}")
        self.hidden_backbone = HiddenStateBackbone(backbone_path, backbone_kind=backbone_kind, trust_remote_code=trust_remote_code, modernbert_num_labels=num_labels, allow_unsafe_torch_load=allow_unsafe_torch_load)
        self.hidden_size = self.hidden_backbone.hidden_size
        self.num_labels = int(num_labels)
        self.use_unet = bool(use_unet)

        allow_transformers_torch_load_on_legacy_torch(allow_unsafe_torch_load, context=f"AMT:{amt_repo_id}")
        loaded = AutoModelForCausalLM.from_pretrained(local_or_remote(amt_repo_id), trust_remote_code=True)
        amt_mod = importlib.import_module(loaded.__class__.__module__)
        AssociativeMemoryCell = getattr(amt_mod, "AssociativeMemoryCell")
        AssociativeRecurrentWrapper = getattr(amt_mod, "AssociativeRecurrentWrapper")
        layers_attr = amt_kwargs.pop("layers_attr", "hidden_backbone.encoder.encoder.layer" if backbone_kind == "gena" else "hidden_backbone.encoder.layers")
        base_model = _BackboneHiddenForMemory(self.hidden_backbone)
        logger.info("[AMT] repo=%s layers_attr=%s hidden=%d use_unet=%s", amt_repo_id, layers_attr, self.hidden_size, self.use_unet)
        memory_cell = AssociativeMemoryCell(
            base_model=base_model,
            num_mem_tokens=int(amt_kwargs.pop("num_mem_tokens", 16)),
            d_mem=int(amt_kwargs.pop("d_mem", 32)),
            layers_attr=layers_attr,
            wrap_pos=bool(amt_kwargs.pop("wrap_pos", False)),
            correction=bool(amt_kwargs.pop("correction", True)),
            n_heads=int(amt_kwargs.pop("n_heads", 1)),
            use_denom=bool(amt_kwargs.pop("use_denom", True)),
            gating=bool(amt_kwargs.pop("gating", False)),
            freeze_mem=False,
            act_on=bool(amt_kwargs.pop("act_on", False)),
            max_hop=int(amt_kwargs.pop("max_hop", 4)),
            act_type=amt_kwargs.pop("act_type", "associative"),
            constant_depth=bool(amt_kwargs.pop("constant_depth", False)),
            act_format=amt_kwargs.pop("act_format", "linear"),
            noisy_halting=bool(amt_kwargs.pop("noisy_halting", False)),
            attend_to_previous_input=bool(amt_kwargs.get("attend_to_previous_input", False)),
            use_sink=bool(amt_kwargs.pop("use_sink", False)),
        )
        self.amt = AssociativeRecurrentWrapper(
            memory_cell,
            segment_size=int(amt_kwargs.pop("segment_size", 128)),
            segment_alignment=amt_kwargs.pop("segment_alignment", "left"),
            sliding_window=bool(amt_kwargs.pop("sliding_window", False)),
            attend_to_previous_input=bool(amt_kwargs.pop("attend_to_previous_input", False)),
            act_on=bool(amt_kwargs.pop("act_on", False)),
            time_penalty=float(amt_kwargs.pop("time_penalty", 0.0)),
        )
        if amt_kwargs:
            raise RuntimeError(f"Unused AMT parameters: {sorted(amt_kwargs.keys())}")

        if self.use_unet:
            self.unet_cycles = int(unet_cycles)
            if self.unet_cycles < 1:
                raise RuntimeError("unet_cycles must be >= 1")
            self.nucleotide_embedding = nn.Embedding(int(nucleotide_vocab_size), self.hidden_size)
            self.unet_input_dim = self.hidden_size * 2
            self.unet = UNET1DSegmentationHead(self.unet_input_dim, self.unet_input_dim, output_channels_list=unet_channels)
            self.activation_fn = nn.SiLU()
            self.fc = nn.Linear(self.unet_input_dim, self.num_labels)
            logger.info("[AMTTokenClassifier] UNET hidden=%d input=%d labels=%d cycles=%d", self.hidden_size, self.unet_input_dim, self.num_labels, self.unet_cycles)
        else:
            self.classifier = nn.Linear(self.hidden_size, self.num_labels)
            logger.info("[AMTTokenClassifier] plain hidden=%d labels=%d", self.hidden_size, self.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, labels_mask=None, pos_weight=None, embedding_repeater=None, letter_level_tokens=None, letter_level_labels=None, letter_level_labels_mask=None, **kwargs):
        out = self.amt(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.logits
        if hidden.shape[-1] != self.hidden_size:
            raise RuntimeError(f"AMT hidden size mismatch: expected {self.hidden_size}, got {hidden.shape[-1]}")
        if not self.use_unet:
            logits = self.classifier(hidden)
            mask = labels_mask if labels_mask is not None else attention_mask.bool()
            loss = BCEWithLogitsLoss()(logits[mask.bool()].float(), labels[mask.bool()].float()) if labels is not None else None
            return TokenClassifierOutput(loss=loss, logits=logits)
        if input_ids.shape[0] != 1:
            raise RuntimeError("AMT+UNET requires batch size 1")
        if embedding_repeater is None or letter_level_tokens is None or letter_level_labels_mask is None:
            raise RuntimeError("AMT+UNET requires embedding_repeater, letter_level_tokens, and letter_level_labels_mask")
        valid_token_mask = labels_mask[0].bool() if labels_mask is not None else attention_mask[0].bool()
        token_hidden = hidden[0, valid_token_mask, :].unsqueeze(0)
        lmask = letter_level_labels_mask[0].bool()
        repeater = embedding_repeater[0][lmask].long()
        if repeater.numel() == 0:
            raise RuntimeError("AMT repeater is empty")
        if repeater.min().item() < 0 or repeater.max().item() >= token_hidden.shape[1]:
            raise RuntimeError(f"AMT repeater range [{repeater.min().item()}, {repeater.max().item()}] incompatible with token length {token_hidden.shape[1]}")
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
        logits = F.pad(logits, (0, 0, 0, letter_level_tokens.shape[1] - logits.shape[1]))
        return TokenClassifierOutput(loss=(loss / self.unet_cycles if target is not None else None), logits=logits)
