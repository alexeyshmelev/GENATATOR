from dataclasses import dataclass
from typing import Any, Dict, Optional
import importlib
import types

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, ModernBertForTokenClassification

from .losses import masked_bce_with_logits, binary_sequence_loss
from .unet import UNet1D


@dataclass
class ModelOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor
    extra: Optional[Dict[str, torch.Tensor]] = None

    def __getitem__(self, key):
        if key == "loss":
            return self.loss
        if key == "logits":
            return self.logits
        if self.extra is not None and key in self.extra:
            return self.extra[key]
        raise KeyError(key)


def last_hidden(output: Any) -> torch.Tensor:
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state
    if hasattr(output, "logits") and output.logits is not None and output.logits.dim() == 3:
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    raise RuntimeError("Backbone output does not contain hidden states")


class RMTBackbone(nn.Module):
    def __init__(self, base: nn.Module, cfg: Dict[str, Any]):
        super().__init__()
        self.base = base
        self.input_embeddings = base.get_input_embeddings()
        self.hidden_size = self.input_embeddings.embedding_dim
        self.num_mem_tokens = int(cfg.get("num_mem_tokens", 10))
        self.segment_size = int(cfg.get("segment_size", 512))
        self.max_segments = int(cfg.get("max_segments", 10000))
        self.bptt_depth = int(cfg.get("bptt_depth", -1))
        self.cls_token_id = int(cfg["cls_token_id"])
        self.sep_token_id = int(cfg["sep_token_id"])
        self.pad_token_id = int(cfg["pad_token_id"])
        self.memory = nn.Parameter(torch.empty(self.num_mem_tokens, self.hidden_size))
        nn.init.normal_(self.memory, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        bsz, total_len = input_ids.shape
        out = input_ids.new_zeros((bsz, total_len, self.hidden_size), dtype=torch.float32).to(self.memory.device)
        for b in range(bsz):
            valid = input_ids[b][attention_mask[b].bool()]
            valid = valid[: self.segment_size * self.max_segments]
            mem = self.memory.unsqueeze(0)
            cursor = 0
            hidden_segments = []
            nseg = max(1, (len(valid) + self.segment_size - 1) // self.segment_size)
            for seg_i in range(nseg):
                seg = valid[cursor : cursor + self.segment_size]
                cursor += len(seg)
                if self.bptt_depth > -1 and nseg - seg_i > self.bptt_depth:
                    mem = mem.detach()
                cls = input_ids.new_tensor([[self.cls_token_id]])
                sep = input_ids.new_tensor([[self.sep_token_id]])
                seg = seg.unsqueeze(0)
                ids_for_special = torch.cat([cls, sep, seg[:, :1], sep], dim=1)
                emb = self.input_embeddings(ids_for_special)
                inputs_embeds = torch.cat([emb[:, 0:1], mem, emb[:, 1:2], self.input_embeddings(seg), emb[:, -1:]], dim=1)
                mask = torch.ones(inputs_embeds.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device)
                base_out = self.base(inputs_embeds=inputs_embeds, attention_mask=mask, output_hidden_states=False)
                h = last_hidden(base_out)
                mem = h[:, 1 : 1 + self.num_mem_tokens]
                hs = h[:, 1 + self.num_mem_tokens + 1 : 1 + self.num_mem_tokens + 1 + seg.shape[1]]
                hidden_segments.append(hs.squeeze(0))
            hcat = torch.cat(hidden_segments, dim=0)
            out[b, : hcat.shape[0]] = hcat
        return out


class ARMTBackbone(nn.Module):
    def __init__(self, base_name: str, cfg: Dict[str, Any], local_files_only: bool = False):
        super().__init__()
        armt_repo_id = cfg.get("armt_repo_id", "irodkin/armt-neox-tiny")
        loaded = AutoModelForCausalLM.from_pretrained(armt_repo_id, trust_remote_code=True, local_files_only=local_files_only)
        armt_mod = importlib.import_module(loaded.__class__.__module__)
        AssociativeMemoryCell = getattr(armt_mod, "AssociativeMemoryCell")
        AssociativeRecurrentWrapper = getattr(armt_mod, "AssociativeRecurrentWrapper")
        base = AutoModel.from_pretrained(base_name, trust_remote_code=True, local_files_only=local_files_only)
        self.config = base.config
        cell = AssociativeMemoryCell(
            base_model=base,
            num_mem_tokens=cfg.get("num_mem_tokens", 16),
            d_mem=cfg.get("d_mem", 32),
            layers_attr=cfg.get("layers_attr", "bert.encoder.layer"),
            wrap_pos=cfg.get("wrap_pos", False),
            correction=cfg.get("correction", True),
            n_heads=cfg.get("n_heads", 1),
            use_denom=cfg.get("use_denom", True),
            gating=cfg.get("gating", False),
            freeze_mem=cfg.get("freeze_mem", False),
            act_on=cfg.get("act_on", False),
            max_hop=cfg.get("max_hop", 4),
            act_type=cfg.get("act_type", "associative"),
            constant_depth=cfg.get("constant_depth", False),
            act_format=cfg.get("act_format", "linear"),
            noisy_halting=cfg.get("noisy_halting", False),
            attend_to_previous_input=cfg.get("attend_to_previous_input", False),
            use_sink=cfg.get("use_sink", False),
        )
        self.armt = AssociativeRecurrentWrapper(
            cell,
            segment_size=cfg.get("segment_size", 512),
            segment_alignment=cfg.get("segment_alignment", "left"),
            sliding_window=cfg.get("sliding_window", False),
            attend_to_previous_input=cfg.get("attend_to_previous_input", False),
            act_on=cfg.get("act_on", False),
            time_penalty=cfg.get("time_penalty", 0.0),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.armt(input_ids=input_ids, attention_mask=attention_mask)
        return last_hidden(out)


class LongBackbone(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        model_name = cfg["pretrained_model_name_or_path"]
        local_files_only = cfg.get("local_files_only", False)
        adapter = cfg.get("adapter", {"type": "none"})
        if adapter.get("type", "none") == "armt":
            self.encoder = ARMTBackbone(model_name, adapter, local_files_only=local_files_only)
            self.hidden_size = self.encoder.config.hidden_size
            return
        base_config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=cfg.get("trust_remote_code", True),
            local_files_only=local_files_only,
        )
        for k, v in cfg.get("config_overrides", {}).items():
            setattr(base_config, k, v)
        base = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=cfg.get("trust_remote_code", True),
            config=base_config,
            local_files_only=local_files_only,
        )
        self.hidden_size = base_config.hidden_size if hasattr(base_config, "hidden_size") else base.get_input_embeddings().embedding_dim
        if adapter.get("type", "none") == "rmt":
            self.encoder = RMTBackbone(base, adapter)
        else:
            self.encoder = base

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if isinstance(self.encoder, (RMTBackbone, ARMTBackbone)):
            return self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return last_hidden(self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False))


class ModernBertTokenBCE(nn.Module):
    def __init__(self, cfg: Dict[str, Any], num_labels: int):
        super().__init__()
        self.model = ModernBertForTokenClassification.from_pretrained(
            cfg["pretrained_model_name_or_path"],
            num_labels=num_labels,
            trust_remote_code=cfg.get("trust_remote_code", True),
            local_files_only=cfg.get("local_files_only", False),
            ignore_mismatched_sizes=True,
        )
        self.pos_weight = cfg.get("pos_weight")
        if self.pos_weight is not None:
            self.register_buffer("pos_weight_tensor", torch.tensor(self.pos_weight, dtype=torch.float32))
        else:
            self.pos_weight_tensor = None

    def forward(self, input_ids, attention_mask, labels=None, labels_mask=None, **batch):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        loss = None
        if labels is not None:
            loss = masked_bce_with_logits(out.logits, labels, labels_mask, self.pos_weight_tensor)
        return ModelOutput(loss=loss, logits=out.logits)


class TokenClassifier(nn.Module):
    def __init__(self, cfg: Dict[str, Any], num_labels: int):
        super().__init__()
        self.backbone = LongBackbone(cfg)
        self.dropout = nn.Dropout(cfg.get("dropout", 0.0))
        self.classifier = nn.Linear(self.backbone.hidden_size, num_labels)
        pw = cfg.get("pos_weight")
        self.register_buffer("pos_weight_tensor", torch.tensor(pw, dtype=torch.float32) if pw is not None else None)

    def forward(self, input_ids, attention_mask, labels=None, labels_mask=None, **batch):
        h = self.backbone(input_ids, attention_mask)
        logits = self.classifier(self.dropout(h))
        loss = None
        if labels is not None:
            loss = masked_bce_with_logits(logits, labels, labels_mask, self.pos_weight_tensor)
        return ModelOutput(loss=loss, logits=logits)


class BpeToNucleotideUNet(nn.Module):
    def __init__(self, cfg: Dict[str, Any], num_labels: int):
        super().__init__()
        self.backbone = LongBackbone(cfg)
        nt_dim = int(cfg.get("nucleotide_embedding_dim", self.backbone.hidden_size))
        self.nt_embedding = nn.Embedding(int(cfg.get("nucleotide_vocab_size", 5)), nt_dim)
        in_ch = self.backbone.hidden_size + nt_dim
        hidden_ch = int(cfg.get("unet_hidden_channels", in_ch))
        self.project = nn.Linear(in_ch, hidden_ch)
        self.unet = UNet1D(
            hidden_ch,
            hidden_ch,
            channels=tuple(cfg.get("unet_channels", [192, 384, 768])),
            layers=int(cfg.get("unet_layers", 2)),
        )
        self.classifier = nn.Linear(hidden_ch, num_labels)
        pw = cfg.get("pos_weight")
        self.register_buffer("pos_weight_tensor", torch.tensor(pw, dtype=torch.float32) if pw is not None else None)

    def forward(self, input_ids, attention_mask, token_to_nt, nt_ids, nt_labels=None, nt_labels_mask=None, **batch):
        token_h = self.backbone(input_ids, attention_mask)
        bsz, nt_len = token_to_nt.shape
        safe_index = token_to_nt.clamp(min=0)
        gathered = torch.gather(token_h, 1, safe_index.unsqueeze(-1).expand(bsz, nt_len, token_h.size(-1)))
        nt_emb = self.nt_embedding(nt_ids.clamp(min=0, max=self.nt_embedding.num_embeddings - 1))
        x = self.project(torch.cat([gathered, nt_emb], dim=-1)).transpose(1, 2)
        x = F.silu(self.unet(x)).transpose(1, 2)
        logits = self.classifier(x)
        loss = None
        if nt_labels is not None:
            loss = masked_bce_with_logits(logits, nt_labels, nt_labels_mask, self.pos_weight_tensor)
        return ModelOutput(loss=loss, logits=logits)


class TranscriptTypeClassifier(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.backbone = LongBackbone(cfg)
        self.dropout = nn.Dropout(cfg.get("dropout", 0.0))
        self.classifier = nn.Linear(self.backbone.hidden_size, 1)

    def forward(self, input_ids, attention_mask, transcript_type=None, **batch):
        h = self.backbone(input_ids, attention_mask)
        mask = attention_mask.bool().unsqueeze(-1)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if transcript_type is not None:
            loss = binary_sequence_loss(logits, transcript_type)
        return ModelOutput(loss=loss, logits=logits)


def build_model(cfg: Dict[str, Any], task: str) -> nn.Module:
    model_cfg = cfg["model"]
    num_labels = int(model_cfg.get("num_labels", len(cfg["task"].get("label_names", []))))
    family = model_cfg["family"]
    if task == "transcript_type":
        return TranscriptTypeClassifier(model_cfg)
    if family == "modernbert_token_classifier":
        return ModernBertTokenBCE(model_cfg, num_labels=num_labels)
    if model_cfg.get("label_mode") == "nucleotide_unet":
        return BpeToNucleotideUNet(model_cfg, num_labels=num_labels)
    return TokenClassifier(model_cfg, num_labels=num_labels)
