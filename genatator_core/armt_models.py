from __future__ import annotations

import importlib
import types
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers import AutoModel, AutoModelForCausalLM, ModernBertForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput


class _HiddenAsLogits(nn.Module):
    def __init__(self, model: nn.Module, hidden_size: int):
        super().__init__()
        self.model = model
        self.config = getattr(model, "config", None)
        if self.config is not None:
            self.config.hidden_size = hidden_size

    def forward(self, *args, **kwargs):
        kwargs.pop("use_cache", None)
        kwargs.pop("past_key_values", None)
        kwargs["output_hidden_states"] = True
        out = self.model(*args, **kwargs)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out.logits
        return TokenClassifierOutput(logits=hidden, hidden_states=getattr(out, "hidden_states", None))


class ARMTTokenClassifier(nn.Module):
    """ARMT token classifier adapted from the supplied ARMT_AnnotationModel.

    Strong starting point from the provided code:
    - imports `AssociativeMemoryCell` and `AssociativeRecurrentWrapper` from the ARMT repo;
    - wraps one loaded backbone path;
    - removes cache arguments before recurrent execution.

    Clear change for this cleaned repo: the wrapper exposes a plain masked BCE token-classifier
    interface used by the unified Trainer and JSON configs.
    """
    def __init__(self, backbone_path: str, num_labels: int, armt_repo_id: str = "irodkin/armt-neox-tiny", trust_remote_code: bool = True, **armt_kwargs):
        super().__init__()
        loaded = AutoModelForCausalLM.from_pretrained(armt_repo_id, trust_remote_code=True)
        armt_mod = importlib.import_module(loaded.__class__.__module__)
        AssociativeMemoryCell = getattr(armt_mod, "AssociativeMemoryCell")
        AssociativeRecurrentWrapper = getattr(armt_mod, "AssociativeRecurrentWrapper")

        if "moderngena" in backbone_path.lower() or "modern" in backbone_path.lower():
            base = ModernBertForTokenClassification.from_pretrained(backbone_path, num_labels=1, trust_remote_code=trust_remote_code)
            hidden_size = base.config.hidden_size
            base.classifier = nn.Identity()
            base_model = _HiddenAsLogits(base.model if hasattr(base, "model") else base, hidden_size)
            layers_attr = armt_kwargs.pop("layers_attr", "model.layers")
        else:
            base = AutoModel.from_pretrained(backbone_path, trust_remote_code=trust_remote_code)
            hidden_size = base.config.hidden_size
            base_model = _HiddenAsLogits(base, hidden_size)
            layers_attr = armt_kwargs.pop("layers_attr", "model.encoder.layer")

        memory_cell = AssociativeMemoryCell(
            base_model=base_model,
            layers_attr=layers_attr,
            num_mem_tokens=int(armt_kwargs.pop("num_mem_tokens", 16)),
            d_mem=int(armt_kwargs.pop("d_mem", 32)),
            wrap_pos=bool(armt_kwargs.pop("wrap_pos", False)),
            correction=bool(armt_kwargs.pop("correction", True)),
            n_heads=int(armt_kwargs.pop("n_heads", 1)),
            use_denom=bool(armt_kwargs.pop("use_denom", True)),
            gating=bool(armt_kwargs.pop("gating", False)),
            freeze_mem=bool(armt_kwargs.pop("freeze_mem", False)),
            act_on=bool(armt_kwargs.pop("act_on", False)),
            max_hop=int(armt_kwargs.pop("max_hop", 4)),
            act_type=armt_kwargs.pop("act_type", "associative"),
            constant_depth=bool(armt_kwargs.pop("constant_depth", False)),
            act_format=armt_kwargs.pop("act_format", "linear"),
            noisy_halting=bool(armt_kwargs.pop("noisy_halting", False)),
            attend_to_previous_input=bool(armt_kwargs.get("attend_to_previous_input", False)),
            use_sink=bool(armt_kwargs.pop("use_sink", False)),
        )
        self.armt = AssociativeRecurrentWrapper(
            memory_cell,
            segment_size=int(armt_kwargs.pop("segment_size", 128)),
            segment_alignment=armt_kwargs.pop("segment_alignment", "left"),
            sliding_window=bool(armt_kwargs.pop("sliding_window", False)),
            attend_to_previous_input=bool(armt_kwargs.pop("attend_to_previous_input", False)),
            act_on=bool(armt_kwargs.pop("act_on", False)),
            time_penalty=float(armt_kwargs.pop("time_penalty", 0.0)),
        )
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, labels_mask=None, **kwargs):
        out = self.armt(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.logits
        logits = self.classifier(hidden)
        loss = None
        if labels is not None:
            if labels_mask is None:
                labels_mask = attention_mask.bool()
            loss = BCEWithLogitsLoss()(logits[labels_mask], labels[labels_mask].float())
        return TokenClassifierOutput(loss=loss, logits=logits)
