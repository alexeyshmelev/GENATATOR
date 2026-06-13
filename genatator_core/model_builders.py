from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from safetensors.torch import load_file as safe_load_file
from transformers import AutoConfig, AutoModel

from .armt_models import ARMTTokenClassifier
from .backbones import BackboneAsLetterLevelTokenClassification
from .config import local_or_remote
from .finding_models import AutoBackboneTokenClassifier, ModernBertForGenatatorFinding
from .legacy_caduceus import (
    CADUSEUS_for_token_classification,
    CADUSEUS_for_token_classification_middle_loss,
    CADUSEUS_for_token_classification_transcript_type,
)
from .transcript_models import ModernBertForTranscriptType
from .utils import get_class


def build_model(cfg: Dict[str, Any], task: str):
    model_cfg = cfg["model"]
    family = model_cfg["family"]
    backbone_path = local_or_remote(model_cfg["backbone_path"])
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))

    if family == "moderngena_token":
        if task.startswith("finding"):
            # Chosen class: ModernBertForGenatatorFinding, internally loads transformers.ModernBertForTokenClassification.
            model = ModernBertForGenatatorFinding(backbone_path, num_labels=int(model_cfg["num_labels"]), dropout=float(model_cfg.get("classifier_dropout", 0.0)), trust_remote_code=trust_remote_code)
        elif task == "transcript_type":
            # Chosen class: ModernBertForTranscriptType, internally loads transformers.ModernBertForTokenClassification.
            model = ModernBertForTranscriptType(backbone_path, trust_remote_code=trust_remote_code)
        else:
            backbone = AutoModel.from_pretrained(backbone_path, trust_remote_code=trust_remote_code)
            model = AutoBackboneTokenClassifier(backbone, num_labels=int(model_cfg["num_labels"]))
    elif family == "plain_token":
        backbone = AutoModel.from_pretrained(backbone_path, trust_remote_code=trust_remote_code)
        model = AutoBackboneTokenClassifier(backbone, num_labels=int(model_cfg["num_labels"]))
    elif family == "caduceus":
        config = AutoConfig.from_pretrained(backbone_path, trust_remote_code=trust_remote_code)
        if "bidirectional_weight_tie" in model_cfg:
            config.bidirectional_weight_tie = bool(model_cfg["bidirectional_weight_tie"])
        backbone = AutoModel.from_pretrained(backbone_path, trust_remote_code=trust_remote_code, config=config)
        class_name = model_cfg.get("class_name")
        if class_name:
            cls = get_class(class_name)
            model = cls(backbone)
        elif task == "transcript_type":
            # Chosen class: CADUSEUS_for_token_classification_transcript_type from your provided code.
            model = CADUSEUS_for_token_classification_transcript_type(backbone)
        elif model_cfg.get("middle_loss", True):
            # Chosen class: CADUSEUS_for_token_classification_middle_loss from your provided code.
            model = CADUSEUS_for_token_classification_middle_loss(backbone)
        else:
            model = CADUSEUS_for_token_classification(backbone)
    elif family == "gena_rmt":
        # Chosen class is configurable. Default large class is the provided cycles=3 repeater.
        base = BackboneAsLetterLevelTokenClassification(backbone_path, trust_remote_code=trust_remote_code)
        cls_path = model_cfg.get("class_name", "genatator_core.legacy_rmt:RMTEncoderForLetterLevelTokenClassificationUNETsegmentedRepeaterLargeCycles3")
        cls = get_class(cls_path)
        model = cls(base, tokenizer=cfg["_tokenizer"], **model_cfg.get("rmt", {}))
    elif family == "armt":
        # Chosen class: ARMTTokenClassifier, adapted from the ARMT_AnnotationModel you provided.
        model = ARMTTokenClassifier(
            backbone_path=backbone_path,
            num_labels=int(model_cfg["num_labels"]),
            trust_remote_code=trust_remote_code,
            **model_cfg.get("armt", {}),
        )
    else:
        raise ValueError(f"Unknown model family: {family}")

    if not bool(model_cfg.get("backbone_trainable", True)):
        freeze_backbone(model)
    checkpoint = model_cfg.get("checkpoint_path")
    if checkpoint:
        load_finetuned_weights(model, checkpoint)
    return model


def freeze_backbone(model) -> None:
    for name, module in model.named_children():
        if name in {"model", "backbone", "base_model", "caduseus_model", "bert", "armt"}:
            for p in module.parameters():
                p.requires_grad = False


def load_finetuned_weights(model, checkpoint_path: str) -> None:
    p = Path(checkpoint_path).expanduser()
    if p.is_dir():
        if (p / "model.safetensors").exists():
            state = safe_load_file(str(p / "model.safetensors"))
        elif (p / "pytorch_model.bin").exists():
            state = torch.load(p / "pytorch_model.bin", map_location="cpu")
        else:
            state = torch.load(p, map_location="cpu")
    elif p.suffix == ".safetensors":
        state = safe_load_file(str(p))
    else:
        state = torch.load(p, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    clean = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(clean, strict=False)
