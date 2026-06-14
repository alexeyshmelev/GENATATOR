from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import torch
from safetensors.torch import load_file as safe_load_file
from transformers import AutoConfig, AutoModel

from .amt_models import AMTTokenClassifier
from .backbones import HiddenStateBackbone
from .config import local_or_remote
from .legacy_caduceus import CaduceusMiddleLossTokenClassifier, CaduceusTranscriptTypeMiddleLossClassifier
from .legacy_rmt import RMTEncoderForLetterLevelTokenClassificationUNETsegmentedRepeater
from .token_models import PlainTokenClassifier, TokenClassifierWithUNet, TranscriptTypeClassifier

logger = logging.getLogger(__name__)


def build_model(cfg: Dict[str, Any], task: str):
    model_cfg = cfg["model"]
    family = model_cfg["family"]
    backbone_kind = model_cfg.get("backbone_kind", family)
    backbone_path = local_or_remote(model_cfg["backbone_path"])
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
    num_labels = _num_labels_for_task(task)
    logger.info("[build_model] task=%s family=%s backbone_kind=%s backbone_path=%s num_labels=%d", task, family, backbone_kind, backbone_path, num_labels)

    if family == "caduceus":
        if backbone_kind != "caduceus":
            raise RuntimeError("family='caduceus' requires backbone_kind='caduceus'")
        config = AutoConfig.from_pretrained(backbone_path, trust_remote_code=trust_remote_code)
        config.bidirectional_weight_tie = bool(model_cfg.get("bidirectional_weight_tie", False))
        logger.info("[caduceus] loading AutoModel path=%s bidirectional_weight_tie=%s", backbone_path, config.bidirectional_weight_tie)
        backbone = AutoModel.from_pretrained(backbone_path, config=config, trust_remote_code=trust_remote_code)
        model = CaduceusTranscriptTypeMiddleLossClassifier(backbone) if task == "transcript_type" else CaduceusMiddleLossTokenClassifier(backbone, num_labels=num_labels)

    elif family == "plain":
        if backbone_kind not in {"gena", "moderngena"}:
            raise RuntimeError(f"plain family is for GENA/ModernGENA only, got backbone_kind={backbone_kind}")
        if task == "transcript_type":
            model = TranscriptTypeClassifier(backbone_path, backbone_kind, trust_remote_code=trust_remote_code)
        else:
            model = PlainTokenClassifier(backbone_path, backbone_kind, num_labels=num_labels, trust_remote_code=trust_remote_code)

    elif family == "unet":
        if backbone_kind not in {"gena", "moderngena"}:
            raise RuntimeError(f"UNET family is for GENA/ModernGENA only, got backbone_kind={backbone_kind}")
        model = TokenClassifierWithUNet(
            backbone_path,
            backbone_kind,
            num_labels=num_labels,
            trust_remote_code=trust_remote_code,
            nucleotide_vocab_size=int(model_cfg.get("nucleotide_vocab_size", 1000)),
            unet_cycles=int(model_cfg.get("unet_cycles", 1)),
            unet_channels=model_cfg.get("unet_channels"),
        )

    elif family == "rmt":
        if backbone_kind not in {"gena", "moderngena"}:
            raise RuntimeError(f"RMT is allowed only for GENA/ModernGENA, got backbone_kind={backbone_kind}")
        if "_tokenizer" not in cfg:
            raise RuntimeError("RMT build requires cfg['_tokenizer'] set by train/infer entrypoint")
        base_model = HiddenStateBackbone(backbone_path, backbone_kind, trust_remote_code=trust_remote_code, modernbert_num_labels=num_labels)
        rmt_kwargs = dict(model_cfg.get("rmt", {}))
        rmt_kwargs.update({
            "tokenizer": cfg["_tokenizer"],
            "num_labels": num_labels,
            "nucleotide_vocab_size": int(model_cfg.get("nucleotide_vocab_size", 1000)),
            "cycles": int(model_cfg.get("cycles", 3)),
            "unet_channels": model_cfg.get("unet_channels"),
        })
        model = RMTEncoderForLetterLevelTokenClassificationUNETsegmentedRepeater(base_model, **rmt_kwargs)

    elif family == "amt":
        if backbone_kind not in {"gena", "moderngena"}:
            raise RuntimeError(f"AMT is allowed only for GENA/ModernGENA, got backbone_kind={backbone_kind}")
        model = AMTTokenClassifier(
            backbone_path=backbone_path,
            backbone_kind=backbone_kind,
            num_labels=num_labels,
            trust_remote_code=trust_remote_code,
            use_unet=bool(model_cfg.get("use_unet", False)),
            nucleotide_vocab_size=int(model_cfg.get("nucleotide_vocab_size", 1000)),
            unet_cycles=int(model_cfg.get("unet_cycles", 1)),
            unet_channels=model_cfg.get("unet_channels"),
            **model_cfg.get("amt", {}),
        )
    else:
        raise RuntimeError(f"Unsupported model family: {family}")

    checkpoint = model_cfg.get("checkpoint_path")
    if checkpoint:
        load_finetuned_weights(model, checkpoint)
    _assert_all_trainable(model)
    _log_parameters(model)
    return model


def _num_labels_for_task(task: str) -> int:
    if task == "finding_edge":
        return 4
    if task == "finding_region":
        return 2
    if task == "segmentation":
        return 5
    if task == "transcript_type":
        return 1
    raise RuntimeError(task)


def load_finetuned_weights(model, checkpoint_path: str) -> None:
    p = Path(local_or_remote(checkpoint_path)).expanduser()
    logger.info("[checkpoint] loading finetuned checkpoint from %s", p)
    if p.is_dir():
        if (p / "model.safetensors").exists():
            state = safe_load_file(str(p / "model.safetensors"))
        elif (p / "pytorch_model.bin").exists():
            state = torch.load(p / "pytorch_model.bin", map_location="cpu")
        else:
            raise RuntimeError(f"Checkpoint directory has neither model.safetensors nor pytorch_model.bin: {p}")
    elif p.suffix == ".safetensors":
        state = safe_load_file(str(p))
    else:
        state = torch.load(p, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    clean = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(clean, strict=False)
    logger.info("[checkpoint] missing_keys=%d unexpected_keys=%d", len(missing), len(unexpected))
    if missing:
        logger.info("[checkpoint] missing=%s", missing)
    if unexpected:
        logger.info("[checkpoint] unexpected=%s", unexpected)


def _assert_all_trainable(model) -> None:
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    if frozen:
        raise RuntimeError(f"All parameters must be trainable, but frozen parameters were found: {frozen[:20]} total={len(frozen)}")


def _log_parameters(model) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("[parameters] total=%d trainable=%d frozen=%d", total, trainable, total - trainable)
