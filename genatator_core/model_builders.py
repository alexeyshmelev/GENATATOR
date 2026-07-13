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
from .legacy_caduceus import CaduceusMiddleLossTokenClassifier, CaduceusTranscriptTypeMiddleLossClassifier, infer_caduceus_hidden_size
from .legacy_rmt import RMTEncoderForLetterLevelTokenClassificationUNETsegmentedRepeater
from .token_models import PlainTokenClassifier, TokenClassifierWithUNet, TranscriptTypeClassifier
from .torch_compat import allow_transformers_torch_load_on_legacy_torch, trusted_torch_load
from .unet import DEFAULT_UNET_CHUNK_SIZE

logger = logging.getLogger(__name__)


def default_memory_segment_size(backbone_kind: str) -> int:
    if backbone_kind == "gena":
        return 512
    if backbone_kind == "moderngena":
        return 1024
    raise RuntimeError(f"Memory segment defaults are defined only for GENA/ModernGENA, got {backbone_kind!r}")


def model_uses_unet(model_cfg: Dict[str, Any]) -> bool:
    family = model_cfg.get("family")
    return family in {"unet", "rmt"} or (family == "amt" and bool(model_cfg.get("use_unet", False)))


def normalize_unet_chunk_size(model_cfg: Dict[str, Any]) -> int | None:
    """Materialize one uniform UNET chunk-size field in the model config."""

    if not model_uses_unet(model_cfg):
        return None
    configured = model_cfg.get("unet_chunk_size")
    legacy = model_cfg.get("rmt", {}).get("unet_sub_model_input_size") if model_cfg.get("family") == "rmt" else None
    if configured is not None and legacy is not None and int(configured) != int(legacy):
        raise RuntimeError(
            "Conflicting UNET chunk sizes: model.unet_chunk_size="
            f"{configured} and model.rmt.unet_sub_model_input_size={legacy}"
        )
    value = int(configured if configured is not None else (legacy if legacy is not None else DEFAULT_UNET_CHUNK_SIZE))
    if value <= 0:
        raise RuntimeError(f"model.unet_chunk_size must be positive, got {value}")
    model_cfg["unet_chunk_size"] = value
    return value


def _nucleotide_vocab_size(model_cfg: Dict[str, Any]) -> int:
    value = model_cfg.get("nucleotide_vocab_size")
    if value in (None, "", "auto"):
        raise RuntimeError("nucleotide_vocab_size was not inferred before build_model. Entry points must call prepare_nucleotide_tokenizer().")
    return int(value)


def build_model(cfg: Dict[str, Any], task: str):
    model_cfg = cfg["model"]
    family = model_cfg["family"]
    if task == "transcript_type" and family not in {"plain", "caduceus"}:
        raise RuntimeError(
            "Transcript-type classification is implemented only for family='plain' and family='caduceus'; "
            f"got family={family!r}"
        )
    unet_chunk_size = normalize_unet_chunk_size(model_cfg)
    backbone_kind = model_cfg.get("backbone_kind", family)
    backbone_path = local_or_remote(model_cfg["backbone_path"])
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
    allow_unsafe_torch_load = bool(model_cfg.get("allow_unsafe_torch_load_with_torch_lt_2_6", True))
    allow_transformers_torch_load_on_legacy_torch(allow_unsafe_torch_load, context=f"build_model:{family}:{backbone_path}")
    num_labels = _num_labels_for_task(task)
    logger.info("[build_model] task=%s family=%s backbone_kind=%s backbone_path=%s num_labels=%d", task, family, backbone_kind, backbone_path, num_labels)

    if family == "caduceus":
        if backbone_kind != "caduceus":
            raise RuntimeError("family='caduceus' requires backbone_kind='caduceus'")
        config = AutoConfig.from_pretrained(backbone_path, trust_remote_code=trust_remote_code)
        # GENATATOR intentionally trains Caduceus with untied bidirectional
        # projections. This is forced regardless of either the downloaded HF
        # config or a user-supplied JSON value.
        config.bidirectional_weight_tie = False
        model_cfg["bidirectional_weight_tie"] = False
        logger.info("[caduceus] loading AutoModel path=%s bidirectional_weight_tie=false (forced)", backbone_path)
        hidden_size = int(model_cfg.get("hidden_size") or infer_caduceus_hidden_size(config, backbone_path))
        if "hidden_size" in model_cfg:
            logger.info("[caduceus.shape] using explicit model.hidden_size=%d from config", hidden_size)
        backbone = AutoModel.from_pretrained(backbone_path, config=config, trust_remote_code=trust_remote_code)
        model = CaduceusTranscriptTypeMiddleLossClassifier(backbone, hidden_size=hidden_size) if task == "transcript_type" else CaduceusMiddleLossTokenClassifier(backbone, num_labels=num_labels, hidden_size=hidden_size)

    elif family == "plain":
        if backbone_kind not in {"gena", "moderngena"}:
            raise RuntimeError(f"plain family is for GENA/ModernGENA only, got backbone_kind={backbone_kind}")
        if task == "transcript_type":
            model = TranscriptTypeClassifier(backbone_path, backbone_kind, trust_remote_code=trust_remote_code, allow_unsafe_torch_load=allow_unsafe_torch_load)
        else:
            model = PlainTokenClassifier(backbone_path, backbone_kind, num_labels=num_labels, trust_remote_code=trust_remote_code, allow_unsafe_torch_load=allow_unsafe_torch_load)

    elif family == "unet":
        if backbone_kind not in {"gena", "moderngena"}:
            raise RuntimeError(f"UNET family is for GENA/ModernGENA only, got backbone_kind={backbone_kind}")
        model = TokenClassifierWithUNet(
            backbone_path,
            backbone_kind,
            num_labels=num_labels,
            trust_remote_code=trust_remote_code,
            nucleotide_vocab_size=_nucleotide_vocab_size(model_cfg),
            unet_cycles=int(model_cfg.get("unet_cycles", 1)),
            unet_channels=model_cfg.get("unet_channels"),
            unet_chunk_size=int(unet_chunk_size),
            allow_unsafe_torch_load=allow_unsafe_torch_load,
        )

    elif family == "rmt":
        if backbone_kind not in {"gena", "moderngena"}:
            raise RuntimeError(f"RMT is allowed only for GENA/ModernGENA, got backbone_kind={backbone_kind}")
        if "_tokenizer" not in cfg:
            raise RuntimeError("RMT build requires cfg['_tokenizer'] set by train/infer entrypoint")
        base_model = HiddenStateBackbone(backbone_path, backbone_kind, trust_remote_code=trust_remote_code, modernbert_num_labels=num_labels, allow_unsafe_torch_load=allow_unsafe_torch_load)
        rmt_kwargs = dict(model_cfg.get("rmt", {}))
        legacy_input_size = rmt_kwargs.pop("input_size", None)
        configured_segment_size = rmt_kwargs.get("segment_size")
        if legacy_input_size is not None and configured_segment_size is not None and int(legacy_input_size) != int(configured_segment_size):
            raise RuntimeError(
                "Conflicting RMT segment sizes: model.rmt.input_size="
                f"{legacy_input_size} and model.rmt.segment_size={configured_segment_size}"
            )
        rmt_kwargs["segment_size"] = int(
            configured_segment_size
            if configured_segment_size is not None
            else (legacy_input_size if legacy_input_size is not None else default_memory_segment_size(backbone_kind))
        )
        rmt_kwargs.setdefault("max_n_segments", 10000)
        rmt_kwargs.update({
            "tokenizer": cfg["_tokenizer"],
            "num_labels": num_labels,
            "nucleotide_vocab_size": _nucleotide_vocab_size(model_cfg),
            "cycles": int(model_cfg.get("cycles", 1)),
            "unet_channels": model_cfg.get("unet_channels"),
            "unet_chunk_size": int(unet_chunk_size),
        })
        model = RMTEncoderForLetterLevelTokenClassificationUNETsegmentedRepeater(base_model, **rmt_kwargs)

    elif family == "amt":
        if backbone_kind not in {"gena", "moderngena"}:
            raise RuntimeError(f"AMT is allowed only for GENA/ModernGENA, got backbone_kind={backbone_kind}")
        use_unet = bool(model_cfg.get("use_unet", False))
        amt_kwargs = dict(model_cfg.get("amt", {}))
        amt_kwargs.setdefault("segment_size", default_memory_segment_size(backbone_kind))
        model = AMTTokenClassifier(
            backbone_path=backbone_path,
            backbone_kind=backbone_kind,
            num_labels=num_labels,
            trust_remote_code=trust_remote_code,
            use_unet=use_unet,
            # This value is unused by plain AMT.  Do not require a nucleotide
            # tokenizer/vocabulary unless a UNET is actually present.
            nucleotide_vocab_size=_nucleotide_vocab_size(model_cfg) if use_unet else 1,
            unet_cycles=int(model_cfg.get("unet_cycles", 1)),
            unet_channels=model_cfg.get("unet_channels"),
            unet_chunk_size=int(unet_chunk_size) if use_unet else DEFAULT_UNET_CHUNK_SIZE,
            allow_unsafe_torch_load=allow_unsafe_torch_load,
            **amt_kwargs,
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
            state = trusted_torch_load(p / "pytorch_model.bin", map_location="cpu")
        else:
            raise RuntimeError(f"Checkpoint directory has neither model.safetensors nor pytorch_model.bin: {p}")
    elif p.suffix == ".safetensors":
        state = safe_load_file(str(p))
    else:
        state = trusted_torch_load(p, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    clean = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(clean, strict=False)
    logger.info("[checkpoint] missing_keys=%d unexpected_keys=%d", len(missing), len(unexpected))
    compatibility_buffer_suffixes = ("position_ids", "token_type_ids")

    def is_compatibility_buffer(name: str) -> bool:
        return any(name == suffix or name.endswith(f".{suffix}") for suffix in compatibility_buffer_suffixes)

    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing_trainable = [name for name in missing if name in trainable_names]
    disallowed_missing = [name for name in missing if not is_compatibility_buffer(name)]
    disallowed_unexpected = [name for name in unexpected if not is_compatibility_buffer(name)]
    if missing_trainable or disallowed_missing or disallowed_unexpected:
        raise RuntimeError(
            "Finetuned checkpoint is incompatible with the requested model; refusing a partial load. "
            f"missing_trainable={missing_trainable[:20]} (total={len(missing_trainable)}), "
            f"missing_keys={disallowed_missing[:20]} (total={len(disallowed_missing)}), "
            f"unexpected_keys={disallowed_unexpected[:20]} (total={len(disallowed_unexpected)}). "
            "Only non-trainable position_ids/token_type_ids compatibility buffers may differ."
        )
    allowed_missing = [name for name in missing if is_compatibility_buffer(name)]
    allowed_unexpected = [name for name in unexpected if is_compatibility_buffer(name)]
    if allowed_missing or allowed_unexpected:
        logger.info(
            "[checkpoint] allowed compatibility-buffer differences missing=%s unexpected=%s",
            allowed_missing,
            allowed_unexpected,
        )


def _assert_all_trainable(model) -> None:
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    if frozen:
        raise RuntimeError(f"All parameters must be trainable, but frozen parameters were found: {frozen[:20]} total={len(frozen)}")


def _log_parameters(model) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("[parameters] total=%d trainable=%d frozen=%d", total, trainable, total - trainable)
