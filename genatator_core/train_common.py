from __future__ import annotations

import logging
from typing import Any, Dict
import inspect

from transformers import Trainer, TrainingArguments

from .config import load_json, save_json
from .data import GenatatorCollator, GenatatorDataset, make_tokenizer
from .metrics_training import metric_for_task
from .model_builders import build_model
from .utils import ensure_dir, set_seed

logger = logging.getLogger(__name__)


def dataset_family_from_model(model_cfg: Dict[str, Any]) -> str:
    family = model_cfg["family"]
    if family == "caduceus":
        return "nucleotide"
    if family == "unet":
        return "bpe_unet"
    if family == "rmt":
        return "rmt_unet"
    if family == "amt" and bool(model_cfg.get("use_unet", False)):
        return "amt_unet"
    return "bpe"


def label_names_for(task: str, dataset_family: str):
    if task in {"finding_edge", "finding_region"}:
        return ["letter_level_labels", "letter_level_labels_mask"] if dataset_family in {"nucleotide", "bpe_unet", "rmt_unet", "amt_unet"} else ["labels", "labels_mask"]
    if task == "segmentation":
        return ["letter_level_labels", "letter_level_labels_mask"]
    if task == "transcript_type":
        return ["transcript_type"]
    raise RuntimeError(task)


def validate_rules(cfg: Dict[str, Any], task: str) -> None:
    model_cfg = cfg["model"]
    family = model_cfg["family"]
    backbone_kind = model_cfg.get("backbone_kind", family)
    tr = cfg["training"]
    train_bs = int(tr.get("per_device_train_batch_size", 1))
    eval_bs = int(tr.get("per_device_eval_batch_size", 1))
    needs_bs1 = family in {"rmt", "unet"} or (family == "amt" and bool(model_cfg.get("use_unet", False)))
    if needs_bs1 and (train_bs != 1 or eval_bs != 1):
        raise RuntimeError("RMT, AMT+UNET, and plain+UNET models require per-device train/eval batch size 1")
    if family in {"rmt", "amt"} and backbone_kind not in {"gena", "moderngena"}:
        raise RuntimeError(f"{family.upper()} is only valid for GENA/ModernGENA. Got backbone_kind={backbone_kind}")
    if family == "rmt" and backbone_kind == "caduceus":
        raise RuntimeError("RMT must not be adapted to Caduceus")
    if task == "segmentation" and backbone_kind in {"gena", "moderngena"}:
        if family not in {"unet", "rmt"} and not (family == "amt" and bool(model_cfg.get("use_unet", False))):
            raise RuntimeError("Segmentation with GENA/ModernGENA requires nucleotide resolution: family='unet', family='rmt', or family='amt' with use_unet=true")
    if family in {"unet", "rmt"} or (family == "amt" and bool(model_cfg.get("use_unet", False))):
        if not model_cfg.get("nucleotide_tokenizer_path"):
            raise RuntimeError("UNET/RMT/AMT+UNET models require model.nucleotide_tokenizer_path")
    if "freeze" in str(model_cfg).lower():
        raise RuntimeError("Freezing options are not supported: all parameters are always trainable")
    logger.info("[rules] task=%s family=%s backbone_kind=%s train_bs=%d eval_bs=%d dataset_family=%s", task, family, backbone_kind, train_bs, eval_bs, dataset_family_from_model(model_cfg))


def train_from_config(config_path: str, task: str) -> None:
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
    cfg = load_json(config_path)
    validate_rules(cfg, task)
    set_seed(int(cfg.get("seed", 42)))
    output_dir = ensure_dir(cfg["training"]["output_dir"])
    save_json(cfg, output_dir / "config.json")

    model_cfg = cfg["model"]
    dataset_family = dataset_family_from_model(model_cfg)
    tokenizer = make_tokenizer(model_cfg["tokenizer_path"], trust_remote_code=bool(model_cfg.get("trust_remote_code", True)))
    if model_cfg.get("padding_side"):
        tokenizer.padding_side = model_cfg["padding_side"]
    elif model_cfg.get("backbone_kind") == "caduceus":
        tokenizer.padding_side = "left"
        logger.info("[tokenizer.main] using Caduceus default padding_side=left")
    nucleotide_tokenizer = None
    if model_cfg.get("nucleotide_tokenizer_path"):
        nucleotide_tokenizer = make_tokenizer(model_cfg["nucleotide_tokenizer_path"], trust_remote_code=bool(model_cfg.get("trust_remote_code", True)))
        if model_cfg.get("nucleotide_padding_side"):
            nucleotide_tokenizer.padding_side = model_cfg["nucleotide_padding_side"]
    logger.info("[tokenizer.main] path=%s pad=%s cls=%s sep=%s padding_side=%s", model_cfg["tokenizer_path"], tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.padding_side)
    if nucleotide_tokenizer is not None:
        logger.info("[tokenizer.nucleotide] path=%s pad=%s cls=%s sep=%s padding_side=%s", model_cfg["nucleotide_tokenizer_path"], nucleotide_tokenizer.pad_token_id, nucleotide_tokenizer.cls_token_id, nucleotide_tokenizer.sep_token_id, nucleotide_tokenizer.padding_side)
    cfg["_tokenizer"] = tokenizer

    train_data_cfg = dict(cfg["train_dataset"])
    train_data_cfg["model_family"] = dataset_family
    eval_data_cfg = dict(cfg["eval_dataset"])
    eval_data_cfg["model_family"] = dataset_family
    logger.info("[dataset.train] %s", train_data_cfg)
    logger.info("[dataset.eval] %s", eval_data_cfg)
    train_dataset = GenatatorDataset(train_data_cfg, task=task, tokenizer=tokenizer, nucleotide_tokenizer=nucleotide_tokenizer, is_train=True)
    eval_dataset = GenatatorDataset(eval_data_cfg, task=task, tokenizer=tokenizer, nucleotide_tokenizer=nucleotide_tokenizer, is_train=False)

    model = build_model(cfg, task=task)
    tr = cfg["training"]
    ta_kwargs = dict(
        output_dir=str(output_dir),
        overwrite_output_dir=bool(tr.get("overwrite_output_dir", False)),
        max_steps=int(tr.get("max_steps", -1)),
        num_train_epochs=float(tr.get("num_train_epochs", 1.0)),
        per_device_train_batch_size=int(tr.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(tr.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(tr.get("gradient_accumulation_steps", 1)),
        learning_rate=float(tr.get("learning_rate", 5e-5)),
        weight_decay=float(tr.get("weight_decay", 1e-4)),
        warmup_steps=int(tr.get("warmup_steps", 1000)),
        lr_scheduler_type=tr.get("lr_scheduler_type", "constant_with_warmup"),
        logging_strategy="steps",
        logging_steps=int(tr.get("logging_steps", 100)),
        eval_steps=int(tr.get("eval_steps", 1000)),
        save_strategy="steps",
        save_steps=int(tr.get("save_steps", 1000)),
        save_total_limit=int(tr.get("save_total_limit", 3)),
        save_safetensors=bool(tr.get("save_safetensors", True)),
        load_best_model_at_end=bool(tr.get("load_best_model_at_end", False)),
        metric_for_best_model=tr.get("metric_for_best_model"),
        greater_is_better=tr.get("greater_is_better"),
        report_to=["tensorboard"],
        logging_dir=str(output_dir / "tensorboard"),
        disable_tqdm=False,
        dataloader_num_workers=int(tr.get("dataloader_num_workers", 4)),
        bf16=bool(tr.get("bf16", False)),
        fp16=bool(tr.get("fp16", False)),
        remove_unused_columns=False,
        label_names=label_names_for(task, dataset_family),
        seed=int(cfg.get("seed", 42)),
    )
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in ta_params:
        ta_kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in ta_params:
        ta_kwargs["evaluation_strategy"] = "steps"
    else:
        raise RuntimeError("Installed transformers.TrainingArguments supports neither eval_strategy nor evaluation_strategy")
    args = TrainingArguments(**ta_kwargs)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=GenatatorCollator(),
        compute_metrics=metric_for_task(task),
    )
    resume = tr.get("resume_from_checkpoint") or None
    logger.info("[training] resume_from_checkpoint=%s", resume)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output_dir / "final_model"))
    trainer.save_state()
