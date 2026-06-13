from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from transformers import Trainer, TrainingArguments

from .config import load_json, save_json
from .data import GenatatorCollator, GenatatorDataset, make_tokenizer
from .metrics_training import metric_for_task
from .model_builders import build_model
from .utils import ensure_dir, set_seed


def _dataset_family(model_family: str) -> str:
    return "nucleotide" if model_family == "caduceus" else "bpe"


def _label_names(task: str):
    if task in {"finding_edge", "finding_region"}:
        return ["labels", "labels_mask"]
    if task == "segmentation":
        return ["letter_level_labels", "letter_level_labels_mask"]
    if task == "transcript_type":
        return ["transcript_type"]
    raise ValueError(task)


def _validate_batch_rules(cfg: Dict[str, Any]) -> None:
    family = cfg["model"]["family"]
    if family == "gena_rmt":
        tr = cfg["training"]
        if int(tr.get("per_device_train_batch_size", 1)) != 1 or int(tr.get("per_device_eval_batch_size", 1)) != 1:
            raise ValueError("RMT repeater models with cycles=3 must use per_device_train_batch_size=1 and per_device_eval_batch_size=1.")


def train_from_config(config_path: str, task: str) -> None:
    cfg = load_json(config_path)
    _validate_batch_rules(cfg)
    set_seed(int(cfg.get("seed", 42)))
    output_dir = ensure_dir(cfg["training"]["output_dir"])
    save_json(cfg, output_dir / "config.json")

    model_family = cfg["model"]["family"]
    tokenizer = make_tokenizer(cfg["model"]["tokenizer_path"], trust_remote_code=bool(cfg["model"].get("trust_remote_code", True)))
    if cfg["model"].get("padding_side"):
        tokenizer.padding_side = cfg["model"]["padding_side"]
    nucleotide_tokenizer = None
    if cfg["model"].get("nucleotide_tokenizer_path"):
        nucleotide_tokenizer = make_tokenizer(cfg["model"]["nucleotide_tokenizer_path"], trust_remote_code=bool(cfg["model"].get("trust_remote_code", True)))
        if cfg["model"].get("nucleotide_padding_side"):
            nucleotide_tokenizer.padding_side = cfg["model"]["nucleotide_padding_side"]
    cfg["_tokenizer"] = tokenizer

    train_data_cfg = dict(cfg["train_dataset"])
    train_data_cfg["model_family"] = _dataset_family(model_family)
    eval_data_cfg = dict(cfg["eval_dataset"])
    eval_data_cfg["model_family"] = _dataset_family(model_family)

    train_dataset = GenatatorDataset(train_data_cfg, task=task, tokenizer=tokenizer, nucleotide_tokenizer=nucleotide_tokenizer)
    eval_dataset = GenatatorDataset(eval_data_cfg, task=task, tokenizer=tokenizer, nucleotide_tokenizer=nucleotide_tokenizer)
    model = build_model(cfg, task=task)

    tr = cfg["training"]
    args = TrainingArguments(
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
        eval_strategy="steps",
        eval_steps=int(tr.get("eval_steps", 1000)),
        save_strategy="steps",
        save_steps=int(tr.get("save_steps", 1000)),
        save_total_limit=int(tr.get("save_total_limit", 3)),
        save_safetensors=bool(tr.get("save_safetensors", True)),
        load_best_model_at_end=bool(tr.get("load_best_model_at_end", True)),
        metric_for_best_model=tr.get("metric_for_best_model"),
        greater_is_better=tr.get("greater_is_better"),
        report_to=["tensorboard"],
        logging_dir=str(output_dir / "tensorboard"),
        disable_tqdm=False,
        dataloader_num_workers=int(tr.get("dataloader_num_workers", 4)),
        bf16=bool(tr.get("bf16", False)),
        fp16=bool(tr.get("fp16", False)),
        remove_unused_columns=False,
        label_names=_label_names(task),
        seed=int(cfg.get("seed", 42)),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=GenatatorCollator(),
        compute_metrics=metric_for_task(task),
    )
    resume = tr.get("resume_from_checkpoint") or None
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output_dir / "final_model"))
    trainer.save_state()
