from __future__ import annotations

import logging
from typing import Any, Dict
import inspect
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm.auto import tqdm
from transformers import Trainer, TrainingArguments

from .config import load_json
from .data import GenatatorCollator, GenatatorDataset, make_tokenizer, nucleotide_token_ids
from .metrics_training import (
    EDGE_CLASS_NAMES,
    REGION_CLASS_NAMES,
    SEGMENTATION_CLASS_INDEX,
    _safe_binary_average_precision,
    metric_for_task,
    metric_names_for_task,
    segmentation_interval_predictions,
    sigmoid,
)
from .intervals import f1_from_counts, interval_counts
from .model_builders import build_model, normalize_unet_chunk_size
from .run_management import (
    BestCheckpointEvaluationConfigCallback,
    EvaluationConfigManager,
    atomic_save_json,
    create_timestamped_run_dir,
    is_world_process_zero,
)
from .torch_compat import allow_transformers_torch_load_on_legacy_torch
from .utils import set_seed

logger = logging.getLogger(__name__)


class GenatatorTrainer(Trainer):
    """Trainer with an explicit sequential sampler for chromosome smoke runs.

    Normal training keeps the standard Transformers sampler. Smoke configs set
    ``training.sequential_train=true`` so the complete chromosome is traversed in
    genomic order. This prevents repeated 10 Mb parquet block reloads caused by
    random window access and makes one epoch exactly one pass over every window.
    """

    def __init__(
        self,
        *args,
        sequential_train: bool = False,
        allow_legacy_torch_load: bool = True,
        genatator_task: str | None = None,
        **kwargs,
    ):
        self.sequential_train = bool(sequential_train)
        self.allow_legacy_torch_load = bool(allow_legacy_torch_load)
        self.genatator_task = genatator_task or "unknown"
        super().__init__(*args, **kwargs)

    def _enable_trusted_checkpoint_loading(self, context: str) -> None:
        allow_transformers_torch_load_on_legacy_torch(
            self.allow_legacy_torch_load,
            context=context,
        )

    def _load_best_model(self):
        self._enable_trusted_checkpoint_loading("GenatatorTrainer._load_best_model")
        logger.info(
            "[checkpoint.best] restoring best checkpoint=%s metric=%s best_metric=%s",
            self.state.best_model_checkpoint,
            self.args.metric_for_best_model,
            self.state.best_metric,
        )
        return super()._load_best_model()

    def _load_from_checkpoint(self, *args, **kwargs):
        self._enable_trusted_checkpoint_loading("GenatatorTrainer._load_from_checkpoint")
        return super()._load_from_checkpoint(*args, **kwargs)

    @staticmethod
    def _output_value(outputs, name: str):
        if isinstance(outputs, dict):
            return outputs.get(name)
        return getattr(outputs, name, None)

    @staticmethod
    def _batch_size_from_inputs(inputs: Dict[str, Any]) -> int:
        for value in inputs.values():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return int(value.shape[0])
        return 1

    @staticmethod
    def _labels_and_mask_from_inputs(task: str, inputs: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if task in {"finding_edge", "finding_region"}:
            if "letter_level_labels" in inputs:
                labels = inputs["letter_level_labels"]
                mask = inputs["letter_level_labels_mask"]
            else:
                labels = inputs["labels"]
                mask = inputs["labels_mask"]
            return labels, mask.bool()
        if task == "segmentation":
            return inputs["letter_level_labels"], inputs["letter_level_labels_mask"].bool()
        raise RuntimeError(f"No token labels for task={task}")

    def _new_streaming_state(self) -> Dict[str, Any]:
        task = self.genatator_task
        state: Dict[str, Any] = {"loss_sum": 0.0, "loss_count": 0}
        if task == "finding_edge":
            state["class_names"] = EDGE_CLASS_NAMES
            state["scores"] = [[] for _ in EDGE_CLASS_NAMES]
            state["refs"] = [[] for _ in EDGE_CLASS_NAMES]
        elif task == "finding_region":
            state["class_names"] = REGION_CLASS_NAMES
            state["scores"] = [[] for _ in REGION_CLASS_NAMES]
            state["refs"] = [[] for _ in REGION_CLASS_NAMES]
        elif task == "segmentation":
            state["counts"] = {name: [0, 0, 0] for name in SEGMENTATION_CLASS_INDEX}
        elif task == "transcript_type":
            state["correct"] = 0
            state["total"] = 0
        else:
            raise RuntimeError(f"Unsupported streaming evaluation task={task}")
        return state

    def _update_streaming_state(self, state: Dict[str, Any], inputs: Dict[str, Any], outputs: Any) -> None:
        loss = self._output_value(outputs, "loss")
        batch_size = self._batch_size_from_inputs(inputs)
        if isinstance(loss, torch.Tensor):
            state["loss_sum"] += float(loss.detach().float().cpu().item()) * batch_size
            state["loss_count"] += batch_size
        logits = self._output_value(outputs, "logits")
        if logits is None:
            return
        task = self.genatator_task
        if task in {"finding_edge", "finding_region"}:
            labels_t, mask_t = self._labels_and_mask_from_inputs(task, inputs)
            logits_np = logits.detach().float().cpu().numpy()
            labels_np = labels_t.detach().float().cpu().numpy()
            mask_np = mask_t.detach().cpu().numpy().astype(bool)
            probs = sigmoid(logits_np)
            for c in range(len(state["class_names"])):
                refs = labels_np[:, :, c][mask_np]
                scores = probs[:, :, c][mask_np]
                if refs.size:
                    state["refs"][c].append(refs.astype(np.float32, copy=False))
                    state["scores"][c].append(scores.astype(np.float32, copy=False))
        elif task == "segmentation":
            labels_t, mask_t = self._labels_and_mask_from_inputs(task, inputs)
            logits_np = logits.detach().float().cpu().numpy()
            labels_np = labels_t.detach().float().cpu().numpy()
            mask_np = mask_t.detach().cpu().numpy().astype(bool)
            for class_name, channel_index in SEGMENTATION_CLASS_INDEX.items():
                tp = fp = fn = 0
                decoded = segmentation_interval_predictions(logits_np, class_name)
                for sample_index in range(labels_np.shape[0]):
                    valid = mask_np[sample_index]
                    if not np.any(valid):
                        continue
                    references = (labels_np[sample_index, valid, channel_index] >= 0.5).astype(np.int8)
                    predictions = decoded[sample_index, valid]
                    sample_tp, sample_fp, sample_fn = interval_counts(references, predictions)
                    tp += sample_tp
                    fp += sample_fp
                    fn += sample_fn
                state["counts"][class_name][0] += tp
                state["counts"][class_name][1] += fp
                state["counts"][class_name][2] += fn
        elif task == "transcript_type":
            refs = inputs["transcript_type"].detach().cpu().numpy().reshape(-1).astype(np.int64)
            scores = sigmoid(logits.detach().float().cpu().numpy().reshape(-1))
            preds = (scores >= 0.5).astype(np.int64)
            state["correct"] += int((preds == refs).sum())
            state["total"] += int(refs.size)

    def _finalize_streaming_state(self, state: Dict[str, Any], metric_key_prefix: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        loss_count = max(1, int(state.get("loss_count", 0)))
        metrics[f"{metric_key_prefix}_loss"] = float(state.get("loss_sum", 0.0) / loss_count)
        task = self.genatator_task
        if task in {"finding_edge", "finding_region"}:
            defined_values = []
            total_dropped = 0
            for c, class_name in enumerate(state["class_names"]):
                refs = np.concatenate(state["refs"][c], axis=0) if state["refs"][c] else np.asarray([], dtype=np.float32)
                scores = np.concatenate(state["scores"][c], axis=0) if state["scores"][c] else np.asarray([], dtype=np.float32)
                ap, defined, positives, negatives, dropped = _safe_binary_average_precision(refs, scores)
                total_dropped += dropped
                metrics[f"{metric_key_prefix}_pr_auc_{class_name}"] = float(ap)
                metrics[f"{metric_key_prefix}_pr_auc_{class_name}_defined"] = float(defined)
                metrics[f"{metric_key_prefix}_pr_auc_{class_name}_positives"] = float(positives)
                metrics[f"{metric_key_prefix}_pr_auc_{class_name}_negatives"] = float(negatives)
                metrics[f"{metric_key_prefix}_pr_auc_{class_name}_dropped_nonfinite"] = float(dropped)
                if defined:
                    defined_values.append(ap)
            metrics[f"{metric_key_prefix}_pr_auc_defined_channels"] = float(len(defined_values))
            metrics[f"{metric_key_prefix}_pr_auc_mean"] = float(np.mean(defined_values)) if defined_values else 0.0
            metrics[f"{metric_key_prefix}_pr_auc_dropped_nonfinite_total"] = float(total_dropped)
        elif task == "segmentation":
            for class_name, (tp, fp, fn) in state["counts"].items():
                metrics[f"{metric_key_prefix}_interval_f1_{class_name}"] = float(f1_from_counts(tp, fp, fn))
                metrics[f"{metric_key_prefix}_interval_tp_{class_name}"] = float(tp)
                metrics[f"{metric_key_prefix}_interval_fp_{class_name}"] = float(fp)
                metrics[f"{metric_key_prefix}_interval_fn_{class_name}"] = float(fn)
        elif task == "transcript_type":
            total = max(1, int(state["total"]))
            metrics[f"{metric_key_prefix}_accuracy"] = float(state["correct"] / total)
        return metrics

    def _streaming_evaluate_rank0(self, eval_dataset=None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            return {}
        dataloader = DataLoader(
            dataset,
            sampler=SequentialSampler(dataset),
            batch_size=int(self.args.per_device_eval_batch_size),
            collate_fn=self.data_collator,
            num_workers=int(self.args.dataloader_num_workers),
            pin_memory=bool(self.args.dataloader_pin_memory),
        )
        model = self.accelerator.unwrap_model(self.model) if hasattr(self, "accelerator") else self.model
        model.eval()
        state = self._new_streaming_state()
        total = len(dataloader) if hasattr(dataloader, "__len__") else None
        logger.info(
            "[rank0_streaming_eval] task=%s batches=%s batch_size=%s no_ddp_allgather=true cpu_metric_accumulation=true",
            self.genatator_task,
            total,
            self.args.per_device_eval_batch_size,
        )
        pbar = tqdm(total=total, desc=f"{metric_key_prefix}:{self.genatator_task}", disable=not self.is_world_process_zero())
        with torch.no_grad():
            for batch in dataloader:
                inputs = self._prepare_inputs(batch)
                outputs = model(**inputs)
                self._update_streaming_state(state, inputs, outputs)
                del outputs, inputs, batch
                pbar.update(1)
        pbar.close()
        metrics = self._finalize_streaming_state(state, metric_key_prefix)
        metrics[f"{metric_key_prefix}_samples"] = float(len(dataset)) if hasattr(dataset, "__len__") else 0.0
        return metrics

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """Rank-0 streaming evaluation that avoids distributed all-gather of large tensors.

        Hugging Face's default prediction loop all-gathers logits/labels across
        all ranks. For nucleotide-level validation this can allocate huge GPU
        tensors or hang in NCCL when one rank is a straggler. Here rank 0 runs a
        sequential validation pass with the unwrapped model, moves every batch's
        predictions to CPU immediately, writes the small metric dictionary to
        disk, and the other ranks wait for that file without using NCCL.
        """
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        step = int(getattr(self.state, "global_step", 0))
        sync_dir = Path(self.args.output_dir) / "rank0_eval_metrics"
        sync_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = sync_dir / f"{metric_key_prefix}_step_{step}.json"
        start_time = time.time()
        if rank == 0:
            if metrics_path.exists():
                metrics_path.unlink()
            metrics = self._streaming_evaluate_rank0(eval_dataset=eval_dataset, metric_key_prefix=metric_key_prefix)
            atomic_save_json(metrics, metrics_path)
            logger.info("[rank0_streaming_eval] wrote metrics=%s", metrics_path)
        else:
            timeout_s = float(getattr(self.args, "eval_rank0_timeout_seconds", 86400.0))
            logger.info(
                "[rank0_streaming_eval] rank=%d waiting for rank0 metrics file=%s without NCCL collectives",
                rank,
                metrics_path,
            )
            while True:
                if metrics_path.exists() and metrics_path.stat().st_mtime >= start_time - 1.0:
                    with open(metrics_path) as fh:
                        metrics = json.load(fh)
                    break
                if time.time() - start_time > timeout_s:
                    raise TimeoutError(f"Timed out waiting for rank0 evaluation metrics at {metrics_path}")
                time.sleep(5.0)
        if rank == 0:
            self.log(metrics)
        return metrics

    def _get_train_sampler(self, *args, **kwargs):
        if self.sequential_train:
            dataset = args[0] if args else kwargs.get("train_dataset", self.train_dataset)
            logger.info(
                "[training.sampler] SequentialSampler enabled; every selected sample/window "
                "is visited once per epoch in dataset order"
            )
            return SequentialSampler(dataset)
        return super()._get_train_sampler(*args, **kwargs)


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




def needs_nucleotide_tokenizer(model_cfg: Dict[str, Any]) -> bool:
    family = model_cfg["family"]
    return family in {"unet", "rmt"} or (family == "amt" and bool(model_cfg.get("use_unet", False)))


def tokenizer_vocab_size(tokenizer) -> int:
    try:
        n = len(tokenizer)
    except TypeError:
        n = None
    vocab_size = getattr(tokenizer, "vocab_size", None)
    vals = [int(x) for x in (n, vocab_size) if x is not None]
    if not vals:
        raise RuntimeError("Could not infer tokenizer vocabulary size")
    return max(vals)


def prepare_nucleotide_tokenizer(model_cfg: Dict[str, Any], tokenizer):
    if not needs_nucleotide_tokenizer(model_cfg):
        return None
    legacy_path = model_cfg.pop("nucleotide_tokenizer_path", None)
    if legacy_path and str(legacy_path) != str(model_cfg["tokenizer_path"]):
        logger.warning(
            "[tokenizer.nucleotide_ids] ignoring legacy nucleotide_tokenizer_path=%s; "
            "single-nucleotide ids are always read from tokenizer_path=%s",
            legacy_path,
            model_cfg["tokenizer_path"],
        )
    if model_cfg.pop("nucleotide_padding_side", None) is not None:
        logger.warning(
            "[tokenizer.nucleotide_ids] nucleotide_padding_side is ignored because the main tokenizer is reused"
        )
    ids = nucleotide_token_ids(tokenizer)
    vocab_size = tokenizer_vocab_size(tokenizer)
    configured = model_cfg.get("nucleotide_vocab_size")
    if configured in (None, "", "auto"):
        model_cfg["nucleotide_vocab_size"] = int(vocab_size)
        logger.info(
            "[tokenizer.nucleotide_ids] source=main tokenizer ids=%s auto nucleotide_vocab_size=%d",
            ids,
            vocab_size,
        )
    else:
        configured_i = int(configured)
        if configured_i < vocab_size:
            raise RuntimeError(
                f"model.nucleotide_vocab_size={configured_i} is smaller than main tokenizer vocab size {vocab_size}. "
                "Set it to null/auto or to a value >= tokenizer vocab size."
            )
        model_cfg["nucleotide_vocab_size"] = configured_i
    return tokenizer

def validate_rules(cfg: Dict[str, Any], task: str) -> None:
    model_cfg = cfg["model"]
    family = model_cfg["family"]
    unet_chunk_size = normalize_unet_chunk_size(model_cfg)
    backbone_kind = model_cfg.get("backbone_kind", family)
    tr = cfg["training"]
    train_bs = int(tr.get("per_device_train_batch_size", 1))
    eval_bs = int(tr.get("per_device_eval_batch_size", 1))
    if train_bs <= 0 or eval_bs <= 0:
        raise RuntimeError("per-device train/eval batch sizes must be positive")
    if family == "caduceus":
        # This project always trains Caduceus with untied bidirectional weights.
        model_cfg["bidirectional_weight_tie"] = False
    if backbone_kind == "gena" and family in {"plain", "unet"}:
        for dataset_name in ("train_dataset", "eval_dataset"):
            max_bpe_tokens = int(cfg[dataset_name].get("max_bpe_tokens", 0))
            if max_bpe_tokens > 512:
                raise RuntimeError(
                    f"Direct/plain GENA requires {dataset_name}.max_bpe_tokens <= 512; "
                    f"got {max_bpe_tokens}. Use RMT/AMT for longer BPE inputs."
                )
    if task == "transcript_type" and family not in {"plain", "caduceus"}:
        raise RuntimeError(
            "Transcript-type classification is implemented only for family='plain' and family='caduceus'; "
            f"got family={family!r}"
        )
    if family in {"rmt", "amt"} and backbone_kind not in {"gena", "moderngena"}:
        raise RuntimeError(f"{family.upper()} is only valid for GENA/ModernGENA. Got backbone_kind={backbone_kind}")
    if family == "rmt" and backbone_kind == "caduceus":
        raise RuntimeError("RMT must not be adapted to Caduceus")
    if task == "segmentation" and backbone_kind in {"gena", "moderngena"}:
        if family not in {"unet", "rmt"} and not (family == "amt" and bool(model_cfg.get("use_unet", False))):
            raise RuntimeError("Segmentation with GENA/ModernGENA requires nucleotide resolution: family='unet', family='rmt', or family='amt' with use_unet=true")
    if needs_nucleotide_tokenizer(model_cfg):
        logger.info("[rules] BPE-to-nucleotide ids will be read from model.tokenizer_path")
    if "freeze" in str(model_cfg).lower():
        raise RuntimeError("Freezing options are not supported: all parameters are always trainable")
    logger.info(
        "[rules] task=%s family=%s backbone_kind=%s train_bs=%d eval_bs=%d dataset_family=%s unet_chunk_size=%s",
        task,
        family,
        backbone_kind,
        train_bs,
        eval_bs,
        dataset_family_from_model(model_cfg),
        unet_chunk_size,
    )


def train_from_config(config_path: str, task: str) -> None:
    # Keep routine training output limited to the Trainer/TQDM progress bar.
    # Warnings and errors remain visible, while per-module INFO diagnostics are
    # suppressed during training.
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)
    cfg = load_json(config_path)
    validate_rules(cfg, task)
    set_seed(int(cfg.get("seed", 42)))
    tr = cfg["training"]
    configured_output_dir = str(Path(tr["output_dir"]).expanduser().resolve())
    output_dir = create_timestamped_run_dir(tr, config_path=config_path)
    tr["output_base_dir"] = configured_output_dir
    tr["output_dir"] = str(output_dir)
    tr["overwrite_output_dir"] = False
    logger.info(
        "[training.run] base_output_dir=%s effective_output_dir=%s custom_prefix=%s",
        configured_output_dir,
        output_dir,
        tr.get("custom_prefix", ""),
    )

    model_cfg = cfg["model"]
    dataset_family = dataset_family_from_model(model_cfg)
    tokenizer = make_tokenizer(model_cfg["tokenizer_path"], trust_remote_code=bool(model_cfg.get("trust_remote_code", True)))
    if model_cfg.get("padding_side"):
        tokenizer.padding_side = model_cfg["padding_side"]
    elif model_cfg.get("backbone_kind") == "caduceus":
        tokenizer.padding_side = "left"
        logger.info("[tokenizer.main] using Caduceus default padding_side=left")
    nucleotide_tokenizer = prepare_nucleotide_tokenizer(model_cfg, tokenizer)
    logger.info("[tokenizer.main] path=%s pad=%s cls=%s sep=%s padding_side=%s", model_cfg["tokenizer_path"], tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.padding_side)
    if nucleotide_tokenizer is not None:
        logger.info("[tokenizer.nucleotide_ids] source=main path=%s vocab_size=%s", model_cfg["tokenizer_path"], model_cfg.get("nucleotide_vocab_size"))

    # At this point tokenizer-dependent values such as nucleotide_vocab_size
    # are resolved, while runtime-only objects have not yet entered cfg.
    evaluation_config_manager = EvaluationConfigManager(cfg, task=task, run_dir=output_dir)
    if is_world_process_zero():
        atomic_save_json(cfg, output_dir / "training_config.json")
        # Keep the historical filename for callers that already consume it.
        atomic_save_json(cfg, output_dir / "config.json")
    evaluation_config_manager.write_initial()
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
    logging_strategy = str(tr.get("logging_strategy", "steps"))
    evaluation_strategy = str(tr.get("evaluation_strategy", tr.get("eval_strategy", "steps")))
    save_strategy = str(tr.get("save_strategy", "steps"))
    logger.info(
        "[training.strategy] logging=%s evaluation=%s save=%s epochs=%s max_steps=%s",
        logging_strategy,
        evaluation_strategy,
        save_strategy,
        tr.get("num_train_epochs", 1.0),
        tr.get("max_steps", -1),
    )
    logger.info(
        "[metrics.validation] task=%s ordered_metrics=%s final_benchmark_metrics_run_only_in_inference=true",
        task,
        list(metric_names_for_task(task)),
    )
    logging_interval = int(tr.get("logging_interval", tr.get("logging_steps", 100)))
    eval_interval = int(tr.get("eval_interval", tr.get("eval_steps", logging_interval)))
    save_interval = int(tr.get("save_interval", tr.get("save_steps", eval_interval)))
    logger.info(
        "[training.intervals] logging_interval=%d eval_interval=%d save_interval=%d",
        logging_interval,
        eval_interval,
        save_interval,
    )
    logger.info(
        "[training.evaluation_memory] eval_accumulation_steps=%s; validation logits/labels are flushed to CPU at this interval",
        tr.get("eval_accumulation_steps", 1),
    )
    ta_kwargs = dict(
        output_dir=str(output_dir),
        # Every invocation owns a newly created timestamped directory.
        overwrite_output_dir=False,
        max_steps=int(tr.get("max_steps", -1)),
        num_train_epochs=float(tr.get("num_train_epochs", 1.0)),
        per_device_train_batch_size=int(tr.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(tr.get("per_device_eval_batch_size", 1)),
        # Critical for long nucleotide-level validation. Transformers otherwise
        # keeps all prediction/label tensors on the GPU until the whole eval
        # loop finishes; for transcript/chromosome tasks this can require tens
        # of GiB. eval_accumulation_steps=1 moves each evaluated batch to CPU
        # immediately before the next batch is processed. Metrics are still
        # computed on the same predictions and labels; only the accumulation
        # device changes.
        eval_accumulation_steps=int(tr.get("eval_accumulation_steps", 1)),
        gradient_accumulation_steps=int(tr.get("gradient_accumulation_steps", 1)),
        learning_rate=float(tr.get("learning_rate", 5e-5)),
        weight_decay=float(tr.get("weight_decay", 1e-4)),
        warmup_steps=int(tr.get("warmup_steps", 1000)),
        lr_scheduler_type=tr.get("lr_scheduler_type", "constant_with_warmup"),
        logging_strategy=logging_strategy,
        logging_steps=logging_interval,
        eval_steps=eval_interval,
        save_strategy=save_strategy,
        save_steps=save_interval,
        save_total_limit=int(tr.get("save_total_limit", 3)),
        save_safetensors=bool(tr.get("save_safetensors", False)),
        load_best_model_at_end=bool(tr.get("load_best_model_at_end", False)),
        metric_for_best_model=tr.get("metric_for_best_model"),
        greater_is_better=tr.get("greater_is_better"),
        report_to=tr.get("report_to", "tensorboard"),
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
        ta_kwargs["eval_strategy"] = evaluation_strategy
    elif "evaluation_strategy" in ta_params:
        ta_kwargs["evaluation_strategy"] = evaluation_strategy
    else:
        raise RuntimeError("Installed transformers.TrainingArguments supports neither eval_strategy nor evaluation_strategy")
    args = TrainingArguments(**ta_kwargs)
    args.eval_rank0_timeout_seconds = float(tr.get("eval_rank0_timeout_seconds", 86400.0))
    logger.info("[training.evaluation_memory] rank0_streaming_evaluation=true; no distributed all-gather of validation logits/labels")
    trainer = GenatatorTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=GenatatorCollator(),
        compute_metrics=metric_for_task(task),
        sequential_train=bool(tr.get("sequential_train", False)),
        allow_legacy_torch_load=bool(
            model_cfg.get("allow_unsafe_torch_load_with_torch_lt_2_6", True)
        ),
        genatator_task=task,
        callbacks=[BestCheckpointEvaluationConfigCallback(evaluation_config_manager)],
    )
    resume = tr.get("resume_from_checkpoint") or None
    if isinstance(resume, bool):
        raise RuntimeError(
            "training.resume_from_checkpoint must be an explicit checkpoint path. "
            "Boolean auto-resume cannot search a newly created timestamped run directory."
        )
    logger.info("[training] resume_from_checkpoint=%s", resume)
    train_result = trainer.train(resume_from_checkpoint=resume)
    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    trainer.save_state()
    if trainer.is_world_process_zero():
        atomic_save_json(dict(train_result.metrics), output_dir / "train_metrics.json")
        best = getattr(trainer.state, "best_model_checkpoint", None)
        if best and Path(best).expanduser().exists():
            evaluation_config_manager.update_checkpoint(best, selection="best", copy_to=final_model_dir)
        else:
            evaluation_config_manager.update_checkpoint(
                final_model_dir,
                selection="final_model_no_best",
                copy_to=final_model_dir,
            )
