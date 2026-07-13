from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from transformers import TrainerCallback


_SAFE_PREFIX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def world_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_world_process_zero() -> bool:
    return world_rank() == 0


def atomic_save_json(obj: Dict[str, Any], path: str | Path) -> Path:
    """Write JSON completely before making it visible to other ranks."""

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return target


def _custom_prefix(value: Any) -> str:
    prefix = str(value or "").strip()
    if not prefix:
        return ""
    if not _SAFE_PREFIX.fullmatch(prefix) or ".." in prefix:
        raise RuntimeError(
            "training.custom_prefix may contain only letters, digits, '.', '_' and '-', "
            "must start with a letter or digit, and must not contain '..'"
        )
    return prefix


def _exclusive_timestamped_child(base_dir: Path, prefix: str) -> Path:
    for attempt in range(1000):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = f"{prefix}_{stamp}" if prefix else stamp
        if attempt:
            name = f"{name}_{attempt:03d}"
        candidate = base_dir / name
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate.resolve()
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not allocate a unique timestamped run directory under {base_dir}")


def _launch_identity(config_path: str | Path) -> tuple[str, bool]:
    explicit = str(os.environ.get("GENATATOR_LAUNCH_ID", "")).strip()
    elastic = str(os.environ.get("TORCHELASTIC_RUN_ID", "")).strip()
    if explicit:
        shared = f"explicit:{explicit}"
        robust = True
    elif elastic and elastic.lower() != "none":
        shared = f"elastic:{elastic}"
        robust = True
    else:
        # Workers launched by one local torchrun agent share a parent PID.  This
        # prevents a stale manifest from a previous launch from being reused.
        shared = (
            f"local:{os.environ.get('MASTER_ADDR', '')}:{os.environ.get('MASTER_PORT', '')}:"
            f"{world_size()}:{os.getppid()}:{os.environ.get('TORCHELASTIC_RESTART_COUNT', '0')}"
        )
        robust = int(os.environ.get("GROUP_WORLD_SIZE", "1")) <= 1
    material = f"{Path(config_path).expanduser().resolve()}|{shared}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20], robust


def create_timestamped_run_dir(
    training_cfg: Dict[str, Any],
    *,
    config_path: str | Path,
    timeout_seconds: float = 120.0,
) -> Path:
    """Create one timestamped run child shared safely by all DDP ranks."""

    base_dir = Path(training_cfg["output_dir"]).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    prefix = _custom_prefix(training_cfg.get("custom_prefix", ""))
    launch_hash, robust_multinode_identity = _launch_identity(config_path)
    if world_size() > 1 and not robust_multinode_identity:
        raise RuntimeError(
            "A multi-node launch needs a shared launch identity. Set GENATATOR_LAUNCH_ID "
            "to the same unique value on every node (or use torchrun with a non-'none' rendezvous id)."
        )
    manifest = base_dir / f".genatator-run-{launch_hash}.json"

    if is_world_process_zero():
        run_dir = _exclusive_timestamped_child(base_dir, prefix)
        payload = {
            "created_at": time.time(),
            "run_dir": str(run_dir),
            "world_size": world_size(),
            "rank0_pid": os.getpid(),
        }
        atomic_save_json(payload, manifest)
        # Stable discovery is useful to launchers and smoke tests.  It is not
        # used for DDP rendezvous, so concurrent experiment families remain
        # independent as long as they have distinct configured base dirs.
        atomic_save_json(payload, base_dir / "latest_run.json")
        return run_dir

    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        try:
            with open(manifest, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            run_dir = Path(payload["run_dir"]).expanduser().resolve()
            if run_dir.is_dir() and int(payload.get("world_size", -1)) == world_size():
                return run_dir
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
        time.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for rank 0 to publish the run directory in {manifest}")


def _absolute_output(run_dir: Path, name: str) -> str:
    return str((run_dir / "evaluation" / name).resolve())


def build_evaluation_config(cfg: Dict[str, Any], *, task: str, run_dir: str | Path) -> Dict[str, Any]:
    """Build a directly runnable task-specific evaluation configuration."""

    run_dir = Path(run_dir).resolve()
    model_cfg = copy.deepcopy(cfg["model"])
    # Inference owns checkpoint loading. Keeping both fields non-null loads
    # weights twice and can silently mix two different checkpoints.
    model_cfg["checkpoint_path"] = None
    dataset_cfg = copy.deepcopy(cfg["eval_dataset"])
    training_cfg = cfg["training"]
    requested = copy.deepcopy(cfg.get("evaluation") or {})
    true_gff = cfg.get("true_gff", requested.get("true_gff"))

    # Every automatically generated evaluation is intentionally restricted to
    # the held-out human chromosome used by the project benchmark.
    dataset_cfg["genomes"] = ["GCF_009914755.1"]
    dataset_cfg["chromosomes"] = ["NC_060944.1"]
    # Final evaluation must not inherit smoke/debug row or window limits from
    # a training configuration.
    for limiting_key in (
        "max_rows",
        "max_windows",
        "streaming_max_rows",
        "streaming_max_scanned_rows",
        "streaming_trim_rows",
    ):
        dataset_cfg.pop(limiting_key, None)

    common = {
        "device": requested.get("device", "cuda"),
        "checkpoint_path": None,
        "batch_size": int(requested.get("batch_size", training_cfg.get("per_device_eval_batch_size", 1))),
        "use_reverse_complement": bool(requested.get("use_reverse_complement", True)),
        "true_gff": true_gff,
    }

    generated = {
        "task": task,
        "run_dir": str(run_dir),
        "checkpoint_selection": "pending",
        "best_checkpoint": None,
    }

    if task in {"finding_edge", "finding_region"}:
        # Gene finding has a dedicated held-out test split.
        dataset_cfg["split"] = "test"
        dataset_cfg.pop("statuses", None)
        common["metrics_json"] = _absolute_output(run_dir, f"{task}_metrics.json")
        return {
            "task": task,
            "model": model_cfg,
            "dataset": dataset_cfg,
            "inference": common,
            "_generated": generated,
        }

    if task == "segmentation":
        # Final evaluation uses every transcript/isoform from val-human and
        # gathers non-overlapping model-sized chunks over complete transcripts.
        dataset_cfg["config_name"] = "val-human"
        dataset_cfg["split"] = "validation"
        dataset_cfg.pop("statuses", None)
        dataset_cfg.pop("random_crop", None)
        dataset_cfg.pop("overlap", None)
        dataset_cfg["full_transcript_chunks"] = True
        common.update(
            {
                "use_cds_heuristic": bool(requested.get("use_cds_heuristic", True)),
                "coordinate_mode": requested.get("coordinate_mode", "transcript"),
                "empty_segment_policy": requested.get("empty_segment_policy", "error"),
                "output_gff": _absolute_output(run_dir, "segmentation_predictions.gff"),
                "metrics_json": _absolute_output(run_dir, "segmentation_metrics.json"),
            }
        )
        return {
            "model": model_cfg,
            "dataset": dataset_cfg,
            "inference": common,
            "_generated": generated,
        }

    if task == "transcript_type":
        dataset_cfg["config_name"] = "val-human"
        dataset_cfg["split"] = "validation"
        dataset_cfg.pop("statuses", None)
        dataset_cfg.pop("random_crop", None)
        dataset_cfg.pop("overlap", None)
        common.update(
            {
                "threshold": float(requested.get("threshold", 0.5)),
                "output_tsv": _absolute_output(run_dir, "transcript_type_predictions.tsv"),
                "metrics_json": _absolute_output(run_dir, "transcript_type_metrics.json"),
            }
        )
        return {
            "model": model_cfg,
            "dataset": dataset_cfg,
            "inference": common,
            "_generated": generated,
        }
    raise RuntimeError(f"Unsupported task for automatic evaluation config: {task}")


class EvaluationConfigManager:
    def __init__(self, cfg: Dict[str, Any], *, task: str, run_dir: str | Path):
        self.run_dir = Path(run_dir).resolve()
        self.path = self.run_dir / "evaluation_config.json"
        self.config = build_evaluation_config(cfg, task=task, run_dir=self.run_dir)

    def write_initial(self) -> None:
        if is_world_process_zero():
            atomic_save_json(self.config, self.path)

    def update_checkpoint(
        self,
        checkpoint: str | Path,
        *,
        selection: str,
        copy_to: str | Path | None = None,
    ) -> None:
        if not is_world_process_zero():
            return
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.exists():
            return
        self.config["inference"]["checkpoint_path"] = str(checkpoint_path)
        generated = self.config.setdefault("_generated", {})
        generated["checkpoint_selection"] = str(selection)
        if selection == "best":
            generated["best_checkpoint"] = str(checkpoint_path)
        atomic_save_json(self.config, self.path)

        destinations = set()
        try:
            checkpoint_path.relative_to(self.run_dir)
            destinations.add(checkpoint_path)
        except ValueError:
            # A resumed TrainerState may retain the best checkpoint from an
            # older run.  Reference it from the new root config, but never
            # modify artifacts owned by that earlier run.
            pass
        if copy_to is not None:
            copy_path = Path(copy_to).expanduser().resolve()
            try:
                copy_path.relative_to(self.run_dir)
                destinations.add(copy_path)
            except ValueError:
                pass
        for directory in destinations:
            if directory.is_dir():
                atomic_save_json(self.config, directory / "evaluation_config.json")


class BestCheckpointEvaluationConfigCallback(TrainerCallback):
    """Keep the run evaluation config pointed at Trainer's current best."""

    def __init__(self, manager: EvaluationConfigManager):
        self.manager = manager

    def on_save(self, args, state, control, **kwargs):
        if not bool(getattr(state, "is_world_process_zero", is_world_process_zero())):
            return control
        best = getattr(state, "best_model_checkpoint", None)
        if best:
            current = Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
            self.manager.update_checkpoint(best, selection="best", copy_to=current)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if not bool(getattr(state, "is_world_process_zero", is_world_process_zero())):
            return control
        best = getattr(state, "best_model_checkpoint", None)
        if best:
            self.manager.update_checkpoint(best, selection="best")
        return control
