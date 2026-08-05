from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from filelock import FileLock
from transformers import TrainerCallback


_SAFE_PREFIX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CHECKPOINT_NAME = re.compile(r"^checkpoint-(\d+)$")
_CONFIG_FINGERPRINT_VERSION = 1
MANUAL_CONFIG_PLACEHOLDER = "<manually_insert_value_here>"
MANUAL_DATASET_LENGTH_FIELDS_PLACEHOLDER = (
    "<manually_insert_model_dependent_length_fields_here>"
)
FINDING_POSTPROCESS_DEFAULTS = {
    "low_pass_fraction": 0.05,
    "peak_prominence": 0.15,
    "peak_distance": 50,
    "peak_height": None,
    "interval_window_size": 2_000_000,
    "max_pairs_per_seed": 10,
    "prob_threshold": 0.5,
    "zero_fraction_drop_threshold": 0.01,
    "pairing_progress_every": 1000,
}


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
    elastic_restart = str(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")).strip()
    if explicit:
        shared = f"explicit:{explicit}:restart:{elastic_restart}"
        robust = True
    elif elastic and elastic.lower() != "none":
        shared = f"elastic:{elastic}:restart:{elastic_restart}"
        robust = True
    else:
        # Workers launched by one local torchrun agent share a parent PID.  This
        # prevents a stale manifest from a previous launch from being reused.
        shared = (
            f"local:{os.environ.get('MASTER_ADDR', '')}:{os.environ.get('MASTER_PORT', '')}:"
            f"{world_size()}:{os.getppid()}:{elastic_restart}"
        )
        robust = int(os.environ.get("GROUP_WORLD_SIZE", "1")) <= 1
    material = f"{Path(config_path).expanduser().resolve()}|{shared}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20], robust


@dataclass(frozen=True)
class TrainingRunPlan:
    """One launch's immutable output directory and checkpoint decision."""

    run_dir: Path
    resume_from_checkpoint: Path | None
    config_fingerprint: str
    recovery_checkpoint: Path | None
    source_run_dir: Path | None
    automatic_restart: bool


def canonical_training_config(
    cfg: Dict[str, Any],
    *,
    output_base_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Return the semantic configuration used for automatic-restart matching.

    A saved ``training_config.json`` contains its timestamped child as
    ``training.output_dir`` and the original directory as
    ``training.output_base_dir``. Those runtime fields must compare equal to the
    source configuration that launched it. ``overwrite_output_dir`` is also
    generated/forced by the runner and has no training effect.
    """

    normalized = copy.deepcopy(cfg)
    training = normalized.get("training")
    if not isinstance(training, dict):
        raise RuntimeError("Training config must contain a training object")

    if output_base_dir is None:
        output_base_dir = training.get("output_base_dir", training.get("output_dir"))
    if output_base_dir in (None, ""):
        raise RuntimeError("training.output_dir must be set")
    training["output_dir"] = str(Path(output_base_dir).expanduser().resolve())
    training.pop("output_base_dir", None)
    training.pop("overwrite_output_dir", None)
    training.setdefault("automatic_restart", True)
    return normalized


def training_config_fingerprint(
    cfg: Dict[str, Any],
    *,
    output_base_dir: str | Path | None = None,
) -> str:
    """Hash normalized JSON, ignoring formatting and dictionary key order."""

    normalized = canonical_training_config(
        cfg,
        output_base_dir=output_base_dir,
    )
    material = json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    versioned = f"genatator-training-config-v{_CONFIG_FINGERPRINT_VERSION}\n{material}"
    return hashlib.sha256(versioned.encode("utf-8")).hexdigest()


def _direct_run_child(value: Any, base_dir: Path) -> Path | None:
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError):
        return None
    if resolved.parent != base_dir or not resolved.is_dir():
        return None
    return resolved


def _model_weights_are_complete(checkpoint: Path) -> bool:
    for name in (
        "pytorch_model.bin",
        "model.safetensors",
        "adapter_model.bin",
        "adapter_model.safetensors",
    ):
        if (checkpoint / name).is_file():
            return True

    for name in (
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
        "adapter_model.bin.index.json",
        "adapter_model.safetensors.index.json",
    ):
        index_path = checkpoint / name
        if not index_path.is_file():
            continue
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            shards = set(payload["weight_map"].values())
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            AttributeError,
        ):
            continue
        if shards and all((checkpoint / str(shard)).is_file() for shard in shards):
            return True
    return False


def is_complete_trainer_checkpoint(checkpoint: str | Path) -> bool:
    """Whether a standard Transformers checkpoint can faithfully resume."""

    checkpoint_path = Path(checkpoint).expanduser()
    match = _CHECKPOINT_NAME.fullmatch(checkpoint_path.name)
    if match is None or not checkpoint_path.is_dir():
        return False
    try:
        state = json.loads(
            (checkpoint_path / "trainer_state.json").read_text(encoding="utf-8")
        )
        state_step = int(state["global_step"])
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ):
        return False
    if state_step != int(match.group(1)):
        return False
    if not _model_weights_are_complete(checkpoint_path):
        return False
    return (
        (checkpoint_path / "optimizer.pt").is_file()
        and (checkpoint_path / "scheduler.pt").is_file()
    )


def newest_complete_checkpoint(run_dir: str | Path) -> Path | None:
    """Return the largest complete ``checkpoint-N`` inside one run only."""

    run_path = Path(run_dir).expanduser().resolve()
    candidates: list[tuple[int, Path]] = []
    try:
        children = list(run_path.iterdir())
    except OSError:
        return None
    for child in children:
        match = _CHECKPOINT_NAME.fullmatch(child.name)
        if match is not None and child.is_dir():
            candidates.append((int(match.group(1)), child))
    for _, checkpoint in sorted(candidates, key=lambda item: item[0], reverse=True):
        if is_complete_trainer_checkpoint(checkpoint):
            return checkpoint.resolve()
    return None


def _load_json_object(path: Path) -> Dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _latest_run_fingerprint(
    payload: Dict[str, Any],
    *,
    run_dir: Path,
    output_base_dir: Path,
) -> str | None:
    saved_config_path = run_dir / "training_config.json"
    if saved_config_path.exists():
        saved_config = _load_json_object(saved_config_path)
        if saved_config is None:
            return None
        try:
            return training_config_fingerprint(
                saved_config,
                output_base_dir=output_base_dir,
            )
        except (RuntimeError, TypeError, ValueError):
            return None

    try:
        version = int(payload.get("config_fingerprint_version", -1))
    except (TypeError, ValueError):
        return None
    if version != _CONFIG_FINGERPRINT_VERSION:
        return None
    fingerprint = payload.get("config_fingerprint")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        return None
    return fingerprint


def _recorded_recovery_checkpoint(payload: Dict[str, Any]) -> Path | None:
    value = payload.get("recovery_checkpoint")
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        checkpoint = Path(value).expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    return checkpoint if is_complete_trainer_checkpoint(checkpoint) else None


def _automatic_resume_checkpoint(
    *,
    output_base_dir: Path,
    config_fingerprint: str,
) -> tuple[Path | None, Path | None]:
    """Inspect exactly the manifest-designated latest run, never its siblings."""

    payload = _load_json_object(output_base_dir / "latest_run.json")
    if payload is None:
        return None, None
    latest_run = _direct_run_child(payload.get("run_dir"), output_base_dir)
    if latest_run is None:
        return None, None
    latest_fingerprint = _latest_run_fingerprint(
        payload,
        run_dir=latest_run,
        output_base_dir=output_base_dir,
    )
    if latest_fingerprint != config_fingerprint:
        return None, None
    checkpoint = newest_complete_checkpoint(latest_run)
    if checkpoint is None:
        checkpoint = _recorded_recovery_checkpoint(payload)
    return checkpoint, latest_run


def _checkpoint_from_payload(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise RuntimeError("Published resume checkpoint must be a string or null")
    return Path(value).expanduser().resolve()


def _plan_from_payload(payload: Dict[str, Any], base_dir: Path) -> TrainingRunPlan | None:
    run_dir = _direct_run_child(payload.get("run_dir"), base_dir)
    if run_dir is None:
        return None
    try:
        fingerprint = str(payload["config_fingerprint"])
        automatic_restart = payload["automatic_restart"]
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            return None
        if type(automatic_restart) is not bool:
            return None
        source_value = payload.get("source_run_dir")
        source_run = (
            _direct_run_child(source_value, base_dir)
            if source_value not in (None, "")
            else None
        )
        if source_value not in (None, "") and source_run is None:
            return None
        return TrainingRunPlan(
            run_dir=run_dir,
            resume_from_checkpoint=_checkpoint_from_payload(
                payload.get("resume_from_checkpoint")
            ),
            config_fingerprint=fingerprint,
            recovery_checkpoint=_checkpoint_from_payload(
                payload.get("recovery_checkpoint")
            ),
            source_run_dir=source_run,
            automatic_restart=automatic_restart,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None


def create_training_run(
    cfg: Dict[str, Any],
    *,
    config_path: str | Path,
    timeout_seconds: float = 120.0,
) -> TrainingRunPlan:
    """Create a timestamped run and resolve its explicit or automatic resume.

    Automatic matching consults only ``latest_run.json``. A matching latest
    continuation with no local checkpoint can pass its recorded recovery
    checkpoint forward, preventing repeated HPS restarts before the next save
    from losing the last durable state.
    """

    training = cfg.get("training")
    if not isinstance(training, dict):
        raise RuntimeError("Training config must contain a training object")
    configured_base = training.get("output_base_dir", training.get("output_dir"))
    if configured_base in (None, ""):
        raise RuntimeError("training.output_dir must be set")
    base_dir = Path(configured_base).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    prefix = _custom_prefix(training.get("custom_prefix", ""))
    automatic_restart = training.get("automatic_restart", True)
    if type(automatic_restart) is not bool:
        raise RuntimeError("training.automatic_restart must be true or false")

    manual_resume = training.get("resume_from_checkpoint")
    if isinstance(manual_resume, bool):
        if manual_resume:
            raise RuntimeError(
                "training.resume_from_checkpoint must be an explicit checkpoint path"
            )
        manual_resume = None
    elif manual_resume not in (None, "") and not isinstance(
        manual_resume, (str, os.PathLike)
    ):
        raise RuntimeError(
            "training.resume_from_checkpoint must be an explicit checkpoint path or null"
        )
    manual_checkpoint = (
        Path(manual_resume).expanduser().resolve()
        if manual_resume not in (None, "")
        else None
    )

    fingerprint = training_config_fingerprint(cfg, output_base_dir=base_dir)
    launch_hash, robust_multinode_identity = _launch_identity(config_path)
    if world_size() > 1 and not robust_multinode_identity:
        raise RuntimeError(
            "A multi-node launch needs a shared launch identity. Set GENATATOR_LAUNCH_ID "
            "to the same unique value on every node (or use torchrun with a non-'none' rendezvous id)."
        )
    launch_manifest = base_dir / f".genatator-run-{launch_hash}.json"

    if is_world_process_zero():
        lock = FileLock(
            str(base_dir / ".genatator-run-selection.lock"),
            timeout=float(timeout_seconds),
        )
        with lock:
            source_run: Path | None = None
            if manual_checkpoint is not None:
                resume = manual_checkpoint
            elif automatic_restart:
                resume, source_run = _automatic_resume_checkpoint(
                    output_base_dir=base_dir,
                    config_fingerprint=fingerprint,
                )
            else:
                resume = None

            run_dir = _exclusive_timestamped_child(base_dir, prefix)
            payload: Dict[str, Any] = {
                "automatic_restart": automatic_restart,
                "config_fingerprint": fingerprint,
                "config_fingerprint_version": _CONFIG_FINGERPRINT_VERSION,
                "created_at": time.time(),
                "rank0_pid": os.getpid(),
                "recovery_checkpoint": str(resume) if resume is not None else None,
                "resume_from_checkpoint": str(resume) if resume is not None else None,
                "run_dir": str(run_dir),
                "source_run_dir": str(source_run) if source_run is not None else None,
                "world_size": world_size(),
            }
            atomic_save_json(payload, launch_manifest)
            atomic_save_json(payload, base_dir / "latest_run.json")
        plan = _plan_from_payload(payload, base_dir)
        if plan is None:
            raise RuntimeError("Internal error: invalid rank-0 training run plan")
        return plan

    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        payload = _load_json_object(launch_manifest)
        if payload is not None:
            plan = _plan_from_payload(payload, base_dir)
            if (
                plan is not None
                and int(payload.get("world_size", -1)) == world_size()
                and plan.config_fingerprint == fingerprint
            ):
                return plan
        time.sleep(0.1)
    raise TimeoutError(
        f"Timed out waiting for rank 0 to publish the training run plan in {launch_manifest}"
    )


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
    # Training never uses reverse-complement augmentation. Generated inference
    # configs expose the inference-only switches explicitly with project defaults.
    true_gff = cfg.get("true_gff")

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
        "device": "cuda",
        "checkpoint_path": None,
        "batch_size": 1,
        "use_reverse_complement": True,
        "true_gff": true_gff,
    }

    generated = {
        "task": task,
        "run_dir": str(run_dir),
        "checkpoint_selection": "pending",
        "best_checkpoint": None,
    }

    if task in {"finding_edge", "finding_region"}:
        # Gene finding has a dedicated held-out test split. Its final benchmark
        # always runs the joint edge + region pipeline so it can emit GFF3 and
        # the official annotation metrics, not only per-stage PR-AUC.
        # The dataset uses the complete T2T assembly identifier.  Filtering is
        # exact, so the shorter NCBI accession used by the transcript datasets
        # would select no gene-finding test rows.
        dataset_cfg["genomes"] = ["GCF_009914755.1_T2T-CHM13v2.0"]
        dataset_cfg["chromosomes"] = ["NC_060944.1"]
        dataset_cfg["split"] = "test"
        dataset_cfg.pop("statuses", None)
        trained_stage = task.removeprefix("finding_")
        manual_stage = "region" if trained_stage == "edge" else "edge"
        generated["trained_stage"] = trained_stage
        generated["manual_stage"] = manual_stage
        trained_stage_config = {
            "model": model_cfg,
            "dataset": dataset_cfg,
            "inference": {
                "checkpoint_path": None,
                "batch_size": 1,
            },
        }
        manual_dataset_config = copy.deepcopy(dataset_cfg)
        # Dataset identity and benchmark filters are known. Only the length
        # fields depend on whether the counterpart is nucleotide- or BPE-based.
        for length_key in (
            "max_nucleotides",
            "max_bpe_tokens",
            "average_bpe_token_length",
        ):
            manual_dataset_config.pop(length_key, None)
        # Replace this marker entry with either max_nucleotides (Caduceus) or
        # max_bpe_tokens + average_bpe_token_length (GENA/ModernGENA).
        manual_dataset_config[MANUAL_DATASET_LENGTH_FIELDS_PLACEHOLDER] = (
            MANUAL_CONFIG_PLACEHOLDER
        )
        manual_stage_config = {
            # The counterpart may use any shipped architecture. A generic
            # partial model object can silently instantiate the wrong class
            # (for example plain AMT instead of AMT + UNET), so require the
            # complete trained model block to be pasted here explicitly.
            "model": MANUAL_CONFIG_PLACEHOLDER,
            "dataset": manual_dataset_config,
            "inference": {
                "checkpoint_path": MANUAL_CONFIG_PLACEHOLDER,
                "batch_size": 1,
            },
        }
        stages = {
            trained_stage: trained_stage_config,
            manual_stage: manual_stage_config,
        }
        return {
            "task": "finding",
            "edge": stages["edge"],
            "region": stages["region"],
            "postprocess": copy.deepcopy(FINDING_POSTPROCESS_DEFAULTS),
            "inference": {
                "device": "cuda",
                "batch_size": 1,
                "use_reverse_complement": True,
                "output_gff": _absolute_output(run_dir, "finding_predictions.gff"),
                "true_gff": true_gff or MANUAL_CONFIG_PLACEHOLDER,
                "metrics_json": _absolute_output(run_dir, "finding_metrics.json"),
                "metrics_csv": _absolute_output(run_dir, "finding_auc_metrics.csv"),
                "k_values": [0, 50, 100, 250, 500],
                "use_strand": True,
                "empty_gff_policy": "error",
            },
            "_generated": generated,
        }

    if task == "segmentation":
        # Final evaluation uses every transcript/isoform from val-human and
        # gathers non-overlapping model-sized chunks over complete transcripts.
        dataset_cfg["genomes"] = ["GCF_009914755.1"]
        dataset_cfg["chromosomes"] = ["NC_060944.1"]
        dataset_cfg["config_name"] = "val-human"
        dataset_cfg["split"] = "validation"
        dataset_cfg.pop("statuses", None)
        dataset_cfg.pop("random_crop", None)
        dataset_cfg.pop("overlap", None)
        dataset_cfg.pop("target_group", None)
        dataset_cfg["full_transcript_chunks"] = True
        common.update(
            {
                "use_cds_heuristic": True,
                "coordinate_mode": "transcript",
                "empty_segment_policy": "error",
                "output_gff": _absolute_output(run_dir, "segmentation_predictions.gff"),
                "metrics_json": _absolute_output(run_dir, "segmentation_metrics.json"),
            }
        )
        return {
            "task": task,
            "model": model_cfg,
            "dataset": dataset_cfg,
            "inference": common,
            "_generated": generated,
        }

    if task == "transcript_type":
        dataset_cfg["genomes"] = ["GCF_009914755.1"]
        dataset_cfg["chromosomes"] = ["NC_060944.1"]
        dataset_cfg["config_name"] = "val-human"
        dataset_cfg["split"] = "validation"
        dataset_cfg.pop("statuses", None)
        dataset_cfg.pop("random_crop", None)
        dataset_cfg.pop("overlap", None)
        dataset_cfg.pop("target_group", None)
        common.update(
            {
                "threshold": 0.5,
                "output_tsv": _absolute_output(run_dir, "transcript_type_predictions.tsv"),
                "metrics_json": _absolute_output(run_dir, "transcript_type_metrics.json"),
            }
        )
        return {
            "task": task,
            "model": model_cfg,
            "dataset": dataset_cfg,
            "inference": common,
            "_generated": generated,
        }
    raise RuntimeError(f"Unsupported task for automatic evaluation config: {task}")


class EvaluationConfigManager:
    def __init__(self, cfg: Dict[str, Any], *, task: str, run_dir: str | Path):
        self.task = str(task)
        self.run_dir = Path(run_dir).resolve()
        self.path = self.run_dir / "evaluation_config.json"
        self.config = build_evaluation_config(cfg, task=task, run_dir=self.run_dir)

    def _checkpoint_inference_config(self) -> Dict[str, Any]:
        if self.task in {"finding_edge", "finding_region"}:
            stage = self.task.removeprefix("finding_")
            return self.config[stage]["inference"]
        return self.config["inference"]

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
        self._checkpoint_inference_config()["checkpoint_path"] = str(checkpoint_path)
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
