#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from tqdm.auto import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_index import (  # noqa: E402
    GeneFindingSelection,
    TranscriptSelection,
    prepare_gene_finding_selection,
    prepare_transcript_selection,
)

MODELS = {
    "caduceus_ps": {"kind": "caduceus", "path": "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"},
    "caduceus_ph": {"kind": "caduceus", "path": "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"},
    "gena_base": {"kind": "gena", "path": "AIRI-Institute/gena-lm-bert-base-lastln-t2t"},
    "gena_large": {"kind": "gena", "path": "AIRI-Institute/gena-lm-bert-large-t2t"},
    "moderngena_base": {"kind": "moderngena", "path": "AIRI-Institute/moderngena-base"},
    "moderngena_large": {"kind": "moderngena", "path": "AIRI-Institute/moderngena-large"},
}
DEFAULT_CHROMOSOME = "NC_060944.1"

SMOKE_EPOCHS = 4
SMOKE_LR = 1e-4


def write_json(path: Path, obj: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path


def aliases_from_reference_gff(reference_gff: Path, requested_chromosome: str) -> List[str]:
    aliases = [requested_chromosome]
    if requested_chromosome.lower().startswith("chr"):
        aliases.append(requested_chromosome[3:])
    elif requested_chromosome.isdigit():
        aliases.append(f"chr{requested_chromosome}")
    with reference_gff.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            seqid = line.split("\t", 1)[0].strip()
            if seqid and seqid not in aliases:
                aliases.insert(0, seqid)
            break
    return list(dict.fromkeys(aliases))


def model_cfg(model_name: str, family: str, extra: Optional[dict] = None) -> dict:
    info = MODELS[model_name]
    cfg = {
        "family": family,
        "backbone_kind": info["kind"],
        "backbone_path": info["path"],
        "tokenizer_path": info["path"],
        "trust_remote_code": True,
        "allow_unsafe_torch_load_with_torch_lt_2_6": True,
        "checkpoint_path": None,
    }
    if info["kind"] == "caduceus":
        cfg["bidirectional_weight_tie"] = False
        cfg["padding_side"] = "left"
    if family in {"unet", "rmt"} or (family == "amt" and bool((extra or {}).get("use_unet", False))):
        # Single-nucleotide IDs are read from the main model tokenizer.
        cfg.update({
            "vocab_size": None,
            "unet_chunk_size": 8192,
        })
    if extra:
        cfg.update(extra)
    return cfg


def smoke_training(
    output_dir: Path,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    task: str,
) -> dict:
    if task in {"finding_edge", "finding_region"}:
        best_metric = "loss"
        greater_is_better = False
    elif task == "segmentation":
        best_metric = "interval_f1_exon"
        greater_is_better = True
    elif task == "transcript_type":
        best_metric = "accuracy"
        greater_is_better = True
    else:
        raise RuntimeError(f"Unsupported smoke training task: {task}")
    return {
        "output_dir": str(output_dir),
        "custom_prefix": "smoke",
        "overwrite_output_dir": True,
        "max_steps": -1,
        "num_train_epochs": float(epochs),
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": float(learning_rate),
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "lr_scheduler_type": "constant",
        "logging_strategy": "epoch",
        "logging_interval": 1,
        "logging_steps": 1,
        "evaluation_strategy": "epoch",
        "eval_steps": 1,
        "save_strategy": "epoch",
        "save_steps": 1,
        "patience": 100,
        "save_total_limit": 1,
        "save_safetensors": False,
        "load_best_model_at_end": True,
        "metric_for_best_model": best_metric,
        "greater_is_better": greater_is_better,
        "dataloader_num_workers": 0,
        "bf16": False,
        "fp16": False,
        "resume_from_checkpoint": None,
        "sequential_train": True,
    }


def _length_fields(family: str, max_nt: int, max_tok: int) -> dict:
    if family == "caduceus":
        return {"max_nucleotides": int(max_nt)}
    if max_tok <= 0:
        raise RuntimeError(f"Smoke max_tok must be positive, got {max_tok}")
    return {
        "max_bpe_tokens": int(max_tok),
        "average_bpe_token_length": float(max_nt) / float(max_tok),
    }


def finding_data(path: Path, aliases: List[str], family: str, max_nt: int, max_tok: int) -> dict:
    cfg = {
        "path": str(path),
        "split": "test",
        "genomes": None,
        "chromosomes": aliases,
        "overlap": 0.5,
        "target_group": "primary",
        "prewindowed": False,
        "max_rows": None,
        "max_windows": None,
        "streaming": False,
    }
    cfg.update(_length_fields(family, max_nt, max_tok))
    return cfg


def transcript_data(path: Path, aliases: List[str], family: str, max_nt: int, max_tok: int, random_crop: bool | None = None) -> dict:
    cfg = {
        "path": str(path),
        "split": "test",
        "genomes": None,
        "chromosomes": aliases,
        "crop_margin": 500,
        "statuses": None,
        "max_rows": None,
        "streaming": False,
        "loader": "direct_parquet",
        "parquet_batch_size": 64,
    }
    if random_crop is not None:
        cfg["random_crop"] = bool(random_crop)
    cfg.update(_length_fields(family, max_nt, max_tok))
    return cfg


def make_finding_train_config(
    work: Path,
    selection: GeneFindingSelection,
    aliases: List[str],
    model_name: str,
    task: str,
    variant: str,
) -> Path:
    max_tok = 64 if task == "edge" else 128
    max_nt = 512 if task == "edge" else 1024
    data_path = selection.selected_index_path
    family = "caduceus" if MODELS[model_name]["kind"] == "caduceus" else variant
    extra = None
    if family == "unet":
        extra = {"unet_cycles": 1}
    elif family == "rmt":
        memory_tokens = 10 if MODELS[model_name]["kind"] == "gena" else 20
        extra = {
            "cycles": 1,
            "rmt": {
                    "segment_size": 64,
                    "max_n_segments": 8,
                    "num_mem_tokens": memory_tokens,
                    "bptt_depth": -1,
                },
        }
    elif family == "amt":
        memory_tokens = 10 if MODELS[model_name]["kind"] == "gena" else 20
        extra = {
            "use_unet": False,
            "amt": {
                "amt_repo_id": "irodkin/armt-neox-tiny",
                "num_mem_tokens": memory_tokens,
                "d_mem": 64,
                "segment_size": 64 - memory_tokens,
                "segment_alignment": "left",
                "sliding_window": False,
                "wrap_pos": False,
                "correction": True,
                "n_heads": 1,
                "use_denom": True,
                "gating": False,
                "act_on": False,
            },
        }
    name = f"finding_{task}_{model_name}_{family}"
    bs = 1
    dataset = finding_data(data_path, aliases, family, max_nt, max_tok)
    cfg = {
        "seed": 42,
        "task": f"finding_{task}",
        "model": model_cfg(model_name, family, extra),
        # Smoke protocol: train and validation use the same
        # chromosome-selected test samples.
        "train_dataset": dataset,
        "eval_dataset": dict(dataset),
        "true_gff": None,
        "training": smoke_training(work / name, bs, SMOKE_EPOCHS, SMOKE_LR, f"finding_{task}"),
    }
    return write_json(work / "configs" / f"{name}.json", cfg)


def make_seg_train_config(
    work: Path,
    selection: TranscriptSelection,
    aliases: List[str],
    model_name: str,
    variant: str,
) -> Path:
    kind = MODELS[model_name]["kind"]
    if kind == "caduceus":
        family, extra, max_nt, max_tok, bs = "caduceus", None, 512, 512, 1
    else:
        family, max_nt, max_tok, bs = variant, 512, 64, 1
        if family == "unet":
            extra = {"unet_cycles": 1}
        elif family == "rmt":
            memory_tokens = 10 if MODELS[model_name]["kind"] == "gena" else 20
            extra = {
                "cycles": 1,
                "rmt": {
                    "segment_size": 64,
                    "max_n_segments": 8,
                    "num_mem_tokens": memory_tokens,
                    "bptt_depth": -1,
                },
            }
        elif family == "amt":
            memory_tokens = 10 if MODELS[model_name]["kind"] == "gena" else 20
            extra = {
                "use_unet": True,
                "unet_cycles": 1,
                "amt": {
                    "amt_repo_id": "irodkin/armt-neox-tiny",
                    "num_mem_tokens": memory_tokens,
                    "d_mem": 64,
                    "segment_size": 64 - memory_tokens,
                    "segment_alignment": "left",
                    "sliding_window": False,
                    "wrap_pos": False,
                    "correction": True,
                    "n_heads": 1,
                    "use_denom": True,
                    "gating": False,
                    "act_on": False,
                },
            }
        else:
            raise RuntimeError(f"Unsupported segmentation family={family}")
    name = f"segmentation_{model_name}_{family}"
    dataset = transcript_data(
        selection.selected_parquet_path, aliases, family, max_nt, max_tok,
        random_crop=(kind == "caduceus"),
    )
    cfg = {
        "seed": 42,
        "task": "segmentation",
        "model": model_cfg(model_name, family, extra),
        "train_dataset": dataset,
        "eval_dataset": dict(dataset),
        "true_gff": None,
        "training": smoke_training(work / name, bs, SMOKE_EPOCHS, SMOKE_LR, "segmentation"),
    }
    return write_json(work / "configs" / f"{name}.json", cfg)


def make_tt_train_config(
    work: Path,
    selection: TranscriptSelection,
    aliases: List[str],
    model_name: str,
) -> Path:
    kind = MODELS[model_name]["kind"]
    family = "caduceus" if kind == "caduceus" else "plain"
    max_nt = 512
    max_tok = 512 if kind == "caduceus" else 64
    bs = 1
    name = f"transcript_type_{model_name}_{family}"
    dataset = transcript_data(selection.selected_parquet_path, aliases, family, max_nt, max_tok)
    cfg = {
        "seed": 42,
        "task": "transcript_type",
        "model": model_cfg(model_name, family),
        "train_dataset": dataset,
        "eval_dataset": dict(dataset),
        "true_gff": None,
        "training": smoke_training(work / name, bs, SMOKE_EPOCHS, SMOKE_LR, "transcript_type"),
    }
    return write_json(work / "configs" / f"{name}.json", cfg)


def make_finding_infer_config(
    work: Path,
    selection: GeneFindingSelection,
    aliases: List[str],
    model_name: str,
    variant: str,
    true_gff: str,
) -> Path:
    edge_train = work / f"finding_edge_{model_name}_{variant}"
    region_train = work / f"finding_region_{model_name}_{variant}"
    edge_train_cfg = json.loads((work / "configs" / f"finding_edge_{model_name}_{variant}.json").read_text())
    region_train_cfg = json.loads((work / "configs" / f"finding_region_{model_name}_{variant}.json").read_text())
    edge_cfg = {
        "model": edge_train_cfg["model"],
        "dataset": finding_data(
            selection.selected_index_path,
            aliases,
            edge_train_cfg["model"]["family"],
            512,
            64,
        ),
        "inference": {"checkpoint_path": str(edge_train / "final_model"), "batch_size": 1},
    }
    region_cfg = {
        "model": region_train_cfg["model"],
        "dataset": finding_data(
            selection.selected_index_path,
            aliases,
            region_train_cfg["model"]["family"],
            1024,
            128,
        ),
        "inference": {"checkpoint_path": str(region_train / "final_model"), "batch_size": 1},
    }
    cfg = {
        "edge": edge_cfg,
        "region": region_cfg,
        "postprocess": {
            "low_pass_fraction": 0.05,
            "peak_prominence": 0.15,
            "peak_distance": 50,
            "peak_height": None,
            "interval_window_size": 2_000_000,
            "max_pairs_per_seed": 10,
            "prob_threshold": 0.5,
            "zero_fraction_drop_threshold": 0.01,
            "pairing_progress_every": 1000,
        },
        "inference": {
            "device": "cuda",
            "batch_size": 1,
            "use_reverse_complement": True,
            "output_gff": str(work / f"finding_{model_name}_{variant}.gff"),
            "true_gff": true_gff,
            "metrics_json": str(work / f"finding_{model_name}_{variant}.metrics.json"),
            "k_values": [0, 50, 100, 250, 500],
            "use_strand": True,
            "empty_gff_policy": "best_interval",
            "empty_gff_min_interval_len": 64,
            "empty_gff_max_records": 1,
        },
    }
    return write_json(work / "configs" / f"infer_finding_{model_name}_{variant}.json", cfg)


def make_seg_infer_config(
    work: Path,
    selection: TranscriptSelection,
    aliases: List[str],
    model_name: str,
    variant: str,
    true_gff: str,
) -> Path:
    family = "caduceus" if MODELS[model_name]["kind"] == "caduceus" else variant
    train_dir = work / f"segmentation_{model_name}_{family}"
    train_cfg = json.loads((work / "configs" / f"segmentation_{model_name}_{family}.json").read_text())
    cfg = {
        "model": train_cfg["model"],
        "dataset": {**transcript_data(
            selection.selected_parquet_path,
            aliases,
            family,
            512,
            512 if family == "caduceus" else 64,
        ), "full_transcript_chunks": True},
        "inference": {
            "device": "cuda",
            "checkpoint_path": str(train_dir / "final_model"),
            "batch_size": 1,
            "use_reverse_complement": True,
            "use_cds_heuristic": True,
            "threshold": 0.5,
            "empty_segment_policy": "best_interval",
            "coordinate_mode": "transcript",
            "output_gff": str(work / f"segmentation_{model_name}_{family}.gff"),
            "true_gff": true_gff,
            "metrics_json": str(work / f"segmentation_{model_name}_{family}.metrics.json"),
        },
    }
    return write_json(work / "configs" / f"infer_segmentation_{model_name}_{family}.json", cfg)


def make_tt_infer_config(
    work: Path,
    selection: TranscriptSelection,
    aliases: List[str],
    model_name: str,
) -> Path:
    family = "caduceus" if MODELS[model_name]["kind"] == "caduceus" else "plain"
    train_dir = work / f"transcript_type_{model_name}_{family}"
    train_cfg = json.loads((work / "configs" / f"transcript_type_{model_name}_{family}.json").read_text())
    cfg = {
        "model": train_cfg["model"],
        "dataset": transcript_data(
            selection.selected_parquet_path,
            aliases,
            family,
            512,
            512 if family == "caduceus" else 64,
        ),
        "inference": {
            "device": "cuda",
            "checkpoint_path": str(train_dir / "final_model"),
            "batch_size": 1,
            "use_reverse_complement": True,
            "threshold": 0.5,
            "true_gff": None,
            "output_tsv": str(work / f"transcript_type_{model_name}_{family}.tsv"),
            "metrics_json": str(work / f"transcript_type_{model_name}_{family}.metrics.json"),
        },
    }
    return write_json(work / "configs" / f"infer_transcript_type_{model_name}_{family}.json", cfg)


def build_jobs(
    work: Path,
    true_gff: str,
    gf_selection: GeneFindingSelection,
    tx_selection: TranscriptSelection,
    aliases: List[str],
) -> List[dict]:
    jobs: List[dict] = []
    for model_name, info in MODELS.items():
        variants = ["caduceus"] if info["kind"] == "caduceus" else ["plain", "unet"]
        if model_name == "moderngena_base":
            variants += ["rmt", "amt"]
        for variant in variants:
            for task in ["edge", "region"]:
                cfg = make_finding_train_config(work, gf_selection, aliases, model_name, task, variant)
                output_dir = work / f"finding_{task}_{model_name}_{variant}"
                jobs.append(
                    {
                        "name": f"train_finding_{task}_{model_name}_{variant}",
                        "kind": "train",
                        "output_dir": str(output_dir),
                        "cmd": [sys.executable, "finding/train.py", "--config", str(cfg)],
                        "deps": [],
                    }
                )
            infer_cfg = make_finding_infer_config(work, gf_selection, aliases, model_name, variant, true_gff)
            jobs.append(
                {
                    "name": f"infer_finding_{model_name}_{variant}",
                    "kind": "infer",
                    "cmd": [sys.executable, "finding/infer.py", "--config", str(infer_cfg)],
                    "deps": [
                        f"train_finding_edge_{model_name}_{variant}",
                        f"train_finding_region_{model_name}_{variant}",
                    ],
                }
            )

    for model_name, info in MODELS.items():
        seg_variants = ["caduceus"] if info["kind"] == "caduceus" else ["unet"]
        if model_name == "moderngena_base":
            seg_variants += ["rmt", "amt"]
        for variant in seg_variants:
            cfg = make_seg_train_config(work, tx_selection, aliases, model_name, variant)
            family = "caduceus" if info["kind"] == "caduceus" else variant
            output_dir = work / f"segmentation_{model_name}_{family}"
            jobs.append(
                {
                    "name": f"train_segmentation_{model_name}_{family}",
                    "kind": "train",
                    "output_dir": str(output_dir),
                    "cmd": [sys.executable, "segmentation/train.py", "--config", str(cfg)],
                    "deps": [],
                }
            )
            infer_cfg = make_seg_infer_config(work, tx_selection, aliases, model_name, variant, true_gff)
            jobs.append(
                {
                    "name": f"infer_segmentation_{model_name}_{family}",
                    "kind": "infer",
                    "cmd": [sys.executable, "segmentation/infer.py", "--config", str(infer_cfg)],
                    "deps": [f"train_segmentation_{model_name}_{family}"],
                }
            )

        cfg = make_tt_train_config(work, tx_selection, aliases, model_name)
        family = "caduceus" if info["kind"] == "caduceus" else "plain"
        output_dir = work / f"transcript_type_{model_name}_{family}"
        jobs.append(
            {
                "name": f"train_transcript_type_{model_name}_{family}",
                "kind": "train",
                "output_dir": str(output_dir),
                "cmd": [sys.executable, "transcript_type/train.py", "--config", str(cfg)],
                "deps": [],
            }
        )
        infer_cfg = make_tt_infer_config(work, tx_selection, aliases, model_name)
        jobs.append(
            {
                "name": f"infer_transcript_type_{model_name}_{family}",
                "kind": "infer",
                "cmd": [sys.executable, "transcript_type/infer.py", "--config", str(infer_cfg)],
                "deps": [f"train_transcript_type_{model_name}_{family}"],
            }
        )
    return jobs


def _latest_run_dir(output_base: Path) -> Path:
    manifest = output_base / "latest_run.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        run_dir = Path(str(payload["run_dir"])).expanduser()
        if not run_dir.is_absolute():
            run_dir = output_base / run_dir
        if run_dir.exists():
            return run_dir.resolve()
    candidates = [
        p for p in output_base.iterdir()
        if p.is_dir() and ((p / "trainer_state.json").exists() or (p / "final_model").exists())
    ] if output_base.exists() else []
    if not candidates:
        raise RuntimeError(
            f"Could not discover timestamped training run under {output_base}; "
            f"missing/invalid {manifest}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime).resolve()


def _refresh_inference_checkpoint_paths(config_path: Path) -> None:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    changed = False

    def visit(node):
        nonlocal changed
        if isinstance(node, dict):
            inference = node.get("inference")
            if isinstance(inference, dict) and inference.get("checkpoint_path"):
                configured = Path(str(inference["checkpoint_path"])).expanduser()
                if configured.name == "final_model":
                    run_dir = _latest_run_dir(configured.parent)
                    resolved = run_dir / "final_model"
                    if not resolved.exists():
                        raise RuntimeError(f"Timestamped smoke run has no final_model: {resolved}")
                    inference["checkpoint_path"] = str(resolved)
                    changed = True
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(cfg)
    if changed:
        write_json(config_path, cfg)


def _training_loss_summary(run_dir: Path) -> dict:
    state_path = run_dir / "trainer_state.json"
    if not state_path.exists():
        raise RuntimeError(f"Training job did not write trainer_state.json: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    history = state.get("log_history", [])
    train_losses = [float(x["loss"]) for x in history if "loss" in x and "eval_loss" not in x]
    eval_losses = [float(x["eval_loss"]) for x in history if "eval_loss" in x]
    summary = {
        "train_loss_first": train_losses[0] if train_losses else None,
        "train_loss_last": train_losses[-1] if train_losses else None,
        "eval_loss_first": eval_losses[0] if eval_losses else None,
        "eval_loss_last": eval_losses[-1] if eval_losses else None,
        "train_loss_values": train_losses,
        "eval_loss_values": eval_losses,
    }
    if len(train_losses) >= 2 and train_losses[0] != 0:
        summary["train_loss_relative_change"] = (train_losses[-1] - train_losses[0]) / abs(train_losses[0])
    else:
        summary["train_loss_relative_change"] = None
    if len(eval_losses) >= 2 and eval_losses[0] != 0:
        summary["eval_loss_relative_change"] = (eval_losses[-1] - eval_losses[0]) / abs(eval_losses[0])
    else:
        summary["eval_loss_relative_change"] = None
    return summary


def run_scheduler(jobs: List[dict], gpus: List[str], work: Path) -> dict:
    pending = {j["name"]: j for j in jobs}
    done: dict = {}
    running: dict = {}
    free_gpus = list(gpus)
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    progress = tqdm(total=len(jobs), desc="smoke train/validation/test jobs")
    try:
        while pending or running:
            launched = True
            while launched and free_gpus:
                launched = False
                for name, job in list(pending.items()):
                    if all(dep in done for dep in job["deps"]):
                        if job["kind"] == "infer" and "--config" in job["cmd"]:
                            config_i = job["cmd"].index("--config") + 1
                            _refresh_inference_checkpoint_paths(Path(job["cmd"][config_i]))
                        gpu = free_gpus.pop(0)
                        env = os.environ.copy()
                        env["CUDA_VISIBLE_DEVICES"] = gpu
                        env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
                        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
                        env["GENATATOR_SMOKE_ENFORCE_LOCAL_DATA"] = "1"
                        log_path = logs / f"{name}.log"
                        fh = open(log_path, "w", encoding="utf-8")
                        proc = subprocess.Popen(
                            job["cmd"],
                            cwd=str(REPO),
                            env=env,
                            stdout=fh,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        running[name] = {
                            "proc": proc,
                            "fh": fh,
                            "gpu": gpu,
                            "start": time.time(),
                            "log": log_path,
                            "job": job,
                        }
                        progress.set_postfix_str(f"launched={name} gpu={gpu}")
                        del pending[name]
                        launched = True
                        break
            time.sleep(2)
            for name, state in list(running.items()):
                ret = state["proc"].poll()
                if ret is None:
                    continue
                state["fh"].close()
                duration = time.time() - state["start"]
                if ret != 0:
                    tail = state["log"].read_text(encoding="utf-8", errors="replace").splitlines()[-100:]
                    raise RuntimeError(
                        f"Smoke job failed: {name} exit_code={ret} gpu={state['gpu']} log={state['log']} "
                        f"cmd={' '.join(state['job']['cmd'])}\n--- log tail ---\n" + "\n".join(tail)
                    )
                result = {"duration_s": duration, "log": str(state["log"]), "kind": state["job"]["kind"]}
                if state["job"]["kind"] == "train":
                    run_dir = _latest_run_dir(Path(state["job"]["output_dir"]))
                    result["run_dir"] = str(run_dir)
                    result["training_loss_summary"] = _training_loss_summary(run_dir)
                done[name] = result
                free_gpus.append(state["gpu"])
                del running[name]
                progress.update(1)
                progress.set_postfix_str(f"done={name}")
        return done
    except Exception:
        for state in running.values():
            try:
                os.killpg(state["proc"].pid, signal.SIGTERM)
            except Exception:
                pass
            try:
                state["fh"].close()
            except Exception:
                pass
        raise
    finally:
        progress.close()


def collect_metric_files(work: Path) -> List[Path]:
    return sorted(work.glob("**/*.metrics.json")) + sorted(work.glob("**/trainer_state.json"))


def window_count(length: int, context: int, overlap: float = 0.5) -> int:
    if length <= context:
        return 1
    step = max(1, int(context * (1.0 - overlap)))
    return ((length - context + step - 1) // step) + 1


def write_summary(
    work: Path,
    done: dict,
    gf_selection: GeneFindingSelection,
    tx_selection: TranscriptSelection,
) -> Path:
    lines = [
        "# GENATATOR smoke-test summary",
        "",
        "## Selected real test data",
        "",
        f"- Gene-finding source split: `test`",
        f"- Gene-finding source test samples scanned: **{gf_selection.source_samples_scanned}**",
        f"- Gene-finding selected chromosome blocks found during scan: **{gf_selection.selected_blocks}**",
        f"- Gene-finding source chromosome length from selected blocks: **{gf_selection.source_chromosome_length:,} nt**",
        f"- Gene-finding smoke training length: **{gf_selection.assembled_length:,} nt** ({gf_selection.smoke_fraction:.0%} of the first selected block)",
        f"- Gene-finding edge windows per epoch (512 nt, 50% overlap): **{window_count(gf_selection.assembled_length, 512):,}**",
        f"- Gene-finding region windows per epoch (1024 nt, 50% overlap): **{window_count(gf_selection.assembled_length, 1024):,}**",
        f"- Transcript source rows scanned: **{tx_selection.source_rows_scanned}**",
        f"- Transcript source: `val-human/validation` (held-out chromosome test role)",
        f"- Selected transcript rows: **{tx_selection.selected_rows}**",
        f"- Selected transcript nucleotides: **{tx_selection.total_nucleotides:,} nt**",
        f"- Transcript-type counts: `{json.dumps(tx_selection.transcript_type_counts, ensure_ascii=False)}`",
        "",
        f"Total jobs: {len(done)}",
        "",
        "## Jobs",
        "",
        "| job | kind | duration_s | loss summary | log |",
        "|---|---|---:|---|---|",
    ]
    for name, st in done.items():
        loss = st.get("training_loss_summary")
        loss_text = ""
        if loss:
            loss_text = (
                f"train {loss['train_loss_first']}→{loss['train_loss_last']}; "
                f"eval {loss['eval_loss_first']}→{loss['eval_loss_last']}"
            )
        lines.append(f"| `{name}` | {st['kind']} | {st['duration_s']:.1f} | {loss_text} | `{st['log']}` |")
    lines += ["", "## Training loss details", ""]
    for name, st in done.items():
        if "training_loss_summary" not in st:
            continue
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(st["training_loss_summary"], indent=2))
        lines.append("```")
        lines.append("")
    lines += ["", "## Metrics files", ""]
    for path in collect_metric_files(work):
        rel = path.relative_to(work)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            preview = json.dumps(data, ensure_ascii=False)[:1500]
        except Exception as exc:
            preview = f"Could not parse: {exc}"
        lines += [f"### `{rel}`", "", "```json", preview, "```", ""]
    summary = work / "summary.md"
    summary.write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all GENATATOR smoke tests on one real held-out chromosome."
    )
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs; overrides --num-gpus.")
    parser.add_argument("--reference-gff", required=True)
    parser.add_argument("--requested-chromosome", default=DEFAULT_CHROMOSOME)
    parser.add_argument("--work-dir", default="smoke_tests/runs")
    parser.add_argument("--index-dir", default="smoke_tests/indexes")
    parser.add_argument("--selected-data-dir", default="smoke_tests/selected_data")
    parser.add_argument("--smoke-cache-dir", default=None, help="Backward-compatible alias for --selected-data-dir. Selection indexes still stay under smoke_tests/indexes by default.")
    parser.add_argument("--gene-finding-dataset-path", default=None, help="Optional local root/file for the gene-finding dataset test split.")
    parser.add_argument("--gene-finding-local-parquet", default=None, help="Backward-compatible alias for --gene-finding-dataset-path.")
    parser.add_argument("--segmentation-dataset-path", default=None, help="Optional local val-human parquet or repository root.")
    parser.add_argument("--segmentation-local-parquet", default=None, help="Backward-compatible alias for --segmentation-dataset-path.")
    parser.add_argument("--hf-local-files-only", action="store_true")
    parser.add_argument("--refresh-index", action="store_true")
    parser.add_argument("--metadata-batch-size", type=int, default=16)
    parser.add_argument("--smoke-epochs", type=int, default=4, help="The only smoke-training size control: complete passes over every selected sample/window.")
    args = parser.parse_args()

    global SMOKE_EPOCHS
    SMOKE_EPOCHS = int(args.smoke_epochs)
    if SMOKE_EPOCHS < 1:
        raise RuntimeError("Smoke tests require at least 1 epoch")

    reference_gff = Path(args.reference_gff).expanduser().resolve()
    if not reference_gff.exists():
        raise FileNotFoundError(reference_gff)
    aliases = aliases_from_reference_gff(reference_gff, args.requested_chromosome)
    print(f"Requested chromosome aliases: {aliases}")
    print(
        f"Smoke training protocol: epochs={SMOKE_EPOCHS}; gene finding uses the configured selected "
        f"real test subset; segmentation/transcript-type use all selected transcripts; "
        f"learning_rate={SMOKE_LR}. The runner records metrics but does not enforce any metric-based success criterion."
    )

    index_dir = (REPO / args.index_dir).resolve()
    if args.smoke_cache_dir:
        selected_data_dir = Path(args.smoke_cache_dir).expanduser().resolve()
        print("[compatibility] --smoke-cache-dir is being used as the selected-data directory")
    else:
        selected_data_dir = (REPO / args.selected_data_dir).resolve()
    gene_finding_dataset_path = args.gene_finding_dataset_path or args.gene_finding_local_parquet
    segmentation_dataset_path = args.segmentation_dataset_path or args.segmentation_local_parquet
    print(f"Persistent smoke index directory: {index_dir}")
    print(f"Persistent selected-data directory: {selected_data_dir}")

    gf_selection = prepare_gene_finding_selection(
        chromosome=args.requested_chromosome,
        aliases=aliases,
        index_dir=index_dir,
        selected_data_dir=selected_data_dir,
        local_dataset_path=gene_finding_dataset_path,
        local_files_only=args.hf_local_files_only,
        refresh=args.refresh_index,
    )
    tx_selection = prepare_transcript_selection(
        chromosome=args.requested_chromosome,
        aliases=aliases,
        index_dir=index_dir,
        selected_data_dir=selected_data_dir,
        local_dataset_path=segmentation_dataset_path,
        local_files_only=args.hf_local_files_only,
        refresh=args.refresh_index,
        batch_size=args.metadata_batch_size,
    )

    gpus = args.gpus.split(",") if args.gpus else [str(i) for i in range(args.num_gpus)]
    if not gpus:
        raise RuntimeError("At least one GPU is required")
    work = (REPO / args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(work, str(reference_gff), gf_selection, tx_selection, aliases)
    write_json(work / "jobs.json", jobs)
    done = run_scheduler(jobs, gpus, work)
    summary = write_summary(work, done, gf_selection, tx_selection)
    print(f"Smoke tests completed. Summary: {summary}")


if __name__ == "__main__":
    main()
