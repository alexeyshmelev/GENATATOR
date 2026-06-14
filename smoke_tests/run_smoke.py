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
from typing import Dict, List

REPO = Path(__file__).resolve().parents[1]

MODELS = {
    "caduceus_ps": {"kind": "caduceus", "path": "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"},
    "caduceus_ph": {"kind": "caduceus", "path": "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"},
    "gena_base": {"kind": "gena", "path": "AIRI-Institute/gena-lm-bert-base-lastln-t2t"},
    "gena_large": {"kind": "gena", "path": "AIRI-Institute/gena-lm-bert-large-t2t"},
    "moderngena_base": {"kind": "moderngena", "path": "AIRI-Institute/moderngena-base"},
    "moderngena_large": {"kind": "moderngena", "path": "AIRI-Institute/moderngena-large"},
}
NUC_TOKENIZER = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"


def write_json(path: Path, obj: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path


def tiny_training(output_dir: str, metric: str, bs: int) -> dict:
    return {
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "max_steps": 2,
        "per_device_train_batch_size": bs,
        "per_device_eval_batch_size": bs,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-5,
        "weight_decay": 1e-4,
        "warmup_steps": 0,
        "lr_scheduler_type": "constant",
        "logging_steps": 1,
        "eval_steps": 1,
        "save_steps": 1,
        "save_total_limit": 1,
        "load_best_model_at_end": False,
        "metric_for_best_model": metric,
        "greater_is_better": True,
        "dataloader_num_workers": 0,
        "bf16": False,
        "fp16": False,
        "resume_from_checkpoint": None,
    }


def model_cfg(model_name: str, family: str, extra: dict | None = None) -> dict:
    info = MODELS[model_name]
    cfg = {
        "family": family,
        "backbone_kind": info["kind"],
        "backbone_path": info["path"],
        "tokenizer_path": info["path"],
        "trust_remote_code": True,
        "checkpoint_path": None,
    }
    if info["kind"] == "caduceus":
        cfg["bidirectional_weight_tie"] = False
    if family in {"unet", "rmt"} or (family == "amt" and (extra or {}).get("use_unet", False)):
        cfg.update({"nucleotide_tokenizer_path": NUC_TOKENIZER, "nucleotide_vocab_size": 1000})
    if extra:
        cfg.update(extra)
    return cfg


def finding_data(split: str, max_nt: int, max_tok: int, test: bool = False) -> dict:
    cfg = {
        "path": "AIRI-Institute/genatator-gene-finding-dataset",
        "split": split,
        "genomes": ["GCF_009914755.1"] if test else [],
        "chromosomes": ["NC_060944.1"] if test else [],
        "max_nucleotides": max_nt,
        "max_tokens": max_tok,
        "overlap": 0.5,
        "target_group": "primary",
        "max_windows": 1 if test else 2,
    }
    return cfg


def seg_data(config_name: str, split: str, max_nt: int, max_tok: int, test: bool = False) -> dict:
    return {
        "path": "AIRI-Institute/genatator-gene-segmentation-dataset",
        "config_name": config_name,
        "split": split,
        "genomes": ["GCF_009914755.1"] if test else [],
        "chromosomes": ["NC_060944.1"] if test else [],
        "max_nucleotides": max_nt,
        "max_tokens": max_tok,
        "overlap": 0.5,
        "crop_margin": 500,
        "random_crop": split == "train",
        "statuses": [1],
        "max_rows": 2,
    }


def make_finding_train_config(work: Path, model_name: str, task: str, variant: str) -> Path:
    max_tok = 64 if task == "edge" else 128
    max_nt = 512 if task == "edge" else 1024
    family = "caduceus" if MODELS[model_name]["kind"] == "caduceus" else variant
    extra = None
    if family == "unet":
        extra = {"unet_cycles": 1}
    elif family == "rmt":
        extra = {"cycles": 3, "rmt": {"input_size": 64, "max_n_segments": 8, "num_mem_tokens": 4, "bptt_depth": -1, "unet_sub_model_input_size": 512}}
    elif family == "amt":
        extra = {"use_unet": False, "amt": {"amt_repo_id": "irodkin/armt-neox-tiny", "num_mem_tokens": 4, "d_mem": 16, "segment_size": 64}}
    name = f"finding_{task}_{model_name}_{family}"
    cfg = {
        "seed": 42,
        "model": model_cfg(model_name, family, extra),
        "train_dataset": finding_data("train", max_nt, max_tok, test=False),
        "eval_dataset": finding_data("validation", max_nt, max_tok, test=False),
        "training": tiny_training(str(work / name), "auc_mean", bs=1 if family in {"unet", "rmt"} else 2),
    }
    return write_json(work / "configs" / f"{name}.json", cfg)


def make_seg_train_config(work: Path, model_name: str, variant: str) -> Path:
    kind = MODELS[model_name]["kind"]
    if kind == "caduceus":
        family = "caduceus"; extra = None; max_nt = max_tok = 512; bs = 1
    else:
        family = variant; max_nt = 512; max_tok = 64; bs = 1
        if family == "unet":
            extra = {"unet_cycles": 1}
        elif family == "rmt":
            extra = {"cycles": 3, "rmt": {"input_size": 64, "max_n_segments": 8, "num_mem_tokens": 4, "bptt_depth": -1, "unet_sub_model_input_size": 512}}
        elif family == "amt":
            extra = {"use_unet": True, "unet_cycles": 1, "amt": {"amt_repo_id": "irodkin/armt-neox-tiny", "num_mem_tokens": 4, "d_mem": 16, "segment_size": 64}}
        else:
            raise RuntimeError(f"Segmentation variant must be unet/rmt/amt for {model_name}, got {variant}")
    name = f"segmentation_{model_name}_{family}"
    cfg = {"seed": 42, "model": model_cfg(model_name, family, extra), "train_dataset": seg_data("train-human", "train", max_nt, max_tok, False), "eval_dataset": seg_data("val-human", "validation", max_nt, max_tok, True), "training": tiny_training(str(work / name), "interval_f1_mean", bs=bs)}
    return write_json(work / "configs" / f"{name}.json", cfg)


def make_tt_train_config(work: Path, model_name: str) -> Path:
    kind = MODELS[model_name]["kind"]
    family = "caduceus" if kind == "caduceus" else "plain"
    max_nt = max_tok = 512 if kind == "caduceus" else 512
    if kind != "caduceus":
        max_tok = 64
    name = f"transcript_type_{model_name}_{family}"
    cfg = {"seed": 42, "model": model_cfg(model_name, family), "train_dataset": seg_data("train-human", "train", max_nt, max_tok, False), "eval_dataset": seg_data("val-human", "validation", max_nt, max_tok, True), "training": tiny_training(str(work / name), "accuracy", bs=1 if kind == "caduceus" else 2)}
    return write_json(work / "configs" / f"{name}.json", cfg)


def make_finding_infer_config(work: Path, model_name: str, variant: str, true_gff: str) -> Path:
    edge_train = work / f"finding_edge_{model_name}_{variant}"
    region_train = work / f"finding_region_{model_name}_{variant}"
    edge_cfg = json.loads((work / "configs" / f"finding_edge_{model_name}_{variant}.json").read_text())
    region_cfg = json.loads((work / "configs" / f"finding_region_{model_name}_{variant}.json").read_text())
    edge_cfg = {"model": edge_cfg["model"], "dataset": finding_data("test", 512, 64, True), "inference": {"checkpoint_path": str(edge_train / "final_model"), "batch_size": 1}}
    region_cfg = {"model": region_cfg["model"], "dataset": finding_data("test", 1024, 128, True), "inference": {"checkpoint_path": str(region_train / "final_model"), "batch_size": 1}}
    cfg = {"edge": edge_cfg, "region": region_cfg, "postprocess": {"lp_frac": 0.05, "pk_prom": 0.1, "pk_dist": 50, "pk_height": None, "interval_window_size": 2000000, "max_pairs_per_seed": 2, "prob_threshold": 0.5, "zero_fraction_drop_threshold": 0.5}, "inference": {"device": "cuda", "use_reverse_complement": False, "output_gff": str(work / f"finding_{model_name}_{variant}.gff"), "true_gff": true_gff, "metrics_json": str(work / f"finding_{model_name}_{variant}.metrics.json"), "k_values": [0, 50, 100, 250, 500], "use_strand": True}}
    return write_json(work / "configs" / f"infer_finding_{model_name}_{variant}.json", cfg)


def make_seg_infer_config(work: Path, model_name: str, variant: str, true_gff: str) -> Path:
    family = "caduceus" if MODELS[model_name]["kind"] == "caduceus" else variant
    train_dir = work / f"segmentation_{model_name}_{family}"
    train_cfg = json.loads((work / "configs" / f"segmentation_{model_name}_{family}.json").read_text())
    cfg = {"model": train_cfg["model"], "dataset": seg_data("val-human", "validation", 512, 64 if family != "caduceus" else 512, True), "inference": {"device": "cuda", "checkpoint_path": str(train_dir / "final_model"), "batch_size": 1, "use_reverse_complement": False, "threshold": 0.5, "output_gff": str(work / f"segmentation_{model_name}_{family}.gff"), "true_gff": true_gff, "metrics_json": str(work / f"segmentation_{model_name}_{family}.metrics.json")}}
    return write_json(work / "configs" / f"infer_segmentation_{model_name}_{family}.json", cfg)


def make_tt_infer_config(work: Path, model_name: str) -> Path:
    family = "caduceus" if MODELS[model_name]["kind"] == "caduceus" else "plain"
    train_dir = work / f"transcript_type_{model_name}_{family}"
    train_cfg = json.loads((work / "configs" / f"transcript_type_{model_name}_{family}.json").read_text())
    cfg = {"model": train_cfg["model"], "dataset": seg_data("val-human", "validation", 512, 64 if family != "caduceus" else 512, True), "inference": {"device": "cuda", "checkpoint_path": str(train_dir / "final_model"), "batch_size": 1, "use_reverse_complement": False, "threshold": 0.5, "output_tsv": str(work / f"transcript_type_{model_name}_{family}.tsv"), "metrics_json": str(work / f"transcript_type_{model_name}_{family}.metrics.json")}}
    return write_json(work / "configs" / f"infer_transcript_type_{model_name}_{family}.json", cfg)


def build_jobs(work: Path, true_gff: str) -> List[dict]:
    jobs = []
    finding_variants_by_model = {}
    for model_name, info in MODELS.items():
        variants = ["caduceus"] if info["kind"] == "caduceus" else ["plain", "unet"]
        if model_name == "moderngena_base":
            variants += ["rmt", "amt"]
        finding_variants_by_model[model_name] = variants
        for variant in variants:
            for task in ["edge", "region"]:
                cfg = make_finding_train_config(work, model_name, task, variant)
                jobs.append({"name": f"train_finding_{task}_{model_name}_{variant}", "cmd": [sys.executable, "finding/train.py", "--task", task, "--config", str(cfg)], "deps": []})
            infer_cfg = make_finding_infer_config(work, model_name, variant, true_gff)
            jobs.append({"name": f"infer_finding_{model_name}_{variant}", "cmd": [sys.executable, "finding/infer.py", "--config", str(infer_cfg)], "deps": [f"train_finding_edge_{model_name}_{variant}", f"train_finding_region_{model_name}_{variant}"]})

    for model_name, info in MODELS.items():
        seg_variants = ["caduceus"] if info["kind"] == "caduceus" else ["unet"]
        if model_name == "moderngena_base":
            seg_variants += ["rmt", "amt"]
        for variant in seg_variants:
            cfg = make_seg_train_config(work, model_name, variant)
            family = "caduceus" if info["kind"] == "caduceus" else variant
            jobs.append({"name": f"train_segmentation_{model_name}_{family}", "cmd": [sys.executable, "segmentation/train.py", "--config", str(cfg)], "deps": []})
            infer_cfg = make_seg_infer_config(work, model_name, variant, true_gff)
            jobs.append({"name": f"infer_segmentation_{model_name}_{family}", "cmd": [sys.executable, "segmentation/infer.py", "--config", str(infer_cfg)], "deps": [f"train_segmentation_{model_name}_{family}"]})

        cfg = make_tt_train_config(work, model_name)
        family = "caduceus" if info["kind"] == "caduceus" else "plain"
        jobs.append({"name": f"train_transcript_type_{model_name}_{family}", "cmd": [sys.executable, "transcript_type/train.py", "--config", str(cfg)], "deps": []})
        infer_cfg = make_tt_infer_config(work, model_name)
        jobs.append({"name": f"infer_transcript_type_{model_name}_{family}", "cmd": [sys.executable, "transcript_type/infer.py", "--config", str(infer_cfg)], "deps": [f"train_transcript_type_{model_name}_{family}"]})
    return jobs


def run_scheduler(jobs: List[dict], gpus: List[str], work: Path) -> dict:
    pending = {j["name"]: j for j in jobs}
    done: dict = {}
    running: dict = {}
    free_gpus = list(gpus)
    logs = work / "logs"; logs.mkdir(parents=True, exist_ok=True)
    try:
        while pending or running:
            launched = True
            while launched and free_gpus:
                launched = False
                for name, job in list(pending.items()):
                    if all(dep in done for dep in job["deps"]):
                        gpu = free_gpus.pop(0)
                        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu; env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
                        log_path = logs / f"{name}.log"
                        fh = open(log_path, "w", encoding="utf-8")
                        start = time.time()
                        proc = subprocess.Popen(job["cmd"], cwd=str(REPO), env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
                        running[name] = {"proc": proc, "fh": fh, "gpu": gpu, "start": start, "log": log_path, "cmd": job["cmd"]}
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
                    raise RuntimeError(f"Smoke job failed: {name} exit_code={ret} gpu={state['gpu']} log={state['log']} cmd={' '.join(state['cmd'])}")
                done[name] = {"duration_s": duration, "log": str(state["log"])}
                free_gpus.append(state["gpu"])
                del running[name]
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


def collect_metric_files(work: Path) -> List[Path]:
    return sorted(work.glob("**/*.metrics.json")) + sorted(work.glob("**/trainer_state.json"))


def write_summary(work: Path, done: dict) -> Path:
    lines = ["# GENATATOR smoke-test summary", "", f"Total jobs: {len(done)}", "", "## Jobs", "", "| job | duration_s | log |", "|---|---:|---|"]
    for name, st in done.items():
        lines.append(f"| `{name}` | {st['duration_s']:.1f} | `{st['log']}` |")
    lines += ["", "## Metrics files", ""]
    for p in collect_metric_files(work):
        rel = p.relative_to(work)
        try:
            data = json.loads(p.read_text())
            preview = json.dumps(data, ensure_ascii=False)[:1000]
        except Exception as e:
            preview = f"Could not parse: {e}"
        lines.append(f"### `{rel}`")
        lines.append("")
        lines.append("```json")
        lines.append(preview)
        lines.append("```")
        lines.append("")
    path = work / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser(description="Run real-data smoke tests for all GENATATOR tasks and model families.")
    ap.add_argument("--num-gpus", type=int, required=True)
    ap.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs. Overrides --num-gpus list generation.")
    ap.add_argument("--reference-gff", type=str, required=True, help="Human T2T chr20 reference GFF/GFF3 supplied by the user.")
    ap.add_argument("--work-dir", type=str, default="smoke_tests/runs")
    args = ap.parse_args()
    true_gff = Path(args.reference_gff).expanduser()
    if not true_gff.exists():
        raise FileNotFoundError(f"Reference GFF does not exist: {true_gff}. Smoke tests never use dummy GFF files.")
    gpus = args.gpus.split(",") if args.gpus else [str(i) for i in range(args.num_gpus)]
    if not gpus:
        raise RuntimeError("At least one GPU is required")
    work = (REPO / args.work_dir).resolve(); work.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(work, str(true_gff))
    (work / "jobs.json").write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    done = run_scheduler(jobs, gpus, work)
    summary = write_summary(work, done)
    print(f"Smoke tests completed. Summary: {summary}")


if __name__ == "__main__":
    main()
