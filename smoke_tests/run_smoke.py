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
CHR20_ALIASES = ["NC_060944.1", "chr20", "20"]


def add_chr_alias_from_reference_gff(reference_gff: str) -> None:
    # The user supplies the chr20 reference GFF. We keep the smoke configs pinned to chr20,
    # but we add the exact seqid from the GFF because local references may use NC_060944.1, chr20, or 20.
    path = Path(reference_gff)
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            seqid = line.split("\t", 1)[0].strip()
            if seqid and seqid not in CHR20_ALIASES:
                CHR20_ALIASES.insert(0, seqid)
            break


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
        cfg["padding_side"] = "left"
    if family in {"unet", "rmt"} or (family == "amt" and (extra or {}).get("use_unet", False)):
        cfg.update({"nucleotide_tokenizer_path": NUC_TOKENIZER, "nucleotide_vocab_size": 1000})
    if extra:
        cfg.update(extra)
    return cfg


def default_smoke_cache_dir() -> Path:
    return Path(os.environ.get("GENATATOR_SMOKE_CACHE_DIR", str(Path.home() / ".cache" / "genatator_smoke"))).expanduser().resolve()


def _metadata_dict(row: dict) -> dict:
    meta = row.get("metadata", {})
    if isinstance(meta, (bytes, bytearray)):
        meta = meta.decode("utf-8", errors="replace")
    if isinstance(meta, dict):
        return dict(meta)
    if isinstance(meta, str):
        text = meta.strip()
        if text.startswith("{"):
            return json.loads(text)
        # Segmentation dataset metadata is compact:
        # <transcript_id>|<gene_id>|<transcript_type>|<strand>|<genome>|<chrom>|<start>:<end>
        if "|" in text:
            parts = text.split("|")
            start, end = 0, 0
            if len(parts) > 6 and ":" in parts[6]:
                a, b = parts[6].split(":", 1)
                start, end = int(a), int(b)
            return {
                "transcript_id": parts[0] if len(parts) > 0 else "",
                "gene_id": parts[1] if len(parts) > 1 else "",
                "transcript_type": parts[2] if len(parts) > 2 else "",
                "strand": parts[3] if len(parts) > 3 else "+",
                "genome": parts[4] if len(parts) > 4 else "",
                "chrom": parts[5] if len(parts) > 5 else "",
                "start": start,
                "end": end,
            }
    raise RuntimeError(f"Unsupported metadata type in smoke cache row: {type(meta)}")


def _row_chrom(row: dict) -> str:
    meta = _metadata_dict(row)
    return str(meta.get("chrom", meta.get("chromosome", meta.get("seqid", ""))))


def _is_chr20_row(row: dict) -> bool:
    chrom = _row_chrom(row)
    aliases = {chrom}
    if chrom.lower().startswith("chr"):
        aliases.add(chrom[3:])
    elif chrom.isdigit():
        aliases.add(f"chr{chrom}")
    return bool(set(CHR20_ALIASES) & aliases)


DEFAULT_GF_REMOTE_PARQUET = (
    "data/test/part-00000/"
    "GCF_009914755.1_T2T-CHM13v2.0__NC_060944.1__000000000000_000010000000.parquet"
)

SEGMENTATION_REPO_ID = "AIRI-Institute/genatator-gene-segmentation-dataset"
DEFAULT_SEGMENTATION_REMOTE_PARQUET = "val-human/data.parquet"



def _json_safe(value):
    # pyarrow usually returns Python scalars/lists, but this makes the cache writer
    # robust to numpy scalars if the local parquet backend changes.
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _download_or_reuse_hf_file(repo_id: str, filename: str, repo_type: str = "dataset", local_files_only: bool = False) -> str:
    """Resolve one exact HF file, preferring the local HF cache without network.

    This avoids the `datasets.load_dataset(...)` resolver storm. We first try
    local_files_only=True, which is a pure cache lookup. Only if the exact file is
    not already cached do we make one remote request for that exact parquet file.
    """
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, local_files_only=True)
        print(f"Using already cached HF file: {path}")
        return path
    except Exception as cache_error:
        if local_files_only:
            raise RuntimeError(
                f"Required HF file is not present in the local cache and --hf-local-files-only was set: {repo_id}/{filename}. "
                "Either run once with network access after the HF rate limit resets, or pass --gene-finding-local-parquet."
            ) from cache_error
        print(
            "HF file is not in the local cache yet; downloading exactly one parquet file, not the whole dataset:\n"
            f"  repo={repo_id}\n  file={filename}"
        )
        return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, local_files_only=False)


def _read_one_parquet_row(parquet_path: Path) -> dict:
    import pyarrow.parquet as pq

    table = pq.read_table(str(parquet_path), columns=["dna_sequence", "targets", "metadata"])
    rows = table.to_pylist()
    if len(rows) == 0:
        raise RuntimeError(f"Parquet file contains zero rows: {parquet_path}")
    if len(rows) > 1:
        print(f"Warning: parquet file contains {len(rows)} rows; using the first row for smoke cache: {parquet_path}")
    return dict(rows[0])


def prepare_gene_finding_smoke_cache(
    cache_dir: Path,
    row_slice: str,
    keep_len: int,
    remote_parquet: str,
    local_parquet: str | None = None,
    hf_local_files_only: bool = False,
) -> Path:
    """Create or reuse a tiny persistent JSONL cache from one real HF chr20 test parquet.

    Important: this function deliberately does NOT call datasets.load_dataset. Even
    split slices can trigger thousands of resolver requests for this sharded dataset.
    Instead, we resolve/download exactly one known chr20 parquet file, trim it, and
    reuse the resulting JSONL forever unless the cache file is removed.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_tag = Path(local_parquet).stem if local_parquet else remote_parquet.replace("/", "__").replace(".parquet", "")
    source_tag = source_tag.replace("[", "_").replace(":", "_").replace("]", "_")
    out = cache_dir / f"gene_finding_{source_tag}_first_{keep_len}.jsonl"
    if out.exists() and out.stat().st_size > 0:
        print(f"Using persistent real gene-finding smoke cache: {out}")
        return out

    if local_parquet:
        parquet_path = Path(local_parquet).expanduser().resolve()
        if not parquet_path.exists():
            raise FileNotFoundError(f"--gene-finding-local-parquet does not exist: {parquet_path}")
        print(f"Preparing persistent real gene-finding smoke cache from local parquet: {parquet_path}")
    else:
        print(
            "Preparing persistent real gene-finding smoke cache from one exact HF parquet file.\n"
            f"Legacy row-slice argument is kept for compatibility but is not used for loading: {row_slice}."
        )
        parquet_path = Path(_download_or_reuse_hf_file(
            repo_id="AIRI-Institute/genatator-gene-finding-dataset",
            filename=remote_parquet,
            repo_type="dataset",
            local_files_only=hf_local_files_only,
        ))

    row = _read_one_parquet_row(parquet_path)
    dna = str(row["dna_sequence"])
    n = min(len(dna), keep_len)
    row["dna_sequence"] = dna[:n]
    row["targets"] = row["targets"][:n]

    meta = _metadata_dict(row)
    meta["start"] = int(meta.get("start", 0))
    meta["end"] = int(meta["start"] + n)
    row["metadata"] = meta
    row = _json_safe(row)

    chrom = str(meta.get("chrom", meta.get("chromosome", "")))
    if CHR20_ALIASES and chrom and chrom not in CHR20_ALIASES:
        print(f"Warning: cached gene-finding row chrom={chrom!r} is not in current chr20 aliases {CHR20_ALIASES}")
    with out.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"Saved persistent real gene-finding smoke cache: {out} kept_len={n} chrom={chrom} source_parquet={parquet_path}")
    return out


def _read_segmentation_rows_from_parquet(parquet_path: Path, keep_len: int, max_rows: int, batch_size: int = 16) -> list[dict]:
    """Read only a few real chr20 transcript rows from one parquet file.

    This function intentionally iterates over small record batches instead of
    materializing a Hugging Face Dataset or a whole Parquet row group. The smoke
    cache is tiny, but the released Parquet files can be hundreds of megabytes,
    so each matching row is trimmed immediately before it is appended to the
    JSONL cache.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(parquet_path))
    rows: list[dict] = []
    scanned = 0
    observed: dict[str, int] = {}
    columns = ["dna_sequence", "labels", "metadata", "status"]

    for batch in pf.iter_batches(batch_size=max(1, int(batch_size)), columns=columns):
        batch_rows = batch.to_pylist()
        for raw in batch_rows:
            scanned += 1
            try:
                meta = _metadata_dict(raw)
            except Exception:
                continue
            chrom = str(meta.get("chrom", meta.get("chromosome", meta.get("seqid", ""))))
            if chrom:
                observed[chrom] = observed.get(chrom, 0) + 1
            row_stub = {"metadata": meta, "status": raw.get("status")}
            if not _is_chr20_row(row_stub):
                continue

            # For smoke tests we only need real chr20 examples. Do not require
            # status == 1 here: val-human keeps all isoforms, and the goal is to
            # verify training/inference code paths without scanning/downloading
            # additional shards if the first chr20 rows are not representative.
            row = dict(raw)
            dna = str(row["dna_sequence"])
            n = min(len(dna), int(keep_len))
            row["dna_sequence"] = dna[:n]
            row["labels"] = row["labels"][:n]
            meta = _metadata_dict(row)
            meta["start"] = int(meta.get("start", 0))
            meta["end"] = int(meta["start"] + n)
            row["metadata"] = meta
            rows.append(_json_safe(row))
            if len(rows) >= int(max_rows):
                print(
                    f"Selected {len(rows)} segmentation smoke rows from {parquet_path} "
                    f"after scanning {scanned} rows; observed_chroms={dict(list(observed.items())[:20])}"
                )
                return rows

    print(
        f"Selected {len(rows)} segmentation smoke rows from {parquet_path} "
        f"after scanning {scanned} rows; observed_chroms={dict(list(observed.items())[:20])}"
    )
    return rows


def prepare_segmentation_smoke_cache(
    cache_dir: Path,
    keep_len: int,
    max_rows: int,
    remote_parquet: str | None = None,
    local_parquet: str | None = None,
    hf_local_files_only: bool = False,
    max_parquet_files: int = 8,
    parquet_batch_size: int = 16,
) -> Path:
    """Create/reuse tiny persistent JSONL cache from real val-human chr20 transcript rows.

    This function deliberately does NOT call datasets.load_dataset. The datasets
    loader can prepare/scan large shards and blow RAM even for streaming. Here we
    resolve one parquet at a time, read metadata row-groups first, and copy only a
    few trimmed chr20 rows to a local JSONL cache.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    if remote_parquet is None and local_parquet is None:
        remote_parquet = DEFAULT_SEGMENTATION_REMOTE_PARQUET
    source_label = Path(local_parquet).stem if local_parquet else (remote_parquet or "auto_val_human_chr20")
    source_label = source_label.replace("/", "__").replace(".parquet", "")
    out = cache_dir / f"segmentation_{source_label}_rows_{max_rows}_first_{keep_len}.jsonl"
    if out.exists() and out.stat().st_size > 0:
        print(f"Using persistent real segmentation/transcript-type smoke cache: {out}")
        return out

    print(
        "Preparing persistent real segmentation/transcript-type smoke cache without datasets.load_dataset.\n"
        "The code will use only val-human/data.parquet by default, iterate small Parquet batches, "
        "and keep only tiny real chr20 rows."
    )

    parquet_paths: list[Path] = []
    if local_parquet:
        p = Path(local_parquet).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--segmentation-local-parquet does not exist: {p}")
        parquet_paths = [p]
    elif remote_parquet:
        parquet_paths = [Path(_download_or_reuse_hf_file(
            repo_id=SEGMENTATION_REPO_ID,
            filename=remote_parquet,
            repo_type="dataset",
            local_files_only=hf_local_files_only,
        ))]
    else:
        # Default smoke behavior: use only the human held-out validation parquet.
        # Do not auto-discover train-human or train-multi-specie shards here; the
        # smoke test is meant to stay on T2T chr20 and must not download species
        # or human-training data.
        parquet_paths = [Path(_download_or_reuse_hf_file(
            repo_id=SEGMENTATION_REPO_ID,
            filename=remote_parquet,
            repo_type="dataset",
            local_files_only=hf_local_files_only,
        ))]

    rows: list[dict] = []
    used_files = []
    for parquet_path in parquet_paths:
        used_files.append(str(parquet_path))
        rows.extend(_read_segmentation_rows_from_parquet(parquet_path, keep_len, max_rows - len(rows), batch_size=parquet_batch_size))
        if len(rows) >= max_rows:
            break

    if not rows:
        raise RuntimeError(
            "Could not build segmentation smoke cache: selected zero chr20 rows. "
            f"chr20 aliases={CHR20_ALIASES}; inspected parquet files={used_files}. "
            "Pass --segmentation-local-parquet or --segmentation-remote-parquet pointing to a val-human chr20 shard."
        )

    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"Saved persistent real segmentation/transcript-type smoke cache: {out} rows={len(rows)} files={used_files}")
    return out

def finding_data(cache_path: Path, max_nt: int, max_tok: int, inference_subset: bool = False) -> dict:
    # Gene-finding smoke tests use a tiny local cache produced from the real HF
    # `test` split. This avoids slow streamed scanning and repeated remote reads.
    return {
        "path": str(cache_path),
        "split": "train",
        "genomes": None,
        "chromosomes": CHR20_ALIASES,
        "max_nucleotides": max_nt,
        "max_tokens": max_tok,
        "overlap": 0.5,
        "target_group": "primary",
        "max_rows": 1,
        "max_windows": 1 if inference_subset else 2,
        "streaming": False,
    }


def seg_data(cache_path: Path, max_nt: int, max_tok: int, split: str = "train") -> dict:
    # Smoke segmentation/transcript-type uses a tiny persistent cache produced
    # from real HF val-human chr20 rows. It never touches the remote HF dataset
    # during per-model train/eval/infer jobs.
    return {
        "path": str(cache_path),
        "split": "train",
        "genomes": None,
        "chromosomes": CHR20_ALIASES,
        "max_nucleotides": max_nt,
        "max_tokens": max_tok,
        "overlap": 0.5,
        "crop_margin": 500,
        "random_crop": split == "train",
        "statuses": None,
        "max_rows": 2,
        "streaming": False,
    }

def make_finding_train_config(work: Path, gf_cache: Path, model_name: str, task: str, variant: str) -> Path:
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
        "train_dataset": finding_data(gf_cache, max_nt, max_tok, inference_subset=False),
        "eval_dataset": finding_data(gf_cache, max_nt, max_tok, inference_subset=False),
        "training": tiny_training(str(work / name), "auc_mean", bs=1 if family in {"unet", "rmt"} else 2),
    }
    return write_json(work / "configs" / f"{name}.json", cfg)


def make_seg_train_config(work: Path, seg_cache: Path, model_name: str, variant: str) -> Path:
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
    cfg = {"seed": 42, "model": model_cfg(model_name, family, extra), "train_dataset": seg_data(seg_cache, max_nt, max_tok, "train"), "eval_dataset": seg_data(seg_cache, max_nt, max_tok, "validation"), "training": tiny_training(str(work / name), "interval_f1_mean", bs=bs)}
    return write_json(work / "configs" / f"{name}.json", cfg)


def make_tt_train_config(work: Path, seg_cache: Path, model_name: str) -> Path:
    kind = MODELS[model_name]["kind"]
    family = "caduceus" if kind == "caduceus" else "plain"
    max_nt = max_tok = 512 if kind == "caduceus" else 512
    if kind != "caduceus":
        max_tok = 64
    name = f"transcript_type_{model_name}_{family}"
    cfg = {"seed": 42, "model": model_cfg(model_name, family), "train_dataset": seg_data(seg_cache, max_nt, max_tok, "train"), "eval_dataset": seg_data(seg_cache, max_nt, max_tok, "validation"), "training": tiny_training(str(work / name), "accuracy", bs=1 if kind == "caduceus" else 2)}
    return write_json(work / "configs" / f"{name}.json", cfg)


def make_finding_infer_config(work: Path, gf_cache: Path, model_name: str, variant: str, true_gff: str) -> Path:
    edge_train = work / f"finding_edge_{model_name}_{variant}"
    region_train = work / f"finding_region_{model_name}_{variant}"
    edge_cfg = json.loads((work / "configs" / f"finding_edge_{model_name}_{variant}.json").read_text())
    region_cfg = json.loads((work / "configs" / f"finding_region_{model_name}_{variant}.json").read_text())
    edge_cfg = {"model": edge_cfg["model"], "dataset": finding_data(gf_cache, 512, 64, inference_subset=True), "inference": {"checkpoint_path": str(edge_train / "final_model"), "batch_size": 1}}
    region_cfg = {"model": region_cfg["model"], "dataset": finding_data(gf_cache, 1024, 128, inference_subset=True), "inference": {"checkpoint_path": str(region_train / "final_model"), "batch_size": 1}}
    cfg = {"edge": edge_cfg, "region": region_cfg, "postprocess": {"lp_frac": 0.05, "pk_prom": 0.1, "pk_dist": 50, "pk_height": None, "interval_window_size": 2000000, "max_pairs_per_seed": 2, "prob_threshold": 0.5, "zero_fraction_drop_threshold": 0.5, "pairing_progress_every": None}, "inference": {"device": "cuda", "use_reverse_complement": False, "output_gff": str(work / f"finding_{model_name}_{variant}.gff"), "true_gff": true_gff, "metrics_json": str(work / f"finding_{model_name}_{variant}.metrics.json"), "k_values": [0, 50, 100, 250, 500], "use_strand": True, "empty_gff_policy": "best_interval", "empty_gff_min_interval_len": 64, "empty_gff_max_records": 1}}
    return write_json(work / "configs" / f"infer_finding_{model_name}_{variant}.json", cfg)


def make_seg_infer_config(work: Path, seg_cache: Path, model_name: str, variant: str, true_gff: str) -> Path:
    family = "caduceus" if MODELS[model_name]["kind"] == "caduceus" else variant
    train_dir = work / f"segmentation_{model_name}_{family}"
    train_cfg = json.loads((work / "configs" / f"segmentation_{model_name}_{family}.json").read_text())
    cfg = {"model": train_cfg["model"], "dataset": seg_data(seg_cache, 512, 64 if family != "caduceus" else 512, "validation"), "inference": {"device": "cuda", "checkpoint_path": str(train_dir / "final_model"), "batch_size": 1, "use_reverse_complement": False, "threshold": 0.5, "empty_segment_policy": "best_interval", "coordinate_mode": "transcript", "output_gff": str(work / f"segmentation_{model_name}_{family}.gff"), "true_gff": true_gff, "metrics_json": str(work / f"segmentation_{model_name}_{family}.metrics.json")}}
    return write_json(work / "configs" / f"infer_segmentation_{model_name}_{family}.json", cfg)


def make_tt_infer_config(work: Path, seg_cache: Path, model_name: str) -> Path:
    family = "caduceus" if MODELS[model_name]["kind"] == "caduceus" else "plain"
    train_dir = work / f"transcript_type_{model_name}_{family}"
    train_cfg = json.loads((work / "configs" / f"transcript_type_{model_name}_{family}.json").read_text())
    cfg = {"model": train_cfg["model"], "dataset": seg_data(seg_cache, 512, 64 if family != "caduceus" else 512, "validation"), "inference": {"device": "cuda", "checkpoint_path": str(train_dir / "final_model"), "batch_size": 1, "use_reverse_complement": False, "threshold": 0.5, "output_tsv": str(work / f"transcript_type_{model_name}_{family}.tsv"), "metrics_json": str(work / f"transcript_type_{model_name}_{family}.metrics.json")}}
    return write_json(work / "configs" / f"infer_transcript_type_{model_name}_{family}.json", cfg)


def build_jobs(work: Path, true_gff: str, gf_cache: Path, seg_cache: Path) -> List[dict]:
    jobs = []
    finding_variants_by_model = {}
    for model_name, info in MODELS.items():
        variants = ["caduceus"] if info["kind"] == "caduceus" else ["plain", "unet"]
        if model_name == "moderngena_base":
            variants += ["rmt", "amt"]
        finding_variants_by_model[model_name] = variants
        for variant in variants:
            for task in ["edge", "region"]:
                cfg = make_finding_train_config(work, gf_cache, model_name, task, variant)
                jobs.append({"name": f"train_finding_{task}_{model_name}_{variant}", "cmd": [sys.executable, "finding/train.py", "--task", task, "--config", str(cfg)], "deps": []})
            infer_cfg = make_finding_infer_config(work, gf_cache, model_name, variant, true_gff)
            jobs.append({"name": f"infer_finding_{model_name}_{variant}", "cmd": [sys.executable, "finding/infer.py", "--config", str(infer_cfg)], "deps": [f"train_finding_edge_{model_name}_{variant}", f"train_finding_region_{model_name}_{variant}"]})

    for model_name, info in MODELS.items():
        seg_variants = ["caduceus"] if info["kind"] == "caduceus" else ["unet"]
        if model_name == "moderngena_base":
            seg_variants += ["rmt", "amt"]
        for variant in seg_variants:
            cfg = make_seg_train_config(work, seg_cache, model_name, variant)
            family = "caduceus" if info["kind"] == "caduceus" else variant
            jobs.append({"name": f"train_segmentation_{model_name}_{family}", "cmd": [sys.executable, "segmentation/train.py", "--config", str(cfg)], "deps": []})
            infer_cfg = make_seg_infer_config(work, seg_cache, model_name, variant, true_gff)
            jobs.append({"name": f"infer_segmentation_{model_name}_{family}", "cmd": [sys.executable, "segmentation/infer.py", "--config", str(infer_cfg)], "deps": [f"train_segmentation_{model_name}_{family}"]})

        cfg = make_tt_train_config(work, seg_cache, model_name)
        family = "caduceus" if info["kind"] == "caduceus" else "plain"
        jobs.append({"name": f"train_transcript_type_{model_name}_{family}", "cmd": [sys.executable, "transcript_type/train.py", "--config", str(cfg)], "deps": []})
        infer_cfg = make_tt_infer_config(work, seg_cache, model_name)
        jobs.append({"name": f"infer_transcript_type_{model_name}_{family}", "cmd": [sys.executable, "transcript_type/infer.py", "--config", str(infer_cfg)], "deps": [f"train_transcript_type_{model_name}_{family}"]})
    return jobs


def assert_smoke_configs_use_local_data(work: Path) -> None:
    bad = []
    for cfg_path in sorted((work / "configs").glob("*.json")):
        text = cfg_path.read_text(encoding="utf-8")
        if "AIRI-Institute/genatator-gene-" in text:
            bad.append(str(cfg_path))
    if bad:
        raise RuntimeError(
            "Smoke configs still reference the remote gene-finding HF dataset. "
            "This would trigger slow downloads/rate limits. Delete stale configs or use the updated smoke runner. "
            f"Bad configs: {bad[:10]}"
        )


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
                        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu; env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", ""); env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1"); env["GENATATOR_SMOKE_ENFORCE_LOCAL_DATA"] = "1"
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
                    tail = ""
                    try:
                        tail = state["log"].read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
                        tail = "\n".join(tail)
                    except Exception as e:
                        tail = f"<could not read log tail: {e}>"
                    raise RuntimeError(
                        f"Smoke job failed: {name} exit_code={ret} gpu={state['gpu']} log={state['log']} "
                        f"cmd={' '.join(state['cmd'])}\n--- log tail ---\n{tail}"
                    )
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
    ap.add_argument("--gene-finding-row-slice", type=str, default="test[286:287]", help="Backward-compatible label only. Smoke loading now uses --gene-finding-remote-parquet or --gene-finding-local-parquet instead of load_dataset split slices.")
    ap.add_argument("--gene-finding-remote-parquet", type=str, default=DEFAULT_GF_REMOTE_PARQUET, help="Exact parquet file inside the HF gene-finding dataset repo to use for smoke tests. This avoids load_dataset resolver storms.")
    ap.add_argument("--gene-finding-local-parquet", type=str, default=None, help="Optional local parquet file for gene-finding smoke cache creation. Use this to avoid any HF request.")
    ap.add_argument("--hf-local-files-only", action="store_true", help="Do not contact Hugging Face while preparing smoke caches. Requires the exact parquet to be present in the local HF cache, or --gene-finding-local-parquet.")
    ap.add_argument("--gene-finding-cache-len", type=int, default=1536, help="Number of real nucleotides to keep from the selected gene-finding smoke row.")
    ap.add_argument("--segmentation-cache-len", type=int, default=768, help="Number of real nucleotides to keep from each selected transcript-level smoke row.")
    ap.add_argument("--segmentation-cache-rows", type=int, default=2, help="Number of real transcript-level chr20 rows to keep for segmentation/transcript-type smoke tests.")
    ap.add_argument("--segmentation-remote-parquet", type=str, default=DEFAULT_SEGMENTATION_REMOTE_PARQUET, help="Exact parquet file inside the HF segmentation dataset repo to use for smoke tests. Defaults to val-human/data.parquet. The smoke runner never auto-downloads train-human or train-multi-specie shards.")
    ap.add_argument("--segmentation-local-parquet", type=str, default=None, help="Optional local parquet file for segmentation/transcript-type smoke cache creation. Use this to avoid any HF request.")
    ap.add_argument("--segmentation-max-parquet-files", type=int, default=1, help="Deprecated compatibility option. Smoke tests now use exactly --segmentation-remote-parquet by default.")
    ap.add_argument("--segmentation-parquet-batch-size", type=int, default=16, help="Small Parquet record-batch size used while extracting chr20 transcript smoke rows.")
    ap.add_argument("--smoke-cache-dir", type=str, default=None, help="Persistent real-data smoke cache directory. Defaults to $GENATATOR_SMOKE_CACHE_DIR or ~/.cache/genatator_smoke.")
    args = ap.parse_args()
    true_gff = Path(args.reference_gff).expanduser()
    if not true_gff.exists():
        raise FileNotFoundError(f"Reference GFF does not exist: {true_gff}. Smoke tests never use dummy GFF files.")
    add_chr_alias_from_reference_gff(str(true_gff))
    print(f"Smoke tests will filter real HF data to chr20 aliases: {CHR20_ALIASES}")
    gpus = args.gpus.split(",") if args.gpus else [str(i) for i in range(args.num_gpus)]
    if not gpus:
        raise RuntimeError("At least one GPU is required")
    work = (REPO / args.work_dir).resolve(); work.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.smoke_cache_dir).expanduser().resolve() if args.smoke_cache_dir else default_smoke_cache_dir()
    print(f"Persistent smoke real-data cache directory: {cache_dir}")
    gf_cache = prepare_gene_finding_smoke_cache(
        cache_dir=cache_dir,
        row_slice=args.gene_finding_row_slice,
        keep_len=args.gene_finding_cache_len,
        remote_parquet=args.gene_finding_remote_parquet,
        local_parquet=args.gene_finding_local_parquet,
        hf_local_files_only=args.hf_local_files_only,
    )
    seg_cache = prepare_segmentation_smoke_cache(
        cache_dir=cache_dir,
        keep_len=args.segmentation_cache_len,
        max_rows=args.segmentation_cache_rows,
        remote_parquet=args.segmentation_remote_parquet,
        local_parquet=args.segmentation_local_parquet,
        hf_local_files_only=args.hf_local_files_only,
        max_parquet_files=args.segmentation_max_parquet_files,
        parquet_batch_size=args.segmentation_parquet_batch_size,
    )
    jobs = build_jobs(work, str(true_gff), gf_cache, seg_cache)
    assert_smoke_configs_use_local_data(work)
    (work / "jobs.json").write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    done = run_scheduler(jobs, gpus, work)
    summary = write_summary(work, done)
    print(f"Smoke tests completed. Summary: {summary}")


if __name__ == "__main__":
    main()
