from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, IterableDataset as HFIterableDataset, load_dataset, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tqdm.auto import tqdm

from .config import is_local, local_or_remote
from .utils import reverse_complement

logger = logging.getLogger(__name__)


BPE_DATASET_FAMILIES = {"bpe", "bpe_unet", "rmt_unet", "amt_unet"}
TRANSCRIPT_TASKS = {"segmentation", "transcript_type"}


def resolve_dataset_lengths(cfg: Dict[str, Any], task: str) -> Dict[str, Any]:
    """Validate public length fields and add private, derived runtime lengths.

    BPE configs describe token capacity and an empirical nucleotide/token ratio.
    They must not carry a separately tuned nucleotide cap. Nucleotide configs use
    an exact nucleotide length and do not expose BPE-only fields.
    """
    resolved = copy.deepcopy(cfg)
    family = str(resolved.get("model_family", "bpe"))

    if family == "nucleotide":
        if "max_nucleotides" not in resolved:
            raise RuntimeError("Nucleotide dataset configs require max_nucleotides")
        unexpected = [k for k in ("max_bpe_tokens", "average_bpe_token_length") if k in resolved]
        if unexpected:
            raise RuntimeError(
                "Nucleotide dataset configs must not define BPE-only length fields: "
                f"{unexpected}"
            )
        max_nucleotides = int(resolved["max_nucleotides"])
        max_tokens = max_nucleotides
    elif family in BPE_DATASET_FAMILIES:
        if "max_nucleotides" in resolved:
            raise RuntimeError(
                "BPE dataset configs must not define max_nucleotides; configure "
                "max_bpe_tokens and average_bpe_token_length instead"
            )
        missing = [k for k in ("max_bpe_tokens", "average_bpe_token_length") if k not in resolved]
        if missing:
            raise RuntimeError(f"BPE dataset config is missing required fields: {missing}")
        max_tokens = int(resolved["max_bpe_tokens"])
        average_token_length = float(resolved["average_bpe_token_length"])
        if average_token_length <= 0 or not math.isfinite(average_token_length):
            raise RuntimeError(
                "average_bpe_token_length must be a finite positive number, got "
                f"{resolved['average_bpe_token_length']!r}"
            )
        max_nucleotides = max(1, int(max_tokens * average_token_length))
    else:
        raise RuntimeError(f"Unsupported dataset model_family={family!r}")

    if max_nucleotides <= 0 or max_tokens <= 0:
        raise RuntimeError(
            f"Resolved dataset lengths must be positive: nt={max_nucleotides} tokens={max_tokens}"
        )

    if task in TRANSCRIPT_TASKS:
        if "overlap" in resolved:
            raise RuntimeError(f"{task} does not use overlapping transcript crops and must not define overlap")
        if int(resolved.get("crop_margin", 500)) < 1:
            raise RuntimeError("crop_margin must be at least 1 nucleotide")
    elif task.startswith("finding"):
        overlap = float(resolved.get("overlap", 0.5))
        if not 0.0 <= overlap < 1.0:
            raise RuntimeError(f"Gene-finding overlap must satisfy 0 <= overlap < 1, got {overlap}")
    else:
        raise RuntimeError(f"Unknown dataset task={task!r}")

    resolved["_task"] = task
    resolved["_resolved_max_nucleotides"] = max_nucleotides
    resolved["_resolved_max_tokens"] = max_tokens
    logger.info(
        "[dataset.lengths] task=%s family=%s max_nt=%d max_tokens=%d average_bpe_token_length=%s",
        task,
        family,
        max_nucleotides,
        max_tokens,
        resolved.get("average_bpe_token_length"),
    )
    return resolved


def _load_jsonl_rows(path: Path) -> MaterializedRows:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return MaterializedRows(rows)


def _nested_target_scalar_to_numpy(scalar: Any) -> np.ndarray:
    """Convert one Arrow L x C target scalar without expanding to nested Python lists."""
    import pyarrow as pa

    outer = scalar.values
    if pa.types.is_fixed_size_list(outer.type):
        width = int(outer.type.list_size)
        flat = outer.values.to_numpy(zero_copy_only=False)
        return np.asarray(flat, dtype=np.float32).reshape(len(outer), width)
    if pa.types.is_list(outer.type) or pa.types.is_large_list(outer.type):
        offsets = outer.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
        widths = np.diff(offsets)
        if widths.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        if not np.all(widths == widths[0]):
            raise RuntimeError("Gene-finding target channel width is not constant")
        width = int(widths[0])
        start = int(offsets[0])
        stop = int(offsets[-1])
        flat = outer.values.slice(start, stop - start).to_numpy(zero_copy_only=False)
        return np.asarray(flat, dtype=np.float32).reshape(len(outer), width)
    raise RuntimeError(f"Unsupported Arrow targets type: {outer.type}")


def _metadata_value_to_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{"):
            return dict(json.loads(text))
        if "|" in text:
            parts = text.split("|")
            start = end = 0
            if len(parts) > 6 and ":" in parts[6]:
                start_s, end_s = parts[6].split(":", 1)
                start, end = int(start_s), int(end_s)
            return {
                "transcript_id": parts[0] if len(parts) > 0 else "",
                "gene_id": parts[1] if len(parts) > 1 else "",
                "transcript_type": parts[2] if len(parts) > 2 else "",
                "strand": parts[3] if len(parts) > 3 else "+",
                "genome": parts[4] if len(parts) > 4 else "",
                "chrom": parts[5] if len(parts) > 5 else "",
                "start": start,
                "end": end,
                "chrom_length": end,
            }
    raise RuntimeError(f"Unsupported metadata value in finding parquet: {type(value)}")


def _read_parquet_block_row(
    parquet_path: str,
    target_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Read one finding block directly into a plain Python/NumPy dictionary."""
    import pyarrow.parquet as pq

    table = pq.read_table(
        str(parquet_path),
        columns=["dna_sequence", "targets", "metadata"],
        memory_map=True,
        use_threads=False,
    )
    if table.num_rows != 1:
        raise RuntimeError(
            f"Expected exactly one row in gene-finding parquet block {parquet_path}, "
            f"got {table.num_rows}"
        )
    dna = str(table.column("dna_sequence")[0].as_py()).upper()
    all_targets = _nested_target_scalar_to_numpy(table.column("targets")[0])
    if target_indices is None:
        targets = np.ascontiguousarray(all_targets, dtype=np.float32)
    else:
        targets = np.ascontiguousarray(all_targets[:, list(target_indices)], dtype=np.float32)
    metadata = _metadata_value_to_dict(table.column("metadata")[0].as_py())
    if len(dna) != targets.shape[0]:
        raise RuntimeError(
            f"DNA/target length mismatch in {parquet_path}: dna={len(dna)} targets={targets.shape}"
        )
    return {"dna_sequence": dna, "targets": targets, "metadata": metadata}


@dataclass(frozen=True)
class ParsedMetadata:
    transcript_id: str = ""
    gene_id: str = ""
    transcript_type: str = ""
    strand: str = "+"
    genome: str = ""
    chrom: str = ""
    start: int = 0
    end: int = 0
    chrom_length: int = 0


class MaterializedRows:
    """Tiny in-memory dataset adapter used for HF streaming smoke slices.

    It deliberately avoids `datasets.Dataset.from_list(...)` because gene-finding rows
    can contain millions of nucleotide-level labels. Converting those rows to Arrow for
    a two-step smoke test can use far more RAM than the actual model smoke needs.
    """

    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows
        keys = set()
        for row in rows:
            keys.update(row.keys())
        self.column_names = sorted(keys)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.rows[key]
        if isinstance(key, slice):
            return self.rows[key]
        if isinstance(key, str):
            return [row.get(key) for row in self.rows]
        raise TypeError(f"Unsupported MaterializedRows key type: {type(key)}")


def parse_metadata(value: Any) -> ParsedMetadata:
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return ParsedMetadata(
            transcript_id=str(value.get("transcript_id", value.get("id", ""))),
            gene_id=str(value.get("gene_id", "")),
            transcript_type=str(value.get("transcript_type", value.get("type", ""))),
            strand=str(value.get("strand", "+")),
            genome=str(value.get("genome", value.get("assembly", ""))),
            chrom=str(value.get("chrom", value.get("chromosome", value.get("seqid", "")))),
            start=int(value.get("start", 0)),
            end=int(value.get("end", value.get("sequence_length", 0))),
            chrom_length=int(value.get("chrom_length", value.get("end", value.get("sequence_length", 0)))),
        )
    if isinstance(value, str) and value.strip().startswith("{"):
        return parse_metadata(json.loads(value))
    if isinstance(value, str) and "|" in value:
        parts = value.split("|")
        region = parts[6] if len(parts) > 6 else "0:0"
        start_s, end_s = region.split(":")
        return ParsedMetadata(
            transcript_id=parts[0],
            gene_id=parts[1] if len(parts) > 1 else "",
            transcript_type=parts[2] if len(parts) > 2 else "",
            strand=parts[3] if len(parts) > 3 else "+",
            genome=parts[4] if len(parts) > 4 else "",
            chrom=parts[5] if len(parts) > 5 else "",
            start=int(start_s),
            end=int(end_s),
            chrom_length=int(end_s),
        )
    return ParsedMetadata()


def _norm_id(x: Any) -> str:
    return str(x or "").strip()


def _chrom_aliases(chrom: Any) -> set[str]:
    value = _norm_id(chrom)
    aliases = {value}
    low = value.lower()
    if low.startswith("chr") and len(value) > 3:
        aliases.add(value[3:])
    elif value and value.isdigit():
        aliases.add(f"chr{value}")
    return aliases


def _matches_any(value: Any, allowed: set[str], is_chrom: bool = False) -> bool:
    if not allowed:
        return True
    if is_chrom:
        return bool(_chrom_aliases(value) & allowed)
    return _norm_id(value) in allowed


def _metadata_summary_from_rows(rows: List[Dict[str, Any]], limit: int = 20) -> Dict[str, Any]:
    genomes: Dict[str, int] = {}
    chroms: Dict[str, int] = {}
    pairs: Dict[str, int] = {}
    for row in rows:
        meta = parse_metadata(row.get("metadata", {}))
        g = meta.genome or "<empty>"
        c = meta.chrom or "<empty>"
        genomes[g] = genomes.get(g, 0) + 1
        chroms[c] = chroms.get(c, 0) + 1
        pairs[f"{g}|{c}"] = pairs.get(f"{g}|{c}", 0) + 1
    def top(d):
        return sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    return {"genomes": top(genomes), "chromosomes": top(chroms), "genome_chrom_pairs": top(pairs)}


def _row_matches_cfg(row: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    genomes = set(_norm_id(x) for x in (cfg.get("genomes") or []))
    chromosomes = set()
    for x in (cfg.get("chromosomes") or []):
        chromosomes |= _chrom_aliases(x)
    statuses = cfg.get("statuses")
    statuses = set(int(x) for x in statuses) if statuses is not None else None
    meta = parse_metadata(row.get("metadata", {}))
    if not _matches_any(meta.genome, genomes):
        return False
    if not _matches_any(meta.chrom, chromosomes, is_chrom=True):
        return False
    if statuses is not None:
        if "status" not in row:
            raise RuntimeError("statuses filter was requested but streamed row has no status column")
        if int(row["status"]) not in statuses:
            return False
    return True


def _maybe_trim_streaming_row(row: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not bool(cfg.get("streaming_trim_rows", False)):
        return row
    if "dna_sequence" not in row:
        return row

    dna = str(row["dna_sequence"])
    max_nt = int(cfg.get("_resolved_max_nucleotides", len(dna)))
    task = str(cfg.get("_task", ""))
    if task.startswith("finding"):
        overlap = float(cfg.get("overlap", 0.5))
        max_windows = int(cfg.get("max_windows") or 1)
        step = max(1, int(max_nt * (1.0 - overlap)))
        keep_len = max_nt + max(0, max_windows - 1) * step
    else:
        overlap = 0.0
        max_windows = 1
        # Random transcript cropping must retain the complete source row so any
        # valid start can be sampled. Beginning-only crops need only the model
        # context. Full-transcript evaluation also keeps the complete row.
        if bool(cfg.get("random_crop", False)) or bool(cfg.get("full_transcript_chunks", False)):
            keep_len = len(dna)
        else:
            keep_len = max_nt
    keep_len = min(len(dna), keep_len)

    trimmed = dict(row)
    trimmed["dna_sequence"] = dna[:keep_len]
    if "targets" in trimmed:
        trimmed["targets"] = trimmed["targets"][:keep_len]
    if "labels" in trimmed:
        trimmed["labels"] = trimmed["labels"][:keep_len]

    meta = parse_metadata(trimmed.get("metadata", {}))
    meta_dict = dict(trimmed.get("metadata", {})) if isinstance(trimmed.get("metadata", {}), dict) else None
    if meta_dict is not None:
        meta_dict["start"] = int(meta.start)
        meta_dict["end"] = int(meta.start + keep_len)
        meta_dict.setdefault("chrom_length", int(meta.chrom_length))
        trimmed["metadata"] = meta_dict
    elif isinstance(trimmed.get("metadata"), str) and trimmed["metadata"].strip().startswith("{"):
        meta_dict = json.loads(trimmed["metadata"])
        meta_dict["start"] = int(meta.start)
        meta_dict["end"] = int(meta.start + keep_len)
        meta_dict.setdefault("chrom_length", int(meta.chrom_length))
        trimmed["metadata"] = json.dumps(meta_dict)

    logger.info(
        "[dataset.streaming.trim] original_len=%d kept_len=%d max_nt=%d max_windows=%d overlap=%.3f",
        len(dna), keep_len, max_nt, max_windows, overlap,
    )
    return trimmed


def _materialize_streaming_dataset(iterable, cfg: Dict[str, Any]) -> MaterializedRows:
    max_rows = int(cfg.get("max_rows") or cfg.get("streaming_max_rows") or 32)
    max_scanned = int(cfg.get("streaming_max_scanned_rows") or 100000)
    rows = []
    scanned = 0
    observed: List[Dict[str, Any]] = []
    observed_limit = int(cfg.get("streaming_observed_metadata_limit") or 2000)
    for row in iterable:
        scanned += 1
        if len(observed) < observed_limit:
            observed.append({"metadata": row.get("metadata", {}), "status": row.get("status")})
        if _row_matches_cfg(row, cfg):
            rows.append(_maybe_trim_streaming_row(row, cfg))
            if len(rows) >= max_rows:
                break
        if scanned >= max_scanned:
            break
    if not rows:
        summary = _metadata_summary_from_rows(observed)
        raise RuntimeError(
            "Streaming dataset materialization selected zero rows after "
            f"scanning {scanned}. filters: genomes={cfg.get('genomes')} "
            f"chromosomes={cfg.get('chromosomes')} statuses={cfg.get('statuses')}. "
            f"Observed metadata summary from first {len(observed)} scanned rows: {json.dumps(summary, ensure_ascii=False)}"
        )
    logger.info("[dataset.streaming] materialized_rows=%d scanned_rows=%d max_rows=%d observed=%s", len(rows), scanned, max_rows, json.dumps(_metadata_summary_from_rows(rows), ensure_ascii=False))
    return MaterializedRows(rows)



# -----------------------------------------------------------------------------
# Direct parquet loading for transcript-level datasets
# -----------------------------------------------------------------------------
# Hugging Face `datasets.load_dataset(...)` may fail on these transcript parquet
# files with a PyArrow "List index overflow" while it is preparing an Arrow
# cache. The model logic does not require the HF Dataset object here, so for the
# GENATATOR segmentation dataset we read parquet files directly with PyArrow in
# bounded record batches and materialize the requested rows into RAM ourselves.

def _looks_like_segmentation_dataset_ref(path: str, cfg: Dict[str, Any]) -> bool:
    ref = str(path)
    if "genatator-gene-segmentation-dataset" in ref:
        return True
    if cfg.get("loader") == "direct_parquet":
        return True
    return False


def _repo_parquet_files(repo_id: str, cfg: Dict[str, Any]) -> List[str]:
    from huggingface_hub import HfApi

    config_name = cfg.get("config_name")
    split = str(cfg.get("split", "train")).strip("/")
    data_files = cfg.get("data_files")
    if data_files:
        files = data_files.get(split, data_files) if isinstance(data_files, dict) else data_files
        if isinstance(files, str):
            return [files]
        return list(files)

    from filelock import FileLock

    cache_root = Path(
        cfg.get("finding_index_cache_dir")
        or os.environ.get("GENATATOR_CACHE_DIR")
        or (Path.home() / ".cache" / "genatator")
    ).expanduser().resolve() / "repo_manifests"
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_key = hashlib.sha256(
        f"{repo_id}|{cfg.get('revision', '')}".encode("utf-8")
    ).hexdigest()
    manifest_path = cache_root / f"{manifest_key}.json"
    with FileLock(str(manifest_path) + ".lock"):
        if manifest_path.exists():
            files = list(json.loads(manifest_path.read_text(encoding="utf-8"))["files"])
        else:
            files = list(
                HfApi().list_repo_files(
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision=cfg.get("revision"),
                )
            )
            temporary = manifest_path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"files": files}), encoding="utf-8")
            os.replace(temporary, manifest_path)
    if config_name:
        config_prefix = str(config_name).strip("/") + "/"
        prefixes = [f"{config_prefix}{split}/", config_prefix]
    else:
        # Gene-finding repository layout: data/train, data/validation, data/test.
        prefixes = [f"data/{split}/", f"{split}/"]

    for prefix in prefixes:
        parquet_files = sorted(
            f for f in files
            if f.startswith(prefix) and f.endswith(".parquet")
            and (prefix.endswith(f"{split}/") or "/" not in f[len(prefix):])
        )
        if parquet_files:
            return parquet_files
    raise RuntimeError(
        f"No parquet files found in repo={repo_id} config={config_name} split={split}; "
        f"searched prefixes={prefixes}"
    )


def _local_parquet_files(path: Path, cfg: Dict[str, Any]) -> List[Path]:
    data_files = cfg.get("data_files")
    split = str(cfg.get("split", "train"))
    if data_files:
        files = data_files.get(split, data_files) if isinstance(data_files, dict) else data_files
        if isinstance(files, str):
            files = [files]
        return [Path(f).expanduser().resolve() for f in files]
    if path.is_file() and path.suffix.lower() == ".parquet":
        return [path.resolve()]
    if path.is_dir():
        config_name = cfg.get("config_name")
        candidates = []
        if config_name:
            candidates.extend([path / str(config_name) / split, path / str(config_name)])
        else:
            candidates.extend([path / "data" / split, path / split])
        for root in candidates:
            if root.exists():
                files = sorted(root.rglob("*.parquet"))
                if files:
                    return files
        return sorted(path.rglob("*.parquet"))
    return []


def _download_or_reuse_hf_parquets(repo_id: str, filenames: List[str], cfg: Dict[str, Any]) -> List[Path]:
    """Resolve one shared local file manifest so DDP ranks do not repeat 20k Hub calls."""
    from filelock import FileLock
    from huggingface_hub import hf_hub_download

    cache_root = Path(
        cfg.get("finding_index_cache_dir")
        or os.environ.get("GENATATOR_CACHE_DIR")
        or (Path.home() / ".cache" / "genatator")
    ).expanduser().resolve() / "download_manifests"
    cache_root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    digest.update(repo_id.encode())
    digest.update(str(cfg.get("revision", "")).encode())
    for filename in filenames:
        digest.update(filename.encode())
    manifest = cache_root / f"{digest.hexdigest()}.json"
    lock = FileLock(str(manifest) + ".lock")
    local_only = bool(cfg.get("local_files_only", False) or cfg.get("hf_local_files_only", False))

    with lock:
        if manifest.exists():
            paths = [Path(value).expanduser().resolve() for value in json.loads(manifest.read_text())["paths"]]
            if len(paths) == len(filenames) and all(path.exists() for path in paths):
                return paths

        paths: List[Path] = []
        for filename in tqdm(filenames, desc=f"download/reuse {repo_id} parquet files", unit="file"):
            kwargs = {
                "repo_id": repo_id,
                "repo_type": "dataset",
                "filename": filename,
                "local_files_only": local_only,
            }
            for key in ("revision", "cache_dir", "token"):
                if cfg.get(key) is not None:
                    kwargs[key] = cfg[key]
            paths.append(Path(hf_hub_download(**kwargs)).resolve())
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps({"paths": [str(path) for path in paths]}), encoding="utf-8")
        os.replace(temporary, manifest)
        return paths


def _arrow_2d_scalar_to_numpy(scalar: Any, dtype=np.float32) -> np.ndarray:
    """Convert one Arrow scalar containing a 2D list/fixed-list to numpy."""
    try:
        arr = _nested_target_scalar_to_numpy(scalar)
        return arr.astype(dtype, copy=False)
    except Exception:
        value = scalar.as_py() if hasattr(scalar, "as_py") else scalar
        return np.asarray(value, dtype=dtype)


def _direct_parquet_transcript_rows_from_files(files: List[Path], cfg: Dict[str, Any]) -> MaterializedRows:
    import pyarrow.parquet as pq

    genomes = set(_norm_id(x) for x in (cfg.get("genomes") or []))
    chromosomes = set()
    for x in (cfg.get("chromosomes") or []):
        chromosomes |= _chrom_aliases(x)
    statuses = cfg.get("statuses")
    statuses = set(int(x) for x in statuses) if statuses is not None else None
    batch_size = int(cfg.get("parquet_batch_size") or cfg.get("metadata_batch_size") or 64)
    task = str(cfg.get("_task", "segmentation"))
    needs_labels = task != "transcript_type"

    rows: List[Dict[str, Any]] = []
    observed = []
    total_rows = 0
    total_selected = 0
    disk_size = sum(p.stat().st_size for p in files if p.exists())
    logger.info(
        "[direct_parquet.location] files=%d disk_size=%s chromosomes=%s genomes=%s statuses=%s batch_size=%d",
        len(files), _human_bytes(disk_size), sorted(chromosomes), sorted(genomes), statuses, batch_size,
    )

    pbar_files = tqdm(files, desc="scan/load selected transcript parquet files", unit="file")
    for path in pbar_files:
        pf = pq.ParquetFile(str(path))
        schema_names = set(pf.schema_arrow.names)
        required = {"dna_sequence", "metadata"}
        if needs_labels:
            required.add("labels")
        missing = sorted(required - schema_names)
        if missing:
            raise RuntimeError(f"Transcript parquet {path} is missing required columns: {missing}")
        has_status = "status" in schema_names
        if statuses is not None and not has_status:
            raise RuntimeError(f"statuses filter was requested but transcript parquet {path} has no status column")
        columns = ["dna_sequence", "metadata"]
        if needs_labels:
            columns.append("labels")
        if has_status:
            columns.append("status")
        # Read in bounded batches. This avoids building the HF Arrow cache and
        # avoids holding unselected rows permanently.
        for batch in pf.iter_batches(
            batch_size=batch_size,
            columns=columns,
            use_threads=False,
        ):
            names = batch.schema.names
            col_meta = batch.column(names.index("metadata"))
            col_status = batch.column(names.index("status")) if "status" in names else None
            col_dna = batch.column(names.index("dna_sequence"))
            col_labels = batch.column(names.index("labels")) if "labels" in names else None
            total_rows += batch.num_rows
            for i in range(batch.num_rows):
                meta_value = col_meta[i].as_py()
                meta = parse_metadata(meta_value)
                if len(observed) < 2000:
                    observed.append({"metadata": meta_value, "status": col_status[i].as_py() if col_status is not None else None})
                if not _matches_any(meta.genome, genomes):
                    continue
                if not _matches_any(meta.chrom, chromosomes, is_chrom=True):
                    continue
                status_value = int(col_status[i].as_py()) if col_status is not None else None
                if statuses is not None and status_value not in statuses:
                    continue
                dna = str(col_dna[i].as_py()).upper()
                selected_row = {
                    "dna_sequence": dna,
                    "metadata": meta_value,
                    "source_parquet": str(path),
                }
                if col_labels is not None:
                    labels = _arrow_2d_scalar_to_numpy(col_labels[i], dtype=np.float32)
                    if len(dna) != labels.shape[0]:
                        raise RuntimeError(
                            f"DNA/labels length mismatch in {path}: row={i} dna={len(dna)} labels={labels.shape}"
                        )
                    selected_row["labels"] = labels
                if col_status is not None:
                    selected_row["status"] = status_value
                rows.append(selected_row)
                total_selected += 1
            pbar_files.set_postfix(rows=total_rows, selected=total_selected)

    if not rows:
        summary = _metadata_summary_from_rows(observed)
        raise RuntimeError(
            "Direct parquet loader selected zero transcript rows. "
            f"filters: genomes={cfg.get('genomes')} chromosomes={cfg.get('chromosomes')} statuses={cfg.get('statuses')}. "
            f"Observed metadata summary from first {len(observed)} rows: {json.dumps(summary, ensure_ascii=False)}"
        )
    metas = [parse_metadata(r.get("metadata", {})) for r in rows]
    chrom_counts: Dict[str, int] = {}
    type_counts: Dict[str, int] = {}
    total_nt = 0
    for r, m in zip(rows, metas):
        chrom_counts[m.chrom] = chrom_counts.get(m.chrom, 0) + 1
        type_counts[m.transcript_type] = type_counts.get(m.transcript_type, 0) + 1
        total_nt += len(str(r.get("dna_sequence", "")))
    logger.info(
        "[direct_parquet.selected] rows=%d source_rows=%d total_nt=%d expected_label_ram=%s chrom_counts=%s transcript_types=%s",
        len(rows), total_rows, total_nt, _human_bytes(total_nt * 5 * 4 if needs_labels else 0), chrom_counts, type_counts,
    )
    return MaterializedRows(rows)


def _load_segmentation_direct_parquet(cfg: Dict[str, Any]) -> MaterializedRows:
    path = cfg["path"]
    ref = local_or_remote(path)
    if is_local(path):
        files = _local_parquet_files(Path(ref).expanduser(), cfg)
        if not files:
            raise RuntimeError(f"No local parquet files found for direct parquet loader: {ref}")
    else:
        filenames = _repo_parquet_files(str(ref), cfg)
        logger.info(
            "[direct_parquet.remote] repo=%s config_name=%s parquet_files=%d first_files=%s",
            ref, cfg.get("config_name"), len(filenames), filenames[:5],
        )
        files = _download_or_reuse_hf_parquets(str(ref), filenames, cfg)
    return _direct_parquet_transcript_rows_from_files(files, cfg)


def _finding_parquet_paths(cfg: Dict[str, Any]) -> List[Path]:
    path = cfg["path"]
    ref = local_or_remote(path)
    if is_local(path):
        files = _local_parquet_files(Path(ref).expanduser(), cfg)
        if not files:
            raise RuntimeError(f"No local finding parquet files found for {ref}")
        return files
    filenames = _repo_parquet_files(str(ref), cfg)
    logger.info(
        "[finding.direct.remote] repo=%s split=%s parquet_files=%d",
        ref,
        cfg.get("split", "train"),
        len(filenames),
    )
    return _download_or_reuse_hf_parquets(str(ref), filenames, cfg)


def _finding_index_cache_path(files: Sequence[Path], cfg: Dict[str, Any]) -> Path:
    cache_root = Path(
        cfg.get("finding_index_cache_dir")
        or os.environ.get("GENATATOR_CACHE_DIR")
        or (Path.home() / ".cache" / "genatator")
    ).expanduser().resolve() / "finding_indexes"
    cache_root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    digest.update(str(cfg.get("path", "")).encode())
    digest.update(str(cfg.get("split", "train")).encode())
    digest.update(str(cfg.get("revision", "")).encode())
    for file_path in files:
        stat = file_path.stat()
        digest.update(str(file_path).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return cache_root / f"{digest.hexdigest()}.json"


def _scan_all_finding_block_metadata(files: Sequence[Path], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Scan only metadata from each finding parquet and share the index across ranks."""
    import pyarrow.parquet as pq
    from filelock import FileLock

    cache_path = _finding_index_cache_path(files, cfg)
    lock = FileLock(str(cache_path) + ".lock")
    with lock:
        if cache_path.exists():
            return list(json.loads(cache_path.read_text(encoding="utf-8"))["blocks"])
        blocks: List[Dict[str, Any]] = []
        for path in tqdm(files, desc="index finding parquet metadata", unit="file"):
            parquet = pq.ParquetFile(str(path))
            names = set(parquet.schema_arrow.names)
            required = {"dna_sequence", "targets", "metadata"}
            missing = sorted(required - names)
            if missing:
                raise RuntimeError(f"Finding parquet {path} is missing columns: {missing}")
            columns = ["metadata"] + (["status"] if "status" in names else [])
            table = parquet.read(columns=columns, use_threads=False)
            if table.num_rows != 1:
                raise RuntimeError(f"Expected one row in finding parquet {path}, got {table.num_rows}")
            row = {
                "parquet_path": str(path.resolve()),
                "metadata": _metadata_value_to_dict(table.column("metadata")[0].as_py()),
            }
            if "status" in columns:
                row["status"] = int(table.column("status")[0].as_py())
            blocks.append(row)
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"blocks": blocks}), encoding="utf-8")
        os.replace(temporary, cache_path)
        return blocks


def load_finding_parquet_groups(cfg: Dict[str, Any]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Build a lightweight block index without creating a Hugging Face Arrow dataset."""
    files = _finding_parquet_paths(cfg)
    blocks = _scan_all_finding_block_metadata(files, cfg)
    genomes = {_norm_id(x) for x in (cfg.get("genomes") or [])}
    chromosomes: set[str] = set()
    for chromosome in cfg.get("chromosomes") or []:
        chromosomes |= _chrom_aliases(chromosome)
    statuses = cfg.get("statuses")
    statuses = {int(x) for x in statuses} if statuses is not None else None
    if statuses is not None and any("status" not in block for block in blocks):
        raise RuntimeError("statuses filter was requested but one or more finding parquet files have no status column")
    max_rows = int(cfg.get("max_rows")) if cfg.get("max_rows") else None

    all_keys = set()
    selected: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    selected_count = 0
    for block in blocks:
        meta = parse_metadata(block["metadata"])
        key = (meta.genome, meta.chrom)
        all_keys.add(key)
        if not _matches_any(meta.genome, genomes):
            continue
        if not _matches_any(meta.chrom, chromosomes, is_chrom=True):
            continue
        if statuses is not None:
            if "status" not in block or int(block["status"]) not in statuses:
                continue
        selected.setdefault(key, []).append(block)
        selected_count += 1
        if max_rows is not None and selected_count >= max_rows:
            break
    if not selected:
        raise RuntimeError(
            "Direct finding parquet loader selected zero blocks. "
            f"genomes={cfg.get('genomes')} chromosomes={cfg.get('chromosomes')} statuses={cfg.get('statuses')}"
        )
    for key in selected:
        selected[key].sort(key=lambda item: parse_metadata(item["metadata"]).start)
    logger.info(
        "[finding.direct.index] available_chromosomes=%d selected_chromosomes=%d selected_blocks=%d keys=%s",
        len(all_keys),
        len(selected),
        selected_count,
        [f"{genome}|{chrom}" for genome, chrom in sorted(selected)],
    )
    return selected


class FindingChromosomeStore:
    """Keep parquet descriptors for every chromosome and one assembled chromosome in RAM."""

    def __init__(self, groups: Dict[Tuple[str, str], List[Dict[str, Any]]], target_indices: Sequence[int]):
        self.groups = groups
        self.target_indices = list(target_indices)
        self.spans: Dict[Tuple[str, str], Tuple[int, int, int]] = {}
        for key, blocks in groups.items():
            metas = [parse_metadata(block["metadata"]) for block in blocks]
            start = min(meta.start for meta in metas)
            end = max(meta.end for meta in metas)
            chrom_length = max([meta.chrom_length for meta in metas] + [end])
            self.spans[key] = (start, end, chrom_length)
        self._cache_key: Optional[Tuple[str, str]] = None
        self._cache: Optional[Dict[str, Any]] = None

    def keys(self) -> List[Tuple[str, str]]:
        return sorted(self.groups)

    def span(self, key: Tuple[str, str]) -> Tuple[int, int, int]:
        return self.spans[key]

    def release(self) -> None:
        self._cache_key = None
        self._cache = None

    def _assemble(self, key: Tuple[str, str]) -> Dict[str, Any]:
        if self._cache_key == key and self._cache is not None:
            return self._cache
        self.release()
        blocks = self.groups[key]
        start, end, chrom_length = self.spans[key]
        length = end - start
        sequence_buffer = bytearray(length)
        targets = np.empty((length, len(self.target_indices)), dtype=np.float32)
        expected_start = start
        for descriptor in blocks:
            expected_meta = parse_metadata(descriptor["metadata"])
            if expected_meta.start != expected_start:
                relation = "overlap" if expected_meta.start < expected_start else "gap"
                raise RuntimeError(
                    f"Finding chromosome blocks contain a {relation} for {key}: "
                    f"expected_start={expected_start} block_start={expected_meta.start}"
                )
            block = _read_parquet_block_row(descriptor["parquet_path"], self.target_indices)
            actual_meta = parse_metadata(block["metadata"])
            if (actual_meta.genome, actual_meta.chrom, actual_meta.start, actual_meta.end) != (
                expected_meta.genome, expected_meta.chrom, expected_meta.start, expected_meta.end
            ):
                raise RuntimeError(f"Finding parquet metadata changed after indexing: {descriptor['parquet_path']}")
            block_length = actual_meta.end - actual_meta.start
            if block_length != len(block["dna_sequence"]):
                raise RuntimeError(
                    f"Finding block metadata length mismatch in {descriptor['parquet_path']}: "
                    f"metadata={block_length} sequence={len(block['dna_sequence'])}"
                )
            offset = actual_meta.start - start
            stop = offset + block_length
            sequence_buffer[offset:stop] = block["dna_sequence"].encode("ascii")
            targets[offset:stop] = block["targets"]
            expected_start = actual_meta.end
            del block
        if expected_start != end:
            raise RuntimeError(f"Finding chromosome assembly ended at {expected_start}, expected {end} for {key}")
        assembled = {
            "dna_sequence": sequence_buffer.decode("ascii"),
            "targets": targets,
            "metadata": {
                "genome": key[0],
                "chrom": key[1],
                "start": start,
                "end": end,
                "chrom_length": chrom_length,
                "strand": "+",
            },
        }
        self._cache_key = key
        self._cache = assembled
        logger.info(
            "[finding.direct.assembly] genome=%s chrom=%s blocks=%d length=%d target_shape=%s; only this chromosome is cached",
            key[0], key[1], len(blocks), length, targets.shape,
        )
        return assembled

    def get_slice(self, key: Tuple[str, str], rel_start: int, rel_end: int) -> Tuple[str, np.ndarray, ParsedMetadata, int]:
        chromosome = self._assemble(key)
        start, end, chrom_length = self.spans[key]
        if rel_start < 0 or rel_end > end - start or rel_end <= rel_start:
            raise RuntimeError(f"Invalid finding window [{rel_start}, {rel_end}) for {key}")
        absolute_start = start + rel_start
        absolute_end = start + rel_end
        metadata = ParsedMetadata(
            genome=key[0], chrom=key[1], start=absolute_start, end=absolute_end,
            chrom_length=chrom_length, strand="+",
        )
        return (
            chromosome["dna_sequence"][rel_start:rel_end],
            chromosome["targets"][rel_start:rel_end],
            metadata,
            0,
        )

def load_dataset_auto(cfg: Dict[str, Any]) -> HFDataset:
    path = cfg["path"]
    split = cfg.get("split", "train")
    if _looks_like_segmentation_dataset_ref(str(path), cfg):
        return _load_segmentation_direct_parquet(cfg)
    name = cfg.get("config_name")
    data_files = cfg.get("data_files")
    ref = local_or_remote(path)
    streaming = bool(cfg.get("streaming", False))
    logger.info("[dataset.load] ref=%s split=%s config_name=%s data_files=%s local=%s streaming=%s", ref, split, name, data_files, is_local(path), streaming)
    if os.environ.get("GENATATOR_SMOKE_ENFORCE_LOCAL_DATA") == "1" and str(ref).startswith("AIRI-Institute/genatator-"):
        raise RuntimeError(
            "Smoke job received a remote GENATATOR dataset path. This is blocked to avoid HF resolver storms/rate limits. "
            f"path={ref!r}, split={split!r}. Regenerate smoke configs with the updated smoke_tests/run_smoke.py, "
            "or point the dataset path to the persistent local JSONL smoke cache."
        )
    if is_local(path):
        p = Path(ref)
        if p.is_dir() and ((p / "dataset_info.json").exists() or (p / "dataset_dict.json").exists()):
            ds = load_from_disk(str(p))
            return ds[split] if isinstance(ds, DatasetDict) else ds
        if p.is_dir():
            if data_files:
                files = data_files.get(split, data_files) if isinstance(data_files, dict) else data_files
                fmt = "parquet" if str(files).endswith(".parquet") else "json"
                return load_dataset(fmt, data_files={split: files}, split=split)
            parquet_files = sorted(str(x) for x in p.rglob("*.parquet"))
            json_files = sorted(str(x) for x in p.rglob("*.jsonl")) + sorted(str(x) for x in p.rglob("*.json"))
            if parquet_files:
                return load_dataset("parquet", data_files={split: parquet_files}, split=split)
            if json_files:
                return load_dataset("json", data_files={split: json_files}, split=split)
        if p.suffix.lower() == ".parquet":
            return load_dataset("parquet", data_files={split: str(p)}, split=split)
        if p.suffix.lower() in {".json", ".jsonl"}:
            # For smoke caches and parquet index files we avoid Arrow conversion: even
            # tiny indices can point to very large genomic blocks, and keeping plain
            # Python rows makes memory behavior explicit.
            return _load_jsonl_rows(p)
        raise RuntimeError(f"Unsupported local dataset path: {p}")
    kwargs = {"path": ref, "split": split}
    if name:
        kwargs["name"] = name
    if data_files:
        kwargs["data_files"] = data_files
    for key in ("revision", "cache_dir", "token"):
        if cfg.get(key) is not None:
            kwargs[key] = cfg[key]
    if streaming:
        iterable = load_dataset(**kwargs, streaming=True)
        return _materialize_streaming_dataset(iterable, cfg)
    return load_dataset(**kwargs)


def filter_row_indices(ds: HFDataset, cfg: Dict[str, Any]) -> List[int]:
    genomes = set(_norm_id(x) for x in (cfg.get("genomes") or []))
    chromosomes = set()
    for x in (cfg.get("chromosomes") or []):
        chromosomes |= _chrom_aliases(x)
    statuses = cfg.get("statuses")
    statuses = set(int(x) for x in statuses) if statuses is not None else None
    max_rows = cfg.get("max_rows")
    metadata_values = ds["metadata"]
    if statuses is not None and "status" not in ds.column_names:
        raise RuntimeError("statuses filter was requested but dataset has no status column")
    status_values = ds["status"] if statuses is not None else None
    indices: List[int] = []
    observed = []
    for i, meta_value in enumerate(metadata_values):
        row_meta = {"metadata": meta_value}
        if len(observed) < 2000:
            observed.append(row_meta)
        meta = parse_metadata(meta_value)
        if not _matches_any(meta.genome, genomes):
            continue
        if not _matches_any(meta.chrom, chromosomes, is_chrom=True):
            continue
        if statuses is not None and status_values is not None and int(status_values[i]) not in statuses:
            continue
        indices.append(i)
        if max_rows and len(indices) >= int(max_rows):
            break
    logger.info("[dataset.filter] selected_rows=%d / %d genomes=%s chromosomes=%s statuses=%s max_rows=%s", len(indices), len(ds), sorted(genomes), sorted(chromosomes), statuses, max_rows)
    if not indices:
        summary = _metadata_summary_from_rows(observed)
        raise RuntimeError(
            "Dataset filter selected zero rows. "
            f"filters: genomes={cfg.get('genomes')} chromosomes={cfg.get('chromosomes')} statuses={cfg.get('statuses')}. "
            f"Observed metadata summary from first {len(observed)} rows: {json.dumps(summary, ensure_ascii=False)}"
        )
    return indices

def make_windows(length: int, max_len: int, overlap: float) -> List[Tuple[int, int]]:
    if length <= max_len:
        return [(0, length)]
    step = max(1, int(max_len * (1.0 - overlap)))
    out = []
    s = 0
    while s < length:
        e = min(length, s + max_len)
        out.append((s, e))
        if e == length:
            break
        s += step
    return out


FINDING_TARGET_NAMES = (
    "primary_tss_+",
    "primary_tss_-",
    "primary_polya_+",
    "primary_polya_-",
    "intragenic_regions_+",
    "intragenic_regions_-",
    "mrna_tss_+",
    "mrna_tss_-",
    "mrna_polya_+",
    "mrna_polya_-",
    "mrna_intragenic_regions_+",
    "mrna_intragenic_regions_-",
)


def _finding_group(cfg: Dict[str, Any]) -> str:
    group = str(cfg.get("target_group", cfg.get("finding_target_group", "primary"))).lower()
    aliases = {
        "primary": "primary",
        "combined": "primary",
        "all": "primary",
        "mrna_lnc": "primary",
        "mrna+lncrna": "primary",
        "mrna+lnc_rna": "primary",
        "mrna": "mrna",
        "mrna_only": "mrna",
        "protein_coding": "mrna",
        "coding": "mrna",
    }
    if group not in aliases:
        raise RuntimeError(
            "Invalid gene-finding target_group. Expected one of primary/combined/all "
            "for mRNA+lncRNA targets or mrna/mrna_only/protein_coding for mRNA-only targets; "
            f"got {cfg.get('target_group')!r}"
        )
    return aliases[group]


def channel_indices(task: str, cfg: Dict[str, Any]) -> List[int]:
    if "target_indices" in cfg:
        return [int(i) for i in cfg["target_indices"]]
    if task == "finding_edge":
        return [0, 1, 2, 3] if _finding_group(cfg) == "primary" else [6, 7, 8, 9]
    if task == "finding_region":
        return [4, 5] if _finding_group(cfg) == "primary" else [10, 11]
    if task in {"segmentation", "transcript_type"}:
        return [0, 1, 2, 3, 4]
    raise RuntimeError(f"Unknown task={task}")


def channel_names_for_task(task: str, cfg: Dict[str, Any]) -> List[str]:
    idx = channel_indices(task, cfg)
    if task.startswith("finding"):
        return [FINDING_TARGET_NAMES[i] for i in idx]
    if task in {"segmentation", "transcript_type"}:
        return ["5UTR", "exon", "intron", "3UTR", "CDS"]
    raise RuntimeError(f"Unknown task={task}")


def token_type_ids_or_zeros(enc: Dict[str, Any], length: int) -> List[int]:
    return list(enc.get("token_type_ids", [0] * length))


def offset_content_mask(offset_mapping: Sequence[Tuple[int, int]], attention_mask: Sequence[int]) -> np.ndarray:
    return np.asarray([(int(a) == 1 and int(e) > int(s)) for (s, e), a in zip(offset_mapping, attention_mask)], dtype=bool)


def max_labels_by_offsets(labels: np.ndarray, offsets: Sequence[Tuple[int, int]], attention_mask: Sequence[int], n_labels: int) -> Tuple[np.ndarray, np.ndarray]:
    y = np.zeros((len(offsets), n_labels), dtype=np.float32)
    mask = np.zeros(len(offsets), dtype=bool)
    for i, ((s, e), a) in enumerate(zip(offsets, attention_mask)):
        if not a or e <= s:
            continue
        s = max(0, min(labels.shape[0], int(s)))
        e = max(0, min(labels.shape[0], int(e)))
        if e <= s:
            continue
        y[i] = labels[s:e].max(axis=0)
        mask[i] = True
    return y, mask


def repeater_from_offsets(offsets: Sequence[Tuple[int, int]], attention_mask: Sequence[int], n_letters: int) -> np.ndarray:
    rep = np.full(n_letters, -100, dtype=np.int64)
    content_index = -1
    for (s, e), a in zip(offsets, attention_mask):
        if not a or e <= s:
            continue
        content_index += 1
        s = max(0, min(n_letters, int(s)))
        e = max(0, min(n_letters, int(e)))
        rep[s:e] = content_index
    return rep


def reverse_complement_task_labels(task: str, labels: np.ndarray) -> np.ndarray:
    """Reverse nucleotide order and remap orientation-dependent target channels."""
    channel_order = {
        "finding_edge": [1, 0, 3, 2],
        "finding_region": [1, 0],
        "segmentation": [3, 1, 2, 0, 4],
    }.get(task)
    reversed_labels = np.asarray(labels)[::-1]
    if channel_order is None:
        return reversed_labels.copy()
    if reversed_labels.ndim != 2 or reversed_labels.shape[1] != len(channel_order):
        raise RuntimeError(
            f"Reverse-complement label shape mismatch for task={task}: {reversed_labels.shape}"
        )
    return reversed_labels[:, channel_order].copy()


def nucleotide_token_ids(tokenizer: PreTrainedTokenizerBase) -> Dict[str, int]:
    """Read the single-nucleotide token ids directly from one model tokenizer."""

    result: Dict[str, int] = {}
    unk = getattr(tokenizer, "unk_token_id", None)
    for nucleotide in ("A", "C", "G", "T", "N"):
        token_id = tokenizer.convert_tokens_to_ids(nucleotide)
        if token_id is None or (unk is not None and int(token_id) == int(unk)):
            if nucleotide == "N":
                # N is optional because the released training datasets normally
                # exclude ambiguous sequence. Fail only when it is actually used.
                continue
            raise RuntimeError(
                f"Tokenizer {type(tokenizer).__name__} does not expose a direct "
                f"single-nucleotide token for {nucleotide!r}."
            )
        result[nucleotide] = int(token_id)
    return result


def nucleotide_ids(seq: str, tokenizer: PreTrainedTokenizerBase, max_len: int) -> np.ndarray:
    token_ids = nucleotide_token_ids(tokenizer)
    ids: List[int] = []
    for ch in seq[:max_len].upper():
        if ch not in token_ids:
            raise RuntimeError(
                f"Tokenizer {type(tokenizer).__name__} has no direct single-nucleotide token id for {ch!r}."
            )
        ids.append(token_ids[ch])
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    if len(ids) < max_len:
        ids += [pad] * (max_len - len(ids))
    return np.asarray(ids[:max_len], dtype=np.int64)




def _human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for unit in units:
        if x < 1024.0 or unit == units[-1]:
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{n} B"


def _row_disk_path(row: Dict[str, Any]) -> Optional[Path]:
    for key in ("parquet_path", "source_parquet", "local_path"):
        val = row.get(key)
        if val:
            p = Path(str(val)).expanduser()
            if p.exists():
                return p.resolve()
    return None


def _slice_targets_for_task(task: str, row: Dict[str, Any], target_indices: List[int]) -> Dict[str, Any]:
    out = dict(row)
    if task.startswith("finding") and "targets" in out:
        arr = np.asarray(out["targets"], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < max(target_indices) + 1:
            raise RuntimeError(f"Gene-finding targets must have shape [L, 12], got {arr.shape}")
        out["targets"] = arr[:, target_indices]
        out["targets_are_selected"] = True
    elif task in {"segmentation", "transcript_type"} and "labels" in out:
        arr = np.asarray(out["labels"], dtype=np.float32)
        out["labels"] = arr
    return out


def _log_selected_dataset_stats(task: str, cfg: Dict[str, Any], raw: Any, row_indices: List[int], target_indices: List[int]) -> None:
    rows = [raw[i] for i in row_indices]
    metas = [parse_metadata(r.get("metadata", {})) for r in rows]
    chromosomes = sorted({m.chrom for m in metas})
    genomes = sorted({m.genome for m in metas})
    genes = sorted({m.gene_id for m in metas if m.gene_id})
    transcripts = sorted({m.transcript_id for m in metas if m.transcript_id})
    disk_paths = sorted({p for r in rows for p in [_row_disk_path(r)] if p is not None})
    if not disk_paths:
        cfg_path = cfg.get("path")
        if cfg_path and Path(str(cfg_path)).expanduser().exists():
            p0 = Path(str(cfg_path)).expanduser().resolve()
            if p0.is_file():
                disk_paths = [p0]
            elif p0.is_dir():
                disk_paths = sorted(list(p0.rglob("*.parquet")) + list(p0.rglob("*.jsonl")) + list(p0.rglob("*.json")))
    disk_size = sum(p.stat().st_size for p in disk_paths if p.exists())
    seq_total = 0
    target_bytes = 0
    if task.startswith("finding"):
        for m in metas:
            length = max(0, int(m.end) - int(m.start))
            seq_total += length
            target_bytes += length * len(target_indices) * 4
        kind = "chromosome_blocks"
        sample_count = len(rows)
    else:
        for r in rows:
            seq_len = len(str(r.get("dna_sequence", "")))
            if seq_len == 0:
                meta = parse_metadata(r.get("metadata", {}))
                seq_len = max(0, meta.end - meta.start)
            seq_total += seq_len
            if task == "segmentation":
                target_bytes += seq_len * 5 * 4
        kind = "transcripts"
        sample_count = len(rows)
    logger.info(
        "[dataset.stats.before_ram] task=%s path=%s split=%s selected_%s=%d parquet_files=%d "
        "chromosomes=%d genomes=%d genes=%d transcripts=%d sequence_nt=%d disk_size=%s "
        "expected_ram_sequence=%s expected_ram_targets=%s target_indices=%s target_names=%s",
        task,
        cfg.get("path"),
        cfg.get("split"),
        kind,
        sample_count,
        len(disk_paths),
        len(chromosomes),
        len(genomes),
        len(genes),
        len(transcripts),
        seq_total,
        _human_bytes(disk_size),
        _human_bytes(seq_total),
        _human_bytes(target_bytes),
        target_indices,
        channel_names_for_task(task, cfg),
    )


def _preload_selected_rows_to_ram(task: str, raw: Any, row_indices: List[int], target_indices: List[int]) -> MaterializedRows:
    rows: List[Dict[str, Any]] = []
    for row_i in tqdm(row_indices, desc=f"load selected {task} rows into CPU RAM", unit="row"):
        row = raw[row_i]
        if isinstance(row, dict) and row.get("parquet_path"):
            full = _read_parquet_block_row(row["parquet_path"])
            rows.append({
                "dna_sequence": full["dna_sequence"],
                "targets": full["targets"][:, target_indices],
                "targets_are_selected": True,
                "metadata": row.get("metadata", {}),
            })
        else:
            rows.append(_slice_targets_for_task(task, row, target_indices))
    logger.info("[dataset.ram] loaded_selected_rows=%d task=%s", len(rows), task)
    return MaterializedRows(rows)

class ChromosomeAssembly:
    def __init__(self, key: Tuple[str, str], rows: List[Tuple[int, ParsedMetadata]], raw: HFDataset):
        self.key = key
        self.rows = sorted(rows, key=lambda x: x[1].start)
        self.raw = raw
        if not self.rows:
            raise RuntimeError(f"Empty chromosome assembly for {key}")
        self.start = min(m.start for _, m in self.rows)
        self.end = max(m.end for _, m in self.rows)
        self.length = self.end - self.start
        self.chrom_length = max(m.chrom_length for _, m in self.rows)
        logger.info("[finding.assembly] genome=%s chrom=%s blocks=%d start=%d end=%d assembled_total_length=%d chrom_length_metadata=%d", key[0], key[1], len(self.rows), self.start, self.end, self.length, self.chrom_length)
        self._parquet_cache_row_i = None
        self._parquet_cache = None

    def get_slice(self, rel_start: int, rel_end: int, target_indices: List[int]) -> Tuple[str, np.ndarray, ParsedMetadata, int]:
        abs_start = self.start + rel_start
        abs_end = self.start + rel_end
        seq_parts: List[str] = []
        lab_parts: List[np.ndarray] = []
        for row_i, meta in self.rows:
            s = max(abs_start, meta.start)
            e = min(abs_end, meta.end)
            if e <= s:
                continue
            row = self.raw[row_i]
            local_s = s - meta.start
            local_e = e - meta.start
            if isinstance(row, dict) and row.get("parquet_path"):
                # Lazy smoke/full-chromosome index row: only the path and metadata
                # are kept in memory. Load the current 10 Mb block on demand.
                if self._parquet_cache_row_i != row_i:
                    logger.info("[finding.parquet_block.load] chrom=%s start=%d end=%d path=%s", meta.chrom, meta.start, meta.end, row["parquet_path"])
                    self._parquet_cache = _read_parquet_block_row(row["parquet_path"])
                    self._parquet_cache_row_i = row_i
                block = self._parquet_cache
                seq_parts.append(str(block["dna_sequence"])[local_s:local_e].upper())
                lab_parts.append(np.asarray(block["targets"][local_s:local_e], dtype=np.float32)[:, target_indices])
            else:
                seq_parts.append(str(row["dna_sequence"])[local_s:local_e].upper())
                lab_parts.append(np.asarray(row["targets"][local_s:local_e], dtype=np.float32)[:, target_indices])
        seq = "".join(seq_parts)
        labels = np.concatenate(lab_parts, axis=0) if lab_parts else np.zeros((0, len(target_indices)), dtype=np.float32)
        if len(seq) != labels.shape[0] or len(seq) != (rel_end - rel_start):
            raise RuntimeError(f"Chromosome assembly slice mismatch for {self.key}: seq={len(seq)} labels={labels.shape} expected={rel_end-rel_start}")
        meta0 = self.rows[0][1]
        meta = ParsedMetadata(genome=meta0.genome, chrom=meta0.chrom, start=abs_start, end=abs_end, chrom_length=self.chrom_length, strand="+")
        return seq, labels, meta, 0


class GenatatorDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: Dict[str, Any], task: str, tokenizer: PreTrainedTokenizerBase, nucleotide_tokenizer: Optional[PreTrainedTokenizerBase] = None, for_inference: bool = False, is_train: bool = False):
        self.task = task
        self.cfg = resolve_dataset_lengths(cfg, task)
        cfg = self.cfg
        self.model_family = str(cfg.get("model_family", "bpe"))
        self.max_nucleotides = int(cfg["_resolved_max_nucleotides"])
        self.max_tokens = int(cfg["_resolved_max_tokens"])
        self.tokenizer = tokenizer
        self.nucleotide_tokenizer = nucleotide_tokenizer or tokenizer
        self.for_inference = for_inference
        self.is_train = is_train
        self.overlap = float(cfg.get("overlap", 0.5)) if task.startswith("finding") else 0.0
        self.target_indices = channel_indices(task, cfg)
        self.crop_margin = int(cfg.get("crop_margin", 500))
        self.random_crop = bool(cfg.get("random_crop", False))
        requested_rc = bool(cfg.get("reverse_complement", False))
        if requested_rc and not for_inference:
            raise RuntimeError("reverse_complement is inference-only and must not be set in training/evaluation datasets")
        self.reverse_complement = requested_rc if for_inference else False
        self.full_transcript_chunks = bool(cfg.get("full_transcript_chunks", False))
        if self.full_transcript_chunks and (not self.for_inference or self.task != "segmentation"):
            raise RuntimeError("full_transcript_chunks is supported only for standalone segmentation inference")
        self.prewindowed = bool(cfg.get("prewindowed", False))
        self.windows: List[Any] = []
        self.finding_window_groups: Dict[Tuple[str, str], List[int]] = {}
        self.finding_store: Optional[FindingChromosomeStore] = None

        if task.startswith("finding") and not self.prewindowed:
            groups = load_finding_parquet_groups(cfg)
            self.finding_store = FindingChromosomeStore(groups, self.target_indices)
            self.target_indices = list(range(len(self.target_indices)))
            self.raw = None
            self.row_indices: List[int] = []
            self._build_finding_windows()
        else:
            self.raw = load_dataset_auto(cfg)
            self.row_indices = filter_row_indices(self.raw, cfg)
            _log_selected_dataset_stats(task, cfg, self.raw, self.row_indices, self.target_indices)
            self.raw = _preload_selected_rows_to_ram(task, self.raw, self.row_indices, self.target_indices)
            self.row_indices = list(range(len(self.raw)))
            if task.startswith("finding"):
                self.target_indices = list(range(len(channel_indices(task, cfg))))
            if task.startswith("finding"):
                self._build_prewindowed_finding_indices()
            else:
                self._build_transcript_indices()

        max_windows = cfg.get("max_windows")
        if max_windows:
            self.windows = self.windows[: int(max_windows)]
            if task.startswith("finding") and self.finding_store is not None:
                self._reindex_finding_window_groups()
        logger.info(
            "[dataset] task=%s family=%s windows=%d max_nt=%d max_tokens=%d overlap=%.3f "
            "random_crop=%s full_transcript_chunks=%s rc=%s is_train=%s",
            task, self.model_family, len(self.windows), self.max_nucleotides, self.max_tokens,
            self.overlap, self.random_crop, self.full_transcript_chunks, self.reverse_complement, self.is_train,
        )

    def _build_prewindowed_finding_indices(self) -> None:
        self.windows = list(self.row_indices)
        by_source_block: Dict[int, int] = {}
        by_chrom: Dict[str, int] = {}
        for row_i in self.row_indices:
            row = self.raw[row_i]
            meta = parse_metadata(row.get("metadata", {}))
            source_start = meta.start
            raw_meta = row.get("metadata", {})
            if isinstance(raw_meta, dict):
                source_start = int(raw_meta.get("smoke_source_block_start", source_start))
            by_source_block[source_start] = by_source_block.get(source_start, 0) + 1
            by_chrom[meta.chrom] = by_chrom.get(meta.chrom, 0) + 1
        logger.info(
            "[finding.prewindowed] task=%s samples=%d chromosomes=%s samples_per_source_block=%s",
            self.task,
            len(self.windows),
            by_chrom,
            dict(sorted(by_source_block.items())),
        )

    def _reindex_finding_window_groups(self) -> None:
        self.finding_window_groups = {}
        for window_index, window in enumerate(self.windows):
            key = window[0]
            self.finding_window_groups.setdefault(key, []).append(window_index)

    def _build_finding_windows(self) -> None:
        if self.finding_store is not None:
            for key in self.finding_store.keys():
                start, end, _ = self.finding_store.span(key)
                for window_start, window_end in make_windows(end - start, self.max_nucleotides, self.overlap):
                    self.windows.append((key, window_start, window_end))
            self._reindex_finding_window_groups()
            logger.info(
                "[finding.direct.windows] chromosomes=%d windows=%d overlap=%.3f",
                len(self.finding_window_groups), len(self.windows), self.overlap,
            )
            return
        grouped: Dict[Tuple[str, str], List[Tuple[int, ParsedMetadata]]] = {}
        for row_i in self.row_indices:
            meta = parse_metadata(self.raw["metadata"][row_i])
            key = (meta.genome, meta.chrom)
            grouped.setdefault(key, []).append((row_i, meta))
        self.assemblies = {k: ChromosomeAssembly(k, rows, self.raw) for k, rows in grouped.items()}
        for key, assembly in self.assemblies.items():
            for window_start, window_end in make_windows(assembly.length, self.max_nucleotides, self.overlap):
                self.windows.append((key, window_start, window_end))
        self._reindex_finding_window_groups()

    def release_finding_cache(self) -> None:
        if self.finding_store is not None:
            self.finding_store.release()

    def _bpe_full_transcript_chunk_bounds(self, dna: str) -> List[Tuple[int, int]]:
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError("Full-transcript BPE chunking requires a fast tokenizer")

        def encoded_length(start: int, end: int) -> int:
            encoded = self.tokenizer(
                dna[start:end],
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            return len(encoded["input_ids"])

        bounds: List[Tuple[int, int]] = []
        start_nt = 0
        while start_nt < len(dna):
            upper = min(len(dna), start_nt + self.max_nucleotides)
            if encoded_length(start_nt, upper) <= self.max_tokens:
                end_nt = upper
            else:
                low = start_nt + 1
                high = upper
                if encoded_length(start_nt, low) > self.max_tokens:
                    raise RuntimeError(
                        "A single nucleotide plus tokenizer special tokens exceeds max_bpe_tokens="
                        f"{self.max_tokens}"
                    )
                # Largest nucleotide endpoint whose independently tokenized chunk
                # fits the model's complete BPE input, including special tokens.
                while low < high:
                    mid = (low + high + 1) // 2
                    if encoded_length(start_nt, mid) <= self.max_tokens:
                        low = mid
                    else:
                        high = mid - 1
                end_nt = low
            if end_nt <= start_nt:
                raise RuntimeError("Full-transcript BPE chunking made no progress")
            bounds.append((start_nt, end_nt))
            start_nt = end_nt
        return bounds

    def _full_transcript_chunk_bounds(self, dna: str) -> List[Tuple[int, int]]:
        if not dna:
            raise RuntimeError("Empty transcript sequence")
        if self.model_family == "nucleotide":
            return make_windows(len(dna), self.max_nucleotides, 0.0)
        return self._bpe_full_transcript_chunk_bounds(dna)

    def _build_transcript_indices(self) -> None:
        self.windows = []
        by_chrom: Dict[str, Dict[str, int]] = {}
        for row_i in self.row_indices:
            row = self.raw[row_i]
            meta = parse_metadata(row.get("metadata", {}))
            chrom = meta.chrom or "<empty>"
            d = by_chrom.setdefault(chrom, {"count": 0, "min_start": None, "max_end": None})
            d["count"] += 1
            d["min_start"] = meta.start if d["min_start"] is None else min(d["min_start"], meta.start)
            d["max_end"] = meta.end if d["max_end"] is None else max(d["max_end"], meta.end)
            if self.full_transcript_chunks:
                original = str(row["dna_sequence"]).upper()
                oriented = reverse_complement(original) if self.reverse_complement else original
                for oriented_start, oriented_end in self._full_transcript_chunk_bounds(oriented):
                    original_start = len(original) - oriented_end if self.reverse_complement else oriented_start
                    self.windows.append((row_i, oriented_start, oriented_end, original_start))
            else:
                self.windows.append(row_i)
        for chrom, d in sorted(by_chrom.items()):
            span = int(d["max_end"] - d["min_start"]) if d["min_start"] is not None and d["max_end"] is not None else 0
            logger.info("[transcript.selection] task=%s chrom=%s transcripts_found=%d metadata_min_start=%s metadata_max_end=%s metadata_span=%d", self.task, chrom, d["count"], d["min_start"], d["max_end"], span)

    def __len__(self) -> int:
        return len(self.windows)

    def _slice_finding(self, idx: int) -> Tuple[str, np.ndarray, ParsedMetadata, int]:
        if self.prewindowed:
            row_i = self.windows[idx]
            row = self.raw[row_i]
            seq = str(row["dna_sequence"]).upper()
            labels = np.asarray(row["targets"], dtype=np.float32)[:, self.target_indices]
            meta = parse_metadata(row.get("metadata", {}))
            if len(seq) != labels.shape[0]:
                raise RuntimeError(
                    f"Prewindowed gene-finding DNA/label mismatch row={row_i}: seq={len(seq)} labels={labels.shape}"
                )
            return seq, labels, meta, 0
        key, s, e = self.windows[idx]
        if self.finding_store is not None:
            return self.finding_store.get_slice(key, s, e)
        return self.assemblies[key].get_slice(s, e, self.target_indices)

    def _crop_transcript(self, seq_len: int) -> Tuple[int, int]:
        if seq_len <= self.max_nucleotides:
            return 0, seq_len
        if not self.random_crop:
            return 0, self.max_nucleotides
        latest_start = max(0, seq_len - self.crop_margin)
        start = int(torch.randint(0, latest_start + 1, (1,)).item())
        return start, min(seq_len, start + self.max_nucleotides)

    def _slice_transcript(self, idx: int) -> Tuple[str, Optional[np.ndarray], ParsedMetadata, int]:
        if self.full_transcript_chunks:
            row_i, s, e, original_start = self.windows[idx]
            row = self.raw[row_i]
            original_seq = str(row["dna_sequence"]).upper()
            seq = reverse_complement(original_seq) if self.reverse_complement else original_seq
            labels = None
            if self.task != "transcript_type":
                if "labels" not in row:
                    raise RuntimeError("Segmentation transcript row is missing labels")
                full_labels = np.asarray(row["labels"], dtype=np.float32)[:, self.target_indices]
                if self.reverse_complement:
                    full_labels = reverse_complement_task_labels(self.task, full_labels)
                labels = full_labels[s:e]
            meta = parse_metadata(row.get("metadata", {}))
            return seq[s:e], labels, meta, int(original_start)

        row_i = self.windows[idx]
        row = self.raw[row_i]
        seq = str(row["dna_sequence"]).upper()
        s, e = self._crop_transcript(len(seq))
        labels = None
        if self.task != "transcript_type":
            if "labels" not in row:
                raise RuntimeError("Segmentation transcript row is missing labels")
            labels = np.asarray(row["labels"][s:e], dtype=np.float32)[:, self.target_indices]
        meta = parse_metadata(row.get("metadata", {}))
        return seq[s:e], labels, meta, s

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.task.startswith("finding"):
            dna, labels, meta, local_start = self._slice_finding(idx)
        else:
            dna, labels, meta, local_start = self._slice_transcript(idx)
        if self.reverse_complement and not self.full_transcript_chunks:
            dna = reverse_complement(dna)
            if labels is not None:
                labels = reverse_complement_task_labels(self.task, labels)
        if self.task == "transcript_type":
            item = self._tokenize_transcript_type(dna, meta, local_start)
        else:
            if labels is None:
                raise RuntimeError(f"Task {self.task} requires nucleotide labels")
            item = self._tokenize_token_task(dna, labels, meta, local_start)
        if self.task == "transcript_type":
            is_lnc = float(meta.transcript_type.lower() in {"lnc_rna", "lncrna", "lnc-rna", "lnc"})
            item["transcript_type"] = torch.tensor([is_lnc], dtype=torch.float32)
        return item

    def _tokenize_basic(
        self,
        dna: str,
        max_len: int,
        return_offsets: bool,
        add_special_tokens: bool = True,
    ) -> Dict[str, Any]:
        kwargs = dict(
            text=dna,
            add_special_tokens=add_special_tokens,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_special_tokens_mask=True,
        )
        if return_offsets:
            if not getattr(self.tokenizer, "is_fast", False):
                raise RuntimeError(
                    f"Tokenizer {type(self.tokenizer).__name__} is not a fast tokenizer and cannot return offsets. "
                    "BPE-resolution models require a fast tokenizer. Use a fast GENA/ModernGENA tokenizer or a nucleotide model."
                )
            kwargs["return_offsets_mapping"] = True
        enc = self.tokenizer(**kwargs)
        if "token_type_ids" not in enc:
            enc["token_type_ids"] = [0] * len(enc["input_ids"])
        if "special_tokens_mask" not in enc:
            enc["special_tokens_mask"] = [0] * len(enc["input_ids"])
        return enc

    @staticmethod
    def _synthetic_offsets_from_content_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
        offsets: List[Tuple[int, int]] = []
        cursor = 0
        for use in mask:
            if bool(use):
                offsets.append((cursor, cursor + 1))
                cursor += 1
            else:
                offsets.append((0, 0))
        return offsets

    def _tokenize_transcript_type(self, dna: str, meta: ParsedMetadata, local_start: int) -> Dict[str, Any]:
        is_nucleotide = self.model_family == "nucleotide"
        nucleotide_special_tokens = int(self.tokenizer.num_special_tokens_to_add(pair=False)) if is_nucleotide else 0
        enc = self._tokenize_basic(
            dna,
            self.max_nucleotides + nucleotide_special_tokens if is_nucleotide else self.max_tokens,
            return_offsets=not is_nucleotide,
            add_special_tokens=True,
        )
        item = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(
                token_type_ids_or_zeros(enc, len(enc["input_ids"])), dtype=torch.long
            ),
        }
        if self.for_inference:
            if is_nucleotide:
                attn = np.asarray(enc["attention_mask"], dtype=np.int64)
                special = np.asarray(enc.get("special_tokens_mask", [0] * len(attn)), dtype=np.int64)
                offsets = self._synthetic_offsets_from_content_mask((attn == 1) & (special == 0))
            else:
                offsets = enc["offset_mapping"]
            item.update({
                "metadata": meta,
                "local_start": local_start,
                "dna_sequence": dna,
                "offset_mapping": offsets,
                "reverse_complement": self.reverse_complement,
            })
        return item

    def _tokenize_token_task(self, dna: str, labels: np.ndarray, meta: ParsedMetadata, local_start: int) -> Dict[str, Any]:
        if self.model_family == "nucleotide":
            nucleotide_special_tokens = int(self.tokenizer.num_special_tokens_to_add(pair=False))
            enc = self._tokenize_basic(
                dna,
                self.max_nucleotides + nucleotide_special_tokens,
                return_offsets=False,
                add_special_tokens=True,
            )
            attn = np.asarray(enc["attention_mask"], dtype=np.int64)
            special = np.asarray(enc.get("special_tokens_mask", [0] * len(attn)), dtype=np.int64)
            mask = (attn == 1) & (special == 0)
            y = np.zeros((len(enc["input_ids"]), labels.shape[1]), dtype=np.float32)
            cursor = 0
            for i, use in enumerate(mask):
                if use and cursor < len(labels):
                    y[i] = labels[cursor]
                    cursor += 1
            item = {
                "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids_or_zeros(enc, len(enc["input_ids"])), dtype=torch.long),
                "letter_level_labels": torch.tensor(y, dtype=torch.float32),
                "letter_level_labels_mask": torch.tensor(mask, dtype=torch.bool),
                "pos_weight": torch.ones((len(enc["input_ids"]), labels.shape[1]), dtype=torch.float32),
            }
            inference_offsets = self._synthetic_offsets_from_content_mask(mask)
        else:
            enc = self._tokenize_basic(dna, self.max_tokens, return_offsets=True)
            token_y, token_mask = max_labels_by_offsets(labels, enc["offset_mapping"], enc["attention_mask"], labels.shape[1])
            item = {
                "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids_or_zeros(enc, len(enc["input_ids"])), dtype=torch.long),
                "labels": torch.tensor(token_y, dtype=torch.float32),
                "labels_mask": torch.tensor(token_mask, dtype=torch.bool),
            }
            inference_offsets = enc["offset_mapping"]
            if self.model_family in {"bpe_unet", "rmt_unet", "amt_unet"}:
                letter_len = self.max_nucleotides
                n = min(len(labels), letter_len)
                rep = repeater_from_offsets(enc["offset_mapping"], enc["attention_mask"], n)
                if len(rep) < letter_len:
                    rep = np.pad(rep, (0, letter_len - len(rep)), constant_values=-100)
                letter_y = np.zeros((letter_len, labels.shape[1]), dtype=np.float32)
                letter_y[:n] = labels[:n]
                letter_mask = np.zeros(letter_len, dtype=bool)
                letter_attention = np.zeros(letter_len, dtype=np.int64)
                letter_attention[:n] = 1
                # Only nucleotide positions that are actually covered by retained
                # BPE tokens can be used by UNET/RMT/AMT repeaters. With small
                # smoke-test max_tokens the tokenizer may truncate before
                # max_nucleotides, leaving -100 in the repeater tail. Do not let
                # those positions enter the loss or repeater indexing.
                letter_mask[:n] = rep[:n] >= 0
                item.update({
                    "letter_level_tokens": torch.tensor(nucleotide_ids(dna, self.nucleotide_tokenizer, letter_len), dtype=torch.long),
                    "letter_level_labels": torch.tensor(letter_y, dtype=torch.float32),
                    "letter_level_labels_mask": torch.tensor(letter_mask, dtype=torch.bool),
                    "letter_level_token_types_ids": torch.zeros(letter_len, dtype=torch.long),
                    "letter_level_attention_mask": torch.tensor(letter_attention, dtype=torch.long),
                    "embedding_repeater": torch.tensor(rep[:letter_len], dtype=torch.long),
                    "pos_weight": torch.ones((self.max_tokens, labels.shape[1]), dtype=torch.float32),
                })
        if self.for_inference:
            item["metadata"] = meta
            item["local_start"] = local_start
            item["dna_sequence"] = dna
            item["offset_mapping"] = inference_offsets
            item["reverse_complement"] = self.reverse_complement
            # Keep nucleotide-resolution truth labels as a plain NumPy array/list.
            # The collator leaves non-tensor values as lists, so these labels are
            # not passed into model.forward. They are used only by inference scripts
            # to compute whole-chromosome PR-AUC for gene-finding tasks.
            item["truth_labels"] = labels.astype(np.float32, copy=False)
        return item


def make_tokenizer(path_or_repo: str, trust_remote_code: bool = True) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(local_or_remote(path_or_repo), trust_remote_code=trust_remote_code, use_fast=True)
    if tok.pad_token_id is None:
        raise RuntimeError(f"Tokenizer {path_or_repo} must define pad_token_id.")
    return tok


class GenatatorCollator:
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in batch[0].keys():
            vals = [b[k] for b in batch]
            if isinstance(vals[0], torch.Tensor):
                out[k] = torch.stack(vals)
            else:
                out[k] = vals
        return out
