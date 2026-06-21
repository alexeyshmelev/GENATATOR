from __future__ import annotations

import copy
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

from .config import is_local, local_or_remote
from .utils import reverse_complement

logger = logging.getLogger(__name__)


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


def _read_parquet_block_row(parquet_path: str) -> Dict[str, Any]:
    """Load one selected chromosome block into RAM.

    Gene-finding smoke jobs traverse chromosome windows sequentially and retain only
    the current selected parquet block. Rejected chromosomes are never loaded here.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(
        str(parquet_path),
        columns=["dna_sequence", "targets"],
        memory_map=True,
        use_threads=False,
    )
    if table.num_rows != 1:
        raise RuntimeError(
            f"Expected exactly one row in gene-finding parquet block {parquet_path}, "
            f"got {table.num_rows}"
        )
    dna = str(table.column("dna_sequence")[0].as_py()).upper()
    targets = _nested_target_scalar_to_numpy(table.column("targets")[0])
    if len(dna) != targets.shape[0]:
        raise RuntimeError(
            f"DNA/target length mismatch in {parquet_path}: dna={len(dna)} targets={targets.shape}"
        )
    logger.info(
        "[finding.parquet_block.ram] loaded selected block only path=%s length=%d target_channels=%d",
        parquet_path,
        len(dna),
        targets.shape[1],
    )
    return {"dna_sequence": dna, "targets": targets}


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

    max_nt = int(cfg.get("max_nucleotides", cfg.get("context_length", len(row["dna_sequence"]))))
    overlap = float(cfg.get("overlap", 0.5))
    max_windows = int(cfg.get("max_windows") or 1)
    step = max(1, int(max_nt * (1.0 - overlap)))
    keep_len = max_nt + max(0, max_windows - 1) * step
    dna = str(row["dna_sequence"])
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

def load_dataset_auto(cfg: Dict[str, Any]) -> HFDataset:
    path = cfg["path"]
    split = cfg.get("split", "train")
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


def channel_indices(task: str, cfg: Dict[str, Any]) -> List[int]:
    if "target_indices" in cfg:
        return [int(i) for i in cfg["target_indices"]]
    group = cfg.get("target_group", "primary")
    if task == "finding_edge":
        return [0, 1, 2, 3] if group in {"primary", "combined", "all"} else [6, 7, 8, 9]
    if task == "finding_region":
        return [4, 5] if group in {"primary", "combined", "all"} else [10, 11]
    if task in {"segmentation", "transcript_type"}:
        return [0, 1, 2, 3, 4]
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


def nucleotide_ids(seq: str, tokenizer: PreTrainedTokenizerBase, max_len: int) -> np.ndarray:
    ids: List[int] = []
    for ch in seq[:max_len]:
        token_id = tokenizer.convert_tokens_to_ids(ch)
        if token_id is None or token_id == tokenizer.unk_token_id:
            tokenized = tokenizer(ch, add_special_tokens=False)["input_ids"]
            if len(tokenized) != 1:
                raise RuntimeError(f"Nucleotide tokenizer must map {ch!r} to exactly one token, got {tokenized}")
            token_id = tokenized[0]
        ids.append(int(token_id))
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    if len(ids) < max_len:
        ids += [pad] * (max_len - len(ids))
    return np.asarray(ids[:max_len], dtype=np.int64)


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
        self.cfg = copy.deepcopy(cfg)
        self.task = task
        self.raw = load_dataset_auto(cfg)
        self.row_indices = filter_row_indices(self.raw, cfg)
        self.tokenizer = tokenizer
        self.nucleotide_tokenizer = nucleotide_tokenizer or tokenizer
        self.for_inference = for_inference
        self.is_train = is_train
        self.model_family = cfg.get("model_family", "bpe")
        self.max_nucleotides = int(cfg.get("max_nucleotides", cfg.get("context_length", 4096)))
        self.max_tokens = int(cfg.get("max_tokens", cfg.get("context_length", 1024)))
        self.overlap = float(cfg.get("overlap", 0.5))
        self.target_indices = channel_indices(task, cfg)
        self.crop_margin = int(cfg.get("crop_margin", 500))
        self.random_crop = bool(cfg.get("random_crop", is_train and task in {"segmentation", "transcript_type"}))
        self.reverse_complement = bool(cfg.get("reverse_complement", False))
        self.prewindowed = bool(cfg.get("prewindowed", False))
        self.windows: List[Any] = []
        if task.startswith("finding"):
            if self.prewindowed:
                self._build_prewindowed_finding_indices()
            else:
                self._build_finding_windows()
        else:
            self._build_transcript_indices()
        max_windows = cfg.get("max_windows")
        if max_windows:
            self.windows = self.windows[: int(max_windows)]
        logger.info("[dataset] task=%s family=%s windows=%d max_nt=%d max_tokens=%d overlap=%.3f rc=%s is_train=%s", task, self.model_family, len(self.windows), self.max_nucleotides, self.max_tokens, self.overlap, self.reverse_complement, self.is_train)

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

    def _build_finding_windows(self) -> None:
        grouped: Dict[Tuple[str, str], List[Tuple[int, ParsedMetadata]]] = {}
        for row_i in self.row_indices:
            meta = parse_metadata(self.raw["metadata"][row_i])
            key = (meta.genome, meta.chrom)
            grouped.setdefault(key, []).append((row_i, meta))
        self.assemblies = {k: ChromosomeAssembly(k, rows, self.raw) for k, rows in grouped.items()}
        for key, assembly in self.assemblies.items():
            for s, e in make_windows(assembly.length, self.max_nucleotides, self.overlap):
                self.windows.append((key, s, e))

    def _build_transcript_indices(self) -> None:
        self.windows = list(self.row_indices)
        by_chrom: Dict[str, Dict[str, int]] = {}
        for row_i in self.row_indices:
            meta = parse_metadata(self.raw["metadata"][row_i])
            chrom = meta.chrom or "<empty>"
            d = by_chrom.setdefault(chrom, {"count": 0, "min_start": None, "max_end": None})
            d["count"] += 1
            d["min_start"] = meta.start if d["min_start"] is None else min(d["min_start"], meta.start)
            d["max_end"] = meta.end if d["max_end"] is None else max(d["max_end"], meta.end)
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
        return self.assemblies[key].get_slice(s, e, self.target_indices)

    def _crop_transcript(self, seq_len: int) -> Tuple[int, int]:
        if seq_len <= self.max_nucleotides:
            return 0, seq_len
        if self.random_crop:
            lo = min(self.crop_margin, max(0, seq_len - self.max_nucleotides))
            hi = max(lo, seq_len - self.max_nucleotides - self.crop_margin)
            start = int(np.random.randint(lo, hi + 1)) if hi > lo else int(lo)
        elif self.for_inference:
            start = 0
        else:
            start = min(self.crop_margin, max(0, seq_len - self.max_nucleotides))
        return start, min(seq_len, start + self.max_nucleotides)

    def _slice_transcript(self, idx: int) -> Tuple[str, np.ndarray, ParsedMetadata, int]:
        row_i = self.windows[idx]
        row = self.raw[row_i]
        seq = str(row["dna_sequence"]).upper()
        s, e = self._crop_transcript(len(seq))
        labels = np.asarray(row["labels"][s:e], dtype=np.float32)[:, self.target_indices]
        meta = parse_metadata(row.get("metadata", {}))
        return seq[s:e], labels, meta, s

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.task.startswith("finding"):
            dna, labels, meta, local_start = self._slice_finding(idx)
        else:
            dna, labels, meta, local_start = self._slice_transcript(idx)
        if self.reverse_complement:
            dna = reverse_complement(dna)
            labels = labels[::-1].copy()
        item = self._tokenize_token_task(dna, labels, meta, local_start)
        if self.task == "transcript_type":
            is_lnc = float(meta.transcript_type.lower() in {"lnc_rna", "lncrna", "lnc-rna", "lnc"})
            item["transcript_type"] = torch.tensor([is_lnc], dtype=torch.float32)
        return item

    def _tokenize_basic(self, dna: str, max_len: int, return_offsets: bool) -> Dict[str, Any]:
        kwargs = dict(
            text=dna,
            add_special_tokens=True,
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

    def _tokenize_token_task(self, dna: str, labels: np.ndarray, meta: ParsedMetadata, local_start: int) -> Dict[str, Any]:
        if self.model_family == "nucleotide":
            enc = self._tokenize_basic(dna, self.max_nucleotides, return_offsets=False)
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
                    "letter_level_attention_mask": torch.tensor(letter_mask, dtype=torch.long),
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
