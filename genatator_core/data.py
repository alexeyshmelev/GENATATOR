from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .config import is_local, local_or_remote


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


def parse_metadata(value: Any) -> ParsedMetadata:
    if isinstance(value, dict):
        return ParsedMetadata(
            transcript_id=str(value.get("transcript_id", "")),
            gene_id=str(value.get("gene_id", "")),
            transcript_type=str(value.get("transcript_type", value.get("type", ""))),
            strand=str(value.get("strand", "+")),
            genome=str(value.get("genome", "")),
            chrom=str(value.get("chrom", value.get("chromosome", ""))),
            start=int(value.get("start", 0)),
            end=int(value.get("end", value.get("sequence_length", 0))),
        )
    if isinstance(value, str) and value.startswith("{"):
        return parse_metadata(json.loads(value))
    if isinstance(value, str) and "|" in value:
        parts = value.split("|")
        region = parts[6] if len(parts) > 6 else "0:0"
        start, end = region.split(":")
        return ParsedMetadata(
            transcript_id=parts[0],
            gene_id=parts[1] if len(parts) > 1 else "",
            transcript_type=parts[2] if len(parts) > 2 else "",
            strand=parts[3] if len(parts) > 3 else "+",
            genome=parts[4] if len(parts) > 4 else "",
            chrom=parts[5] if len(parts) > 5 else "",
            start=int(start),
            end=int(end),
        )
    return ParsedMetadata()


def load_dataset_auto(cfg: Dict[str, Any]) -> HFDataset:
    path = cfg["path"]
    split = cfg.get("split", "train")
    name = cfg.get("config_name")
    data_files = cfg.get("data_files")
    ref = local_or_remote(path)

    if is_local(path):
        p = Path(ref)
        if p.is_dir() and ((p / "dataset_info.json").exists() or (p / "dataset_dict.json").exists()):
            ds = load_from_disk(str(p))
            return ds[split] if isinstance(ds, DatasetDict) else ds
        if p.is_dir():
            parquet_files = sorted(str(x) for x in p.rglob("*.parquet"))
            json_files = sorted(str(x) for x in p.rglob("*.jsonl")) + sorted(str(x) for x in p.rglob("*.json"))
            if parquet_files:
                return load_dataset("parquet", data_files={split: parquet_files}, split=split)
            if json_files:
                return load_dataset("json", data_files={split: json_files}, split=split)
        suffix = p.suffix.lower()
        if suffix == ".parquet":
            return load_dataset("parquet", data_files={split: str(p)}, split=split)
        if suffix in {".json", ".jsonl"}:
            return load_dataset("json", data_files={split: str(p)}, split=split)
        raise ValueError(f"Unsupported local dataset path: {p}")

    kwargs = {"path": ref, "split": split}
    if name:
        kwargs["name"] = name
    if data_files:
        kwargs["data_files"] = data_files
    return load_dataset(**kwargs)


def filter_indices(ds: HFDataset, cfg: Dict[str, Any]) -> List[int]:
    genomes = set(cfg.get("genomes") or [])
    chromosomes = set(cfg.get("chromosomes") or [])
    max_rows = cfg.get("max_rows")
    indices = []
    for i in range(len(ds)):
        meta = parse_metadata(ds[i].get("metadata", {}))
        if genomes and meta.genome not in genomes:
            continue
        if chromosomes and meta.chrom not in chromosomes:
            continue
        indices.append(i)
        if max_rows and len(indices) >= int(max_rows):
            break
    return indices


def make_windows(length: int, max_len: int, overlap: float) -> List[Tuple[int, int]]:
    if length <= max_len:
        return [(0, length)]
    step = max(1, int(max_len * (1.0 - overlap)))
    windows = []
    start = 0
    while start < length:
        end = min(length, start + max_len)
        windows.append((start, end))
        if end == length:
            break
        start += step
    return windows


def channel_indices(task: str, cfg: Dict[str, Any]) -> List[int]:
    if "target_indices" in cfg:
        return [int(i) for i in cfg["target_indices"]]
    group = cfg.get("target_group", "combined")
    if task == "finding_edge":
        return [0, 1, 2, 3] if group == "combined" else [6, 7, 8, 9]
    if task == "finding_region":
        return [4, 5] if group == "combined" else [10, 11]
    if task == "segmentation":
        return [0, 1, 2, 3, 4]
    if task == "transcript_type":
        return [1, 2]
    raise ValueError(task)


def token_type_ids_or_zeros(enc: Dict[str, Any], length: int) -> List[int]:
    return list(enc.get("token_type_ids", [0] * length))


def offset_content_mask(offset_mapping: Sequence[Tuple[int, int]], attention_mask: Sequence[int]) -> np.ndarray:
    return np.asarray([(a == 1 and e > s) for (s, e), a in zip(offset_mapping, attention_mask)], dtype=bool)


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
    ids = []
    for ch in seq[:max_len]:
        token_id = tokenizer.convert_tokens_to_ids(ch)
        if token_id is None or token_id == tokenizer.unk_token_id:
            token_id = tokenizer(ch, add_special_tokens=False)["input_ids"][0]
        ids.append(int(token_id))
    if len(ids) < max_len:
        ids += [int(tokenizer.pad_token_id or 0)] * (max_len - len(ids))
    return np.asarray(ids[:max_len], dtype=np.int64)


class GenatatorDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: Dict[str, Any], task: str, tokenizer: PreTrainedTokenizerBase, nucleotide_tokenizer: Optional[PreTrainedTokenizerBase] = None, for_inference: bool = False):
        self.cfg = cfg
        self.task = task
        self.raw = load_dataset_auto(cfg)
        self.indices = filter_indices(self.raw, cfg)
        self.tokenizer = tokenizer
        self.nucleotide_tokenizer = nucleotide_tokenizer or tokenizer
        self.for_inference = for_inference
        self.model_family = cfg.get("model_family", "bpe")
        self.max_nucleotides = int(cfg.get("max_nucleotides", cfg.get("context_length", 4096)))
        self.max_tokens = int(cfg.get("max_tokens", cfg.get("context_length", 1024)))
        self.overlap = float(cfg.get("overlap", 0.5))
        self.shuffle_starts = bool(cfg.get("shuffle_starts", False))
        self.min_random_length = int(cfg.get("min_random_length", 512))
        self.target_indices = channel_indices(task, cfg)
        self.windows: List[Tuple[int, int, int]] = []
        for row_i in self.indices:
            seq_len = len(self.raw[row_i]["dna_sequence"])
            for s, e in make_windows(seq_len, self.max_nucleotides, self.overlap):
                self.windows.append((row_i, s, e))
        max_windows = cfg.get("max_windows")
        if max_windows:
            self.windows = self.windows[: int(max_windows)]

    def __len__(self) -> int:
        return len(self.windows)

    def _slice(self, idx: int) -> tuple[str, np.ndarray | None, ParsedMetadata, int]:
        row_i, s, e = self.windows[idx]
        row = self.raw[row_i]
        seq = row["dna_sequence"]
        if self.shuffle_starts and len(seq) > self.min_random_length:
            max_start = max(0, len(seq) - self.min_random_length)
            s = int(np.random.randint(0, max_start + 1))
            e = min(len(seq), s + self.max_nucleotides)
        dna = seq[s:e].upper()
        meta = parse_metadata(row.get("metadata", {}))
        labels = None
        if self.task.startswith("finding"):
            labels = np.asarray(row["targets"], dtype=np.float32)[s:e][:, self.target_indices]
        elif self.task in {"segmentation", "transcript_type"}:
            labels = np.asarray(row["labels"], dtype=np.float32)[s:e][:, self.target_indices]
        return dna, labels, meta, s

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        dna, labels, meta, local_start = self._slice(idx)
        if self.task.startswith("finding"):
            return self._tokenize_finding(dna, labels, meta, local_start)
        if self.task == "segmentation":
            return self._tokenize_segmentation(dna, labels, meta, local_start)
        if self.task == "transcript_type":
            return self._tokenize_transcript_type(dna, labels, meta, local_start)
        raise ValueError(self.task)

    def _tokenize_basic(self, dna: str, max_len: int) -> Dict[str, Any]:
        return self.tokenizer(
            dna,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_offsets_mapping=True,
        )

    def _tokenize_finding(self, dna: str, labels: np.ndarray, meta: ParsedMetadata, local_start: int) -> Dict[str, Any]:
        enc = self._tokenize_basic(dna, self.max_tokens)
        y, y_mask = max_labels_by_offsets(labels, enc["offset_mapping"], enc["attention_mask"], labels.shape[1])
        item = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids_or_zeros(enc, len(enc["input_ids"])), dtype=torch.long),
            "labels": torch.tensor(y, dtype=torch.float32),
            "labels_mask": torch.tensor(y_mask, dtype=torch.bool),
        }
        if self.for_inference:
            item["metadata"] = meta
            item["local_start"] = local_start
            item["dna_sequence"] = dna
            item["offset_mapping"] = enc["offset_mapping"]
        return item

    def _tokenize_segmentation(self, dna: str, labels: np.ndarray, meta: ParsedMetadata, local_start: int) -> Dict[str, Any]:
        if self.model_family == "nucleotide":
            enc = self._tokenize_basic(dna, self.max_nucleotides)
            mask = offset_content_mask(enc["offset_mapping"], enc["attention_mask"])
            y = np.zeros((len(enc["input_ids"]), labels.shape[1]), dtype=np.float32)
            cursor = 0
            for i, use in enumerate(mask):
                if use and cursor < len(labels):
                    y[i] = labels[cursor]
                    cursor += 1
            item = {
                "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                "letter_level_labels": torch.tensor(y, dtype=torch.float32),
                "letter_level_labels_mask": torch.tensor(mask, dtype=torch.bool),
            }
        else:
            enc = self._tokenize_basic(dna, self.max_tokens)
            token_y, token_mask = max_labels_by_offsets(labels, enc["offset_mapping"], enc["attention_mask"], labels.shape[1])
            letter_len = self.max_nucleotides
            rep = repeater_from_offsets(enc["offset_mapping"], enc["attention_mask"], min(len(dna), letter_len))
            if len(rep) < letter_len:
                rep = np.pad(rep, (0, letter_len - len(rep)), constant_values=-100)
            letter_y = np.zeros((letter_len, labels.shape[1]), dtype=np.float32)
            n = min(len(labels), letter_len)
            letter_y[:n] = labels[:n]
            letter_mask = np.zeros(letter_len, dtype=bool)
            letter_mask[:n] = True
            item = {
                "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids_or_zeros(enc, len(enc["input_ids"])), dtype=torch.long),
                "labels": torch.tensor(token_y, dtype=torch.float32),
                "labels_mask": torch.tensor(token_mask, dtype=torch.bool),
                "letter_level_tokens": torch.tensor(nucleotide_ids(dna, self.nucleotide_tokenizer, letter_len), dtype=torch.long),
                "letter_level_labels": torch.tensor(letter_y, dtype=torch.float32),
                "letter_level_labels_mask": torch.tensor(letter_mask, dtype=torch.bool),
                "letter_level_token_types_ids": torch.zeros(letter_len, dtype=torch.long),
                "letter_level_attention_mask": torch.tensor(letter_mask, dtype=torch.long),
                "embedding_repeater": torch.tensor(rep[:letter_len], dtype=torch.long),
                "pos_weight": torch.ones((self.max_tokens, labels.shape[1]), dtype=torch.float32),
            }
        if self.for_inference:
            item["metadata"] = meta
            item["local_start"] = local_start
            item["dna_sequence"] = dna
        return item

    def _tokenize_transcript_type(self, dna: str, labels: np.ndarray, meta: ParsedMetadata, local_start: int) -> Dict[str, Any]:
        item = self._tokenize_segmentation(dna, labels, meta, local_start)
        is_lnc = float(meta.transcript_type.lower() in {"lnc_rna", "lncrna", "lncRNA".lower()})
        item["transcript_type"] = torch.tensor([is_lnc], dtype=torch.float32)
        return item


def make_tokenizer(path_or_repo: str, trust_remote_code: bool = True) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(local_or_remote(path_or_repo), trust_remote_code=trust_remote_code, use_fast=True)


class GenatatorCollator:
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        keys = batch[0].keys()
        for k in keys:
            vals = [b[k] for b in batch]
            if isinstance(vals[0], torch.Tensor):
                out[k] = torch.stack(vals)
            else:
                out[k] = vals
        return out
