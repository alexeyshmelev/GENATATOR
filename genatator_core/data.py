import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset

from .utils import dna_to_ids, sliding_windows

FINDING_TARGET_NAMES = [
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
]
SEGMENTATION_LABEL_NAMES = ["5UTR", "exon", "intron", "3UTR", "CDS"]


def parse_segmentation_metadata(meta: str) -> Dict[str, str | int]:
    transcript_id, gene_id, transcript_type, strand, genome, chrom, interval = meta.split("|")
    start, end = interval.split(":")
    return {
        "transcript_id": transcript_id,
        "gene_id": gene_id,
        "transcript_type": transcript_type,
        "strand": strand,
        "genome": genome,
        "chrom": chrom,
        "start": int(start),
        "end": int(end),
    }


def parse_finding_metadata(meta: str) -> Dict[str, Any]:
    return json.loads(meta) if isinstance(meta, str) else dict(meta)


def load_split(data_cfg: Dict[str, Any], split: str):
    if data_cfg["source"] == "hf":
        name = data_cfg.get(f"{split}_name", data_cfg.get("name"))
        if name is None:
            return load_dataset(data_cfg["repo"], split=split)
        return load_dataset(data_cfg["repo"], name, split=split)
    if data_cfg["source"] == "local_parquet":
        files = data_cfg["data_files"][split]
        return load_dataset("parquet", data_files={split: files}, split=split)
    if data_cfg["source"] == "local_dataset_script":
        return load_dataset(data_cfg["path"], data_cfg.get("name"), split=split)
    raise ValueError(data_cfg["source"])


def target_indices_for_finding(model_role: str, target_group: str = "combined") -> list[int]:
    offset = 0 if target_group == "combined" else 6
    if model_role == "edge":
        return [offset + 0, offset + 1, offset + 2, offset + 3]
    if model_role == "region":
        return [offset + 4, offset + 5]
    if model_role == "edge_region":
        return [offset + 0, offset + 1, offset + 2, offset + 3, offset + 4, offset + 5]
    raise ValueError(model_role)


@dataclass
class WindowSpec:
    mode: str
    nucleotide_length: int
    overlap: float
    samples_per_epoch: int
    min_length: int = 512


class GenatatorDataset(Dataset):
    def __init__(self, cfg: Dict[str, Any], split: str, task: str):
        self.cfg = cfg
        self.split = split
        self.task = task
        self.data = load_split(cfg["data"], cfg["data"].get(f"{split}_split", split))
        if cfg["data"].get("representative_only", False) and "status" in self.data.column_names:
            self.data = self.data.filter(lambda x: x["status"] == 1)
        self.tokenizer_cfg = cfg["tokenizer"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_cfg["path"],
            trust_remote_code=self.tokenizer_cfg.get("trust_remote_code", True),
            local_files_only=self.tokenizer_cfg.get("local_files_only", False),
        )
        window_cfg = dict(cfg["window"])
        if split in window_cfg.get("split_modes", {}):
            window_cfg["mode"] = window_cfg["split_modes"][split]
        window_cfg.pop("split_modes", None)
        self.window = WindowSpec(**window_cfg)
        self.label_mode = cfg["model"].get("label_mode", "token")
        self.model_role = cfg["task"].get("model_role", task)
        self.target_group = cfg["task"].get("target_group", "combined")
        self.target_indices = cfg["task"].get("target_indices")
        if self.target_indices is None and task == "finding":
            self.target_indices = target_indices_for_finding(self.model_role, self.target_group)
        self.label_names = cfg["task"].get("label_names")
        self.tokenizer_kind = self.tokenizer_cfg["kind"]
        self.max_tokens = int(self.tokenizer_cfg.get("max_tokens", self.window.nucleotide_length))
        self.pad_token_id = int(self.tokenizer_cfg.get("pad_token_id", self.tokenizer.pad_token_id))
        self.eos_token_id = self.tokenizer_cfg.get("eos_token_id", None)
        if self.eos_token_id is not None:
            self.eos_token_id = int(self.eos_token_id)
        self.pad_side = self.tokenizer_cfg.get("pad_side", "right")
        self.add_eos = bool(self.tokenizer_cfg.get("add_eos", False))
        if self.window.mode == "sliding":
            self.index = self._build_sliding_index()
        else:
            self.index = None

    def _build_sliding_index(self):
        index = []
        for row_i in range(len(self.data)):
            n = len(self.data[row_i]["dna_sequence"])
            for s, e in sliding_windows(n, self.window.nucleotide_length, self.window.overlap):
                index.append((row_i, s, e))
        return index

    def __len__(self):
        if self.index is not None:
            return len(self.index)
        return self.window.samples_per_epoch

    def _row_and_window(self, idx: int):
        if self.index is not None:
            row_i, s, e = self.index[idx]
            row = self.data[row_i]
            return row, s, e
        row_i = random.randrange(len(self.data))
        row = self.data[row_i]
        n = len(row["dna_sequence"])
        if n <= self.window.nucleotide_length:
            return row, 0, n
        max_s = n - self.window.min_length
        s = random.randint(0, max_s)
        e = min(n, s + self.window.nucleotide_length)
        if e - s < self.window.min_length:
            s = max(0, e - self.window.min_length)
        return row, s, e

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row, start, end = self._row_and_window(idx)
        seq = row["dna_sequence"][start:end].upper()
        labels = self._labels(row, start, end)
        metadata = self._metadata(row, start, end)
        if self.tokenizer_kind == "bpe":
            features = self._encode_bpe(seq, labels)
        else:
            features = self._encode_nucleotide(seq, labels)
        if self.task == "transcript_type":
            features["transcript_type"] = torch.tensor([self._transcript_type(row)], dtype=torch.float32)
        features["metadata"] = metadata
        return features

    def _labels(self, row: Dict[str, Any], start: int, end: int) -> Optional[np.ndarray]:
        if self.task == "transcript_type":
            return None
        if self.task == "finding":
            y = np.asarray(row["targets"], dtype=np.float32)[start:end]
            return y[:, self.target_indices]
        y = np.asarray(row["labels"], dtype=np.float32)[start:end]
        if self.target_indices is not None:
            y = y[:, self.target_indices]
        return y

    def _metadata(self, row: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
        if self.task == "finding":
            meta = parse_finding_metadata(row["metadata"])
            base_start = int(meta.get("start", 0))
            meta["window_start"] = base_start + start
            meta["window_end"] = base_start + end
            return meta
        meta = parse_segmentation_metadata(row["metadata"])
        meta["window_start"] = int(meta["start"]) + start
        meta["window_end"] = int(meta["start"]) + end
        return meta

    def _transcript_type(self, row: Dict[str, Any]) -> float:
        if "metadata" in row:
            meta = parse_segmentation_metadata(row["metadata"])
            return 0.0 if meta["transcript_type"] == "mRNA" else 1.0
        return float(row["transcript_type"])

    def _pad_1d(self, x: list[int], length: int, value: int) -> list[int]:
        x = x[:length]
        pad_n = length - len(x)
        if self.pad_side == "left":
            return [value] * pad_n + x
        return x + [value] * pad_n

    def _pad_2d(self, x: np.ndarray, length: int, value: float = -100.0) -> np.ndarray:
        x = x[:length]
        pad_n = length - len(x)
        pad = np.full((pad_n, x.shape[1]), value, dtype=np.float32)
        if self.pad_side == "left":
            return np.concatenate([pad, x], axis=0)
        return np.concatenate([x, pad], axis=0)

    def _encode_nucleotide(self, seq: str, labels: Optional[np.ndarray]) -> Dict[str, Any]:
        max_len = self.max_tokens
        content_len = max_len - int(self.add_eos)
        ids = self.tokenizer.encode(seq[:content_len], add_special_tokens=False)
        ids = ids[:content_len]
        if self.add_eos:
            ids = ids + [self.eos_token_id]
        ids = self._pad_1d(ids, max_len, self.pad_token_id)
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).long()
        label_mask = attention_mask.bool()
        if self.add_eos:
            label_mask = label_mask & (input_ids != self.eos_token_id)
        out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels_mask": label_mask}
        if labels is not None:
            y = labels[:content_len]
            if self.add_eos:
                y = np.concatenate([y, np.full((1, y.shape[1]), -100, dtype=np.float32)], axis=0)
            y = self._pad_2d(y, max_len, -100.0)
            out["labels"] = torch.tensor(y, dtype=torch.float32)
            out["nt_labels"] = out["labels"]
            out["nt_labels_mask"] = label_mask
        return out

    def _encode_bpe(self, seq: str, labels: Optional[np.ndarray]) -> Dict[str, Any]:
        encoded = self.tokenizer(
            seq,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_offsets_mapping=True,
            return_tensors=None,
        )
        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)
        offsets = np.asarray(encoded["offset_mapping"], dtype=np.int64)
        out = {"input_ids": input_ids, "attention_mask": attention_mask}
        if labels is None:
            return out
        token_labels = np.full((self.max_tokens, labels.shape[1]), -100.0, dtype=np.float32)
        for i, (s, e) in enumerate(offsets):
            if attention_mask[i].item() == 0 or e <= s:
                continue
            token_labels[i] = labels[s:e].max(axis=0)
        out["labels"] = torch.tensor(token_labels, dtype=torch.float32)
        out["labels_mask"] = attention_mask.bool()
        if self.label_mode == "nucleotide_unet":
            nt_len = min(len(seq), self.window.nucleotide_length)
            nt_ids = dna_to_ids(seq[:nt_len]).astype(np.int64)
            nt_labels = labels[:nt_len]
            token_to_nt = np.full(self.window.nucleotide_length, -100, dtype=np.int64)
            for i, (s, e) in enumerate(offsets):
                if attention_mask[i].item() == 0 or e <= s:
                    continue
                token_to_nt[s:min(e, self.window.nucleotide_length)] = i
            nt_mask = token_to_nt != -100
            nt_ids = np.pad(nt_ids, (0, self.window.nucleotide_length - len(nt_ids)), constant_values=4)
            nt_labels = np.pad(nt_labels, ((0, self.window.nucleotide_length - len(nt_labels)), (0, 0)), constant_values=-100)
            out["nt_ids"] = torch.tensor(nt_ids, dtype=torch.long)
            out["token_to_nt"] = torch.tensor(token_to_nt, dtype=torch.long)
            out["nt_labels"] = torch.tensor(nt_labels, dtype=torch.float32)
            out["nt_labels_mask"] = torch.tensor(nt_mask, dtype=torch.bool)
        return out


def collate_fn(items: list[Dict[str, Any]]) -> Dict[str, Any]:
    batch: Dict[str, Any] = {}
    keys = [k for k in items[0].keys() if k != "metadata"]
    for k in keys:
        batch[k] = torch.stack([x[k] for x in items])
    batch["metadata"] = [x["metadata"] for x in items]
    return batch


def build_dataset(cfg: Dict[str, Any], split: str, task: str) -> GenatatorDataset:
    return GenatatorDataset(cfg, split=split, task=task)
