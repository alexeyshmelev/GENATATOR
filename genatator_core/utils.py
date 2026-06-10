import os
import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dna_reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1].upper()


def dna_to_ids(seq: str) -> np.ndarray:
    table = np.full(256, 4, dtype=np.int64)
    for i, ch in enumerate(b"ACGT"):
        table[ch] = i
        table[ch + 32] = i
    return table[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


def parse_fasta(path: str | Path) -> Iterator[Tuple[str, str]]:
    name = None
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks).upper()
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if name is not None:
            yield name, "".join(chunks).upper()


def sliding_windows(length: int, window: int, overlap: float) -> list[tuple[int, int]]:
    step = max(1, int(window * (1.0 - overlap)))
    starts = list(range(0, max(1, length), step))
    out = []
    for s in starts:
        e = min(length, s + window)
        if e > s:
            out.append((s, e))
        if e == length:
            break
    return out


def move_to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def main_process(accelerator) -> bool:
    return accelerator.is_main_process
