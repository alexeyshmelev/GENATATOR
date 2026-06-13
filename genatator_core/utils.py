from __future__ import annotations

import importlib
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_class(path: str):
    module, name = path.split(":")
    return getattr(importlib.import_module(module), name)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(table)[::-1].upper()


def tensor_to_list(x: Any):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: tensor_to_list(v) for k, v in x.items()}
    if isinstance(x, list):
        return [tensor_to_list(v) for v in x]
    return x


def first_existing_file(directory: str | Path, names: Iterable[str]) -> Path | None:
    directory = Path(directory).expanduser()
    for name in names:
        p = directory / name
        if p.exists():
            return p
    return None


def clean_env_for_gpu(gpus: str | None) -> Dict[str, str]:
    env = os.environ.copy()
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    return env
