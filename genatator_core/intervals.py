from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

Interval = Tuple[int, int]


def binary_intervals(x: Sequence[int] | np.ndarray) -> List[Interval]:
    arr = np.asarray(x).astype(bool)
    idx = np.flatnonzero(arr)
    if len(idx) == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1) + 1
    groups = np.split(idx, cuts)
    return [(int(g[0]), int(g[-1]) + 1) for g in groups]


def interval_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> tuple[int, int, int]:
    t = set(binary_intervals(y_true))
    p = set(binary_intervals(y_pred))
    tp = len(t & p)
    fp = len(p - t)
    fn = len(t - p)
    return tp, fp, fn


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else float(2 * tp / denom)


def exact_interval_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp, fp, fn = interval_counts(y_true, y_pred)
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    return {"precision": float(precision), "recall": float(recall), "f1": f1_from_counts(tp, fp, fn)}
