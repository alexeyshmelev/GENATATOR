from typing import Dict, List
import numpy as np
from scipy.signal import find_peaks


def lowpass(signal: np.ndarray, keep_fraction: float) -> np.ndarray:
    spec = np.fft.rfft(signal)
    keep = max(1, int(len(spec) * keep_fraction))
    spec[keep:] = 0
    return np.fft.irfft(spec, n=len(signal)).astype(np.float32)


def peaks(track: np.ndarray, keep_fraction: float, prominence: float, distance: int, height=None) -> np.ndarray:
    smooth = lowpass(track, keep_fraction)
    idx, _ = find_peaks(smooth, prominence=prominence, distance=distance, height=height)
    return idx.astype(np.int64)


def build_intervals(edge_probs: np.ndarray, region_probs: np.ndarray, cfg: Dict) -> List[Dict]:
    lp = float(cfg.get("lp_frac", 0.05))
    prom = float(cfg.get("pk_prom", 0.15))
    dist = int(cfg.get("pk_dist", 50))
    height = cfg.get("pk_height")
    window = int(cfg.get("interval_window_size", 2_000_000))
    max_pairs = int(cfg.get("max_pairs_per_seed", 10))
    prob_thr = float(cfg.get("prob_threshold", 0.5))
    zero_thr = float(cfg.get("zero_fraction_drop_threshold", 0.01))

    tss_p = peaks(edge_probs[:, 0], lp, prom, dist, height)
    tss_m = peaks(edge_probs[:, 1], lp, prom, dist, height)
    polya_p = peaks(edge_probs[:, 2], lp, prom, dist, height)
    polya_m = peaks(edge_probs[:, 3], lp, prom, dist, height)
    intervals = []

    for t in tss_p:
        partners = polya_p[(polya_p > t) & (polya_p <= t + window)][:max_pairs]
        for p in partners:
            mask = (region_probs[t:p + 1, 0] >= prob_thr)
            if 1.0 - mask.mean() <= zero_thr:
                intervals.append({"start": int(t), "end": int(p) + 1, "strand": "+"})

    for t in tss_m:
        partners = polya_m[(polya_m < t) & (polya_m >= t - window)][-max_pairs:]
        for p in partners:
            a, b = sorted([int(t), int(p)])
            mask = (region_probs[a:b + 1, 1] >= prob_thr)
            if 1.0 - mask.mean() <= zero_thr:
                intervals.append({"start": a, "end": b + 1, "strand": "-"})

    seen = set()
    unique = []
    for r in intervals:
        key = (r["start"], r["end"], r["strand"])
        if key not in seen:
            unique.append(r); seen.add(key)
    return unique


def write_intervals_tsv(intervals: List[Dict], path: str, chrom: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("chrom\tstart\tend\tstrand\n")
        for r in intervals:
            f.write(f"{chrom}\t{r['start']}\t{r['end']}\t{r['strand']}\n")
