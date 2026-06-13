from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import find_peaks


def lowpass_fft(x: np.ndarray, frac: float) -> np.ndarray:
    spec = np.fft.rfft(x)
    keep = max(1, int(len(spec) * frac))
    spec[keep:] = 0
    return np.fft.irfft(spec, n=len(x))


def peaks(x: np.ndarray, prominence: float, distance: int, height=None) -> np.ndarray:
    kwargs = {"prominence": prominence, "distance": distance}
    if height is not None:
        kwargs["height"] = height
    p, _ = find_peaks(x, **kwargs)
    return p.astype(int)


def pair_intervals(edge_tracks: np.ndarray, region_tracks: np.ndarray, chrom: str, window_size: int, max_pairs_per_seed: int, prob_threshold: float, zero_fraction_drop_threshold: float) -> List[Dict]:
    tss_p, tss_m, polya_p, polya_m = edge_tracks
    intra_p, intra_m = region_tracks
    records: List[Dict] = []
    for strand, tss, polya, intra in [('+', tss_p, polya_p, intra_p), ('-', tss_m, polya_m, intra_m)]:
        if strand == '+':
            seeds = sorted(tss.tolist())
            ends = sorted(polya.tolist())
            for s in seeds:
                partners = [e for e in ends if e > s and e - s <= window_size][:max_pairs_per_seed]
                for e in partners:
                    mask = intra[s:e] >= prob_threshold
                    if len(mask) and float((~mask).mean()) <= zero_fraction_drop_threshold:
                        records.append({"chrom": chrom, "start": s, "end": e + 1, "strand": strand})
        else:
            seeds = sorted(tss.tolist(), reverse=True)
            ends = sorted(polya.tolist(), reverse=True)
            for s in seeds:
                partners = [e for e in ends if e < s and s - e <= window_size][:max_pairs_per_seed]
                for e in partners:
                    a, b = e, s + 1
                    mask = intra[a:b] >= prob_threshold
                    if len(mask) and float((~mask).mean()) <= zero_fraction_drop_threshold:
                        records.append({"chrom": chrom, "start": a, "end": b, "strand": strand})
    seen = set()
    unique = []
    for r in records:
        key = (r["chrom"], r["start"], r["end"], r["strand"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique
