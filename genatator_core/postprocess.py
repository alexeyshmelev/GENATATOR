from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gene-finding FFT + peak-calling code.
# This is intentionally kept equivalent to the public GENATATOR pipeline code:
#   fft_lowpass -> call_peaks_on_segment -> peak_finding_indices ->
#   find_tss_polya_pairs_from_peak_indices -> filter_intervals_by_intragenic_bool.
# Channel order for peak finding is always:
#   TSS+, PolyA+, TSS-, PolyA-
# ---------------------------------------------------------------------------


def lowpass_fft(x: np.ndarray, frac: float) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=float))
    xf = np.fft.rfft(x)
    k = int(np.clip(float(frac), 0.0, 1.0) * len(xf))
    if k < 1:
        k = 1
    xf_lp = np.zeros_like(xf)
    xf_lp[:k] = xf[:k]
    return np.fft.irfft(xf_lp, n=len(x))


def call_peaks_on_segment(
    x: np.ndarray,
    lp_frac: float,
    pk_prom: float,
    pk_dist: int,
    pk_height: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    y_lp = lowpass_fft(x, frac=lp_frac)
    idx, _props = find_peaks(y_lp, prominence=pk_prom, distance=pk_dist, height=pk_height)
    return idx.astype(np.int64), y_lp


def peaks(x: np.ndarray, prominence: float, distance: int, height: Optional[float] = None) -> np.ndarray:
    idx, _ = call_peaks_on_segment(x, lp_frac=1.0, pk_prom=prominence, pk_dist=distance, pk_height=height)
    return idx.astype(int)


def peak_finding_indices(
    x: np.ndarray,
    lp_frac: float,
    pk_prom: float,
    pk_dist: int,
    pk_height: Optional[float],
    coordinate_offset: int = 0,
    log: Optional[logging.Logger] = None,
) -> List[np.ndarray]:
    log = log or logger
    x = np.nan_to_num(np.asarray(x, dtype=float))
    if x.ndim != 2 or x.shape[0] != 4:
        raise RuntimeError(f"peak_finding_indices expects shape (4, N) in order TSS+, PolyA+, TSS-, PolyA-, got {x.shape}")
    log.info(
        "Peak finding | input_shape=%s | lp_frac=%.4f | pk_prom=%.4f | pk_dist=%d | pk_height=%s",
        tuple(x.shape), float(lp_frac), float(pk_prom), int(pk_dist), str(pk_height),
    )
    out: List[np.ndarray] = []
    channel_names = ["TSS+", "PolyA+", "TSS-", "PolyA-"]
    for channel_i, channel_name in enumerate(channel_names):
        idx_local, y_lp = call_peaks_on_segment(
            x[channel_i],
            lp_frac=lp_frac,
            pk_prom=pk_prom,
            pk_dist=pk_dist,
            pk_height=pk_height,
        )
        idx = np.asarray(idx_local, dtype=np.int64) + int(coordinate_offset)
        log.info(
            "Peak finding | channel=%s | raw_max=%.5f | smooth_max=%.5f | peaks=%d | first10=%s",
            channel_name,
            float(np.max(x[channel_i])) if x[channel_i].size else 0.0,
            float(np.max(y_lp)) if y_lp.size else 0.0,
            len(idx),
            idx[:10].tolist(),
        )
        out.append(idx)
    return out


def _choose_k_nearest(seed: int, candidates: np.ndarray, k: int) -> np.ndarray:
    if k is None or k <= 0 or candidates.size <= k:
        return candidates
    dist = np.abs(candidates - seed)
    idx = np.argpartition(dist, k - 1)[:k]
    order = np.lexsort((candidates[idx], dist[idx]))
    return candidates[idx][order]


def find_tss_polya_pairs_from_peak_indices(
    tss_plus_idx: np.ndarray,
    polya_plus_idx: np.ndarray,
    tss_minus_idx: np.ndarray,
    polya_minus_idx: np.ndarray,
    sequence_length: int,
    chrom_name: str,
    window_size: int = 2_000_000,
    k: int = 10,
    progress_every: Optional[int] = None,
    log: Optional[logging.Logger] = None,
) -> List[Dict]:
    log = log or logger
    n = int(sequence_length)
    window_size = min(int(window_size), n)
    tss_plus_idx = np.unique(np.asarray(tss_plus_idx, dtype=np.int64))
    polya_plus_idx = np.unique(np.asarray(polya_plus_idx, dtype=np.int64))
    tss_minus_idx = np.unique(np.asarray(tss_minus_idx, dtype=np.int64))
    polya_minus_idx = np.unique(np.asarray(polya_minus_idx, dtype=np.int64))
    pairs_sign: Dict[Tuple[int, int], str] = {}

    def scan(seeds_tss: np.ndarray, targets_polya: np.ndarray, direction: str, strand_sign: str, label: str) -> None:
        if progress_every:
            log.info("Scanning %s TSS seeds on strand %s", len(seeds_tss), label)
        for ii, seed in enumerate(seeds_tss):
            seed = int(seed)
            if direction == "right":
                start_w = seed
                end_w = min(n, seed + window_size)
            else:
                start_w = max(0, seed - window_size)
                end_w = seed + 1
            left = targets_polya.searchsorted(start_w, side="left")
            right = targets_polya.searchsorted(end_w, side="left")
            if right > left:
                for partner in _choose_k_nearest(seed, targets_polya[left:right], int(k)):
                    partner = int(partner)
                    a, b = (seed, partner) if seed <= partner else (partner, seed)
                    pairs_sign.setdefault((a, b), strand_sign)
            if progress_every and (ii + 1) % int(progress_every) == 0:
                log.info("processed %d/%d seeds; pairs=%d", ii + 1, len(seeds_tss), len(pairs_sign))

    scan(tss_plus_idx, polya_plus_idx, "right", "+", "plus")
    scan(tss_minus_idx, polya_minus_idx, "left", "-", "minus")
    pairs_sorted = sorted(pairs_sign.keys(), key=lambda ab: (ab[0], ab[1]))
    records = [{"chrom": chrom_name, "start": int(a), "end": int(b) + 1, "strand": pairs_sign[(a, b)]} for a, b in pairs_sorted if int(b) + 1 > int(a)]
    log.info("Pairing | constructed %d candidate intervals | first10=%s", len(records), [(r["start"], r["end"], r["strand"]) for r in records[:10]])
    return records


def filter_intervals_by_intragenic_bool(
    records: Sequence[Dict],
    intragenic_plus_mask: np.ndarray,
    intragenic_minus_mask: np.ndarray,
    zero_fraction_drop_threshold: float,
    log: Optional[logging.Logger] = None,
) -> List[Dict]:
    log = log or logger
    kept: List[Dict] = []
    for rec in tqdm(records, desc="Interval filtering", leave=False):
        start = int(rec["start"])
        end = int(rec["end"])
        strand = str(rec.get("strand", "+"))
        if end <= start:
            continue
        if strand == "+":
            mask = intragenic_plus_mask[start:end]
        elif strand == "-":
            mask = intragenic_minus_mask[start:end]
        else:
            mask = np.logical_or(intragenic_plus_mask[start:end], intragenic_minus_mask[start:end])
        if mask.size == 0:
            continue
        zero_fraction = float((mask == 0).mean())
        if zero_fraction <= float(zero_fraction_drop_threshold):
            kept.append(dict(rec))
    log.info("Kept %d / %d intervals after intragenic filtering.", len(kept), len(records))
    return kept


def pair_intervals(
    edge_tracks: np.ndarray,
    region_tracks: np.ndarray,
    chrom: str,
    window_size: int,
    max_pairs_per_seed: int,
    prob_threshold: float,
    zero_fraction_drop_threshold: float,
    lp_frac: float = 0.05,
    pk_prom: float = 0.1,
    pk_dist: int = 50,
    pk_height: Optional[float] = None,
    progress_every: Optional[int] = None,
    log: Optional[logging.Logger] = None,
) -> List[Dict]:
    """Compatibility wrapper around the official gene-finding post-processing path.

    ``edge_tracks`` must be model-order channels: TSS+, TSS-, PolyA+, PolyA-.
    Internally it is reordered to the pipeline peak-calling order:
    TSS+, PolyA+, TSS-, PolyA-.
    """
    log = log or logger
    if edge_tracks.shape[0] != 4:
        raise RuntimeError(f"edge_tracks must have shape (4, N), got {edge_tracks.shape}")
    if region_tracks.shape[0] != 2:
        raise RuntimeError(f"region_tracks must have shape (2, N), got {region_tracks.shape}")
    x = np.stack([edge_tracks[0], edge_tracks[2], edge_tracks[1], edge_tracks[3]], axis=0)
    tss_plus, polya_plus, tss_minus, polya_minus = peak_finding_indices(x, lp_frac, pk_prom, pk_dist, pk_height, log=log)
    pairs = find_tss_polya_pairs_from_peak_indices(
        tss_plus,
        polya_plus,
        tss_minus,
        polya_minus,
        sequence_length=edge_tracks.shape[1],
        chrom_name=chrom,
        window_size=window_size,
        k=max_pairs_per_seed,
        progress_every=progress_every,
        log=log,
    )
    masks_plus = np.asarray(region_tracks[0] > float(prob_threshold), dtype=np.bool_)
    masks_minus = np.asarray(region_tracks[1] > float(prob_threshold), dtype=np.bool_)
    return filter_intervals_by_intragenic_bool(pairs, masks_plus, masks_minus, zero_fraction_drop_threshold, log=log)


def best_interval_records(edge_probs: np.ndarray, region_probs: np.ndarray, chrom: str, max_records: int = 1, min_len: int = 64) -> List[Dict]:
    if edge_probs.ndim != 2 or region_probs.ndim != 2:
        raise RuntimeError(f"best_interval_records expects 2D tracks, got edge={edge_probs.shape}, region={region_probs.shape}")
    if edge_probs.shape[0] < 4 or region_probs.shape[0] < 2:
        raise RuntimeError(f"best_interval_records expects edge>=4 channels and region>=2 channels, got edge={edge_probs.shape}, region={region_probs.shape}")
    length = int(edge_probs.shape[1])
    if length <= 1:
        raise RuntimeError(f"Cannot create a fallback interval on chromosome {chrom}: length={length}")

    candidates: List[Tuple[float, Dict]] = []
    min_len = max(2, min(int(min_len), length))

    s = int(np.nanargmax(edge_probs[0]))      # TSS+
    e = int(np.nanargmax(edge_probs[2]))      # PolyA+
    score = float(edge_probs[0, s] + edge_probs[2, e] + np.nanmax(region_probs[0]))
    if e > s:
        candidates.append((score, {"chrom": chrom, "start": s, "end": e + 1, "strand": "+"}))

    s = int(np.nanargmax(edge_probs[1]))      # TSS-
    e = int(np.nanargmax(edge_probs[3]))      # PolyA-
    score = float(edge_probs[1, s] + edge_probs[3, e] + np.nanmax(region_probs[1]))
    if e < s:
        candidates.append((score, {"chrom": chrom, "start": e, "end": s + 1, "strand": "-"}))

    if not candidates:
        plus_max = float(np.nanmax(region_probs[0]))
        minus_max = float(np.nanmax(region_probs[1]))
        strand = "+" if plus_max >= minus_max else "-"
        channel = 0 if strand == "+" else 1
        center = int(np.nanargmax(region_probs[channel]))
        start = max(0, min(length - min_len, center - min_len // 2))
        end = min(length, start + min_len)
        score = plus_max if strand == "+" else minus_max
        candidates.append((score, {"chrom": chrom, "start": start, "end": end, "strand": strand}))

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
    out: List[Dict] = []
    seen = set()
    for _, rec in candidates:
        if rec["end"] <= rec["start"]:
            continue
        key = (rec["chrom"], rec["start"], rec["end"], rec["strand"])
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
        if len(out) >= int(max_records):
            break
    return out
