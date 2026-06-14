#!/usr/bin/env python
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from genatator_core.config import load_json
from genatator_core.data import GenatatorCollator, GenatatorDataset
from genatator_core.evaluate_gff import evaluate_annotation
from genatator_core.gff import write_finding_gff
from genatator_core.infer_common import prepare_model, sigmoid, undo_reverse_complement_logits
from genatator_core.postprocess import lowpass_fft, pair_intervals, peaks
from genatator_core.train_common import dataset_family_from_model


def ensure_tracks(store: Dict[str, Tuple[np.ndarray, np.ndarray]], chrom: str, length: int, n_channels: int):
    if chrom not in store:
        store[chrom] = (np.zeros((n_channels, length), dtype=np.float32), np.zeros((n_channels, length), dtype=np.float32))
        return
    sums, counts = store[chrom]
    if sums.shape[1] < length:
        add = length - sums.shape[1]
        store[chrom] = (np.pad(sums, ((0, 0), (0, add))), np.pad(counts, ((0, 0), (0, add))))


def project_sample_logits(probs: np.ndarray, model_family: str, batch: dict, b: int, task: str, is_rc: bool) -> np.ndarray:
    if model_family in {"nucleotide", "bpe_unet", "rmt_unet", "amt_unet"}:
        mask = batch["letter_level_labels_mask"][b].detach().cpu().numpy().astype(bool)
        arr = probs[b][mask]
        return undo_reverse_complement_logits(arr, task) if is_rc else arr
    dna_len = len(batch["dna_sequence"][b])
    n_ch = probs.shape[-1]
    tmp = np.zeros((dna_len, n_ch), dtype=np.float32)
    cnt = np.zeros((dna_len, n_ch), dtype=np.float32)
    attn = batch["attention_mask"][b].detach().cpu().numpy()
    for tok_i, ((s, e), a) in enumerate(zip(batch["offset_mapping"][b], attn)):
        if int(a) == 0 or e <= s:
            continue
        s = max(0, min(dna_len, int(s)))
        e = max(0, min(dna_len, int(e)))
        if e <= s:
            continue
        tmp[s:e] += probs[b, tok_i]
        cnt[s:e] += 1.0
    arr = tmp / np.maximum(cnt, 1.0)
    return undo_reverse_complement_logits(arr, task) if is_rc else arr


def predict_tracks(stage_cfg, task: str, device: str, use_reverse_complement: bool):
    model, tokenizer, nucleotide_tokenizer = prepare_model(stage_cfg, task, device)
    data_cfg = dict(stage_cfg["dataset"])
    data_cfg["model_family"] = dataset_family_from_model(stage_cfg["model"])
    tracks: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    passes = [False, True] if use_reverse_complement else [False]
    with torch.no_grad():
        for is_rc in passes:
            data_cfg_pass = dict(data_cfg)
            data_cfg_pass["reverse_complement"] = is_rc
            ds = GenatatorDataset(data_cfg_pass, task=task, tokenizer=tokenizer, nucleotide_tokenizer=nucleotide_tokenizer, for_inference=True)
            dl = DataLoader(ds, batch_size=int(stage_cfg.get("inference", {}).get("batch_size", 1)), collate_fn=GenatatorCollator())
            for batch in tqdm(dl, desc=f"{task}:rc={is_rc}"):
                metas = batch["metadata"]
                starts = batch["local_start"]
                tensor_batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                out = model(**tensor_batch)
                logits = (out["logits"] if isinstance(out, dict) else out.logits).detach().cpu().numpy()
                probs = sigmoid(logits)
                for b in range(probs.shape[0]):
                    vals = project_sample_logits(probs, data_cfg_pass["model_family"], batch, b, task, is_rc)
                    meta = metas[b]
                    chrom = meta.chrom
                    n_ch = vals.shape[-1]
                    base_start = int(meta.start) + int(starts[b])
                    ensure_tracks(tracks, chrom, base_start + vals.shape[0], n_ch)
                    sums, counts = tracks[chrom]
                    end = base_start + vals.shape[0]
                    sums[:, base_start:end] += vals.T
                    counts[:, base_start:end] += 1.0
    return {chrom: sums / np.maximum(counts, 1.0) for chrom, (sums, counts) in tracks.items()}


parser = ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
cfg = load_json(args.config)
device = cfg.get("inference", {}).get("device", "cuda")
use_rc = bool(cfg.get("inference", {}).get("use_reverse_complement", False))
edge_tracks = predict_tracks(cfg["edge"], "finding_edge", device, use_reverse_complement=use_rc)
region_tracks = predict_tracks(cfg["region"], "finding_region", device, use_reverse_complement=use_rc)
records = []
post = cfg.get("postprocess", {})
for chrom in sorted(edge_tracks):
    if chrom not in region_tracks:
        raise RuntimeError(f"Region tracks missing chromosome {chrom}")
    edge = edge_tracks[chrom]
    if edge.shape[0] != 4:
        raise RuntimeError(f"Edge tracks must have 4 channels, got {edge.shape}")
    smooth = np.stack([lowpass_fft(edge[i], float(post.get("lp_frac", 0.05))) for i in range(edge.shape[0])])
    peak_tracks = np.array([
        peaks(smooth[i], prominence=float(post.get("pk_prom", 0.1)), distance=int(post.get("pk_dist", 50)), height=post.get("pk_height"))
        for i in range(4)
    ], dtype=object)
    records.extend(pair_intervals(
        peak_tracks,
        region_tracks[chrom],
        chrom=chrom,
        window_size=int(post.get("interval_window_size", 2_000_000)),
        max_pairs_per_seed=int(post.get("max_pairs_per_seed", 10)),
        prob_threshold=float(post.get("prob_threshold", 0.5)),
        zero_fraction_drop_threshold=float(post.get("zero_fraction_drop_threshold", 0.01)),
    ))
out_gff = cfg["inference"]["output_gff"]
write_finding_gff(records, out_gff)
if cfg["inference"].get("true_gff"):
    evaluate_annotation(
        out_gff,
        cfg["inference"]["true_gff"],
        cfg["inference"].get("metrics_json", str(Path(out_gff).with_suffix(".metrics.json"))),
        k_values=cfg["inference"].get("k_values", [0, 50, 100, 250, 500]),
        use_strand=bool(cfg["inference"].get("use_strand", True)),
    )
