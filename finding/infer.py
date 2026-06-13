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
from genatator_core.data import GenatatorCollator, GenatatorDataset, make_tokenizer
from genatator_core.evaluate_gff import evaluate_annotation
from genatator_core.gff import write_finding_gff
from genatator_core.model_builders import build_model, load_finetuned_weights
from genatator_core.postprocess import lowpass_fft, pair_intervals, peaks


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ensure_tracks(store: Dict[str, Tuple[np.ndarray, np.ndarray]], chrom: str, length: int, n_channels: int):
    if chrom not in store:
        store[chrom] = (np.zeros((n_channels, length), dtype=np.float32), np.zeros((n_channels, length), dtype=np.float32))
        return
    sums, counts = store[chrom]
    if sums.shape[1] < length:
        add = length - sums.shape[1]
        store[chrom] = (np.pad(sums, ((0, 0), (0, add))), np.pad(counts, ((0, 0), (0, add))))


def predict_tracks(stage_cfg, task: str, device: str):
    tokenizer = make_tokenizer(stage_cfg["model"]["tokenizer_path"], trust_remote_code=bool(stage_cfg["model"].get("trust_remote_code", True)))
    if stage_cfg["model"].get("padding_side"):
        tokenizer.padding_side = stage_cfg["model"]["padding_side"]
    stage_cfg["_tokenizer"] = tokenizer
    model = build_model(stage_cfg, task=task)
    ckpt = stage_cfg.get("inference", {}).get("checkpoint_path")
    if ckpt:
        load_finetuned_weights(model, ckpt)
    model.to(device).eval()
    data_cfg = dict(stage_cfg["dataset"])
    data_cfg["model_family"] = "bpe"
    ds = GenatatorDataset(data_cfg, task=task, tokenizer=tokenizer, for_inference=True)
    dl = DataLoader(ds, batch_size=int(stage_cfg.get("inference", {}).get("batch_size", 1)), collate_fn=GenatatorCollator())
    tracks: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    with torch.no_grad():
        for batch in tqdm(dl, desc=task):
            metas = batch.pop("metadata")
            local_starts = batch.pop("local_start")
            offsets = batch.pop("offset_mapping")
            batch.pop("dna_sequence")
            tensor_batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            out = model(**tensor_batch)
            logits = (out["logits"] if isinstance(out, dict) else out.logits).detach().cpu().numpy()
            probs = sigmoid(logits)
            attention = tensor_batch["attention_mask"].detach().cpu().numpy()
            for b in range(probs.shape[0]):
                meta = metas[b]
                chrom = meta.chrom
                n_ch = probs.shape[-1]
                for tok_i, ((s, e), attn) in enumerate(zip(offsets[b], attention[b])):
                    if int(attn) == 0 or e <= s:
                        continue
                    start = int(meta.start) + int(local_starts[b]) + int(s)
                    end = int(meta.start) + int(local_starts[b]) + int(e)
                    ensure_tracks(tracks, chrom, end, n_ch)
                    sums, counts = tracks[chrom]
                    sums[:, start:end] += probs[b, tok_i, :, None]
                    counts[:, start:end] += 1.0
    return {chrom: sums / np.maximum(counts, 1.0) for chrom, (sums, counts) in tracks.items()}


parser = ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
cfg = load_json(args.config)
device = cfg.get("inference", {}).get("device", "cuda")
edge_tracks = predict_tracks(cfg["edge"], "finding_edge", device)
region_tracks = predict_tracks(cfg["region"], "finding_region", device)
records = []
post = cfg.get("postprocess", {})
for chrom in sorted(edge_tracks):
    if chrom not in region_tracks:
        continue
    edge = edge_tracks[chrom]
    smooth = np.stack([lowpass_fft(edge[i], float(post.get("lp_frac", 0.05))) for i in range(edge.shape[0])])
    peak_tracks = np.array([
        peaks(smooth[i], prominence=float(post.get("pk_prom", 0.1)), distance=int(post.get("pk_dist", 50)), height=post.get("pk_height"))
        for i in range(4)
    ], dtype=object)
    chrom_records = pair_intervals(
        peak_tracks,
        region_tracks[chrom],
        chrom=chrom,
        window_size=int(post.get("interval_window_size", 2_000_000)),
        max_pairs_per_seed=int(post.get("max_pairs_per_seed", 10)),
        prob_threshold=float(post.get("prob_threshold", 0.5)),
        zero_fraction_drop_threshold=float(post.get("zero_fraction_drop_threshold", 0.01)),
    )
    records.extend(chrom_records)
out_gff = cfg["inference"]["output_gff"]
write_finding_gff(records, out_gff)
if cfg["inference"].get("true_gff"):
    evaluate_annotation(out_gff, cfg["inference"]["true_gff"], cfg["inference"].get("metrics_json", str(Path(out_gff).with_suffix(".metrics.json"))), k_values=cfg["inference"].get("k_values", [0, 50, 100, 250, 500]), use_strand=bool(cfg["inference"].get("use_strand", True)))
