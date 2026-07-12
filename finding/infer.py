#!/usr/bin/env python
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from genatator_core.config import load_json
from genatator_core.data import GenatatorCollator, GenatatorDataset
from genatator_core.evaluate_gff import evaluate_annotation
from genatator_core.gff import write_finding_gff
from genatator_core.infer_common import (
    prepare_model,
    project_bpe_token_logits_to_nucleotides,
    project_masked_letter_logits_to_nucleotides,
    sigmoid,
    undo_reverse_complement_logits,
)
from genatator_core.postprocess import (
    best_interval_records,
    filter_intervals_by_intragenic_bool,
    find_tss_polya_pairs_from_peak_indices,
    peak_finding_indices,
)
from genatator_core.train_common import dataset_family_from_model

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

TrackStore = Dict[str, Tuple[np.ndarray, np.ndarray]]


def ensure_tracks(store: TrackStore, chrom: str, length: int, n_channels: int) -> None:
    if chrom not in store:
        store[chrom] = (np.zeros((n_channels, length), dtype=np.float32), np.zeros((n_channels, length), dtype=np.float32))
        return
    sums, counts = store[chrom]
    if sums.shape[1] < length:
        add = length - sums.shape[1]
        store[chrom] = (np.pad(sums, ((0, 0), (0, add))), np.pad(counts, ((0, 0), (0, add))))


def project_sample_logits(probs: np.ndarray, model_family: str, batch: dict, b: int, task: str, is_rc: bool) -> np.ndarray:
    """Project one sample to its full nucleotide crop, leaving uncovered bases NaN."""
    dna_len = len(batch["dna_sequence"][b])
    if model_family in {"nucleotide", "bpe_unet", "rmt_unet", "amt_unet"}:
        mask = batch["letter_level_labels_mask"][b].detach().cpu().numpy().astype(bool)
        arr = project_masked_letter_logits_to_nucleotides(probs[b], mask, dna_len)
        return undo_reverse_complement_logits(arr, task) if is_rc else arr
    attn = batch["attention_mask"][b].detach().cpu().numpy()
    arr = project_bpe_token_logits_to_nucleotides(
        probs[b], batch["offset_mapping"][b], attn, dna_len
    )
    return undo_reverse_complement_logits(arr, task) if is_rc else arr


def _finalize_store(store: TrackStore) -> Dict[str, np.ndarray]:
    finalized: Dict[str, np.ndarray] = {}
    for chrom, (sums, counts) in store.items():
        values = np.full_like(sums, np.nan, dtype=np.float32)
        np.divide(sums, counts, out=values, where=counts > 0)
        finalized[chrom] = values
    return finalized


def predict_tracks(stage_cfg, task: str, device: str, use_reverse_complement: bool) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    model, tokenizer, nucleotide_tokenizer = prepare_model(stage_cfg, task, device)
    data_cfg = dict(stage_cfg["dataset"])
    data_cfg["model_family"] = dataset_family_from_model(stage_cfg["model"])
    pred_tracks: TrackStore = {}
    truth_tracks: TrackStore = {}
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
                    ensure_tracks(pred_tracks, chrom, base_start + vals.shape[0], n_ch)
                    sums, counts = pred_tracks[chrom]
                    end = base_start + vals.shape[0]
                    projected = vals.T
                    finite = np.isfinite(projected)
                    sums[:, base_start:end] += np.where(finite, projected, 0.0)
                    counts[:, base_start:end] += finite.astype(np.float32)

                    # Truth labels are gathered only once, in the forward-orientation pass.
                    # They are nucleotide-resolution and are used for whole-chromosome PR-AUC.
                    if not is_rc and "truth_labels" in batch:
                        y = np.asarray(batch["truth_labels"][b], dtype=np.float32)
                        if y.ndim == 2 and y.shape[0] > 0:
                            ensure_tracks(truth_tracks, chrom, base_start + y.shape[0], y.shape[1])
                            ys, yc = truth_tracks[chrom]
                            y_end = base_start + y.shape[0]
                            ys[:, base_start:y_end] += y.T
                            yc[:, base_start:y_end] += 1.0
    return _finalize_store(pred_tracks), _finalize_store(truth_tracks)


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    y_true = (np.asarray(y_true) > 0).astype(np.uint8)
    y_score = np.asarray(y_score).astype(float)
    ok = np.isfinite(y_score)
    y_true = y_true[ok]
    y_score = y_score[ok]
    if y_true.size == 0 or int(y_true.sum()) == 0 or int((1 - y_true).sum()) == 0:
        return None
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y_true, y_score))
    except Exception:
        # Simple exact AP implementation by score ordering.
        order = np.argsort(-y_score, kind="mergesort")
        y = y_true[order]
        tp = np.cumsum(y)
        rank = np.arange(1, len(y) + 1)
        precision = tp / rank
        return float((precision * y).sum() / max(1, int(y.sum())))


def compute_whole_chromosome_pr_auc(pred: Dict[str, np.ndarray], truth: Dict[str, np.ndarray], class_names: List[str]) -> Dict:
    per_chrom: Dict[str, Dict[str, Optional[float]]] = {}
    pooled_scores: List[List[np.ndarray]] = [[] for _ in class_names]
    pooled_truth: List[List[np.ndarray]] = [[] for _ in class_names]
    for chrom in sorted(pred):
        if chrom not in truth:
            continue
        p = pred[chrom]
        y = truth[chrom]
        n = min(p.shape[1], y.shape[1])
        c = min(p.shape[0], y.shape[0], len(class_names))
        per_chrom[chrom] = {}
        for i in range(c):
            yi = y[i, :n]
            pi = p[i, :n]
            ap = average_precision(yi, pi)
            per_chrom[chrom][class_names[i]] = ap
            pooled_scores[i].append(pi)
            pooled_truth[i].append(yi)
    pooled: Dict[str, Optional[float]] = {}
    vals = []
    for i, name in enumerate(class_names):
        if pooled_scores[i]:
            ap = average_precision(np.concatenate(pooled_truth[i]), np.concatenate(pooled_scores[i]))
        else:
            ap = None
        pooled[name] = ap
        if ap is not None:
            vals.append(float(ap))
    return {"per_chromosome": per_chrom, "pooled": pooled, "mean": float(np.mean(vals)) if vals else None}


parser = ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
cfg = load_json(args.config)
device = cfg.get("inference", {}).get("device", "cuda")
use_rc = bool(cfg.get("inference", {}).get("use_reverse_complement", False))

edge_tracks, edge_truth = predict_tracks(cfg["edge"], "finding_edge", device, use_reverse_complement=use_rc)
region_tracks, region_truth = predict_tracks(cfg["region"], "finding_region", device, use_reverse_complement=use_rc)

metrics: Dict[str, object] = {
    "pr_auc": {
        "edge": compute_whole_chromosome_pr_auc(edge_tracks, edge_truth, ["TSS+", "TSS-", "PolyA+", "PolyA-"]),
        "region": compute_whole_chromosome_pr_auc(region_tracks, region_truth, ["intragenic+", "intragenic-"]),
    }
}
logger.info("Whole-chromosome PR-AUC edge=%s", json.dumps(metrics["pr_auc"]["edge"]["pooled"], ensure_ascii=False))
logger.info("Whole-chromosome PR-AUC region=%s", json.dumps(metrics["pr_auc"]["region"]["pooled"], ensure_ascii=False))

records = []
post = cfg.get("postprocess", {})
logger.info(
    "Gene-finding postprocess parameters | lp_frac=%s pk_prom=%s pk_dist=%s pk_height=%s interval_window_size=%s max_pairs_per_seed=%s prob_threshold=%s zero_fraction_drop_threshold=%s pairing_progress_every=%s",
    post.get("lp_frac", 0.05), post.get("pk_prom", 0.1), post.get("pk_dist", 50), post.get("pk_height"),
    post.get("interval_window_size", 2_000_000), post.get("max_pairs_per_seed", 10), post.get("prob_threshold", 0.5),
    post.get("zero_fraction_drop_threshold", 0.01), post.get("pairing_progress_every"),
)
for chrom in sorted(edge_tracks):
    if chrom not in region_tracks:
        raise RuntimeError(f"Region tracks missing chromosome {chrom}")
    # Missing BPE coverage remains NaN for PR-AUC exclusion, but postprocessing
    # needs finite tracks. Treat genuinely uncovered bases as no signal here.
    edge = np.nan_to_num(edge_tracks[chrom], nan=0.0)
    region = np.nan_to_num(region_tracks[chrom], nan=0.0)
    if edge.shape[0] != 4:
        raise RuntimeError(f"Edge tracks must have 4 channels in model order TSS+,TSS-,PolyA+,PolyA-, got {edge.shape}")
    if region.shape[0] != 2:
        raise RuntimeError(f"Region tracks must have 2 channels in order intragenic+,intragenic-, got {region.shape}")

    # The public GENATATOR pipeline peak caller expects TSS+, PolyA+, TSS-, PolyA-.
    edge_for_peak = np.stack([edge[0], edge[2], edge[1], edge[3]], axis=0)
    tss_plus, polya_plus, tss_minus, polya_minus = peak_finding_indices(
        edge_for_peak,
        lp_frac=float(post.get("lp_frac", 0.05)),
        pk_prom=float(post.get("pk_prom", 0.1)),
        pk_dist=int(post.get("pk_dist", 50)),
        pk_height=post.get("pk_height"),
        coordinate_offset=0,
        log=logger,
    )
    pairs = find_tss_polya_pairs_from_peak_indices(
        tss_plus,
        polya_plus,
        tss_minus,
        polya_minus,
        sequence_length=edge.shape[1],
        chrom_name=chrom,
        window_size=int(post.get("interval_window_size", 2_000_000)),
        k=int(post.get("max_pairs_per_seed", 10)),
        progress_every=post.get("pairing_progress_every"),
        log=logger,
    )
    region_plus_mask = np.asarray(region[0] > float(post.get("prob_threshold", 0.5)), dtype=np.bool_)
    region_minus_mask = np.asarray(region[1] > float(post.get("prob_threshold", 0.5)), dtype=np.bool_)
    chrom_records = filter_intervals_by_intragenic_bool(
        pairs,
        intragenic_plus_mask=region_plus_mask,
        intragenic_minus_mask=region_minus_mask,
        zero_fraction_drop_threshold=float(post.get("zero_fraction_drop_threshold", 0.01)),
        log=logger,
    )
    if not chrom_records:
        policy = cfg.get("inference", {}).get("empty_gff_policy", "error")
        if policy == "best_interval":
            chrom_records = best_interval_records(
                edge,
                region,
                chrom=chrom,
                max_records=int(cfg.get("inference", {}).get("empty_gff_max_records", 1)),
                min_len=int(cfg.get("inference", {}).get("empty_gff_min_interval_len", 64)),
            )
            logger.warning(
                "No transcript intervals passed post-processing for chrom=%s; empty_gff_policy=best_interval produced %d explicit best-score interval(s).",
                chrom,
                len(chrom_records),
            )
        else:
            raise RuntimeError(
                f"No transcript intervals were produced for chromosome {chrom}. The official evaluator rejects empty GFF files. "
                "Lower postprocess thresholds or set inference.empty_gff_policy='best_interval' for smoke tests."
            )
    records.extend(chrom_records)

out_gff = cfg["inference"]["output_gff"]
logger.info("Writing gene-finding prediction GFF with %d transcript intervals; each interval is represented as one full-length exon for boundary evaluation.", len(records))
write_finding_gff(records, out_gff)

if cfg["inference"].get("true_gff"):
    metrics["annotation"] = evaluate_annotation(
        out_gff,
        cfg["inference"]["true_gff"],
        output_json=None,
        k_values=cfg["inference"].get("k_values", [0, 50, 100, 250, 500]),
        use_strand=bool(cfg["inference"].get("use_strand", True)),
    )

metrics_json = cfg["inference"].get("metrics_json", str(Path(out_gff).with_suffix(".metrics.json")))
Path(metrics_json).parent.mkdir(parents=True, exist_ok=True)
with open(metrics_json, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
logger.info("Wrote gene-finding inference metrics to %s", metrics_json)
