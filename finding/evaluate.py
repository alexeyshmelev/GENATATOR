#!/usr/bin/env python
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from genatator_core.config import load_json
from genatator_core.data import GenatatorCollator, GenatatorDataset
from genatator_core.infer_common import prepare_model, sigmoid, undo_reverse_complement_logits
from genatator_core.metrics_training import _safe_binary_average_precision
from genatator_core.run_management import atomic_save_json
from genatator_core.train_common import dataset_family_from_model


logger = logging.getLogger(__name__)
TrackStore = Dict[str, Tuple[np.ndarray, np.ndarray]]


def _ensure_track(store: TrackStore, chrom: str, length: int, channels: int) -> None:
    if chrom not in store:
        store[chrom] = (
            np.zeros((channels, length), dtype=np.float32),
            np.zeros((channels, length), dtype=np.float32),
        )
        return
    sums, counts = store[chrom]
    if sums.shape[1] < length:
        extra = length - sums.shape[1]
        store[chrom] = (
            np.pad(sums, ((0, 0), (0, extra))),
            np.pad(counts, ((0, 0), (0, extra))),
        )


def _project_logits(probs: np.ndarray, family: str, batch: dict, sample_index: int, task: str, is_rc: bool) -> np.ndarray:
    dna_length = len(batch["dna_sequence"][sample_index])
    channels = int(probs.shape[-1])
    if family in {"nucleotide", "bpe_unet", "rmt_unet", "amt_unet"}:
        mask = batch["letter_level_labels_mask"][sample_index].detach().cpu().numpy().astype(bool)
        retained = probs[sample_index][mask]
        retained_length = min(dna_length, int(retained.shape[0]))
        values = np.full((dna_length, channels), np.nan, dtype=np.float32)
        values[:retained_length] = retained[:retained_length]
    else:
        sums = np.zeros((dna_length, channels), dtype=np.float32)
        counts = np.zeros(dna_length, dtype=np.float32)
        attention = batch["attention_mask"][sample_index].detach().cpu().numpy()
        for token_index, ((start, end), use) in enumerate(zip(batch["offset_mapping"][sample_index], attention)):
            if not int(use) or int(end) <= int(start):
                continue
            start = max(0, min(dna_length, int(start)))
            end = max(0, min(dna_length, int(end)))
            if end > start:
                sums[start:end] += probs[sample_index, token_index]
                counts[start:end] += 1.0
        values = np.full((dna_length, channels), np.nan, dtype=np.float32)
        covered = counts > 0
        values[covered] = sums[covered] / counts[covered, None]
    return undo_reverse_complement_logits(values, task) if is_rc else values


def _finalize(store: TrackStore) -> Dict[str, np.ndarray]:
    finalized = {}
    for chrom, (sums, counts) in store.items():
        values = np.full_like(sums, np.nan, dtype=np.float32)
        np.divide(sums, counts, out=values, where=counts > 0)
        finalized[chrom] = values
    return finalized


def predict_stage_tracks(cfg: dict, *, task: str, device: str) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    model, tokenizer, nucleotide_tokenizer = prepare_model(cfg, task, device)
    family = dataset_family_from_model(cfg["model"])
    base_data_cfg = dict(cfg["dataset"])
    base_data_cfg["model_family"] = family
    use_rc = bool(cfg.get("inference", {}).get("use_reverse_complement", False))
    prediction_store: TrackStore = {}
    truth_store: TrackStore = {}

    with torch.no_grad():
        for is_rc in ([False, True] if use_rc else [False]):
            data_cfg = dict(base_data_cfg)
            data_cfg["reverse_complement"] = is_rc
            dataset = GenatatorDataset(
                data_cfg,
                task=task,
                tokenizer=tokenizer,
                nucleotide_tokenizer=nucleotide_tokenizer,
                for_inference=True,
            )
            loader = DataLoader(
                dataset,
                batch_size=int(cfg.get("inference", {}).get("batch_size", 1)),
                collate_fn=GenatatorCollator(),
            )
            for batch in tqdm(loader, desc=f"evaluate:{task}:rc={is_rc}"):
                tensor_batch = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
                output = model(**tensor_batch)
                logits = output["logits"] if isinstance(output, dict) else output.logits
                probs = sigmoid(logits.detach().float().cpu().numpy())
                for sample_index in range(int(probs.shape[0])):
                    values = _project_logits(probs, family, batch, sample_index, task, is_rc)
                    metadata = batch["metadata"][sample_index]
                    chrom = metadata.chrom
                    start = int(metadata.start) + int(batch["local_start"][sample_index])
                    end = start + int(values.shape[0])
                    _ensure_track(prediction_store, chrom, end, int(values.shape[1]))
                    sums, counts = prediction_store[chrom]
                    projected = values.T
                    finite = np.isfinite(projected)
                    sums[:, start:end] += np.where(finite, projected, 0.0)
                    counts[:, start:end] += finite.astype(np.float32)

                    if not is_rc:
                        truth = np.asarray(batch["truth_labels"][sample_index], dtype=np.float32)
                        truth_end = start + int(truth.shape[0])
                        _ensure_track(truth_store, chrom, truth_end, int(truth.shape[1]))
                        truth_sums, truth_counts = truth_store[chrom]
                        truth_sums[:, start:truth_end] += truth.T
                        truth_counts[:, start:truth_end] += 1.0
    return _finalize(prediction_store), _finalize(truth_store)


def evaluate_tracks(predictions: Dict[str, np.ndarray], truth: Dict[str, np.ndarray], class_names: List[str]) -> dict:
    per_chromosome = {}
    pooled_references: List[List[np.ndarray]] = [[] for _ in class_names]
    pooled_scores: List[List[np.ndarray]] = [[] for _ in class_names]
    for chrom in sorted(predictions):
        if chrom not in truth:
            continue
        predicted = predictions[chrom]
        reference = truth[chrom]
        length = min(int(predicted.shape[1]), int(reference.shape[1]))
        channels = min(int(predicted.shape[0]), int(reference.shape[0]), len(class_names))
        per_chromosome[chrom] = {}
        for channel in range(channels):
            refs = reference[channel, :length]
            scores = predicted[channel, :length]
            ap, defined, positives, negatives, dropped = _safe_binary_average_precision(refs, scores)
            per_chromosome[chrom][class_names[channel]] = {
                "pr_auc": float(ap),
                "defined": bool(defined),
                "positives": int(positives),
                "negatives": int(negatives),
                "dropped_nonfinite": int(dropped),
            }
            pooled_references[channel].append(refs)
            pooled_scores[channel].append(scores)

    pooled = {}
    defined_values = []
    for channel, class_name in enumerate(class_names):
        refs = np.concatenate(pooled_references[channel]) if pooled_references[channel] else np.asarray([])
        scores = np.concatenate(pooled_scores[channel]) if pooled_scores[channel] else np.asarray([])
        ap, defined, positives, negatives, dropped = _safe_binary_average_precision(refs, scores)
        pooled[class_name] = {
            "pr_auc": float(ap),
            "defined": bool(defined),
            "positives": int(positives),
            "negatives": int(negatives),
            "dropped_nonfinite": int(dropped),
        }
        if defined:
            defined_values.append(float(ap))
    return {
        "per_chromosome": per_chromosome,
        "pooled": pooled,
        "mean_pr_auc": float(np.mean(defined_values)) if defined_values else 0.0,
        "defined_channels": len(defined_values),
    }


def main() -> None:
    parser = ArgumentParser(description="Evaluate one trained GENATATOR gene-finding stage")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    task = str(cfg.get("task", ""))
    if task not in {"finding_edge", "finding_region"}:
        raise RuntimeError("Single-stage finding evaluation config must set task to finding_edge or finding_region")
    class_names = ["TSS+", "TSS-", "PolyA+", "PolyA-"] if task == "finding_edge" else ["intragenic+", "intragenic-"]
    device = cfg.get("inference", {}).get("device", "cuda")
    predictions, truth = predict_stage_tracks(cfg, task=task, device=device)
    metrics = {"task": task, "pr_auc": evaluate_tracks(predictions, truth, class_names)}
    metrics_path = Path(cfg["inference"]["metrics_json"]).expanduser()
    atomic_save_json(metrics, metrics_path)
    logger.info("Wrote %s single-stage metrics to %s", task, metrics_path)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
    main()
