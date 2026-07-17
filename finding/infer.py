#!/usr/bin/env python
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
from genatator_core.run_management import atomic_save_json
from genatator_core.train_common import dataset_family_from_model

logger = logging.getLogger(__name__)
GenomeChromosome = Tuple[str, str]
TrackStore = Dict[GenomeChromosome, Tuple[np.ndarray, np.ndarray]]


def _require_batch_size_one(inference_cfg: Dict) -> None:
    batch_size = int(inference_cfg.get("batch_size", 1))
    if batch_size != 1:
        raise RuntimeError("GENATATOR inference batch_size must be 1 for every task/model")


def ensure_tracks(store: TrackStore, key: GenomeChromosome, length: int, n_channels: int) -> None:
    if key not in store:
        store[key] = (
            np.zeros((n_channels, length), dtype=np.float32),
            np.zeros((n_channels, length), dtype=np.float32),
        )
        return
    sums, counts = store[key]
    if sums.shape[1] < length:
        add = length - sums.shape[1]
        store[key] = (
            np.pad(sums, ((0, 0), (0, add))),
            np.pad(counts, ((0, 0), (0, add))),
        )


def project_sample_logits(
    probs: np.ndarray,
    model_family: str,
    batch: dict,
    sample_index: int,
    task: str,
    is_rc: bool,
) -> np.ndarray:
    """Project one sample to its full nucleotide crop; uncovered bases remain NaN."""
    dna_len = len(batch["dna_sequence"][sample_index])
    if model_family in {"nucleotide", "bpe_unet", "rmt_unet", "amt_unet"}:
        mask = batch["letter_level_labels_mask"][sample_index].detach().cpu().numpy().astype(bool)
        values = project_masked_letter_logits_to_nucleotides(
            probs[sample_index], mask, dna_len
        )
    else:
        attention = batch["attention_mask"][sample_index].detach().cpu().numpy()
        values = project_bpe_token_logits_to_nucleotides(
            probs[sample_index],
            batch["offset_mapping"][sample_index],
            attention,
            dna_len,
        )
    return undo_reverse_complement_logits(values, task) if is_rc else values


def _finalize_store(store: TrackStore) -> Dict[GenomeChromosome, np.ndarray]:
    finalized: Dict[GenomeChromosome, np.ndarray] = {}
    for key, (sums, counts) in store.items():
        values = np.full_like(sums, np.nan, dtype=np.float32)
        np.divide(sums, counts, out=values, where=counts > 0)
        finalized[key] = values
    return finalized


def predict_tracks(
    stage_cfg: Dict,
    task: str,
    device: str,
    use_reverse_complement: bool,
) -> Tuple[Dict[GenomeChromosome, np.ndarray], Dict[GenomeChromosome, np.ndarray]]:
    _require_batch_size_one(stage_cfg.get("inference", {}))
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
            dataset = GenatatorDataset(
                data_cfg_pass,
                task=task,
                tokenizer=tokenizer,
                nucleotide_tokenizer=nucleotide_tokenizer,
                for_inference=True,
            )
            loader = DataLoader(
                dataset,
                batch_size=1,
                num_workers=0,
                collate_fn=GenatatorCollator(),
            )
            for batch in tqdm(loader, desc=f"{task}:rc={is_rc}"):
                metas = batch["metadata"]
                starts = batch["local_start"]
                tensor_batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                    if isinstance(value, torch.Tensor)
                }
                output = model(**tensor_batch)
                logits = output["logits"] if isinstance(output, dict) else output.logits
                probs = sigmoid(logits.detach().float().cpu().numpy())
                for sample_index in range(probs.shape[0]):
                    values = project_sample_logits(
                        probs,
                        data_cfg_pass["model_family"],
                        batch,
                        sample_index,
                        task,
                        is_rc,
                    )
                    meta = metas[sample_index]
                    key = (meta.genome, meta.chrom)
                    base_start = int(meta.start) + int(starts[sample_index])
                    end = base_start + values.shape[0]
                    ensure_tracks(pred_tracks, key, end, values.shape[-1])
                    sums, counts = pred_tracks[key]
                    projected = values.T
                    finite = np.isfinite(projected)
                    sums[:, base_start:end] += np.where(finite, projected, 0.0)
                    counts[:, base_start:end] += finite.astype(np.float32)

                    # Truth is gathered once from the forward orientation.
                    if not is_rc and "truth_labels" in batch:
                        truth = np.asarray(batch["truth_labels"][sample_index], dtype=np.float32)
                        if truth.ndim == 2 and truth.shape[0] > 0:
                            truth_end = base_start + truth.shape[0]
                            ensure_tracks(truth_tracks, key, truth_end, truth.shape[1])
                            truth_sums, truth_counts = truth_tracks[key]
                            truth_sums[:, base_start:truth_end] += truth.T
                            truth_counts[:, base_start:truth_end] += 1.0
            dataset.release_finding_cache()

    return _finalize_store(pred_tracks), _finalize_store(truth_tracks)


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    y_true = (np.asarray(y_true) > 0).astype(np.uint8)
    y_score = np.asarray(y_score, dtype=float)
    finite = np.isfinite(y_score)
    y_true = y_true[finite]
    y_score = y_score[finite]
    if y_true.size == 0 or int(y_true.sum()) == 0 or int((1 - y_true).sum()) == 0:
        return None
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y_true, y_score))
    except Exception:
        order = np.argsort(-y_score, kind="mergesort")
        ordered = y_true[order]
        true_positives = np.cumsum(ordered)
        ranks = np.arange(1, len(ordered) + 1)
        precision = true_positives / ranks
        return float((precision * ordered).sum() / max(1, int(ordered.sum())))


def compute_whole_chromosome_pr_auc(
    predictions: Dict[GenomeChromosome, np.ndarray],
    truth: Dict[GenomeChromosome, np.ndarray],
    class_names: List[str],
) -> Dict:
    per_chromosome: Dict[str, Dict[str, Optional[float]]] = {}
    pooled_scores: List[List[np.ndarray]] = [[] for _ in class_names]
    pooled_truth: List[List[np.ndarray]] = [[] for _ in class_names]
    for key in sorted(predictions):
        if key not in truth:
            continue
        predicted = predictions[key]
        reference = truth[key]
        length = min(predicted.shape[1], reference.shape[1])
        channels = min(predicted.shape[0], reference.shape[0], len(class_names))
        display_name = f"{key[0]}|{key[1]}"
        per_chromosome[display_name] = {}
        for channel in range(channels):
            y_true = reference[channel, :length]
            y_score = predicted[channel, :length]
            score = average_precision(y_true, y_score)
            per_chromosome[display_name][class_names[channel]] = score
            pooled_scores[channel].append(y_score)
            pooled_truth[channel].append(y_true)

    pooled: Dict[str, Optional[float]] = {}
    defined = []
    for channel, name in enumerate(class_names):
        if pooled_scores[channel]:
            score = average_precision(
                np.concatenate(pooled_truth[channel]),
                np.concatenate(pooled_scores[channel]),
            )
        else:
            score = None
        pooled[name] = score
        if score is not None:
            defined.append(float(score))
    return {
        "per_chromosome": per_chromosome,
        "pooled": pooled,
        "mean": float(np.mean(defined)) if defined else None,
    }


def _run_single_stage(cfg: Dict) -> None:
    task = str(cfg.get("task", ""))
    if task not in {"finding_edge", "finding_region"}:
        raise RuntimeError("Single-stage finding inference requires task=finding_edge or task=finding_region")
    inference_cfg = cfg.get("inference", {})
    _require_batch_size_one(inference_cfg)
    device = inference_cfg.get("device", "cuda")
    use_rc = bool(inference_cfg.get("use_reverse_complement", True))
    predictions, truth = predict_tracks(cfg, task, device, use_rc)
    class_names = (
        ["TSS+", "TSS-", "PolyA+", "PolyA-"]
        if task == "finding_edge"
        else ["intragenic+", "intragenic-"]
    )
    metrics = {
        "task": task,
        "pr_auc": compute_whole_chromosome_pr_auc(predictions, truth, class_names),
    }
    metrics_path = Path(inference_cfg.get("metrics_json", f"{task}_metrics.json")).expanduser()
    atomic_save_json(metrics, metrics_path)
    logger.info("Wrote %s inference/evaluation metrics to %s", task, metrics_path)


def _run_full_pipeline(cfg: Dict) -> None:
    inference_cfg = cfg.get("inference", {})
    _require_batch_size_one(inference_cfg)
    device = inference_cfg.get("device", "cuda")
    use_rc = bool(inference_cfg.get("use_reverse_complement", True))

    edge_tracks, edge_truth = predict_tracks(
        cfg["edge"], "finding_edge", device, use_reverse_complement=use_rc
    )
    region_tracks, region_truth = predict_tracks(
        cfg["region"], "finding_region", device, use_reverse_complement=use_rc
    )
    metrics: Dict[str, object] = {
        "pr_auc": {
            "edge": compute_whole_chromosome_pr_auc(
                edge_tracks, edge_truth, ["TSS+", "TSS-", "PolyA+", "PolyA-"]
            ),
            "region": compute_whole_chromosome_pr_auc(
                region_tracks, region_truth, ["intragenic+", "intragenic-"]
            ),
        }
    }

    genomes = {key[0] for key in edge_tracks}
    if len(genomes) != 1:
        raise RuntimeError(
            "Complete gene-finding GFF inference requires exactly one genome because GFF seqids do not encode assembly IDs; "
            f"selected_genomes={sorted(genomes)}"
        )
    records = []
    post = cfg.get("postprocess", {})
    lp_frac = float(post.get("lp_frac", post.get("low_pass_fraction", 0.05)))
    pk_prom = float(post.get("pk_prom", post.get("peak_prominence", 0.1)))
    pk_dist = int(post.get("pk_dist", post.get("peak_distance", 50)))
    pk_height = post.get("pk_height", post.get("peak_height"))
    for key in sorted(edge_tracks):
        if key not in region_tracks:
            raise RuntimeError(f"Region tracks missing chromosome {key}")
        genome, chrom = key
        edge = np.nan_to_num(edge_tracks[key], nan=0.0)
        region = np.nan_to_num(region_tracks[key], nan=0.0)
        if edge.shape[0] != 4:
            raise RuntimeError(
                "Edge tracks must have 4 channels in model order "
                f"TSS+,TSS-,PolyA+,PolyA-, got {edge.shape}"
            )
        if region.shape[0] != 2:
            raise RuntimeError(
                "Region tracks must have 2 channels in order intragenic+,intragenic-, "
                f"got {region.shape}"
            )

        edge_for_peak = np.stack([edge[0], edge[2], edge[1], edge[3]], axis=0)
        tss_plus, polya_plus, tss_minus, polya_minus = peak_finding_indices(
            edge_for_peak,
            lp_frac=lp_frac,
            pk_prom=pk_prom,
            pk_dist=pk_dist,
            pk_height=pk_height,
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
        chrom_records = filter_intervals_by_intragenic_bool(
            pairs,
            intragenic_plus_mask=np.asarray(
                region[0] > float(post.get("prob_threshold", 0.5)), dtype=np.bool_
            ),
            intragenic_minus_mask=np.asarray(
                region[1] > float(post.get("prob_threshold", 0.5)), dtype=np.bool_
            ),
            zero_fraction_drop_threshold=float(
                post.get("zero_fraction_drop_threshold", 0.01)
            ),
            log=logger,
        )
        if not chrom_records:
            policy = inference_cfg.get("empty_gff_policy", "error")
            if policy == "best_interval":
                chrom_records = best_interval_records(
                    edge,
                    region,
                    chrom=chrom,
                    max_records=int(inference_cfg.get("empty_gff_max_records", 1)),
                    min_len=int(inference_cfg.get("empty_gff_min_interval_len", 64)),
                )
            else:
                raise RuntimeError(
                    f"No transcript intervals were produced for chromosome {chrom}. "
                    "Lower postprocess thresholds or use empty_gff_policy='best_interval'."
                )
        records.extend(chrom_records)

    output_gff = inference_cfg["output_gff"]
    write_finding_gff(records, output_gff)
    if inference_cfg.get("true_gff"):
        metrics["annotation"] = evaluate_annotation(
            output_gff,
            inference_cfg["true_gff"],
            output_json=None,
            k_values=inference_cfg.get("k_values", [0, 50, 100, 250, 500]),
            use_strand=bool(inference_cfg.get("use_strand", True)),
        )
    metrics_path = inference_cfg.get(
        "metrics_json", str(Path(output_gff).with_suffix(".metrics.json"))
    )
    atomic_save_json(metrics, metrics_path)
    logger.info("Wrote gene-finding inference/evaluation metrics to %s", metrics_path)


def main() -> None:
    parser = ArgumentParser(description="Run GENATATOR gene-finding inference and evaluation")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    if "edge" in cfg or "region" in cfg:
        if not ("edge" in cfg and "region" in cfg):
            raise RuntimeError("Full finding pipeline config must contain both edge and region stages")
        _run_full_pipeline(cfg)
    else:
        _run_single_stage(cfg)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
