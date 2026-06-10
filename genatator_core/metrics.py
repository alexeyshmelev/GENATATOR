from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score


def segments_from_binary(x: np.ndarray) -> list[tuple[int, int]]:
    idx = np.where(x.astype(bool))[0]
    if len(idx) == 0:
        return []
    cuts = np.where(np.diff(idx) > 1)[0] + 1
    blocks = np.split(idx, cuts)
    return [(int(b[0]), int(b[-1]) + 1) for b in blocks]


def interval_counts(y: np.ndarray, p: np.ndarray, threshold: float) -> tuple[int, int, int]:
    y_set = set(segments_from_binary(y >= threshold))
    p_set = set(segments_from_binary(p >= threshold))
    tp = len(y_set & p_set)
    fp = len(p_set - y_set)
    fn = len(y_set - p_set)
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def segmentation_interval_metrics(records: list[dict], label_names: list[str], thresholds=(0.5,)) -> Dict[str, float]:
    metrics = {}
    for thr in thresholds:
        for class_i, name in enumerate(label_names):
            tp = fp = fn = 0
            for r in records:
                y = r["labels"][:, class_i]
                p = r["probs"][:, class_i]
                a, b, c = interval_counts(y, p, thr)
                tp += a; fp += b; fn += c
            precision, recall, f1 = prf(tp, fp, fn)
            metrics[f"interval_precision/{name}@{thr}"] = precision
            metrics[f"interval_recall/{name}@{thr}"] = recall
            metrics[f"interval_f1/{name}@{thr}"] = f1
    metrics["opt/interval_f1_exon"] = max(metrics.get(f"interval_f1/exon@{t}", 0.0) for t in thresholds)
    return metrics


def token_metrics(records: list[dict], label_names: list[str]) -> Dict[str, float]:
    y = np.concatenate([r["labels"] for r in records], axis=0)
    p = np.concatenate([r["probs"] for r in records], axis=0)
    metrics = {}
    aucs = []
    for i, name in enumerate(label_names):
        yt = y[:, i]
        pt = p[:, i]
        mask = yt != -100
        yt = yt[mask]
        pt = pt[mask]
        if len(np.unique(yt > 0.5)) == 2:
            metrics[f"pr_auc/{name}"] = float(average_precision_score(yt, pt))
            try:
                metrics[f"roc_auc/{name}"] = float(roc_auc_score(yt > 0.5, pt))
            except ValueError:
                metrics[f"roc_auc/{name}"] = 0.0
        else:
            metrics[f"pr_auc/{name}"] = 0.0
            metrics[f"roc_auc/{name}"] = 0.0
        aucs.append(metrics[f"pr_auc/{name}"])
    metrics["opt/pr_auc_mean"] = float(np.mean(aucs)) if aucs else 0.0
    return metrics


def gene_level_segmentation_metrics(records: list[dict], exon_class: str = "exon", cds_class: str = "CDS", threshold: float = 0.5) -> Dict[str, float]:
    by_gene = defaultdict(list)
    for r in records:
        by_gene[r["metadata"]["gene_id"]].append(r)
    correct_exon = 0
    correct_cds = 0
    for _, rs in by_gene.items():
        true_exons = []
        true_cds = []
        for r in rs:
            labels = r["labels"]
            names = r["label_names"]
            true_exons.append(set(segments_from_binary(labels[:, names.index(exon_class)] >= threshold)))
            if cds_class in names:
                true_cds.append(set(segments_from_binary(labels[:, names.index(cds_class)] >= threshold)))
        gene_ok_exon = False
        gene_ok_cds = False
        for r in rs:
            probs = r["probs"]
            names = r["label_names"]
            pred_exon = set(segments_from_binary(probs[:, names.index(exon_class)] >= threshold))
            if pred_exon in true_exons:
                gene_ok_exon = True
            if cds_class in names:
                pred_cds = set(segments_from_binary(probs[:, names.index(cds_class)] >= threshold))
                if pred_cds in true_cds:
                    gene_ok_cds = True
        correct_exon += int(gene_ok_exon)
        correct_cds += int(gene_ok_cds)
    total = len(by_gene)
    return {
        "gene_level/exon_correct": float(correct_exon),
        "gene_level/cds_correct": float(correct_cds),
        "gene_level/exon_recall": correct_exon / total if total else 0.0,
        "gene_level/cds_recall": correct_cds / total if total else 0.0,
    }


def kx_metrics(pred: list[dict], ref: list[dict], k: int) -> Dict[str, float]:
    matched_pred = set()
    matched_genes = set()
    for pi, p in enumerate(pred):
        for r in ref:
            if p.get("strand") != r.get("strand"):
                continue
            if abs(int(p["start"]) - int(r["start"])) <= k and abs(int(p["end"]) - int(r["end"])) <= k:
                matched_pred.add(pi)
                matched_genes.add(r.get("gene_id", f'{r["start"]}:{r["end"]}:{r.get("strand", ".")}'))
                break
    ref_genes = {r.get("gene_id", f'{r["start"]}:{r["end"]}:{r.get("strand", ".")}') for r in ref}
    tp_int = len(matched_pred)
    fp_int = len(pred) - tp_int
    tp_gene = len(matched_genes)
    fn_gene = len(ref_genes) - tp_gene
    precision = tp_int / (tp_int + fp_int) if (tp_int + fp_int) else 0.0
    recall = tp_gene / (tp_gene + fn_gene) if (tp_gene + fn_gene) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {f"kx{k}/precision": precision, f"kx{k}/recall": recall, f"kx{k}/f1": f1, f"kx{k}/tp_gene": float(tp_gene)}
