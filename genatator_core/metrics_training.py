from __future__ import annotations

from typing import Any, Dict

import evaluate
import numpy as np

from .intervals import exact_interval_f1


def _pred_array(predictions: Any) -> np.ndarray:
    if isinstance(predictions, (tuple, list)):
        return np.asarray(predictions[0])
    return np.asarray(predictions)


def _labels_and_mask(label_ids: Any):
    if isinstance(label_ids, (tuple, list)):
        return np.asarray(label_ids[0]), np.asarray(label_ids[1]).astype(bool)
    labels = np.asarray(label_ids)
    mask = np.ones(labels.shape[:2], dtype=bool)
    return labels, mask


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def finding_auc_metrics(eval_pred) -> Dict[str, float]:
    logits = _pred_array(eval_pred.predictions)
    labels, mask = _labels_and_mask(eval_pred.label_ids)
    probs = sigmoid(logits)
    roc_auc = evaluate.load("roc_auc")
    metrics: Dict[str, float] = {}
    aucs = []
    for c in range(labels.shape[-1]):
        y = labels[:, :, c][mask].astype(int)
        p = probs[:, :, c][mask]
        auc = roc_auc.compute(references=y, prediction_scores=p)["roc_auc"]
        metrics[f"auc_channel_{c}"] = float(auc)
        aucs.append(float(auc))
    metrics["auc_mean"] = float(np.mean(aucs))
    return metrics


def segmentation_interval_metrics(eval_pred) -> Dict[str, float]:
    logits = _pred_array(eval_pred.predictions)
    labels, mask = _labels_and_mask(eval_pred.label_ids)
    probs = sigmoid(logits)
    out: Dict[str, float] = {}
    names = {1: "exon", 4: "cds"}
    f1s = []
    for idx, name in names.items():
        tp = fp = fn = 0
        per_sample = []
        for i in range(labels.shape[0]):
            valid = mask[i]
            if valid.sum() == 0:
                continue
            y = (labels[i, valid, idx] >= 0.5).astype(int)
            p = (probs[i, valid, idx] >= 0.5).astype(int)
            m = exact_interval_f1(y, p)
            per_sample.append(m["f1"])
        val = float(np.mean(per_sample)) if per_sample else 0.0
        out[f"interval_f1_{name}"] = val
        f1s.append(val)
    out["interval_f1_mean"] = float(np.mean(f1s))
    return out


def transcript_type_metrics(eval_pred) -> Dict[str, float]:
    logits = _pred_array(eval_pred.predictions).reshape(-1)
    labels = np.asarray(eval_pred.label_ids)
    if isinstance(eval_pred.label_ids, (tuple, list)):
        labels = np.asarray(eval_pred.label_ids[-1])
    labels = labels.reshape(-1).astype(int)
    pred = (sigmoid(logits) >= 0.5).astype(int)
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    return {
        "accuracy": float(accuracy.compute(predictions=pred, references=labels)["accuracy"]),
        "f1": float(f1.compute(predictions=pred, references=labels, average="binary")["f1"]),
        "precision": float(precision.compute(predictions=pred, references=labels, average="binary")["precision"]),
        "recall": float(recall.compute(predictions=pred, references=labels, average="binary")["recall"]),
    }


def metric_for_task(task: str):
    if task in {"finding_edge", "finding_region"}:
        return finding_auc_metrics
    if task == "segmentation":
        return segmentation_interval_metrics
    if task == "transcript_type":
        return transcript_type_metrics
    raise ValueError(task)
