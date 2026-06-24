from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Dict, Sequence

import evaluate
import numpy as np
from sklearn.metrics import average_precision_score

from .intervals import f1_from_counts, interval_counts


EDGE_CLASS_NAMES: tuple[str, ...] = ("TSS+", "TSS-", "PolyA+", "PolyA-")
REGION_CLASS_NAMES: tuple[str, ...] = ("intragenic+", "intragenic-")
SEGMENTATION_CLASS_INDEX = {"exon": 1, "CDS": 4}


def _pred_array(predictions: Any) -> np.ndarray:
    if isinstance(predictions, (tuple, list)):
        return np.asarray(predictions[0])
    return np.asarray(predictions)


def _labels_and_mask(label_ids: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(label_ids, (tuple, list)):
        if len(label_ids) != 2:
            raise RuntimeError(
                "Token-level metrics expect exactly two label tensors: labels and labels_mask; "
                f"received {len(label_ids)} tensors"
            )
        labels = np.asarray(label_ids[0])
        mask = np.asarray(label_ids[1]).astype(bool)
    else:
        labels = np.asarray(label_ids)
        if labels.ndim < 2:
            raise RuntimeError(f"Token-level labels must have at least 2 dimensions, got {labels.shape}")
        mask = np.ones(labels.shape[:2], dtype=bool)
    if labels.shape[:2] != mask.shape:
        raise RuntimeError(
            f"Label/mask shape mismatch: labels={labels.shape} mask={mask.shape}"
        )
    return labels, mask


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    positive = x >= 0
    out = np.empty_like(x, dtype=np.float64)
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _validate_token_metric_shapes(
    logits: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    expected_classes: int,
    task_name: str,
) -> None:
    if logits.ndim != 3 or labels.ndim != 3:
        raise RuntimeError(
            f"{task_name} metrics require [batch, length, classes] tensors; "
            f"logits={logits.shape} labels={labels.shape}"
        )
    if logits.shape != labels.shape:
        raise RuntimeError(
            f"{task_name} logits/labels shape mismatch: logits={logits.shape} labels={labels.shape}"
        )
    if logits.shape[-1] != expected_classes:
        raise RuntimeError(
            f"{task_name} expected {expected_classes} output classes, got {logits.shape[-1]}"
        )
    if mask.shape != logits.shape[:2]:
        raise RuntimeError(
            f"{task_name} mask shape mismatch: mask={mask.shape} logits={logits.shape}"
        )


def finding_pr_auc_metrics(
    eval_pred: Any,
    *,
    class_names: Sequence[str],
    task_name: str,
) -> Dict[str, float]:
    """Compute nucleotide/token-level PR-AUC independently for each finding class.

    Boundary targets are smooth signals in the released dataset. For PR-AUC they
    are treated as positive wherever the target signal is greater than zero, which
    preserves the complete labeled boundary region rather than truncating all
    fractional targets to zero.
    """
    logits = _pred_array(eval_pred.predictions)
    labels, mask = _labels_and_mask(eval_pred.label_ids)
    _validate_token_metric_shapes(logits, labels, mask, len(class_names), task_name)
    probabilities = sigmoid(logits)

    metrics: Dict[str, float] = {}
    for channel_index, class_name in enumerate(class_names):
        references = (labels[:, :, channel_index][mask] > 0.0).astype(np.int8)
        scores = probabilities[:, :, channel_index][mask]
        positives = int(references.sum())
        negatives = int(references.size - positives)
        if positives == 0 or negatives == 0:
            raise RuntimeError(
                f"Cannot compute PR-AUC for {task_name} class {class_name!r}: "
                f"positives={positives} negatives={negatives}. Use an evaluation "
                "set containing both positive and negative examples for every class."
            )
        metrics[f"pr_auc_{class_name}"] = float(
            average_precision_score(references, scores)
        )
    return metrics



def finding_edge_pr_auc_metrics(eval_pred: Any) -> Dict[str, float]:
    return finding_pr_auc_metrics(
        eval_pred,
        class_names=EDGE_CLASS_NAMES,
        task_name="finding_edge",
    )


def finding_region_pr_auc_metrics(eval_pred: Any) -> Dict[str, float]:
    return finding_pr_auc_metrics(
        eval_pred,
        class_names=REGION_CLASS_NAMES,
        task_name="finding_region",
    )

def segmentation_interval_metrics(eval_pred: Any) -> Dict[str, float]:
    """Compute exact interval-level F1 for exon and CDS only.

    Counts are accumulated across all validation transcripts before F1 is derived.
    UTR and intron channels are intentionally excluded from training-time metrics.
    """
    logits = _pred_array(eval_pred.predictions)
    labels, mask = _labels_and_mask(eval_pred.label_ids)
    _validate_token_metric_shapes(logits, labels, mask, 5, "segmentation")
    probabilities = sigmoid(logits)

    metrics: Dict[str, float] = {}
    for class_name, channel_index in SEGMENTATION_CLASS_INDEX.items():
        tp = fp = fn = 0
        for sample_index in range(labels.shape[0]):
            valid = mask[sample_index]
            if not np.any(valid):
                continue
            references = (labels[sample_index, valid, channel_index] >= 0.5).astype(np.int8)
            predictions = (probabilities[sample_index, valid, channel_index] >= 0.5).astype(np.int8)
            sample_tp, sample_fp, sample_fn = interval_counts(references, predictions)
            tp += sample_tp
            fp += sample_fp
            fn += sample_fn
        metrics[f"interval_f1_{class_name}"] = f1_from_counts(tp, fp, fn)
    return metrics


@lru_cache(maxsize=1)
def _accuracy_metric():
    return evaluate.load("accuracy")


def transcript_type_accuracy(eval_pred: Any) -> Dict[str, float]:
    logits = _pred_array(eval_pred.predictions).reshape(-1)
    label_ids = eval_pred.label_ids
    if isinstance(label_ids, (tuple, list)):
        if len(label_ids) != 1:
            raise RuntimeError(
                "Transcript-type evaluation expects exactly one label tensor, "
                f"received {len(label_ids)}"
            )
        label_ids = label_ids[0]
    references = np.asarray(label_ids).reshape(-1).astype(np.int64)
    predictions = (sigmoid(logits) >= 0.5).astype(np.int64)
    if predictions.shape != references.shape:
        raise RuntimeError(
            f"Transcript-type prediction/reference shape mismatch: "
            f"predictions={predictions.shape} references={references.shape}"
        )
    result = _accuracy_metric().compute(
        predictions=predictions,
        references=references,
    )
    return {"accuracy": float(result["accuracy"])}


def metric_for_task(task: str) -> Callable[[Any], Dict[str, float]]:
    if task == "finding_edge":
        return finding_edge_pr_auc_metrics
    if task == "finding_region":
        return finding_region_pr_auc_metrics
    if task == "segmentation":
        return segmentation_interval_metrics
    if task == "transcript_type":
        return transcript_type_accuracy
    raise ValueError(f"Unsupported task: {task}")

def metric_names_for_task(task: str) -> tuple[str, ...]:
    if task == "finding_edge":
        return tuple(f"pr_auc_{name}" for name in EDGE_CLASS_NAMES)
    if task == "finding_region":
        return tuple(f"pr_auc_{name}" for name in REGION_CLASS_NAMES)
    if task == "segmentation":
        return ("interval_f1_exon", "interval_f1_CDS")
    if task == "transcript_type":
        return ("accuracy",)
    raise ValueError(f"Unsupported task: {task}")

