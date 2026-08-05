from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

try:
    from genatator_core.metrics_training import (
        EDGE_CLASS_NAMES,
        finding_edge_pr_auc_metrics,
    )
    from genatator_core.train_common import (
        GenatatorTrainer,
        filter_training_logs,
    )
except ImportError:
    EDGE_CLASS_NAMES = ()
    finding_edge_pr_auc_metrics = None
    GenatatorTrainer = None
    filter_training_logs = None


@unittest.skipIf(
    finding_edge_pr_auc_metrics is None or GenatatorTrainer is None,
    "training metric dependencies are not installed",
)
class TrainingMetricLoggingTests(unittest.TestCase):
    def test_finding_metrics_contain_only_per_class_pr_and_roc_auc(self) -> None:
        labels = np.asarray(
            [
                [
                    [0.0, 1.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0, 0.0],
                ]
            ],
            dtype=np.float32,
        )
        logits = np.where(labels > 0, 4.0, -4.0).astype(np.float32)
        mask = np.ones(labels.shape[:2], dtype=bool)
        metrics = finding_edge_pr_auc_metrics(
            SimpleNamespace(predictions=logits, label_ids=(labels, mask))
        )
        expected = {
            metric
            for name in EDGE_CLASS_NAMES
            for metric in (f"pr_auc_{name}", f"roc_auc_{name}")
        }
        self.assertEqual(set(metrics), expected)
        self.assertTrue(all(value == 1.0 for value in metrics.values()))

    def test_streaming_finding_metrics_have_exact_validation_keys(self) -> None:
        trainer = object.__new__(GenatatorTrainer)
        trainer.genatator_task = "finding_edge"
        references = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        scores = np.asarray([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
        state = {
            "loss_sum": 0.5,
            "loss_count": 2,
            "class_names": EDGE_CLASS_NAMES,
            "refs": [[references] for _ in EDGE_CLASS_NAMES],
            "scores": [[scores] for _ in EDGE_CLASS_NAMES],
        }
        metrics = trainer._finalize_streaming_state(state, "eval")
        expected = {"eval_loss"} | {
            metric
            for name in EDGE_CLASS_NAMES
            for metric in (f"eval_pr_auc_{name}", f"eval_roc_auc_{name}")
        }
        self.assertEqual(set(metrics), expected)

    def test_finding_log_filter_removes_bookkeeping_and_final_summaries(self) -> None:
        validation = filter_training_logs(
            {
                "eval_loss": 0.25,
                "eval_pr_auc_TSS+": 0.8,
                "eval_roc_auc_TSS+": 0.9,
                "eval_pr_auc_TSS+_defined": 1.0,
                "eval_samples": 100,
                "epoch": 2.0,
            },
            "finding_edge",
        )
        self.assertEqual(
            validation,
            {
                "eval_loss": 0.25,
                "eval_pr_auc_TSS+": 0.8,
                "eval_roc_auc_TSS+": 0.9,
            },
        )
        summary = filter_training_logs(
            {
                "train_loss": 0.4,
                "train_runtime": 12.0,
                "train_steps_per_second": 3.0,
                "total_flos": 123.0,
                "epoch": 2.0,
            },
            "finding_region",
        )
        self.assertEqual(summary, {"loss": 0.4, "epoch": 2.0})

    def test_other_tasks_never_emit_train_loss_or_flos(self) -> None:
        filtered = filter_training_logs(
            {
                "train_loss": 0.4,
                "total_flos": 123.0,
                "train_runtime": 12.0,
                "epoch": 2.0,
            },
            "segmentation",
        )
        self.assertNotIn("train_loss", filtered)
        self.assertNotIn("total_flos", filtered)
        self.assertEqual(filtered["loss"], 0.4)
        self.assertEqual(filtered["train_runtime"], 12.0)


if __name__ == "__main__":
    unittest.main()
