from __future__ import annotations

import unittest
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

try:
    import torch
    from transformers import Trainer, TrainingArguments, default_data_collator

    from genatator_core.metrics_training import (
        EDGE_CLASS_NAMES,
        REGION_CLASS_NAMES,
        finding_edge_pr_auc_metrics,
        finding_region_pr_auc_metrics,
        metric_names_for_task,
        sigmoid,
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
    def test_finding_metrics_retain_per_class_and_add_mean_pr_and_roc_auc(self) -> None:
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
        expected = {"mean_pr_auc", "mean_roc_auc"} | {
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
        expected = {"eval_loss", "eval_mean_pr_auc", "eval_mean_roc_auc"} | {
            metric
            for name in EDGE_CLASS_NAMES
            for metric in (f"eval_pr_auc_{name}", f"eval_roc_auc_{name}")
        }
        self.assertEqual(set(metrics), expected)

    def test_channel_means_match_between_standard_and_streaming_validation(self) -> None:
        # Different prevalences and rankings make micro-averaging, using one
        # channel, and mixing PR-AUC with ROC-AUC observably wrong.
        labels = np.asarray([[
            [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1],
            [1, 0, 1, 0], [1, 1, 0, 1],
        ]], dtype=np.float32)
        logits = np.asarray([[
            [-3, -2, -1, 3], [2, 1, 2, -2], [-1, 0, -3, -1],
            [1, 3, 1, 0], [-9, -9, 9, -9],
        ]], dtype=np.float32)
        mask = np.asarray([[True, True, True, True, False]])
        for task, names, metric in (
            ("finding_edge", EDGE_CLASS_NAMES, finding_edge_pr_auc_metrics),
            ("finding_region", REGION_CLASS_NAMES, finding_region_pr_auc_metrics),
        ):
            with self.subTest(task=task):
                count = len(names)
                result = metric(SimpleNamespace(
                    predictions=logits[..., :count],
                    label_ids=(labels[..., :count], mask),
                ))
                self.assertEqual(set(result), set(metric_names_for_task(task)))
                self.assertNotEqual(result["mean_pr_auc"], result["mean_roc_auc"])
                for kind in ("pr_auc", "roc_auc"):
                    self.assertAlmostEqual(
                        result[f"mean_{kind}"],
                        sum(result[f"{kind}_{name}"] for name in names) / count,
                    )
                trainer = object.__new__(GenatatorTrainer)
                trainer.genatator_task = task
                state = trainer._new_streaming_state()
                for c in range(count):
                    refs = labels[..., c][mask]
                    scores = sigmoid(logits[..., c])[mask]
                    # Unequal batch sizes must be concatenated before AUC.
                    state["refs"][c] = [refs[:1], refs[1:]]
                    state["scores"][c] = [scores[:1], scores[1:]]
                streaming = trainer._finalize_streaming_state(state, "validation")
                self.assertEqual(
                    {k: streaming[f"validation_{k}"] for k in result}, result
                )

    def test_undefined_channels_keep_zero_in_fixed_denominator_mean(self) -> None:
        labels = np.asarray([[[0, 0], [1, 0], [0, 0], [1, 0]]])
        logits = np.asarray([[[-2, 0], [2, 0], [-1, 0], [1, 0]]])
        for mask, expected in (
            (np.ones((1, 4), dtype=bool), 0.5),
            (np.zeros((1, 4), dtype=bool), 0.0),
        ):
            metrics = finding_region_pr_auc_metrics(SimpleNamespace(
                predictions=logits, label_ids=(labels, mask)
            ))
            self.assertEqual(metrics["mean_pr_auc"], expected)
            self.assertEqual(metrics["mean_roc_auc"], expected)

    def test_trainer_evaluation_paths_return_same_means_without_extra_inference(self) -> None:
        class FixedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.calls = 0

            def forward(self, input_ids, labels=None, labels_mask=None):
                self.calls += 1
                logits = input_ids.float() + self.anchor * 0
                return {"loss": logits.sum() * 0, "logits": logits}

        for task, count, metric in (
            ("finding_edge", 4, finding_edge_pr_auc_metrics),
            ("finding_region", 2, finding_region_pr_auc_metrics),
        ):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                labels = torch.tensor([[0., 1., 0., 1.], [1., 0., 1., 0.]])[:, :count]
                dataset = [
                    {"input_ids": (labels * 2 - 1) * scale,
                     "labels": labels,
                     "labels_mask": torch.ones(2, dtype=torch.bool)}
                    for scale in (1., -2.)
                ]
                model = FixedModel()
                trainer = GenatatorTrainer(
                    model=model,
                    args=TrainingArguments(
                        output_dir=directory, use_cpu=True, report_to=[],
                        per_device_eval_batch_size=2, dataloader_pin_memory=False,
                        label_names=["labels", "labels_mask"], disable_tqdm=True,
                    ),
                    eval_dataset=dataset, data_collator=default_data_collator,
                    compute_metrics=metric, genatator_task=task,
                )
                ordinary = Trainer.evaluate(trainer)
                self.assertEqual(model.calls, 1)
                with patch.dict(os.environ, {"RANK": "0"}):
                    streaming = trainer.evaluate()
                self.assertEqual(model.calls, 3)  # One streaming call per example.
                for kind in ("pr_auc", "roc_auc"):
                    key = f"eval_mean_{kind}"
                    self.assertAlmostEqual(ordinary[key], streaming[key])
                # Nonzero finding ranks consume the same completed metric
                # dictionary and notify callbacks without evaluating again.
                metrics_path = Path(directory) / "rank0_eval_metrics/eval_step_0.json"
                os.utime(metrics_path, None)
                with patch.dict(os.environ, {"RANK": "1"}):
                    follower = trainer.evaluate()
                self.assertEqual(follower, streaming)
                self.assertEqual(model.calls, 3)

    def test_finding_log_filter_removes_bookkeeping_and_final_summaries(self) -> None:
        validation = filter_training_logs(
            {
                "eval_loss": 0.25,
                "eval_pr_auc_TSS+": 0.8,
                "eval_roc_auc_TSS+": 0.9,
                "eval_mean_pr_auc": 0.7,
                "eval_mean_roc_auc": 0.85,
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
                "eval_mean_pr_auc": 0.7,
                "eval_mean_roc_auc": 0.85,
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
