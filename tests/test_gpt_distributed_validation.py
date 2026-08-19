from __future__ import annotations

import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


try:
    import torch

    from genatator_core.gpt_head import EXON_LABEL_INDEX, INTRON_LABEL_INDEX
    from genatator_core.train_common import (
        AtomicValidationIndexQueue,
        GenatatorTrainer,
        _GPT_GROUND_TRUTH_INPUTS,
        _maybe_relaunch_gpt_training,
        gpt_torchrun_relaunch_command,
    )
except ImportError:
    torch = None
    GenatatorTrainer = None


@unittest.skipIf(torch is None, "torch/transformers are not installed")
class GPTDistributedValidationTests(unittest.TestCase):
    def test_normal_gpt_cli_relaunches_one_process_per_visible_gpu(self) -> None:
        command = gpt_torchrun_relaunch_command(
            model_cfg={"family": "gpt"},
            config_path="relative/config.json",
            cuda_device_count=4,
            distributed_world_size=1,
            executable="/env/bin/python",
        )

        self.assertIsNotNone(command)
        self.assertEqual(command[:5], [
            "/env/bin/python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
        ])
        self.assertEqual(command[-2], "--config")
        self.assertTrue(Path(command[-1]).is_absolute())

    def test_relaunch_is_gpt_only_and_not_nested(self) -> None:
        shared = dict(
            config_path="config.json",
            cuda_device_count=8,
            executable="python",
        )
        self.assertIsNone(
            gpt_torchrun_relaunch_command(
                model_cfg={"family": "unet"},
                distributed_world_size=1,
                **shared,
            )
        )
        self.assertIsNone(
            gpt_torchrun_relaunch_command(
                model_cfg={"family": "gpt"},
                distributed_world_size=8,
                **shared,
            )
        )
        self.assertIsNone(
            gpt_torchrun_relaunch_command(
                model_cfg={"family": "gpt"},
                config_path="config.json",
                cuda_device_count=1,
                distributed_world_size=1,
            )
        )

    def test_relaunch_makes_repository_importable_to_workers(self) -> None:
        with (
            patch.object(torch.cuda, "device_count", return_value=2),
            patch("os.execvpe", side_effect=RuntimeError("captured")) as execvpe,
            patch.dict(os.environ, {"WORLD_SIZE": "1", "PYTHONPATH": ""}, clear=False),
            self.assertRaisesRegex(RuntimeError, "captured"),
        ):
            _maybe_relaunch_gpt_training(
                {"family": "gpt"},
                "config.json",
            )

        _, _, environment = execvpe.call_args.args
        repository_root = str(Path(__file__).resolve().parent.parent)
        self.assertEqual(environment["PYTHONPATH"], repository_root)

    def test_atomic_queue_assigns_next_transcript_to_first_available_rank(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            queue = AtomicValidationIndexQueue.create(
                Path(temporary) / "queue",
                total=5,
                timeout_seconds=5.0,
            )

            self.assertEqual(queue.claim(rank=0), 0)
            self.assertEqual(queue.claim(rank=1), 1)
            # Rank 1 finished first, so it claims the next global transcript;
            # indices are not pre-sharded by rank.
            self.assertEqual(queue.claim(rank=1), 2)
            self.assertEqual(queue.claim(rank=0), 3)
            self.assertEqual(queue.claim(rank=1), 4)
            self.assertIsNone(queue.claim(rank=0))

    def test_atomic_queue_claims_every_index_once_under_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            queue = AtomicValidationIndexQueue.create(
                Path(temporary) / "queue",
                total=31,
                timeout_seconds=5.0,
            )
            claimed = []
            claimed_lock = threading.Lock()

            def consume(rank: int) -> None:
                while True:
                    index = queue.claim(rank)
                    if index is None:
                        return
                    with claimed_lock:
                        claimed.append(index)

            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(consume, range(4)))

            self.assertEqual(sorted(claimed), list(range(31)))
            self.assertEqual(len(claimed), len(set(claimed)))

    def test_validation_calls_generate_without_any_ground_truth(self) -> None:
        class GenerateOnlyModel:
            def __init__(self):
                self.kwargs = None

            def __call__(self, **kwargs):
                raise AssertionError("teacher-forced forward must never run")

            def generate(self, **kwargs):
                self.kwargs = kwargs
                logits = torch.zeros((1, 3, 5), dtype=torch.float32)
                logits[0, 0, INTRON_LABEL_INDEX] = 3.0
                logits[0, 1, EXON_LABEL_INDEX] = 3.0
                logits[0, 2, INTRON_LABEL_INDEX] = 3.0
                return logits

        labels = torch.zeros((1, 3, 5), dtype=torch.float32)
        labels[0, 0, INTRON_LABEL_INDEX] = 1.0
        labels[0, 1, EXON_LABEL_INDEX] = 1.0
        labels[0, 2, INTRON_LABEL_INDEX] = 1.0
        inputs = {
            "input_ids": torch.tensor([[5, 6, 7]]),
            "labels": torch.ones((1, 3, 5)),
            "labels_mask": torch.ones((1, 3), dtype=torch.bool),
            "letter_level_labels": labels,
            "letter_level_labels_mask": torch.ones((1, 3), dtype=torch.bool),
            "pos_weight": torch.ones((1, 1, 5)),
        }
        trainer = object.__new__(GenatatorTrainer)
        trainer.genatator_task = "segmentation"
        model = GenerateOnlyModel()

        outputs = trainer._gpt_autoregressive_validation_outputs(model, inputs)

        self.assertTrue(torch.isfinite(outputs["loss"]))
        self.assertEqual(tuple(outputs["logits"].shape), (1, 3, 5))
        self.assertIsNotNone(model.kwargs)
        self.assertFalse(_GPT_GROUND_TRUTH_INPUTS.intersection(model.kwargs))
        self.assertEqual(set(model.kwargs), {"input_ids"})

    def test_validation_rejects_more_than_one_transcript_per_gpu(self) -> None:
        trainer = object.__new__(GenatatorTrainer)
        trainer.genatator_task = "segmentation"
        inputs = {
            "input_ids": torch.ones((2, 3), dtype=torch.long),
            "letter_level_labels": torch.zeros((2, 3, 5)),
            "letter_level_labels_mask": torch.ones((2, 3), dtype=torch.bool),
        }

        with self.assertRaisesRegex(RuntimeError, "exactly one transcript per GPU"):
            trainer._gpt_autoregressive_validation_outputs(Mock(), inputs)

    def test_rank_payload_merge_requires_every_transcript_exactly_once(self) -> None:
        trainer = object.__new__(GenatatorTrainer)
        trainer.genatator_task = "segmentation"
        empty_counts = {
            "exon": [0, 0, 0],
            "CDS": [0, 0, 0],
        }
        payloads = [
            {
                "claimed_indices": [0, 2],
                "loss_count": 2,
                "loss_sum": 1.0,
                "counts": {
                    "exon": [1, 0, 1],
                    "CDS": [0, 0, 0],
                },
            },
            {
                "claimed_indices": [1],
                "loss_count": 1,
                "loss_sum": 2.0,
                "counts": empty_counts,
            },
        ]

        metrics = trainer._merge_gpt_validation_payloads(
            payloads,
            dataset_size=3,
            metric_key_prefix="eval",
        )
        self.assertAlmostEqual(metrics["eval_loss"], 1.0)
        self.assertEqual(metrics["eval_interval_tp_exon"], 1.0)
        self.assertEqual(metrics["eval_interval_fn_exon"], 1.0)

        payloads[1]["claimed_indices"] = [2]
        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            trainer._merge_gpt_validation_payloads(
                payloads,
                dataset_size=3,
                metric_key_prefix="eval",
            )

    def test_gpt_evaluate_dispatch_preserves_non_gpt_rank0_path(self) -> None:
        metrics = {"eval_loss": 0.25}
        for is_gpt in (False, True):
            with self.subTest(is_gpt=is_gpt), tempfile.TemporaryDirectory() as temporary:
                trainer = object.__new__(GenatatorTrainer)
                trainer.args = SimpleNamespace(output_dir=temporary)
                trainer.state = SimpleNamespace(global_step=10)
                trainer.control = object()
                trainer.callback_handler = Mock()
                trainer.callback_handler.on_evaluate.return_value = trainer.control
                trainer.log = Mock()
                trainer.gpt_validation = is_gpt
                trainer._distributed_gpt_evaluate = Mock(return_value=metrics)
                trainer._streaming_evaluate_rank0 = Mock(return_value=metrics)

                with patch.dict(
                    os.environ,
                    {"RANK": "0", "WORLD_SIZE": "1"},
                    clear=False,
                ):
                    result = trainer.evaluate()

                self.assertEqual(result, metrics)
                if is_gpt:
                    trainer._distributed_gpt_evaluate.assert_called_once()
                    trainer._streaming_evaluate_rank0.assert_not_called()
                else:
                    trainer._streaming_evaluate_rank0.assert_called_once()
                    trainer._distributed_gpt_evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
