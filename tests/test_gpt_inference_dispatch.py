from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch

    from genatator_core.infer_common import (
        _predict_once,
        merge_rank_strided_results,
        model_logits_for_inference,
        predict_dataset_logits,
    )
    from segmentation.infer import _distributed_runtime
except ImportError:
    torch = None
    model_logits_for_inference = None


@unittest.skipIf(torch is None, "torch dependencies are not installed")
class GPTInferenceDispatchTests(unittest.TestCase):
    def test_gpt_inference_uses_generate_even_when_reference_labels_exist(self) -> None:
        class FakeModel:
            def __init__(self):
                self.generated = False

            def __call__(self, **kwargs):
                raise AssertionError("GPT inference must not use teacher-forced forward")

            def generate(self, **kwargs):
                self.generated = True
                self.received_labels = "letter_level_labels" in kwargs
                return torch.ones((1, 3, 5))

        model = FakeModel()
        logits = model_logits_for_inference(
            model,
            {"letter_level_labels": torch.zeros((1, 3, 5))},
            task="segmentation",
            model_cfg={"family": "gpt"},
        )
        self.assertTrue(model.generated)
        self.assertTrue(model.received_labels)
        self.assertEqual(tuple(logits.shape), (1, 3, 5))

    def test_non_gpt_inference_uses_forward(self) -> None:
        class FakeModel:
            def __call__(self, **kwargs):
                return {"logits": torch.zeros((1, 2, 5))}

            def generate(self, **kwargs):
                raise AssertionError("Non-GPT inference must not call generate")

        logits = model_logits_for_inference(
            FakeModel(),
            {},
            task="segmentation",
            model_cfg={"family": "unet"},
        )
        self.assertEqual(tuple(logits.shape), (1, 2, 5))

    def test_gpt_inference_forces_forward_only_even_when_rc_is_requested(self) -> None:
        cfg = {
            "model": {"family": "gpt"},
            "inference": {"use_reverse_complement": True},
        }
        expected = [{"row": "forward"}]
        with patch(
            "genatator_core.infer_common._predict_once",
            return_value=expected,
        ) as predict_once:
            rows = predict_dataset_logits(cfg, task="segmentation", device="cpu")

        self.assertIs(rows, expected)
        predict_once.assert_called_once()
        args, kwargs = predict_once.call_args
        self.assertEqual(args[1:3], ("segmentation", "cpu"))
        self.assertIs(kwargs["reverse_complement"], False)
        self.assertEqual(kwargs["rank"], 0)
        self.assertEqual(kwargs["world_size"], 1)

    def test_private_gpt_inference_path_rejects_reverse_complement(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "forbidden for GPT segmentation"):
            _predict_once(
                {"model": {"family": "gpt"}},
                task="segmentation",
                device="cpu",
                reverse_complement=True,
            )

    def test_non_gpt_inference_still_honors_reverse_complement(self) -> None:
        metadata = object()
        forward = [
            {
                "metadata": metadata,
                "local_start": 0,
                "logits": np.asarray([[1.0]], dtype=np.float32),
            }
        ]
        reverse = [
            {
                "metadata": metadata,
                "local_start": 0,
                "logits": np.asarray([[3.0]], dtype=np.float32),
            }
        ]
        cfg = {
            "model": {"family": "unet"},
            "inference": {"use_reverse_complement": True},
        }
        with patch(
            "genatator_core.infer_common._predict_once",
            side_effect=(forward, reverse),
        ) as predict_once:
            rows = predict_dataset_logits(cfg, task="segmentation", device="cpu")

        self.assertEqual(predict_once.call_count, 2)
        np.testing.assert_allclose(rows[0]["logits"], [[2.0]])

    def test_distributed_gpt_dispatch_forwards_rank_topology(self) -> None:
        cfg = {"model": {"family": "gpt"}, "inference": {"num_transcripts": 5}}
        with patch(
            "genatator_core.infer_common._predict_once",
            return_value=[],
        ) as predict_once:
            predict_dataset_logits(
                cfg,
                task="segmentation",
                device="cuda:2",
                rank=2,
                world_size=4,
            )

        self.assertEqual(predict_once.call_count, 1)
        self.assertEqual(predict_once.call_args.kwargs["rank"], 2)
        self.assertEqual(predict_once.call_args.kwargs["world_size"], 4)

    def test_distributed_non_gpt_inference_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "only for GPT segmentation"):
            predict_dataset_logits(
                {"model": {"family": "unet"}},
                task="segmentation",
                device="cuda:0",
                rank=0,
                world_size=2,
            )

    def test_rank_strided_results_are_restored_to_dataset_order(self) -> None:
        def row(ordinal: int, selected: int = 5):
            return {
                "ordinal": ordinal,
                "_inference_transcript_ordinal": ordinal,
                "_inference_selected_transcripts": selected,
            }

        rank_results = [
            [row(0), row(3)],
            [row(1), row(4)],
            [row(2)],
        ]
        merged = merge_rank_strided_results(rank_results)
        self.assertEqual([row["ordinal"] for row in merged], list(range(5)))
        self.assertNotIn("_inference_transcript_ordinal", merged[0])

    def test_rank_strided_results_allow_ranks_without_transcripts(self) -> None:
        rank_results = [
            [
                {
                    "ordinal": 0,
                    "_inference_transcript_ordinal": 0,
                    "_inference_selected_transcripts": 2,
                }
            ],
            [
                {
                    "ordinal": 1,
                    "_inference_transcript_ordinal": 1,
                    "_inference_selected_transcripts": 2,
                }
            ],
            [],
            [],
        ]
        merged = merge_rank_strided_results(rank_results)
        self.assertEqual([row["ordinal"] for row in merged], [0, 1])

    def test_rank_strided_merge_rejects_incomplete_partition(self) -> None:
        def row(ordinal: int):
            return {
                "_inference_transcript_ordinal": ordinal,
                "_inference_selected_transcripts": 5,
            }

        with self.assertRaisesRegex(RuntimeError, "incomplete or out of order"):
            merge_rank_strided_results(
                [[row(0), row(3)], [row(1)], [row(2)]]
            )

    def test_torchrun_rank_is_bound_to_its_local_gpu(self) -> None:
        cfg = {
            "model": {"family": "gpt"},
            "inference": {"device": "cuda"},
        }
        environment = {
            "RANK": "2",
            "WORLD_SIZE": "4",
            "LOCAL_RANK": "2",
            "LOCAL_WORLD_SIZE": "4",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("segmentation.infer.torch.cuda.is_available", return_value=True),
            patch("segmentation.infer.torch.cuda.device_count", return_value=4),
            patch("segmentation.infer.torch.cuda.set_device") as set_device,
            patch("segmentation.infer.dist.is_initialized", return_value=False),
            patch("segmentation.infer.dist.init_process_group") as init_group,
        ):
            rank, world_size, device = _distributed_runtime(cfg)

        self.assertEqual((rank, world_size, device), (2, 4, "cuda:2"))
        set_device.assert_called_once_with(2)
        self.assertEqual(init_group.call_args.kwargs["backend"], "nccl")

    def test_plain_python_gpt_inference_rejects_unused_visible_gpus(self) -> None:
        cfg = {
            "model": {"family": "gpt"},
            "inference": {"device": "cuda"},
        }
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("segmentation.infer.torch.cuda.device_count", return_value=8),
            self.assertRaisesRegex(RuntimeError, "torchrun"),
        ):
            _distributed_runtime(cfg)

if __name__ == "__main__":
    unittest.main()
