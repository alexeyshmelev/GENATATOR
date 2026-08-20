from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch

    from genatator_core.infer_common import (
        _predict_once,
        model_logits_for_inference,
        predict_dataset_logits,
    )
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

if __name__ == "__main__":
    unittest.main()
