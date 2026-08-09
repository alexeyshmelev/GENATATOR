from __future__ import annotations

import unittest

try:
    import torch

    from genatator_core.infer_common import model_logits_for_inference
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


if __name__ == "__main__":
    unittest.main()
