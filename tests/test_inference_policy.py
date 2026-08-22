from __future__ import annotations

import unittest

from genatator_core.inference_policy import (
    gpt_inference_num_transcripts,
    inference_uses_reverse_complement,
    is_gpt_segmentation,
    segmentation_uses_cds_heuristic,
)


class InferencePolicyTests(unittest.TestCase):
    def test_gpt_segmentation_is_the_only_gpt_policy_scope(self) -> None:
        self.assertTrue(is_gpt_segmentation({"family": "gpt"}, "segmentation"))
        self.assertFalse(is_gpt_segmentation({"family": "gpt"}, "finding_edge"))
        self.assertFalse(is_gpt_segmentation({"family": "unet"}, "segmentation"))

    def test_gpt_reverse_complement_is_always_disabled(self) -> None:
        for inference in (
            {},
            {"use_reverse_complement": True},
            {"use_reverse_complement": False},
        ):
            cfg = {"model": {"family": "gpt"}, "inference": inference}
            self.assertFalse(inference_uses_reverse_complement(cfg, "segmentation"))

    def test_non_gpt_reverse_complement_remains_configurable(self) -> None:
        base = {"model": {"family": "unet"}}
        self.assertTrue(inference_uses_reverse_complement(base, "segmentation"))
        self.assertFalse(
            inference_uses_reverse_complement(
                {**base, "inference": {"use_reverse_complement": False}},
                "segmentation",
            )
        )

    def test_gpt_cds_heuristic_is_always_enabled(self) -> None:
        for inference in (
            {},
            {"use_cds_heuristic": True},
            {"use_cds_heuristic": False},
        ):
            cfg = {"model": {"family": "gpt"}, "inference": inference}
            self.assertTrue(segmentation_uses_cds_heuristic(cfg))

    def test_non_gpt_cds_heuristic_remains_configurable(self) -> None:
        base = {"model": {"family": "unet"}}
        self.assertTrue(segmentation_uses_cds_heuristic(base))
        self.assertFalse(
            segmentation_uses_cds_heuristic(
                {**base, "inference": {"use_cds_heuristic": False}}
            )
        )

    def test_gpt_num_transcripts_defaults_to_all_and_accepts_positive_counts(self) -> None:
        base = {"model": {"family": "gpt"}}
        self.assertEqual(gpt_inference_num_transcripts(base, "segmentation"), -1)
        self.assertEqual(
            gpt_inference_num_transcripts(
                {**base, "inference": {"num_transcripts": -1}},
                "segmentation",
            ),
            -1,
        )
        self.assertEqual(
            gpt_inference_num_transcripts(
                {**base, "inference": {"num_transcripts": 17}},
                "segmentation",
            ),
            17,
        )

    def test_gpt_num_transcripts_rejects_invalid_values(self) -> None:
        for value in (0, -2, 1.5, "4", True, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(RuntimeError, "num_transcripts"):
                    gpt_inference_num_transcripts(
                        {
                            "model": {"family": "gpt"},
                            "inference": {"num_transcripts": value},
                        },
                        "segmentation",
                    )

    def test_num_transcripts_is_gpt_segmentation_only(self) -> None:
        for model, task in (
            ({"family": "unet"}, "segmentation"),
            ({"family": "gpt"}, "finding_edge"),
        ):
            with self.subTest(model=model, task=task):
                with self.assertRaisesRegex(RuntimeError, "only for GPT segmentation"):
                    gpt_inference_num_transcripts(
                        {
                            "model": model,
                            "inference": {"num_transcripts": -1},
                        },
                        task,
                    )


if __name__ == "__main__":
    unittest.main()
