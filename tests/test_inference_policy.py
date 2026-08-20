from __future__ import annotations

import unittest

from genatator_core.inference_policy import (
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


if __name__ == "__main__":
    unittest.main()
