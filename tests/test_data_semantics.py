from __future__ import annotations

import unittest

import numpy as np

try:
    from genatator_core.data import (
        GenatatorDataset,
        resolve_dataset_lengths,
        reverse_complement_task_labels,
    )
except ImportError:
    GenatatorDataset = None


@unittest.skipIf(GenatatorDataset is None, "torch/datasets/transformers are not installed")
class DataSemanticsTests(unittest.TestCase):
    def test_bpe_lengths_are_derived_from_token_fields(self) -> None:
        edge = resolve_dataset_lengths(
            {
                "model_family": "bpe",
                "max_bpe_tokens": 1024,
                "average_bpe_token_length": 9.0,
                "overlap": 0.5,
            },
            "finding_edge",
        )
        transcript = resolve_dataset_lengths(
            {
                "model_family": "bpe_unet",
                "max_bpe_tokens": 32768,
                "average_bpe_token_length": 9.0,
            },
            "segmentation",
        )
        self.assertEqual(edge["_resolved_max_nucleotides"], 9216)
        self.assertEqual(edge["_resolved_max_tokens"], 1024)
        self.assertEqual(transcript["_resolved_max_nucleotides"], 294912)

    def test_length_schema_rejects_ambiguous_or_overlapping_transcript_configs(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "must not define max_nucleotides"):
            resolve_dataset_lengths(
                {
                    "model_family": "bpe",
                    "max_bpe_tokens": 1024,
                    "average_bpe_token_length": 9.0,
                    "max_nucleotides": 9216,
                    "overlap": 0.5,
                },
                "finding_edge",
            )
        with self.assertRaisesRegex(RuntimeError, "must not define overlap"):
            resolve_dataset_lengths(
                {
                    "model_family": "bpe",
                    "max_bpe_tokens": 1024,
                    "average_bpe_token_length": 9.0,
                    "overlap": 0.5,
                },
                "transcript_type",
            )

    def test_transcript_crop_is_single_and_deterministic(self) -> None:
        dataset = object.__new__(GenatatorDataset)
        dataset.max_nucleotides = 1000
        dataset.crop_margin = 500
        self.assertEqual(dataset._crop_transcript(800), (0, 800))
        self.assertEqual(dataset._crop_transcript(1200), (200, 1200))
        self.assertEqual(dataset._crop_transcript(3000), (500, 1500))
        self.assertEqual(dataset._crop_transcript(3000), (500, 1500))

    def test_reverse_complement_remaps_orientation_dependent_channels(self) -> None:
        segmentation = np.arange(20).reshape(4, 5)
        edge = np.arange(12).reshape(3, 4)
        region = np.arange(8).reshape(4, 2)
        self.assertTrue(np.array_equal(
            reverse_complement_task_labels("segmentation", segmentation),
            segmentation[::-1][:, [3, 1, 2, 0, 4]],
        ))
        self.assertTrue(np.array_equal(
            reverse_complement_task_labels("finding_edge", edge),
            edge[::-1][:, [1, 0, 3, 2]],
        ))
        self.assertTrue(np.array_equal(
            reverse_complement_task_labels("finding_region", region),
            region[::-1][:, [1, 0]],
        ))


if __name__ == "__main__":
    unittest.main()

