from __future__ import annotations

import unittest

import numpy as np

try:
    from genatator_core.infer_common import (
        project_bpe_token_logits_to_nucleotides,
        project_masked_letter_logits_to_nucleotides,
        undo_reverse_complement_logits,
    )
except ImportError:
    project_bpe_token_logits_to_nucleotides = None


@unittest.skipIf(
    project_bpe_token_logits_to_nucleotides is None,
    "torch/datasets/transformers/safetensors are not installed",
)
class InferenceProjectionTests(unittest.TestCase):
    def test_plain_bpe_truncation_leaves_uncovered_nucleotides_nan(self) -> None:
        logits = np.arange(12, dtype=np.float32).reshape(3, 4)
        projected = project_bpe_token_logits_to_nucleotides(
            logits,
            [(0, 4), (4, 7), (0, 0)],
            np.asarray([1, 1, 0]),
            dna_length=10,
        )
        self.assertEqual(projected.shape, (10, 4))
        self.assertTrue(np.isfinite(projected[:7]).all())
        self.assertTrue(np.isnan(projected[7:]).all())

    def test_masked_unet_truncation_and_rc_keep_full_coordinates(self) -> None:
        logits = np.arange(32, dtype=np.float32).reshape(8, 4)
        forward = project_masked_letter_logits_to_nucleotides(
            logits,
            np.asarray([1, 1, 1, 1, 1, 1, 0, 0], dtype=bool),
            dna_length=10,
        )
        reverse = undo_reverse_complement_logits(forward, "finding_edge")
        self.assertTrue(np.isfinite(forward[:6]).all())
        self.assertTrue(np.isnan(forward[6:]).all())
        self.assertTrue(np.isnan(reverse[:4]).all())
        self.assertTrue(np.isfinite(reverse[4:]).all())
        self.assertTrue(np.array_equal(reverse[4:, 0], forward[:6, 1][::-1]))
        self.assertTrue(np.array_equal(reverse[4:, 2], forward[:6, 3][::-1]))


if __name__ == "__main__":
    unittest.main()

