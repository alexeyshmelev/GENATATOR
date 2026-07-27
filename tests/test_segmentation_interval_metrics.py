from __future__ import annotations

import unittest
import numpy as np

try:
    from genatator_core.metrics_training import segmentation_interval_predictions
except ImportError:
    segmentation_interval_predictions = None


@unittest.skipIf(segmentation_interval_predictions is None, "metric dependencies are not installed")
class SegmentationIntervalDecodingTests(unittest.TestCase):
    def test_exon_competes_only_with_intron(self) -> None:
        logits = np.zeros((1, 4, 5), dtype=np.float32)
        # EXON wins despite both scores being below zero.
        logits[0, 0, [1, 2]] = [-0.1, -2.0]
        # INTRON wins.
        logits[0, 1, [1, 2]] = [10.0, 11.0]
        # UTR scores do not compete with the multilabel EXON track.
        logits[0, 2, [1, 2, 0, 3]] = [2.0, 1.0, 100.0, 100.0]
        # The first/positive class wins ties, matching the benchmark logic.
        logits[0, 3, [1, 2]] = [5.0, 5.0]
        decoded = segmentation_interval_predictions(logits, "exon")
        self.assertEqual(decoded.tolist(), [[1, 0, 1, 1]])

    def test_cds_competes_with_intron_and_both_utrs(self) -> None:
        logits = np.zeros((1, 4, 5), dtype=np.float32)
        logits[0, 0, [4, 2, 0, 3]] = [-0.1, -2.0, -3.0, -4.0]
        logits[0, 1, [4, 2, 0, 3]] = [9.0, 10.0, 0.0, 0.0]
        logits[0, 2, [4, 2, 0, 3]] = [3.0, 2.0, 4.0, 1.0]
        # The first/positive class wins ties, matching the benchmark logic.
        logits[0, 3, [4, 2, 0, 3]] = [5.0, 4.0, 5.0, 3.0]
        decoded = segmentation_interval_predictions(logits, "CDS")
        self.assertEqual(decoded.tolist(), [[1, 0, 0, 1]])


if __name__ == "__main__":
    unittest.main()
