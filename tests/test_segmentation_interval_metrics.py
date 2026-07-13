from __future__ import annotations

import unittest
import numpy as np

try:
    from genatator_core.metrics_training import segmentation_interval_predictions
except ImportError:
    segmentation_interval_predictions = None


@unittest.skipIf(segmentation_interval_predictions is None, "metric dependencies are not installed")
class SegmentationIntervalDecodingTests(unittest.TestCase):
    def test_exon_uses_argmax_over_exon_5utr_3utr(self) -> None:
        logits = np.zeros((1, 4, 5), dtype=np.float32)
        # EXON wins despite every compared score being below zero.
        logits[0, 0, [1, 0, 3]] = [-0.1, -1.0, -2.0]
        # 5UTR wins even though EXON itself is far above a 0.5-probability threshold.
        logits[0, 1, [1, 0, 3]] = [10.0, 11.0, 0.0]
        # 3UTR wins.
        logits[0, 2, [1, 0, 3]] = [2.0, 1.0, 3.0]
        # EXON wins.
        logits[0, 3, [1, 0, 3]] = [5.0, 4.0, 4.0]
        decoded = segmentation_interval_predictions(logits, "exon")
        self.assertEqual(decoded.tolist(), [[1, 0, 0, 1]])

    def test_cds_uses_argmax_over_cds_and_intron(self) -> None:
        logits = np.zeros((1, 3, 5), dtype=np.float32)
        logits[0, 0, [4, 2]] = [-0.1, -2.0]
        logits[0, 1, [4, 2]] = [9.0, 10.0]
        logits[0, 2, [4, 2]] = [3.0, 2.0]
        decoded = segmentation_interval_predictions(logits, "CDS")
        self.assertEqual(decoded.tolist(), [[1, 0, 1]])


if __name__ == "__main__":
    unittest.main()
