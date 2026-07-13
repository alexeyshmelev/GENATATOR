from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from genatator_core.gff import labels_to_segmentation_record


class SegmentationGffDecodingTests(unittest.TestCase):
    def test_decoding_uses_requested_argmax_groups_not_thresholds(self) -> None:
        # Columns: 5UTR, EXON, INTRON, 3UTR, CDS.
        scores = np.asarray(
            [
                [10.0, 9.0, 0.0, 1.0, -5.0],  # 5UTR wins: not exon; intron wins: not CDS
                [1.0, 2.0, 5.0, 0.0, 4.0],   # EXON wins its group; intron beats CDS
                [0.0, 3.0, 1.0, 2.0, 4.0],   # EXON and CDS win
                [0.0, 1.0, 2.0, 4.0, 3.0],   # 3UTR wins: not exon; CDS wins
            ],
            dtype=np.float32,
        )
        meta = SimpleNamespace(
            transcript_type="mRNA",
            transcript_id="tx",
            gene_id="gene",
            chrom="chr",
            start=0,
            end=4,
            strand="+",
        )
        record = labels_to_segmentation_record(meta, scores, threshold=0.999999)
        self.assertEqual(record["exons"], [(1, 3)])
        self.assertEqual(record["cds"], [(2, 4)])


if __name__ == "__main__":
    unittest.main()
