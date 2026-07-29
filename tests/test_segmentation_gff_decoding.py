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
                [10.0, 9.0, 0.0, 1.0, -5.0],  # UTR is ignored for exon; exon wins intron
                [1.0, 2.0, 5.0, 0.0, 4.0],   # Intron beats both exon and CDS
                [0.0, 3.0, 1.0, 2.0, 4.0],   # Exon and CDS both win
                [0.0, 2.0, 2.0, 4.0, 4.0],   # Target wins exon tie and CDS/3UTR tie
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
        self.assertEqual(record["exons"], [(0, 1), (2, 4)])
        self.assertEqual(record["cds"], [(2, 4)])


if __name__ == "__main__":
    unittest.main()
