from __future__ import annotations

import unittest

try:
    from genatator_core.cds_heuristic import infer_cds_with_benchmark_heuristic
except ImportError:
    infer_cds_with_benchmark_heuristic = None


@unittest.skipIf(infer_cds_with_benchmark_heuristic is None, "biopython is not installed")
class BenchmarkCdsHeuristicTests(unittest.TestCase):
    def test_finds_complete_orf_inside_exon(self) -> None:
        result = infer_cds_with_benchmark_heuristic(
            sequence="CCCATGAAATAGGG",
            interval_start=0,
            exons=[(0, 15)],
            strand="+",
        )
        self.assertEqual(result, [(3, 12)])

    def test_splices_exons_and_maps_cds_back_to_disjoint_intervals(self) -> None:
        result = infer_cds_with_benchmark_heuristic(
            sequence="ATGAAACCCCTAG",
            interval_start=100,
            exons=[(100, 106), (110, 113)],
            strand="+",
        )
        self.assertEqual(result, [(100, 106), (110, 113)])

    def test_reverse_strand_coordinates_are_restored(self) -> None:
        # Reverse complement of AATGAAATAGCCCCCC.  In transcript orientation
        # the complete ORF occupies [1, 10); on this genomic strand it maps to
        # [6, 15).
        result = infer_cds_with_benchmark_heuristic(
            sequence="GGGGGGCTATTTCATT",
            interval_start=0,
            exons=[(0, 16)],
            strand="-",
        )
        self.assertEqual(result, [(6, 15)])

    def test_requires_both_start_and_stop_codon(self) -> None:
        result = infer_cds_with_benchmark_heuristic(
            sequence="ATGAAAAAA",
            interval_start=0,
            exons=[(0, 9)],
            strand="+",
        )
        self.assertEqual(result, [])

    def test_no_exons_returns_no_cds(self) -> None:
        result = infer_cds_with_benchmark_heuristic(
            sequence="ATGAAATAG",
            interval_start=0,
            exons=[],
            strand="+",
        )
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
