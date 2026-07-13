from __future__ import annotations

import unittest
import numpy as np

try:
    from genatator_core.data import GenatatorDataset, ParsedMetadata
    from genatator_core.infer_common import aggregate_full_segmentation_chunks
except ImportError:
    GenatatorDataset = None


@unittest.skipIf(GenatatorDataset is None, "runtime dependencies are not installed")
class FullTranscriptGatheringTests(unittest.TestCase):
    def test_nucleotide_chunks_are_non_overlapping_and_complete(self) -> None:
        dataset = object.__new__(GenatatorDataset)
        dataset.model_family = "nucleotide"
        dataset.max_nucleotides = 4
        self.assertEqual(dataset._full_transcript_chunk_bounds("ACGTACGTAA"), [(0, 4), (4, 8), (8, 10)])


    def test_bpe_chunks_are_independent_non_overlapping_and_fit_token_limit(self) -> None:
        class FakeFastTokenizer:
            is_fast = True

            def __call__(self, text, **kwargs):
                # One token per two nucleotides, plus CLS/SEP.
                return {"input_ids": [101] + list(range((len(text) + 1) // 2)) + [102]}

        dataset = object.__new__(GenatatorDataset)
        dataset.model_family = "bpe_unet"
        dataset.max_nucleotides = 20
        dataset.max_tokens = 6
        dataset.tokenizer = FakeFastTokenizer()
        bounds = dataset._full_transcript_chunk_bounds("A" * 21)
        self.assertEqual(bounds[0][0], 0)
        self.assertEqual(bounds[-1][1], 21)
        self.assertTrue(all(a < b for a, b in bounds))
        self.assertTrue(all(bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1)))
        for start, end in bounds:
            self.assertLessEqual(len(dataset.tokenizer("A" * (end - start))["input_ids"]), 6)

    def test_forward_chunks_are_gathered_in_original_order(self) -> None:
        meta = ParsedMetadata(transcript_id="tx1", gene_id="g1", genome="g", chrom="c", start=10, end=16)
        rows = [
            {
                "metadata": meta,
                "dna_sequence": "ACG",
                "local_start": 0,
                "model_family": "nucleotide",
                "reverse_complement": False,
                "logits": np.ones((3, 5), dtype=np.float32),
            },
            {
                "metadata": meta,
                "dna_sequence": "TTA",
                "local_start": 3,
                "model_family": "nucleotide",
                "reverse_complement": False,
                "logits": np.full((3, 5), 2.0, dtype=np.float32),
            },
        ]
        gathered = aggregate_full_segmentation_chunks(rows)
        self.assertEqual(len(gathered), 1)
        self.assertEqual(gathered[0]["dna_sequence"], "ACGTTA")
        self.assertEqual(gathered[0]["local_start"], 0)
        self.assertEqual(gathered[0]["logits"][:, 0].tolist(), [1, 1, 1, 2, 2, 2])

    def test_reverse_complement_chunk_dna_is_restored_before_gathering(self) -> None:
        meta = ParsedMetadata(transcript_id="tx1", gene_id="g1", genome="g", chrom="c", start=10, end=16)
        # RC chunks correspond to original spans [3:6] and [0:3].
        rows = [
            {
                "metadata": meta,
                "dna_sequence": "TAA",  # RC of original TTA
                "local_start": 3,
                "model_family": "nucleotide",
                "reverse_complement": True,
                "logits": np.ones((3, 5), dtype=np.float32),
            },
            {
                "metadata": meta,
                "dna_sequence": "CGT",  # RC of original ACG
                "local_start": 0,
                "model_family": "nucleotide",
                "reverse_complement": True,
                "logits": np.full((3, 5), 2.0, dtype=np.float32),
            },
        ]
        gathered = aggregate_full_segmentation_chunks(rows)
        self.assertEqual(gathered[0]["dna_sequence"], "ACGTTA")


if __name__ == "__main__":
    unittest.main()
