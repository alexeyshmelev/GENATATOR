from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

try:
    from genatator_core.data import FindingChromosomeStore
except ImportError:
    FindingChromosomeStore = None

try:
    from genatator_core.train_common import FindingWindowSampler
except ImportError:
    FindingWindowSampler = None


@unittest.skipIf(FindingChromosomeStore is None, "runtime data dependencies are not installed")
class FindingChromosomeStoreTests(unittest.TestCase):
    def test_one_chromosome_cache_and_genome_chromosome_keys(self) -> None:
        groups = {
            ("genome_a", "chr1"): [
                {"parquet_path": "a0", "metadata": {"genome": "genome_a", "chrom": "chr1", "start": 0, "end": 3, "chrom_length": 6}},
                {"parquet_path": "a1", "metadata": {"genome": "genome_a", "chrom": "chr1", "start": 3, "end": 6, "chrom_length": 6}},
            ],
            ("genome_b", "chr1"): [
                {"parquet_path": "b0", "metadata": {"genome": "genome_b", "chrom": "chr1", "start": 0, "end": 2, "chrom_length": 2}},
            ],
        }
        blocks = {
            "a0": {"dna_sequence": "ACG", "targets": np.ones((3, 2), dtype=np.float32), "metadata": groups[("genome_a", "chr1")][0]["metadata"]},
            "a1": {"dna_sequence": "TTA", "targets": np.full((3, 2), 2.0, dtype=np.float32), "metadata": groups[("genome_a", "chr1")][1]["metadata"]},
            "b0": {"dna_sequence": "GG", "targets": np.full((2, 2), 3.0, dtype=np.float32), "metadata": groups[("genome_b", "chr1")][0]["metadata"]},
        }
        calls = []

        def fake_read(path, target_indices):
            calls.append(path)
            return blocks[path]

        store = FindingChromosomeStore(groups, [0, 1])
        with patch("genatator_core.data._read_parquet_block_row", side_effect=fake_read):
            sequence, targets, metadata, _ = store.get_slice(("genome_a", "chr1"), 0, 6)
            self.assertEqual(sequence, "ACGTTA")
            self.assertEqual(targets.shape, (6, 2))
            self.assertEqual(metadata.genome, "genome_a")
            store.get_slice(("genome_a", "chr1"), 1, 4)
            self.assertEqual(calls, ["a0", "a1"])
            sequence_b, _, metadata_b, _ = store.get_slice(("genome_b", "chr1"), 0, 2)
            self.assertEqual(sequence_b, "GG")
            self.assertEqual(metadata_b.genome, "genome_b")
            self.assertEqual(calls, ["a0", "a1", "b0"])
            self.assertEqual(store._cache_key, ("genome_b", "chr1"))


@unittest.skipIf(FindingWindowSampler is None, "trainer dependencies are not installed")
class FindingWindowSamplerTests(unittest.TestCase):
    def test_global_order_has_unique_equal_rank_lanes(self) -> None:
        class Dataset:
            finding_window_groups = {
                ("g1", "c1"): list(range(0, 5)),
                ("g1", "c2"): list(range(5, 11)),
                ("g2", "c1"): list(range(11, 19)),
            }

            def __len__(self):
                return 19

        with patch.dict(os.environ, {"WORLD_SIZE": "4"}, clear=False):
            sampler = FindingWindowSampler(Dataset(), seed=7)
            order = list(iter(sampler))
        self.assertEqual(len(order), 16)
        self.assertEqual(len(order), len(set(order)))
        lanes = [order[rank::4] for rank in range(4)]
        self.assertTrue(all(len(lane) == 4 for lane in lanes))
        self.assertEqual(len(set().union(*map(set, lanes))), 16)
        for left in range(4):
            for right in range(left + 1, 4):
                self.assertFalse(set(lanes[left]) & set(lanes[right]))


if __name__ == "__main__":
    unittest.main()
