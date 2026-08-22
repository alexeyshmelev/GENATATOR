from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from genatator_core.data import (
        DirectParquetTranscriptIndex,
        GenatatorDataset,
        MaterializedRows,
        _direct_parquet_transcript_index_from_files,
        _load_segmentation_direct_parquet,
        _read_direct_parquet_transcript_row,
    )
except ImportError:
    pa = None
    pq = None
    DirectParquetTranscriptIndex = None


@unittest.skipIf(DirectParquetTranscriptIndex is None, "runtime dependencies are not installed")
class SegmentationDirectLoaderTests(unittest.TestCase):
    @staticmethod
    def _metadata(transcript: str, gene: str, chrom: str, length: int) -> str:
        return json.dumps(
            {
                "transcript_id": transcript,
                "gene_id": gene,
                "transcript_type": "mRNA",
                "strand": "+",
                "genome": "GCF_test",
                "chrom": chrom,
                "start": 0,
                "end": length,
                "chrom_length": 1000,
            }
        )

    @staticmethod
    def _labels(length: int, offset: float) -> list[list[float]]:
        return [
            [offset + float(position * 5 + channel) for channel in range(5)]
            for position in range(length)
        ]

    def _write_parquet(
        self,
        path: Path,
        rows: list[tuple[str, str, str, int]],
    ) -> None:
        sequences = [row[0] for row in rows]
        table = pa.table(
            {
                "dna_sequence": pa.array(sequences),
                "labels": pa.array(
                    [self._labels(len(sequence), float(index * 100)) for index, sequence in enumerate(sequences)],
                    type=pa.list_(pa.list_(pa.float32())),
                ),
                "metadata": pa.array(
                    [
                        self._metadata(transcript, f"gene_{transcript}", chrom, len(sequence))
                        for sequence, transcript, chrom, _ in rows
                    ]
                ),
                "status": pa.array([row[3] for row in rows], type=pa.int64()),
            }
        )
        pq.write_table(table, path, row_group_size=2)

    def _fixture(self, root: Path) -> list[Path]:
        first = root / "part-0.parquet"
        second = root / "part-1.parquet"
        self._write_parquet(
            first,
            [
                ("ACGT", "A1", "chr20", 1),
                ("AAAA", "A2", "chr20", 0),
                ("CCCCCC", "B2", "chr20", 1),
            ],
        )
        self._write_parquet(
            second,
            [
                ("GGGG", "C1", "chr21", 1),
                ("TTTTT", "D2", "chr20", 1),
            ],
        )
        return [first, second]

    @staticmethod
    def _config(**updates):
        cfg = {
            "_task": "segmentation",
            "genomes": ["GCF_test"],
            "chromosomes": ["chr20"],
            "statuses": [1],
            "parquet_batch_size": 2,
        }
        cfg.update(updates)
        return cfg

    def test_index_keeps_only_locations_and_reads_each_row_on_demand(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = self._fixture(root)
            cfg = self._config(segmentation_index_cache_dir=str(root / "cache"))
            with patch(
                "genatator_core.data._arrow_2d_scalar_to_numpy",
                side_effect=AssertionError("labels must not be decoded while indexing"),
            ):
                index = _direct_parquet_transcript_index_from_files(files, cfg)

            self.assertEqual(
                index.selected_indices.tolist(),
                [[0, 0, 0], [0, 1, 0], [1, 0, 0]],
            )
            self.assertEqual(index.selected_indices.dtype, np.uint32)
            self.assertFalse(index.selected_indices.flags.writeable)
            self.assertEqual(
                set(vars(index)),
                {"files", "selected_indices", "column_names"},
            )
            self.assertNotIn("rows", vars(index))
            restored = pickle.loads(pickle.dumps(index))
            self.assertFalse(restored.selected_indices.flags.writeable)

            for cache_file in index.files:
                parquet = pq.ParquetFile(cache_file, pre_buffer=False, memory_map=False)
                try:
                    self.assertTrue(
                        all(
                            parquet.metadata.row_group(group).num_rows == 1
                            for group in range(parquet.num_row_groups)
                        )
                    )
                finally:
                    parquet.close()

            with patch(
                "genatator_core.data._read_direct_parquet_transcript_row",
                wraps=_read_direct_parquet_transcript_row,
            ) as read_row:
                first = index[0]
                again = index[0]
            self.assertEqual(read_row.call_count, 2)
            self.assertEqual(first["dna_sequence"], "ACGT")
            self.assertEqual(first["labels"].shape, (4, 5))
            self.assertEqual(first["metadata"], again["metadata"])

            original_nonzero_offset = index[2]
            self.assertEqual(original_nonzero_offset["dna_sequence"], "TTTTT")

            no_labels = index.read_row(1, include_labels=False)
            self.assertEqual(no_labels["dna_sequence"], "CCCCCC")
            self.assertNotIn("labels", no_labels)

            with patch(
                "genatator_core.data._build_segmentation_row_cache",
                side_effect=AssertionError("complete cache must be reused"),
            ):
                reused = _direct_parquet_transcript_index_from_files(files, cfg)
            self.assertEqual(reused.selected_indices.tolist(), index.selected_indices.tolist())

    def test_max_rows_keeps_the_first_filtered_locations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = self._fixture(root)
            index = _direct_parquet_transcript_index_from_files(
                files,
                self._config(
                    max_rows=2,
                    segmentation_index_cache_dir=str(root / "cache"),
                ),
            )
            self.assertEqual(index.selected_indices.tolist(), [[0, 0, 0], [0, 1, 0]])

    def test_pipe_metadata_is_filtered_before_caching(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "pipe.parquet"
            sequence = "ACGT"
            pq.write_table(
                pa.table(
                    {
                        "dna_sequence": [sequence],
                        "labels": pa.array(
                            [self._labels(len(sequence), 0.0)],
                            type=pa.list_(pa.list_(pa.float32())),
                        ),
                        "metadata": ["P1|gene_P|mRNA|+|GCF_test|chr20|0:4"],
                        "status": pa.array([1], type=pa.int64()),
                    }
                ),
                source,
            )
            index = _direct_parquet_transcript_index_from_files(
                [source],
                self._config(segmentation_index_cache_dir=str(root / "cache")),
            )
            self.assertEqual(len(index), 1)
            self.assertEqual(index[0]["dna_sequence"], sequence)

    def test_regular_segmentation_build_does_not_dereference_the_index(self) -> None:
        index = DirectParquetTranscriptIndex(
            [Path("unused.parquet")],
            [(0, 0, 0), (0, 0, 0)],
            has_status=True,
        )
        cfg = {
            "path": "unused.parquet",
            "loader": "direct_parquet",
            "model_family": "bpe_unet",
            "max_bpe_tokens": 8,
            "average_bpe_token_length": 2.0,
            "crop_margin": 500,
        }
        with (
            patch("genatator_core.data.load_dataset_auto", return_value=index),
            patch.object(index, "read_row", side_effect=AssertionError("unexpected row read")) as read_row,
        ):
            dataset = GenatatorDataset(cfg, "segmentation", tokenizer=object())
        self.assertEqual(read_row.call_count, 0)
        self.assertIsInstance(dataset.row_indices, range)
        self.assertIsInstance(dataset.windows, range)
        self.assertEqual(list(dataset.windows), [0, 1])

    def test_full_transcript_chunks_reread_the_source_without_a_cache(self) -> None:
        index = DirectParquetTranscriptIndex(
            [Path("unused.parquet")],
            [(0, 0, 0)],
            has_status=True,
        )
        sequence = "ACGTACGTA"
        labels = np.arange(len(sequence) * 5, dtype=np.float32).reshape(len(sequence), 5)
        metadata = self._metadata("A1", "gene_A", "chr20", len(sequence))

        def row_from_disk(_index: int, *, include_labels: bool = True):
            row = {"dna_sequence": sequence, "metadata": metadata}
            if include_labels:
                row["labels"] = labels.copy()
            return row

        dataset = object.__new__(GenatatorDataset)
        dataset.raw = index
        dataset.row_indices = range(1)
        dataset.full_transcript_chunks = True
        dataset.model_family = "nucleotide"
        dataset.max_nucleotides = 4
        dataset.reverse_complement = False
        dataset.target_indices = [0, 1, 2, 3, 4]
        dataset.task = "segmentation"

        with patch.object(index, "read_row", side_effect=row_from_disk) as read_row:
            dataset._build_transcript_indices()
            self.assertEqual(
                dataset.windows,
                [(0, 0, 4, 0), (0, 4, 8, 4), (0, 8, 9, 8)],
            )
            self.assertEqual(read_row.call_args_list[0].kwargs, {"include_labels": False})
            chunks = [dataset._slice_transcript(i) for i in range(len(dataset.windows))]

        self.assertEqual(read_row.call_count, 4)
        self.assertEqual([chunk[0] for chunk in chunks], ["ACGT", "ACGT", "A"])
        self.assertEqual([chunk[3] for chunk in chunks], [0, 4, 8])
        self.assertEqual([chunk[1].shape for chunk in chunks], [(4, 5), (4, 5), (1, 5)])

    def test_gpt_inference_shards_transcripts_before_building_chunks(self) -> None:
        index = DirectParquetTranscriptIndex(
            [Path("unused.parquet")],
            [(0, 0, 0)] * 5,
            has_status=True,
        )

        def row_from_disk(row_index: int, *, include_labels: bool = True):
            sequence = "ACGTA"
            row = {
                "dna_sequence": sequence,
                "metadata": self._metadata(
                    f"tx_{row_index}",
                    "shared_gene",
                    "chr20",
                    len(sequence),
                ),
            }
            if include_labels:
                row["labels"] = np.zeros((len(sequence), 5), dtype=np.float32)
            return row

        dataset = object.__new__(GenatatorDataset)
        dataset.cfg = {
            "_gpt_inference_num_transcripts": 5,
            "_gpt_inference_rank": 1,
            "_gpt_inference_world_size": 2,
        }
        dataset.for_inference = True
        dataset.task = "segmentation"
        dataset.full_transcript_chunks = True
        dataset.row_indices = range(5)
        dataset.raw = index
        dataset.model_family = "nucleotide"
        dataset.max_nucleotides = 2
        dataset.reverse_complement = False
        dataset.target_indices = [0, 1, 2, 3, 4]

        with patch.object(index, "read_row", side_effect=row_from_disk):
            dataset._apply_inference_transcript_selection()
            dataset._build_transcript_indices()

        self.assertEqual(dataset.inference_total_transcripts, 5)
        self.assertEqual(dataset.inference_selected_transcripts, 5)
        self.assertEqual(dataset.inference_assigned_ordinals, [1, 3])
        self.assertEqual(dataset.row_indices, [1, 3])
        self.assertEqual({window[0] for window in dataset.windows}, {1, 3})
        self.assertEqual(
            [window[0] for window in dataset.windows],
            [1, 1, 1, 3, 3, 3],
        )

    def test_gpt_inference_rejects_more_than_the_filtered_chromosome(self) -> None:
        dataset = object.__new__(GenatatorDataset)
        dataset.cfg = {
            "_gpt_inference_num_transcripts": 4,
            "_gpt_inference_rank": 0,
            "_gpt_inference_world_size": 1,
        }
        dataset.for_inference = True
        dataset.task = "segmentation"
        dataset.full_transcript_chunks = True
        dataset.row_indices = range(3)

        with self.assertRaisesRegex(
            RuntimeError,
            "requested=4 available=3",
        ):
            dataset._apply_inference_transcript_selection()

    def test_transcript_type_keeps_the_existing_materialized_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = self._fixture(root)
            cfg = self._config(
                _task="transcript_type",
                path=str(files[0]),
                segmentation_index_cache_dir=str(root / "cache"),
            )
            rows = _load_segmentation_direct_parquet(cfg)
            self.assertIsInstance(rows, MaterializedRows)
            self.assertIn("dna_sequence", rows[0])
            self.assertNotIn("labels", rows[0])


if __name__ == "__main__":
    unittest.main()
