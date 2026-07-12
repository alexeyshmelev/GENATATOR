from __future__ import annotations

import unittest

import numpy as np

try:
    from genatator_core.data import (
        GenatatorDataset,
        nucleotide_ids,
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

    def test_caduceus_uses_normal_tokenizer_special_tokens(self) -> None:
        class FakeTokenizer:
            pad_token_id = 0
            unk_token_id = 999

            def __init__(self):
                self.last_kwargs = None

            def num_special_tokens_to_add(self, pair=False):
                return 2

            def convert_tokens_to_ids(self, token):
                return {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}.get(token, self.unk_token_id)

            def __call__(self, **kwargs):
                self.last_kwargs = kwargs
                ids = [101] + [self.convert_tokens_to_ids(ch) for ch in kwargs["text"]] + [102]
                ids = ids[: kwargs["max_length"]]
                attention = [1] * len(ids)
                special = [1] + [0] * max(0, len(ids) - 2) + ([1] if len(ids) > 1 else [])
                while len(ids) < kwargs["max_length"]:
                    ids.append(self.pad_token_id)
                    attention.append(0)
                    special.append(1)
                return {
                    "input_ids": ids,
                    "attention_mask": attention,
                    "token_type_ids": [0] * len(ids),
                    "special_tokens_mask": special,
                }

        dataset = object.__new__(GenatatorDataset)
        dataset.model_family = "nucleotide"
        dataset.max_nucleotides = 4
        dataset.tokenizer = FakeTokenizer()
        item = dataset._tokenize_token_task(
            "ACGT",
            np.arange(4, dtype=np.float32).reshape(4, 1),
            meta=None,
            local_start=0,
        )
        self.assertTrue(dataset.tokenizer.last_kwargs["add_special_tokens"])
        self.assertEqual(dataset.tokenizer.last_kwargs["max_length"], 6)
        self.assertEqual(item["input_ids"].tolist(), [101, 1, 2, 3, 4, 102])
        self.assertEqual(item["letter_level_labels_mask"].tolist(), [False, True, True, True, True, False])

    def test_nucleotide_ids_are_read_directly_from_tokenizer_vocab(self) -> None:
        class DirectTokenizer:
            pad_token_id = 0
            unk_token_id = 999

            def convert_tokens_to_ids(self, token):
                return {"A": 11, "C": 12, "G": 13, "T": 14, "N": 15}.get(token, self.unk_token_id)

            def __call__(self, *args, **kwargs):
                raise AssertionError("nucleotide_ids must not tokenize each nucleotide separately")

        ids = nucleotide_ids("ACGT", DirectTokenizer(), 6)
        self.assertEqual(ids.tolist(), [11, 12, 13, 14, 0, 0])


if __name__ == "__main__":
    unittest.main()

