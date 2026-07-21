from __future__ import annotations

import unittest


try:
    from genatator_core.train_common import normalize_vocab_size_field
except ImportError:
    normalize_vocab_size_field = None


@unittest.skipIf(normalize_vocab_size_field is None, "torch/transformers are not installed")
class VocabSizeConfigTests(unittest.TestCase):
    def test_legacy_field_is_migrated_and_removed(self) -> None:
        model = {"nucleotide_vocab_size": 32_000}
        normalize_vocab_size_field(model)
        self.assertEqual(model["vocab_size"], 32_000)
        self.assertNotIn("nucleotide_vocab_size", model)

    def test_conflicting_fields_are_rejected(self) -> None:
        model = {"vocab_size": 32_000, "nucleotide_vocab_size": 30_000}
        with self.assertRaisesRegex(RuntimeError, "Conflicting vocabulary sizes"):
            normalize_vocab_size_field(model)


if __name__ == "__main__":
    unittest.main()
