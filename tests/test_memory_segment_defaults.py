from __future__ import annotations

import unittest

try:
    from genatator_core.model_builders import (
        default_amt_segment_size,
        default_memory_segment_size,
        default_memory_token_count,
        normalize_memory_wrapper_config,
    )
except ImportError:
    default_amt_segment_size = None
    default_memory_segment_size = None
    default_memory_token_count = None
    normalize_memory_wrapper_config = None


@unittest.skipIf(default_memory_segment_size is None, "torch/transformers are not installed")
class MemorySegmentDefaultTests(unittest.TestCase):
    def test_backbone_specific_defaults(self) -> None:
        self.assertEqual(default_memory_segment_size("gena"), 512)
        self.assertEqual(default_memory_segment_size("moderngena"), 1024)
        self.assertEqual(default_memory_token_count("gena"), 10)
        self.assertEqual(default_memory_token_count("moderngena"), 20)
        self.assertEqual(default_amt_segment_size("gena"), 502)
        self.assertEqual(default_amt_segment_size("moderngena"), 1004)

    def test_wrapper_defaults_are_materialized_for_every_entrypoint(self) -> None:
        amt = {"family": "amt", "backbone_kind": "gena", "amt": {}}
        normalize_memory_wrapper_config(amt)
        self.assertEqual(amt["amt"]["num_mem_tokens"], 10)
        self.assertEqual(amt["amt"]["segment_size"], 502)

        rmt = {"family": "rmt", "backbone_kind": "moderngena", "rmt": {}}
        normalize_memory_wrapper_config(rmt)
        self.assertEqual(rmt["rmt"]["num_mem_tokens"], 20)

    def test_obsolete_explicit_memory_settings_are_rejected(self) -> None:
        amt = {
            "family": "amt",
            "backbone_kind": "gena",
            "amt": {"num_mem_tokens": 16, "segment_size": 512},
        }
        with self.assertRaisesRegex(RuntimeError, "requires num_mem_tokens=10"):
            normalize_memory_wrapper_config(amt)

        rmt = {
            "family": "rmt",
            "backbone_kind": "moderngena",
            "rmt": {"num_mem_tokens": 10, "segment_size": 1024},
        }
        with self.assertRaisesRegex(RuntimeError, "requires num_mem_tokens=20"):
            normalize_memory_wrapper_config(rmt)


if __name__ == "__main__":
    unittest.main()
