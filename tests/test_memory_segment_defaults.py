from __future__ import annotations

import unittest

try:
    from genatator_core.model_builders import default_memory_segment_size
except ImportError:
    default_memory_segment_size = None


@unittest.skipIf(default_memory_segment_size is None, "torch/transformers are not installed")
class MemorySegmentDefaultTests(unittest.TestCase):
    def test_backbone_specific_defaults(self) -> None:
        self.assertEqual(default_memory_segment_size("gena"), 512)
        self.assertEqual(default_memory_segment_size("moderngena"), 1024)


if __name__ == "__main__":
    unittest.main()
