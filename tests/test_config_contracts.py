from __future__ import annotations

import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StaticConfigContractsTest(unittest.TestCase):
    def test_no_cycle_count_in_config_names_or_output_dirs(self) -> None:
        for task in ("finding", "segmentation", "transcript_type"):
            for path in sorted((ROOT / task / "configs").glob("*.json")):
                self.assertNotIn("cycles", path.name)
                cfg = json.loads(path.read_text())
                if isinstance(cfg.get("training"), dict):
                    self.assertNotIn("cycles", cfg["training"]["output_dir"])

    def test_all_training_configs_expose_true_gff(self) -> None:
        for task in ("finding", "segmentation", "transcript_type"):
            for path in sorted((ROOT / task / "configs").glob("*.json")):
                cfg = json.loads(path.read_text())
                if isinstance(cfg.get("training"), dict):
                    self.assertIn("true_gff", cfg)
                    self.assertIsNone(cfg["true_gff"])


if __name__ == "__main__":
    unittest.main()
