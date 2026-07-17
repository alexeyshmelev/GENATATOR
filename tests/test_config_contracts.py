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


    def test_training_configs_are_task_complete_batch_one_and_rc_free(self) -> None:
        valid_tasks = {"finding_edge", "finding_region", "segmentation", "transcript_type"}
        for task_dir in ("finding", "segmentation", "transcript_type"):
            for path in sorted((ROOT / task_dir / "configs").glob("*.json")):
                cfg = json.loads(path.read_text())
                if not isinstance(cfg.get("training"), dict):
                    continue
                self.assertIn(cfg.get("task"), valid_tasks, path)
                self.assertEqual(cfg["training"]["per_device_train_batch_size"], 1, path)
                self.assertEqual(cfg["training"]["per_device_eval_batch_size"], 1, path)
                self.assertNotIn("evaluation", cfg, path)
                self.assertNotIn("reverse_complement", cfg.get("train_dataset", {}), path)
                self.assertNotIn("reverse_complement", cfg.get("eval_dataset", {}), path)
                if task_dir == "finding":
                    self.assertEqual(cfg["training"]["dataloader_num_workers"], 0, path)

    def test_inference_templates_expose_batch_one_and_rc_defaults(self) -> None:
        for task_dir in ("finding", "segmentation", "transcript_type"):
            for path in sorted((ROOT / task_dir / "configs").glob("infer_*.json")):
                cfg = json.loads(path.read_text())
                if "inference" in cfg:
                    self.assertEqual(cfg["inference"]["batch_size"], 1, path)
                    self.assertIs(cfg["inference"]["use_reverse_complement"], True, path)
                    if task_dir == "segmentation":
                        self.assertIs(cfg["inference"]["use_cds_heuristic"], True, path)
                for stage in ("edge", "region"):
                    if stage in cfg:
                        self.assertEqual(cfg[stage]["inference"]["batch_size"], 1, path)

    def test_finding_has_no_standalone_evaluate_script(self) -> None:
        self.assertFalse((ROOT / "finding" / "evaluate.py").exists())

    def test_all_training_configs_expose_true_gff(self) -> None:
        for task in ("finding", "segmentation", "transcript_type"):
            for path in sorted((ROOT / task / "configs").glob("*.json")):
                cfg = json.loads(path.read_text())
                if isinstance(cfg.get("training"), dict):
                    self.assertIn("true_gff", cfg)
                    self.assertIsNone(cfg["true_gff"])


if __name__ == "__main__":
    unittest.main()
