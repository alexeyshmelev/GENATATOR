from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _training_configs():
    for task in ("finding", "segmentation", "transcript_type"):
        for path in sorted((ROOT / task / "configs").rglob("*.json")):
            cfg = _load(path)
            if isinstance(cfg.get("training"), dict):
                yield task, path, cfg


class StaticConfigContractsTest(unittest.TestCase):
    def test_no_cycle_count_in_config_names_or_output_dirs(self) -> None:
        for _, path, cfg in _training_configs():
            self.assertNotIn("cycles", path.name)
            self.assertNotIn("cycles", cfg["training"]["output_dir"])

    def test_training_configs_are_task_complete_batch_one_and_rc_free(self) -> None:
        valid_tasks = {
            "finding_edge",
            "finding_region",
            "segmentation",
            "transcript_type",
        }
        for task_dir, path, cfg in _training_configs():
            self.assertIn(cfg.get("task"), valid_tasks, path)
            self.assertEqual(cfg["training"]["per_device_train_batch_size"], 1, path)
            self.assertEqual(cfg["training"]["per_device_eval_batch_size"], 1, path)
            self.assertNotIn("evaluation", cfg, path)
            self.assertNotIn("reverse_complement", cfg.get("train_dataset", {}), path)
            self.assertNotIn("reverse_complement", cfg.get("eval_dataset", {}), path)
            if task_dir == "finding":
                self.assertEqual(cfg["training"]["dataloader_num_workers"], 0, path)

    def test_no_inference_templates_are_checked_in_outside_experiments(self) -> None:
        unexpected = []
        for task_dir in ("finding", "segmentation", "transcript_type"):
            for path in sorted((ROOT / task_dir / "configs").rglob("*.json")):
                cfg = _load(path)
                if "inference" in cfg and "training" not in cfg:
                    unexpected.append(path)
                if path.name.startswith("infer_") or path.name == "evaluation_config.json":
                    unexpected.append(path)
        self.assertEqual(unexpected, [])

    def test_experiment_inference_templates_use_batch_one_and_model_rc_policy(self) -> None:
        paths = sorted((ROOT / "experiments").rglob("*.json"))
        self.assertGreater(len(paths), 0)
        for path in paths:
            cfg = _load(path)
            inference = cfg.get("inference")
            if isinstance(inference, dict):
                self.assertEqual(inference["batch_size"], 1, path)
                expected_rc = cfg.get("model", {}).get("family") != "gpt"
                self.assertIs(inference["use_reverse_complement"], expected_rc, path)
            for stage in ("edge", "region"):
                if stage in cfg:
                    self.assertEqual(cfg[stage]["inference"]["batch_size"], 1, path)

    def test_finding_has_no_standalone_evaluate_script(self) -> None:
        self.assertFalse((ROOT / "finding" / "evaluate.py").exists())

    def test_all_training_configs_expose_true_gff(self) -> None:
        for _, path, cfg in _training_configs():
            self.assertIn("true_gff", cfg, path)
            self.assertEqual(cfg["true_gff"], "datasets/chr20.gff", path)
            self.assertIs(cfg["training"]["automatic_restart"], True, path)


if __name__ == "__main__":
    unittest.main()
