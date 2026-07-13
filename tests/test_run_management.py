import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


try:
    from genatator_core.run_management import (
        EvaluationConfigManager,
        build_evaluation_config,
        create_timestamped_run_dir,
    )
except ImportError:
    EvaluationConfigManager = None


@unittest.skipIf(EvaluationConfigManager is None, "transformers is not installed")
class RunManagementTests(unittest.TestCase):
    def _config(self, base: Path):
        return {
            "model": {
                "family": "unet",
                "backbone_kind": "moderngena",
                "backbone_path": "backbone",
                "tokenizer_path": "tokenizer",
                "checkpoint_path": None,
                "nucleotide_vocab_size": 42,
                "unet_chunk_size": 8192,
            },
            "eval_dataset": {
                "path": "dataset",
                "config_name": "val-human",
                "split": "validation",
                "statuses": [1],
                "random_crop": False,
            },
            "true_gff": "/tmp/reference.gff",
            "evaluation": {},
            "training": {
                "output_dir": str(base),
                "custom_prefix": "experiment",
                "per_device_eval_batch_size": 3,
            },
        }

    def test_timestamped_runs_are_unique_and_prefixed(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "runs"
            training = self._config(base)["training"]
            environment = {"RANK": "0", "WORLD_SIZE": "1"}
            with patch.dict(os.environ, environment, clear=False):
                first = create_timestamped_run_dir(training, config_path=__file__)
                second = create_timestamped_run_dir(training, config_path=__file__)
            self.assertNotEqual(first, second)
            self.assertTrue(first.name.startswith("experiment_"))
            latest = json.loads((base / "latest_run.json").read_text())
            self.assertEqual(Path(latest["run_dir"]), second)

    def test_segmentation_evaluation_removes_status_filter_and_updates_best(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            cfg = self._config(Path(temporary) / "base")
            evaluation = build_evaluation_config(cfg, task="segmentation", run_dir=run_dir)
            self.assertNotIn("statuses", evaluation["dataset"])
            self.assertNotIn("random_crop", evaluation["dataset"])
            self.assertNotIn("overlap", evaluation["dataset"])
            self.assertTrue(evaluation["dataset"]["full_transcript_chunks"])
            self.assertEqual(evaluation["dataset"]["genomes"], ["GCF_009914755.1"])
            self.assertEqual(evaluation["dataset"]["chromosomes"], ["NC_060944.1"])
            self.assertEqual(cfg["eval_dataset"]["statuses"], [1])
            self.assertTrue(evaluation["inference"]["use_reverse_complement"])
            self.assertTrue(evaluation["inference"]["use_cds_heuristic"])
            self.assertEqual(evaluation["inference"]["true_gff"], "/tmp/reference.gff")

            manager = EvaluationConfigManager(cfg, task="segmentation", run_dir=run_dir)
            manager.write_initial()
            checkpoint = run_dir / "checkpoint-10"
            checkpoint.mkdir()
            manager.update_checkpoint(checkpoint, selection="best")
            written = json.loads((run_dir / "evaluation_config.json").read_text())
            self.assertEqual(Path(written["inference"]["checkpoint_path"]), checkpoint.resolve())
            self.assertEqual(written["_generated"]["checkpoint_selection"], "best")
            self.assertTrue((checkpoint / "evaluation_config.json").is_file())

    def test_finding_evaluation_uses_test_split_and_fixed_chromosome(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            cfg = self._config(Path(temporary) / "base")
            cfg["eval_dataset"].pop("config_name", None)
            evaluation = build_evaluation_config(cfg, task="finding_edge", run_dir=run_dir)
            self.assertEqual(evaluation["dataset"]["split"], "test")
            self.assertEqual(evaluation["dataset"]["genomes"], ["GCF_009914755.1"])
            self.assertEqual(evaluation["dataset"]["chromosomes"], ["NC_060944.1"])
            self.assertTrue(evaluation["inference"]["use_reverse_complement"])
            self.assertEqual(evaluation["inference"]["true_gff"], "/tmp/reference.gff")

    def test_external_resumed_best_is_referenced_but_never_modified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_checkpoint = root / "run_a" / "checkpoint-10"
            old_checkpoint.mkdir(parents=True)
            sentinel = old_checkpoint / "evaluation_config.json"
            sentinel.write_bytes(b"old-run-owned-bytes\n")
            before = sentinel.read_bytes()

            run_b = root / "run_b"
            run_b.mkdir()
            cfg = self._config(root / "base")
            manager = EvaluationConfigManager(cfg, task="segmentation", run_dir=run_b)
            manager.write_initial()
            manager.update_checkpoint(old_checkpoint, selection="best")

            self.assertEqual(sentinel.read_bytes(), before)
            written = json.loads((run_b / "evaluation_config.json").read_text())
            self.assertEqual(Path(written["inference"]["checkpoint_path"]), old_checkpoint.resolve())


if __name__ == "__main__":
    unittest.main()
