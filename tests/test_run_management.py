import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


try:
    from genatator_core.run_management import (
        EvaluationConfigManager,
        FINDING_POSTPROCESS_DEFAULTS,
        MANUAL_CONFIG_PLACEHOLDER,
        MANUAL_DATASET_LENGTH_FIELDS_PLACEHOLDER,
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
                "vocab_size": 42,
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
            "training": {
                "output_dir": str(base),
                "custom_prefix": "experiment",
                "per_device_eval_batch_size": 1,
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
            self.assertEqual(evaluation["inference"]["batch_size"], 1)
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
            self.assertEqual(evaluation["task"], "finding")
            self.assertEqual(evaluation["edge"]["dataset"]["split"], "test")
            self.assertEqual(evaluation["edge"]["dataset"]["genomes"], ["GCF_009914755.1"])
            self.assertEqual(evaluation["edge"]["dataset"]["chromosomes"], ["NC_060944.1"])
            self.assertEqual(evaluation["edge"]["model"]["vocab_size"], 42)
            self.assertNotIn("nucleotide_vocab_size", evaluation["edge"]["model"])
            self.assertIsNone(evaluation["edge"]["inference"]["checkpoint_path"])
            self.assertEqual(evaluation["region"]["model"], MANUAL_CONFIG_PLACEHOLDER)
            self.assertEqual(evaluation["region"]["dataset"]["path"], "dataset")
            self.assertEqual(evaluation["region"]["dataset"]["split"], "test")
            self.assertEqual(
                evaluation["region"]["dataset"][MANUAL_DATASET_LENGTH_FIELDS_PLACEHOLDER],
                MANUAL_CONFIG_PLACEHOLDER,
            )
            self.assertNotIn("max_nucleotides", evaluation["region"]["dataset"])
            self.assertNotIn("max_bpe_tokens", evaluation["region"]["dataset"])
            self.assertNotIn(
                "average_bpe_token_length", evaluation["region"]["dataset"]
            )
            self.assertEqual(
                evaluation["region"]["dataset"]["genomes"],
                ["GCF_009914755.1"],
            )
            self.assertEqual(
                evaluation["region"]["inference"]["checkpoint_path"],
                MANUAL_CONFIG_PLACEHOLDER,
            )
            self.assertEqual(evaluation["postprocess"], FINDING_POSTPROCESS_DEFAULTS)
            self.assertTrue(evaluation["inference"]["use_reverse_complement"])
            self.assertEqual(evaluation["inference"]["true_gff"], "/tmp/reference.gff")
            self.assertEqual(evaluation["inference"]["k_values"], [0, 50, 100, 250, 500])
            self.assertTrue(evaluation["inference"]["use_strand"])
            self.assertNotIn("checkpoint_path", evaluation["inference"])

    def test_region_finding_evaluation_reverses_trained_and_manual_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            cfg = self._config(Path(temporary) / "base")
            cfg["true_gff"] = None
            evaluation = build_evaluation_config(cfg, task="finding_region", run_dir=run_dir)
            self.assertIsInstance(evaluation["region"]["model"], dict)
            self.assertEqual(evaluation["edge"]["model"], MANUAL_CONFIG_PLACEHOLDER)
            self.assertEqual(
                evaluation["inference"]["true_gff"],
                MANUAL_CONFIG_PLACEHOLDER,
            )

    def test_finding_checkpoint_update_targets_only_the_trained_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            cfg = self._config(Path(temporary) / "base")
            manager = EvaluationConfigManager(cfg, task="finding_edge", run_dir=run_dir)
            manager.write_initial()
            checkpoint = run_dir / "checkpoint-10"
            checkpoint.mkdir()
            manager.update_checkpoint(checkpoint, selection="best")

            written = json.loads((run_dir / "evaluation_config.json").read_text())
            self.assertEqual(
                Path(written["edge"]["inference"]["checkpoint_path"]),
                checkpoint.resolve(),
            )
            self.assertEqual(
                written["region"]["inference"]["checkpoint_path"],
                MANUAL_CONFIG_PLACEHOLDER,
            )
            self.assertNotIn("checkpoint_path", written["inference"])

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
