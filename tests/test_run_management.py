import json
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch


try:
    from genatator_core.run_management import (
        EvaluationConfigManager,
        FINDING_POSTPROCESS_DEFAULTS,
        MANUAL_CONFIG_PLACEHOLDER,
        MANUAL_DATASET_LENGTH_FIELDS_PLACEHOLDER,
        build_evaluation_config,
        canonical_training_config,
        create_training_run,
        create_timestamped_run_dir,
        newest_complete_checkpoint,
        training_config_fingerprint,
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
                "learning_rate": 5e-5,
                "resume_from_checkpoint": None,
            },
        }

    def _save_runtime_config(self, cfg: dict, run_dir: Path, base: Path) -> None:
        saved = deepcopy(cfg)
        saved["training"]["output_base_dir"] = str(base.resolve())
        saved["training"]["output_dir"] = str(run_dir.resolve())
        saved["training"]["overwrite_output_dir"] = False
        (run_dir / "training_config.json").write_text(
            json.dumps(saved, indent=2),
            encoding="utf-8",
        )

    def _checkpoint(
        self,
        run_dir: Path,
        step: int,
        *,
        complete: bool = True,
        sharded: bool = False,
    ) -> Path:
        checkpoint = run_dir / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": step}),
            encoding="utf-8",
        )
        if sharded:
            (checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"weights")
            (checkpoint / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "encoder.weight": "model-00001-of-00001.safetensors"
                        }
                    }
                ),
                encoding="utf-8",
            )
        else:
            (checkpoint / "pytorch_model.bin").write_bytes(b"weights")
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
        if complete:
            (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
        return checkpoint.resolve()

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

    def test_fingerprint_normalizes_saved_run_paths_and_restart_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "runs"
            current = self._config(base)
            current["training"].pop("automatic_restart", None)
            current["training"]["overwrite_output_dir"] = True

            saved = deepcopy(current)
            saved["training"]["output_base_dir"] = str(base.resolve())
            saved["training"]["output_dir"] = str(
                (base / "experiment_20260729_120000_000000").resolve()
            )
            saved["training"]["overwrite_output_dir"] = False
            saved["training"]["automatic_restart"] = True

            self.assertEqual(
                canonical_training_config(current),
                canonical_training_config(saved),
            )
            self.assertEqual(
                training_config_fingerprint(current),
                training_config_fingerprint(saved),
            )
            saved["training"]["learning_rate"] = 1e-4
            self.assertNotEqual(
                training_config_fingerprint(current),
                training_config_fingerprint(saved),
            )

    def test_automatic_restart_selects_newest_complete_numeric_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "runs"
            cfg = self._config(base)
            cfg["training"]["automatic_restart"] = True
            environment = {"RANK": "0", "WORLD_SIZE": "1"}
            with patch.dict(os.environ, environment, clear=False):
                first = create_training_run(cfg, config_path=__file__)
                self._save_runtime_config(cfg, first.run_dir, base)
                self._checkpoint(first.run_dir, 9)
                self._checkpoint(first.run_dir, 100, sharded=True)
                self._checkpoint(first.run_dir, 20)

                self.assertEqual(
                    newest_complete_checkpoint(first.run_dir),
                    (first.run_dir / "checkpoint-100").resolve(),
                )
                second = create_training_run(cfg, config_path=__file__)

            self.assertNotEqual(second.run_dir, first.run_dir)
            self.assertEqual(
                second.resume_from_checkpoint,
                (first.run_dir / "checkpoint-100").resolve(),
            )
            self.assertEqual(second.recovery_checkpoint, second.resume_from_checkpoint)
            self.assertEqual(second.source_run_dir, first.run_dir)

    def test_latest_mismatch_never_falls_back_to_older_matching_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "runs"
            cfg_a = self._config(base)
            cfg_a["training"]["automatic_restart"] = True
            cfg_b = deepcopy(cfg_a)
            cfg_b["training"]["learning_rate"] = 1e-4
            environment = {"RANK": "0", "WORLD_SIZE": "1"}
            with patch.dict(os.environ, environment, clear=False):
                run_a = create_training_run(cfg_a, config_path=__file__)
                self._save_runtime_config(cfg_a, run_a.run_dir, base)
                old_matching_checkpoint = self._checkpoint(run_a.run_dir, 10)

                run_b = create_training_run(cfg_b, config_path=__file__)
                self._save_runtime_config(cfg_b, run_b.run_dir, base)
                self._checkpoint(run_b.run_dir, 20)

                next_a = create_training_run(cfg_a, config_path=__file__)

            self.assertIsNone(next_a.resume_from_checkpoint)
            self.assertIsNone(next_a.source_run_dir)
            self.assertNotEqual(next_a.resume_from_checkpoint, old_matching_checkpoint)

    def test_incomplete_newest_checkpoint_falls_back_within_latest_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "runs"
            cfg = self._config(base)
            cfg["training"]["automatic_restart"] = True
            environment = {"RANK": "0", "WORLD_SIZE": "1"}
            with patch.dict(os.environ, environment, clear=False):
                first = create_training_run(cfg, config_path=__file__)
                self._save_runtime_config(cfg, first.run_dir, base)
                complete = self._checkpoint(first.run_dir, 5000)
                self._checkpoint(first.run_dir, 10000, complete=False)
                second = create_training_run(cfg, config_path=__file__)

            self.assertEqual(second.resume_from_checkpoint, complete)
            self.assertEqual(second.source_run_dir, first.run_dir)

    def test_continuation_without_local_save_inherits_recorded_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "runs"
            cfg = self._config(base)
            cfg["training"]["automatic_restart"] = True
            environment = {"RANK": "0", "WORLD_SIZE": "1"}
            with patch.dict(os.environ, environment, clear=False):
                first = create_training_run(cfg, config_path=__file__)
                self._save_runtime_config(cfg, first.run_dir, base)
                durable = self._checkpoint(first.run_dir, 5000)

                second = create_training_run(cfg, config_path=__file__)
                self._save_runtime_config(cfg, second.run_dir, base)
                self.assertEqual(second.recovery_checkpoint, durable)
                # Simulate HPS terminating the continuation before it reaches
                # the next save: second.run_dir has no checkpoint directory.
                third = create_training_run(cfg, config_path=__file__)

            self.assertEqual(third.resume_from_checkpoint, durable)
            self.assertEqual(third.recovery_checkpoint, durable)
            self.assertEqual(third.source_run_dir, second.run_dir)

    def test_false_starts_fresh_and_explicit_checkpoint_takes_precedence(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "runs"
            automatic = self._config(base)
            automatic["training"]["automatic_restart"] = True
            environment = {"RANK": "0", "WORLD_SIZE": "1"}
            with patch.dict(os.environ, environment, clear=False):
                first = create_training_run(automatic, config_path=__file__)
                self._save_runtime_config(automatic, first.run_dir, base)
                previous = self._checkpoint(first.run_dir, 10)

                disabled = deepcopy(automatic)
                disabled["training"]["automatic_restart"] = False
                fresh = create_training_run(disabled, config_path=__file__)
                self.assertIsNone(fresh.resume_from_checkpoint)
                self.assertIsNone(fresh.source_run_dir)

                manual_source = first.run_dir / "manual-checkpoint"
                manual_source.mkdir()
                explicit = deepcopy(automatic)
                explicit["training"]["resume_from_checkpoint"] = str(manual_source)
                manual = create_training_run(explicit, config_path=__file__)

            self.assertEqual(
                manual.resume_from_checkpoint,
                manual_source.resolve(),
            )
            self.assertEqual(manual.recovery_checkpoint, manual_source.resolve())
            self.assertIsNone(manual.source_run_dir)
            self.assertNotEqual(manual.resume_from_checkpoint, previous)

    def test_latest_run_must_be_a_direct_child_of_output_base(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "runs"
            cfg = self._config(base)
            cfg["training"]["automatic_restart"] = True
            outside = root / "outside"
            outside.mkdir()
            checkpoint = self._checkpoint(outside, 10)
            base.mkdir()
            (base / "latest_run.json").write_text(
                json.dumps(
                    {
                        "run_dir": str(outside),
                        "config_fingerprint": training_config_fingerprint(cfg),
                        "config_fingerprint_version": 1,
                        "recovery_checkpoint": str(checkpoint),
                    }
                ),
                encoding="utf-8",
            )
            environment = {"RANK": "0", "WORLD_SIZE": "1"}
            with patch.dict(os.environ, environment, clear=False):
                plan = create_training_run(cfg, config_path=__file__)

            self.assertIsNone(plan.resume_from_checkpoint)
            self.assertIsNone(plan.source_run_dir)

    def test_ddp_workers_receive_one_plan_and_elastic_restarts_get_new_manifests(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "runs"
            cfg = self._config(base)
            cfg["training"]["automatic_restart"] = True
            shared = {
                "WORLD_SIZE": "2",
                "GENATATOR_LAUNCH_ID": "restart-test",
                "TORCHELASTIC_RESTART_COUNT": "0",
            }
            with patch.dict(os.environ, {**shared, "RANK": "0"}, clear=False):
                rank0 = create_training_run(cfg, config_path=__file__)
            with patch.dict(os.environ, {**shared, "RANK": "1"}, clear=False):
                rank1 = create_training_run(cfg, config_path=__file__)
            self.assertEqual(rank1, rank0)

            restarted = dict(shared)
            restarted["TORCHELASTIC_RESTART_COUNT"] = "1"
            with patch.dict(os.environ, {**restarted, "RANK": "0"}, clear=False):
                next_plan = create_training_run(cfg, config_path=__file__)

            self.assertNotEqual(next_plan.run_dir, rank0.run_dir)
            self.assertEqual(
                len(list(base.glob(".genatator-run-*.json"))),
                2,
            )

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
            self.assertEqual(
                evaluation["edge"]["dataset"]["genomes"],
                ["GCF_009914755.1_T2T-CHM13v2.0"],
            )
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
                ["GCF_009914755.1_T2T-CHM13v2.0"],
            )
            self.assertEqual(
                evaluation["region"]["inference"]["checkpoint_path"],
                MANUAL_CONFIG_PLACEHOLDER,
            )
            self.assertEqual(evaluation["postprocess"], FINDING_POSTPROCESS_DEFAULTS)
            self.assertTrue(evaluation["inference"]["use_reverse_complement"])
            self.assertEqual(evaluation["inference"]["true_gff"], "/tmp/reference.gff")
            self.assertEqual(
                Path(evaluation["inference"]["metrics_csv"]),
                (run_dir / "evaluation" / "finding_auc_metrics.csv").resolve(),
            )
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
