from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


try:
    import finding.infer as finding_infer

    _validate_full_pipeline_config = finding_infer._validate_full_pipeline_config
except ImportError:
    finding_infer = None
    _validate_full_pipeline_config = None


@unittest.skipIf(_validate_full_pipeline_config is None, "inference dependencies are not installed")
class FindingPipelineConfigTests(unittest.TestCase):
    def _complete_config(self):
        return {
            "edge": {
                "model": {"family": "plain"},
                "dataset": {"path": "dataset"},
                "inference": {"checkpoint_path": "edge-checkpoint"},
            },
            "region": {
                "model": {"family": "plain"},
                "dataset": {"path": "dataset"},
                "inference": {"checkpoint_path": "region-checkpoint"},
            },
            "inference": {"true_gff": "reference.gff"},
        }

    def test_complete_config_is_accepted(self) -> None:
        _validate_full_pipeline_config(self._complete_config())

    def test_legacy_optional_annotation_and_model_checkpoint_remain_valid(self) -> None:
        cfg = self._complete_config()
        cfg["inference"]["true_gff"] = None
        cfg["edge"]["model"]["checkpoint_path"] = "edge-checkpoint"
        cfg["edge"]["inference"]["checkpoint_path"] = None
        _validate_full_pipeline_config(cfg)

    def test_unresolved_fields_are_reported_before_inference(self) -> None:
        cfg = self._complete_config()
        cfg["region"]["model"] = "<manually_insert_value_here>"
        cfg["region"]["inference"]["checkpoint_path"] = "<manually_insert_value_here>"
        cfg["inference"]["true_gff"] = "<manually_insert_value_here>"
        with self.assertRaisesRegex(
            RuntimeError,
            "region.model, region.inference.checkpoint_path, inference.true_gff",
        ):
            _validate_full_pipeline_config(cfg)

    def test_dataset_length_field_marker_must_be_replaced(self) -> None:
        cfg = self._complete_config()
        cfg["region"]["dataset"] = {
            "path": "dataset",
            "<manually_insert_model_dependent_length_fields_here>": 4096,
        }
        with self.assertRaisesRegex(RuntimeError, "region.dataset"):
            _validate_full_pipeline_config(cfg)

    def test_joint_pipeline_writes_full_exon_gff_and_complete_metrics(self) -> None:
        key = ("GCF_009914755.1", "NC_060944.1")
        edge = np.asarray(
            [
                [0.0, 0.9, 0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.1, 0.9, 0.0],
                [0.0, 0.0, 0.0, 0.1, 0.0],
            ],
            dtype=np.float32,
        )
        edge_truth = (edge > 0.5).astype(np.float32)
        region = np.asarray(
            [
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        def predictions(_stage_cfg, task, _device, use_reverse_complement):
            if task == "finding_edge":
                return {key: edge}, {key: edge_truth}
            return {key: region}, {key: region.copy()}

        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            gff_path = temporary / "predictions.gff"
            metrics_path = temporary / "metrics.json"
            cfg = self._complete_config()
            cfg["postprocess"] = {
                "low_pass_fraction": 0.05,
                "peak_prominence": 0.15,
                "peak_distance": 50,
                "peak_height": None,
                "interval_window_size": 2_000_000,
                "max_pairs_per_seed": 10,
                "prob_threshold": 0.5,
                "zero_fraction_drop_threshold": 0.01,
                "pairing_progress_every": 1000,
            }
            cfg["inference"].update(
                {
                    "device": "cpu",
                    "batch_size": 1,
                    "use_reverse_complement": True,
                    "output_gff": str(gff_path),
                    "metrics_json": str(metrics_path),
                    "empty_gff_policy": "error",
                }
            )

            record = {
                "chrom": "NC_060944.1",
                "start": 1,
                "end": 4,
                "strand": "+",
            }
            with (
                patch.object(finding_infer, "predict_tracks", side_effect=predictions),
                patch.object(
                    finding_infer,
                    "peak_finding_indices",
                    return_value=[np.asarray([1]), np.asarray([3]), np.asarray([]), np.asarray([])],
                ),
                patch.object(
                    finding_infer,
                    "find_tss_polya_pairs_from_peak_indices",
                    return_value=[record],
                ),
                patch.object(
                    finding_infer,
                    "filter_intervals_by_intragenic_bool",
                    return_value=[record],
                ),
                patch.object(
                    finding_infer,
                    "evaluate_annotation",
                    return_value={"official_metric": 1.0},
                ),
            ):
                finding_infer._run_full_pipeline(cfg)

            features = [
                line.split("\t")
                for line in gff_path.read_text().splitlines()
                if line and not line.startswith("#")
            ]
            self.assertEqual([row[2] for row in features], ["gene", "mRNA", "exon"])
            self.assertEqual(len({(row[3], row[4]) for row in features}), 1)
            metrics = json.loads(metrics_path.read_text())
            self.assertIn("edge", metrics["pr_auc"])
            self.assertIn("region", metrics["pr_auc"])
            self.assertEqual(metrics["annotation"], {"official_metric": 1.0})


if __name__ == "__main__":
    unittest.main()
