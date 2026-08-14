from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "experiments" / "first_gpt_model_experiments"
MULTI_TOKEN_PREDICTIONS = (1, 2, 5, 10, 50, 100)
FAMILIES = {
    "caduceus_ps_gpt": {
        "backbone_kind": "caduceus",
        "length_field": "max_nucleotides",
        "length": 32768,
    },
    "moderngena_base_gpt": {
        "backbone_kind": "moderngena",
        "length_field": "max_bpe_tokens",
        "length": 8192,
    },
}


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict), path
    return value


class GPTExperimentConfigTests(unittest.TestCase):
    def test_first_gpt_experiment_matrix_is_exact_and_parallel_safe(self) -> None:
        expected_names = {
            f"{family}_mtp_{multi_token_prediction}.json"
            for family in FAMILIES
            for multi_token_prediction in MULTI_TOKEN_PREDICTIONS
        }
        paths = sorted(EXPERIMENT_ROOT.glob("*.json"))
        self.assertEqual({path.name for path in paths}, expected_names)
        self.assertEqual(len(paths), 12)

        output_dirs = set()
        for family, specification in FAMILIES.items():
            normalized_reference = None
            for multi_token_prediction in MULTI_TOKEN_PREDICTIONS:
                path = EXPERIMENT_ROOT / (
                    f"{family}_mtp_{multi_token_prediction}.json"
                )
                cfg = _load(path)
                self.assertEqual(cfg["task"], "segmentation", path)
                self.assertEqual(
                    cfg["model"]["backbone_kind"],
                    specification["backbone_kind"],
                    path,
                )
                self.assertEqual(cfg["model"]["gpt"]["num_decoder_layers"], 4, path)
                self.assertEqual(
                    cfg["model"]["gpt"]["multi_token_prediction"],
                    multi_token_prediction,
                    path,
                )
                self.assertEqual(cfg["train_dataset"]["config_name"], "train-human", path)
                self.assertEqual(cfg["eval_dataset"]["config_name"], "val-human", path)
                for dataset_name in ("train_dataset", "eval_dataset"):
                    dataset = cfg[dataset_name]
                    self.assertEqual(
                        dataset[specification["length_field"]],
                        specification["length"],
                        path,
                    )

                expected_output = (
                    "runs/experiments/first_gpt_model_experiments/"
                    f"{path.stem}"
                )
                output_dir = cfg["training"]["output_dir"]
                self.assertEqual(output_dir, expected_output, path)
                self.assertNotIn(output_dir, output_dirs, path)
                output_dirs.add(output_dir)

                # Within one encoder family, only the requested MTP setting and
                # its collision-free output namespace may vary.
                normalized = deepcopy(cfg)
                normalized["model"]["gpt"].pop("multi_token_prediction")
                normalized["training"].pop("output_dir")
                if normalized_reference is None:
                    normalized_reference = normalized
                else:
                    self.assertEqual(normalized, normalized_reference, path)

        self.assertEqual(len(output_dirs), 12)

    def test_all_moderngena_amt_training_configs_use_fp32(self) -> None:
        selected = []
        for task in ("finding", "segmentation", "transcript_type"):
            for path in sorted((ROOT / task / "configs").rglob("*.json")):
                cfg = _load(path)
                model = cfg.get("model", {})
                uses_amt = model.get("family") == "amt" or (
                    model.get("family") == "gpt"
                    and isinstance(model.get("amt"), dict)
                )
                if model.get("backbone_kind") != "moderngena" or not uses_amt:
                    continue
                selected.append(path)
                self.assertIs(cfg["training"]["bf16"], False, path)
                self.assertIs(cfg["training"]["fp16"], False, path)

        self.assertEqual(len(selected), 42)


if __name__ == "__main__":
    unittest.main()
