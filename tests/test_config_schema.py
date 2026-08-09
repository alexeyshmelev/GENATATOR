from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[1]
FINDING_TEST_GENOME = ["GCF_009914755.1_T2T-CHM13v2.0"]
FIXED_CHROMOSOME = ["NC_060944.1"]
MANUAL_PLACEHOLDER = "<manually_insert_value_here>"

FINDING_SETUPS = {
    "short_human_mrna_lncrna": {
        "short": True,
        "train_genomes": ["hg38"],
        "train_target_group": "primary",
    },
    "long_human_mrna_lncrna": {
        "short": False,
        "train_genomes": ["hg38"],
        "train_target_group": "primary",
    },
    "long_human_mrna": {
        "short": False,
        "train_genomes": ["hg38"],
        "train_target_group": "mrna",
    },
    "long_multispecies_mrna_lncrna": {
        "short": False,
        "train_genomes": [],
        "train_target_group": "primary",
    },
}

TRANSCRIPT_SETUPS = {
    "short_human": {"short": True, "train_config_name": "train-human"},
    "long_human": {"short": False, "train_config_name": "train-human"},
    "long_multispecies": {
        "short": False,
        "train_config_name": "train-multi-specie",
    },
}

FINDING_POSTPROCESS = {
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


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict), path
    return value


def _training_paths(task: str):
    for path in sorted((REPO / task / "configs").rglob("*.json")):
        cfg = _load(path)
        if isinstance(cfg.get("training"), dict):
            yield path, cfg


def _uses_unet(model: dict, task: str) -> bool:
    if task == "transcript_type" or model.get("family") == "gpt":
        return False
    return model.get("family") in {"unet", "rmt"} or (
        model.get("family") == "amt" and bool(model.get("use_unet", False))
    )


def _expected_context(task: str, path: Path, model: dict) -> tuple[str, int]:
    family = model["family"]
    backbone = model.get("backbone_kind", family)
    if task == "finding":
        short = FINDING_SETUPS[path.parent.name]["short"]
    elif task == "transcript_type":
        short = TRANSCRIPT_SETUPS[path.parent.name]["short"]
    else:
        short = False

    if task == "segmentation":
        if backbone == "caduceus":
            return "max_nucleotides", 250_000
        return "max_bpe_tokens", 30_000

    if family == "caduceus":
        return "max_nucleotides", 8192 if short else 32768
    if backbone == "gena" and family in {"plain", "unet"}:
        return "max_bpe_tokens", 512
    return "max_bpe_tokens", 1024 if short else 4096


def _keys_named(value, key: str) -> list[object]:
    found = []
    if isinstance(value, dict):
        if key in value:
            found.append(value[key])
        for child in value.values():
            found.extend(_keys_named(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_keys_named(child, key))
    return found


def test_all_shipped_training_configs_use_requested_contracts() -> None:
    for task in ("finding", "segmentation", "transcript_type"):
        for path, cfg in _training_paths(task):
            model = cfg["model"]
            training = cfg["training"]
            assert cfg.get("true_gff") == "datasets/chr20.gff", path
            assert training.get("custom_prefix") is not None, path
            assert training["max_steps"] == 500_000, path
            assert training["eval_steps"] == 5000, path
            assert training["save_steps"] == 5000, path
            assert training["patience"] == 50, path
            assert training["automatic_restart"] is True, path
            assert "eval_interval" not in training, path
            assert "save_interval" not in training, path
            assert "nucleotide_vocab_size" not in model, path
            assert "cycles" not in path.name, path

            family = model["family"]
            backbone = model.get("backbone_kind", family)
            is_gpt = family == "gpt"
            wrapper_family = family
            if is_gpt:
                wrapper_family = (
                    "rmt" if "rmt" in model else "amt" if "amt" in model else None
                )
            assert "head_kind" not in model, path
            if backbone == "caduceus":
                assert model["bidirectional_weight_tie"] is False, path
            if wrapper_family == "rmt":
                assert model["rmt"]["segment_size"] == (
                    512 if backbone == "gena" else 1024
                ), path
                assert model["rmt"]["num_mem_tokens"] == (
                    10 if backbone == "gena" else 20
                ), path
                assert model["rmt"]["max_n_segments"] > 0, path
                if task == "transcript_type" or is_gpt:
                    assert "cycles" not in model, path
                    assert "unet_chunk_size" not in model, path
                    if task == "transcript_type":
                        assert "vocab_size" not in model, path
                else:
                    assert model["cycles"] == 1, path
            if wrapper_family == "amt":
                assert model["amt"]["num_mem_tokens"] == (
                    10 if backbone == "gena" else 20
                ), path
                assert model["amt"]["segment_size"] == (
                    502 if backbone == "gena" else 1004
                ), path
                assert model["amt"]["d_mem"] == 64, path
                if task == "transcript_type" or is_gpt:
                    assert model["use_unet"] is False, path
                    assert "unet_chunk_size" not in model, path
                    if task == "transcript_type":
                        assert "vocab_size" not in model, path
            if family == "unet":
                assert model["unet_cycles"] == 1, path
            if wrapper_family == "amt" and model.get("use_unet"):
                assert model["unet_cycles"] == 1, path
            if _uses_unet(model, task):
                assert "vocab_size" in model, path
                assert model["unet_chunk_size"] == 8192, path
            if is_gpt:
                assert task == "segmentation", path
                assert "vocab_size" in model, path
                assert model["gpt"] == {
                    "context_size": 8192,
                    "encoder_lookahead": 8192,
                    "decoder_hidden_size": 256,
                    "decoder_intermediate_size": 1024,
                    "num_decoder_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "dropout_rate": 0.0,
                    "attention_dropout": 0.0,
                    "multi_token_prediction": 3,
                }, path

            length_field, expected_length = _expected_context(task, path, model)
            for dataset_name in ("train_dataset", "eval_dataset"):
                dataset = cfg[dataset_name]
                assert dataset[length_field] == expected_length, path
                if length_field == "max_nucleotides":
                    assert "max_bpe_tokens" not in dataset, path
                    assert "average_bpe_token_length" not in dataset, path
                else:
                    assert dataset["average_bpe_token_length"] == 9.0, path
                    assert "max_nucleotides" not in dataset, path
                if task != "finding":
                    assert "overlap" not in dataset, path

            if task == "segmentation":
                expected_random = backbone == "caduceus"
                assert cfg["train_dataset"]["random_crop"] is expected_random, path
                assert cfg["eval_dataset"]["random_crop"] is expected_random, path
                assert cfg["train_dataset"]["statuses"] == [1], path
                assert cfg["eval_dataset"]["statuses"] == [1], path


def test_config_matrices_and_setup_semantics_are_exact() -> None:
    finding_root = REPO / "finding" / "configs"
    transcript_root = REPO / "transcript_type" / "configs"
    assert not list(finding_root.glob("*.json"))
    assert not list(transcript_root.glob("*.json"))
    assert {path.name for path in finding_root.iterdir() if path.is_dir()} == set(
        FINDING_SETUPS
    )
    assert {path.name for path in transcript_root.iterdir() if path.is_dir()} == set(
        TRANSCRIPT_SETUPS
    )

    backbones = {"gena_base", "gena_large", "moderngena_base", "moderngena_large"}
    finding_stage_names = {
        *{f"{backbone}_{variant}" for backbone in backbones for variant in (
            "plain", "unet", "rmt_unet", "amt_plain", "amt_unet"
        )},
        "caduceus_ph",
        "caduceus_ps",
    }
    expected_finding = {
        f"{stage}_{model}.json"
        for stage in ("edge", "region")
        for model in finding_stage_names
    }
    for setup_name, setup in FINDING_SETUPS.items():
        paths = sorted((finding_root / setup_name).glob("*.json"))
        assert len(paths) == 44
        assert {path.name for path in paths} == expected_finding
        for path in paths:
            cfg = _load(path)
            assert cfg["train_dataset"]["genomes"] == setup["train_genomes"], path
            assert cfg["eval_dataset"]["genomes"] == ["hg38"], path
            assert cfg["train_dataset"]["target_group"] == setup["train_target_group"], path
            assert cfg["eval_dataset"]["target_group"] == "primary", path

    expected_transcript = {
        *{
            f"{backbone}_{variant}.json"
            for backbone in backbones
            for variant in ("plain", "rmt_plain", "amt_plain")
        },
        "caduceus_ph_middle_loss.json",
        "caduceus_ps_middle_loss.json",
    }
    for setup_name, setup in TRANSCRIPT_SETUPS.items():
        paths = sorted((transcript_root / setup_name).glob("*.json"))
        assert len(paths) == 14
        assert {path.name for path in paths} == expected_transcript
        for path in paths:
            cfg = _load(path)
            assert cfg["train_dataset"]["config_name"] == setup["train_config_name"], path
            assert cfg["eval_dataset"]["config_name"] == "val-human", path

    segmentation_expected = {
        f"{backbone}_{variant}.json"
        for backbone in backbones
        for variant in ("unet", "rmt_unet", "amt_unet")
    }
    segmentation_expected |= {
        f"{backbone}_{variant}.json"
        for backbone in backbones
        for variant in ("gpt", "rmt_gpt", "amt_gpt")
    }
    segmentation_expected |= {"caduceus_ph_gpt.json", "caduceus_ps_gpt.json"}
    segmentation_actual = {
        path.name for path, _ in _training_paths("segmentation")
    }
    assert segmentation_expected <= segmentation_actual


def test_target_group_is_exclusive_to_finding_configs() -> None:
    for task in ("segmentation", "transcript_type"):
        for path in sorted((REPO / task / "configs").rglob("*.json")):
            assert not _keys_named(_load(path), "target_group"), path
    for path, cfg in _training_paths("finding"):
        values = _keys_named(cfg, "target_group")
        assert values == [
            FINDING_SETUPS[path.parent.name]["train_target_group"],
            "primary",
        ], path


def test_no_checked_in_inference_config_exists_outside_experiments() -> None:
    outside = []
    for task in ("finding", "segmentation", "transcript_type"):
        for path in sorted((REPO / task / "configs").rglob("*.json")):
            cfg = _load(path)
            if "inference" in cfg and "training" not in cfg:
                outside.append(path)
            if path.name.startswith("infer_") or path.name == "evaluation_config.json":
                outside.append(path)
    assert not outside


def test_massive_finding_experiments_are_exact_cartesian_transfers() -> None:
    finding_root = REPO / "finding" / "configs"
    experiment_root = REPO / "experiments" / "massive_gene_finding_evaluation"
    assert {path.name for path in experiment_root.iterdir() if path.is_dir()} == set(
        FINDING_SETUPS
    )

    output_paths = set()
    for setup_name in FINDING_SETUPS:
        training_root = finding_root / setup_name
        edge_sources = {
            path.stem: _load(path) for path in training_root.glob("edge_*.json")
        }
        region_sources = {
            path.stem: _load(path) for path in training_root.glob("region_*.json")
        }
        expected_names = {
            f"{edge}__{region}.json"
            for edge in edge_sources
            for region in region_sources
        }
        paths = sorted((experiment_root / setup_name).glob("*.json"))
        assert len(paths) == 484
        assert {path.name for path in paths} == expected_names

        for path in paths:
            cfg = _load(path)
            edge_name, region_name = path.stem.split("__", 1)
            for stage, source in (
                ("edge", edge_sources[edge_name]),
                ("region", region_sources[region_name]),
            ):
                stage_cfg = cfg[stage]
                assert stage_cfg["model"] == source["model"], path
                assert stage_cfg["model"]["checkpoint_path"] is None, path
                assert stage_cfg["inference"] == {
                    "checkpoint_path": MANUAL_PLACEHOLDER,
                    "batch_size": 1,
                }, path
                expected_dataset = copy.deepcopy(source["eval_dataset"])
                expected_dataset.update(
                    split="test",
                    genomes=FINDING_TEST_GENOME,
                    chromosomes=FIXED_CHROMOSOME,
                )
                expected_dataset.pop("statuses", None)
                assert stage_cfg["dataset"] == expected_dataset, path

            assert cfg["task"] == "finding", path
            assert cfg["postprocess"] == FINDING_POSTPROCESS, path
            inference = cfg["inference"]
            assert inference["true_gff"] == MANUAL_PLACEHOLDER, path
            assert inference["batch_size"] == 1, path
            assert inference["use_reverse_complement"] is True, path
            pair_root = (
                f"runs/massive_gene_finding_evaluation/{setup_name}/{path.stem}"
            )
            expected_outputs = {
                "output_gff": f"{pair_root}/finding_predictions.gff",
                "metrics_json": f"{pair_root}/finding_metrics.json",
                "metrics_csv": f"{pair_root}/finding_auc_metrics.csv",
            }
            for field, expected in expected_outputs.items():
                assert inference[field] == expected, path
                assert expected not in output_paths, (path, field)
                output_paths.add(expected)

    assert len(output_paths) == 4 * 484 * 3
    small = _load(
        REPO / "experiments" / "small_finding_evaluation_v1" / "evaluation_config.json"
    )
    for stage in ("edge", "region"):
        assert small[stage]["dataset"]["genomes"] == FINDING_TEST_GENOME
        assert small[stage]["dataset"]["chromosomes"] == FIXED_CHROMOSOME


class ShippedConfigSchemaTests(unittest.TestCase):
    def test_requested_contracts(self) -> None:
        test_all_shipped_training_configs_use_requested_contracts()

    def test_complete_matrix(self) -> None:
        test_config_matrices_and_setup_semantics_are_exact()

    def test_target_group_scope(self) -> None:
        test_target_group_is_exclusive_to_finding_configs()

    def test_inference_template_locations(self) -> None:
        test_no_checked_in_inference_config_exists_outside_experiments()

    def test_massive_experiments(self) -> None:
        test_massive_finding_experiments_are_exact_cartesian_transfers()


if __name__ == "__main__":
    unittest.main()
