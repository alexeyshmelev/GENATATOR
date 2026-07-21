from __future__ import annotations

import json
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[1]
FIXED_GENOME = ["GCF_009914755.1"]
FIXED_CHROMOSOME = ["NC_060944.1"]


def _uses_unet(model: dict) -> bool:
    return model.get("family") in {"unet", "rmt"} or (
        model.get("family") == "amt" and bool(model.get("use_unet", False))
    )


def _training_paths(task: str):
    for path in sorted((REPO / task / "configs").glob("*.json")):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(cfg.get("training"), dict):
            yield path, cfg


def test_all_shipped_training_configs_use_requested_contracts() -> None:
    for task in ("finding", "segmentation", "transcript_type"):
        for path, cfg in _training_paths(task):
            model = cfg["model"]
            training = cfg["training"]
            assert cfg.get("true_gff") is None, path
            assert training.get("custom_prefix") is not None, path
            assert training["max_steps"] == 500_000, path
            assert training["eval_steps"] == 1000, path
            assert training["save_steps"] == 1000, path
            assert training["patience"] == 100, path
            assert "eval_interval" not in training, path
            assert "save_interval" not in training, path
            assert "nucleotide_vocab_size" not in model, path
            assert "cycles" not in path.name, path
            if model["family"] == "caduceus":
                assert model["bidirectional_weight_tie"] is False, path
            if model["family"] == "rmt":
                assert model["cycles"] == 1, path
                assert model["rmt"]["segment_size"] == (512 if model["backbone_kind"] == "gena" else 1024), path
                assert model["rmt"]["num_mem_tokens"] == (10 if model["backbone_kind"] == "gena" else 20), path
                assert model["rmt"]["max_n_segments"] > 0, path
            if model["family"] == "amt":
                assert model["amt"]["num_mem_tokens"] == (10 if model["backbone_kind"] == "gena" else 20), path
                assert model["amt"]["segment_size"] == (502 if model["backbone_kind"] == "gena" else 1004), path
            if model["family"] == "unet":
                assert model["unet_cycles"] == 1, path
            if model["family"] == "amt" and model.get("use_unet"):
                assert model["unet_cycles"] == 1, path
            if _uses_unet(model):
                assert "vocab_size" in model, path
                assert model["unet_chunk_size"] == 8192, path

            for dataset_name in ("train_dataset", "eval_dataset"):
                dataset = cfg[dataset_name]
                if model["family"] == "caduceus":
                    assert dataset["max_nucleotides"] == 32768, path
                    assert "max_bpe_tokens" not in dataset, path
                else:
                    assert dataset["max_bpe_tokens"] > 0, path
                    assert dataset["average_bpe_token_length"] > 0, path
                    if model["backbone_kind"] == "gena" and model["family"] in {"plain", "unet"}:
                        assert dataset["max_bpe_tokens"] <= 512, path
                    else:
                        resolved_nt = int(dataset["max_bpe_tokens"] * dataset["average_bpe_token_length"])
                        assert 30000 <= resolved_nt <= 40000, path
                if task != "finding":
                    assert "overlap" not in dataset, path

            if task == "segmentation":
                expected_random = model["family"] == "caduceus"
                assert cfg["train_dataset"]["random_crop"] is expected_random, path
                assert cfg["eval_dataset"]["random_crop"] is expected_random, path
                assert cfg["train_dataset"]["statuses"] == [1], path
                assert cfg["eval_dataset"]["statuses"] == [1], path


def test_config_matrix_is_complete() -> None:
    backbones = {"gena_base", "gena_large", "moderngena_base", "moderngena_large"}
    segmentation_expected = {
        f"{backbone}_{variant}.json"
        for backbone in backbones
        for variant in ("unet", "rmt_unet", "amt_unet")
    }
    segmentation_actual = {p.name for p, _ in _training_paths("segmentation")}
    assert segmentation_expected <= segmentation_actual

    for stage in ("edge", "region"):
        expected = {
            f"{stage}_{backbone}_{variant}.json"
            for backbone in backbones
            for variant in ("plain", "unet", "rmt_unet", "amt_plain", "amt_unet")
        }
        actual = {p.name for p, _ in _training_paths("finding") if p.name.startswith(f"{stage}_")}
        assert expected <= actual

    transcript_expected = {f"{backbone}_plain.json" for backbone in backbones}
    transcript_actual = {p.name for p, _ in _training_paths("transcript_type")}
    assert transcript_expected <= transcript_actual


def test_standalone_evaluation_templates_use_required_subsets() -> None:
    segmentation = json.loads((REPO / "segmentation/configs/infer_caduceus_ps.json").read_text())
    assert segmentation["dataset"]["config_name"] == "val-human"
    assert segmentation["dataset"]["split"] == "validation"
    assert segmentation["dataset"]["genomes"] == FIXED_GENOME
    assert segmentation["dataset"]["chromosomes"] == FIXED_CHROMOSOME
    assert "statuses" not in segmentation["dataset"]
    assert segmentation["dataset"]["full_transcript_chunks"] is True
    assert segmentation["inference"]["use_reverse_complement"] is True

    transcript = json.loads((REPO / "transcript_type/configs/infer_moderngena_base.json").read_text())
    assert transcript["dataset"]["config_name"] == "val-human"
    assert transcript["dataset"]["split"] == "validation"
    assert transcript["dataset"]["genomes"] == FIXED_GENOME
    assert transcript["dataset"]["chromosomes"] == FIXED_CHROMOSOME
    assert "statuses" not in transcript["dataset"]
    assert transcript["inference"]["use_reverse_complement"] is True

    finding = json.loads((REPO / "finding/configs/infer_moderngena_base_plain.json").read_text())
    for stage in ("edge", "region"):
        assert finding[stage]["dataset"]["split"] == "test"
        assert finding[stage]["dataset"]["genomes"] == FIXED_GENOME
        assert finding[stage]["dataset"]["chromosomes"] == FIXED_CHROMOSOME
    assert finding["inference"]["use_reverse_complement"] is True
    assert finding["edge"]["inference"]["checkpoint_path"] == "<manually_insert_value_here>"
    assert finding["region"]["inference"]["checkpoint_path"] == "<manually_insert_value_here>"
    assert finding["inference"]["true_gff"] == "<manually_insert_value_here>"
    assert finding["inference"]["k_values"] == [0, 50, 100, 250, 500]
    assert finding["inference"]["use_strand"] is True
    assert finding["postprocess"] == {
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


class ShippedConfigSchemaTests(unittest.TestCase):
    def test_requested_contracts(self) -> None:
        test_all_shipped_training_configs_use_requested_contracts()

    def test_complete_matrix(self) -> None:
        test_config_matrix_is_complete()

    def test_evaluation_templates(self) -> None:
        test_standalone_evaluation_templates_use_required_subsets()


if __name__ == "__main__":
    unittest.main()
