from __future__ import annotations

import json
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[1]


def _uses_unet(model: dict) -> bool:
    return model.get("family") in {"unet", "rmt"} or (
        model.get("family") == "amt" and bool(model.get("use_unet", False))
    )


def _model_dataset_pairs(path: Path, cfg: dict):
    task_group = path.relative_to(REPO).parts[0]
    if "train_dataset" in cfg:
        transcript_task = task_group in {"segmentation", "transcript_type"}
        return [
            (cfg["model"], cfg["train_dataset"], transcript_task),
            (cfg["model"], cfg["eval_dataset"], transcript_task),
        ]
    if task_group == "finding":
        return [
            (cfg[stage]["model"], cfg[stage]["dataset"], False)
            for stage in ("edge", "region")
        ]
    return [(cfg["model"], cfg["dataset"], True)]


def test_all_shipped_configs_use_canonical_length_and_unet_fields() -> None:
    paths = sorted(
        path
        for task in ("finding", "segmentation", "transcript_type")
        for path in (REPO / task / "configs").glob("*.json")
    )
    assert paths
    for path in paths:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if "training" in cfg:
            assert "custom_prefix" in cfg["training"], path
        for model, dataset, transcript_task in _model_dataset_pairs(path, cfg):
            if model["family"] == "caduceus":
                assert "max_nucleotides" in dataset, path
                assert "max_bpe_tokens" not in dataset, path
                assert "average_bpe_token_length" not in dataset, path
            else:
                assert "max_nucleotides" not in dataset, path
                assert dataset["max_bpe_tokens"] > 0, path
                assert dataset["average_bpe_token_length"] > 0, path
            if transcript_task:
                assert "overlap" not in dataset, path
                assert "random_crop" not in dataset, path
            if _uses_unet(model):
                assert model["unet_chunk_size"] == 8192, path
                assert "unet_sub_model_input_size" not in model.get("rmt", {}), path


def test_training_status_filter_and_full_segmentation_inference_are_distinct() -> None:
    for task in ("segmentation", "transcript_type"):
        for path in (REPO / task / "configs").glob("*.json"):
            cfg = json.loads(path.read_text(encoding="utf-8"))
            if "training" in cfg:
                assert cfg["train_dataset"]["statuses"] == [1], path
                assert cfg["eval_dataset"]["statuses"] == [1], path
    segmentation_infer = json.loads(
        (REPO / "segmentation/configs/infer_caduceus_ps.json").read_text(encoding="utf-8")
    )
    assert "statuses" not in segmentation_infer["dataset"]
    assert segmentation_infer["inference"]["coordinate_mode"] == "transcript"
    assert isinstance(segmentation_infer["inference"]["use_cds_heuristic"], bool)


class ShippedConfigSchemaTests(unittest.TestCase):
    def test_canonical_length_and_unet_fields(self) -> None:
        test_all_shipped_configs_use_canonical_length_and_unet_fields()

    def test_training_and_standalone_status_policies(self) -> None:
        test_training_status_filter_and_full_segmentation_inference_are_distinct()


if __name__ == "__main__":
    unittest.main()
