from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "segmentation" / "configs"


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict), path
    return value


def test_segmentation_training_matrix_uses_multispecies_long_contexts() -> None:
    paths = sorted(CONFIG_ROOT.glob("*.json"))
    assert len(paths) == 28

    counts = {"caduceus": 0, "gena": 0, "moderngena": 0}
    for path in paths:
        cfg = _load(path)
        assert cfg["task"] == "segmentation", path
        assert isinstance(cfg.get("training"), dict), path

        train = cfg["train_dataset"]
        validation = cfg["eval_dataset"]
        assert train["config_name"] == "train-multi-specie", path
        assert train["split"] == "train", path
        assert validation["config_name"] == "val-human", path
        assert validation["split"] == "validation", path

        backbone = cfg["model"].get("backbone_kind", cfg["model"]["family"])
        assert backbone in counts, path
        counts[backbone] += 1

        for dataset in (train, validation):
            if backbone == "caduceus":
                assert dataset["max_nucleotides"] == 250_000, path
                assert "max_bpe_tokens" not in dataset, path
                assert "average_bpe_token_length" not in dataset, path
            else:
                assert dataset["max_bpe_tokens"] == 30_000, path
                assert dataset["average_bpe_token_length"] == 9.0, path
                assert "max_nucleotides" not in dataset, path

    assert counts == {"caduceus": 4, "gena": 12, "moderngena": 12}
