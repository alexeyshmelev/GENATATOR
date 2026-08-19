from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINING_CONFIG_ROOTS = {
    "finding": ROOT / "finding" / "configs",
    "segmentation": ROOT / "segmentation" / "configs",
    "transcript_type": ROOT / "transcript_type" / "configs",
    "experiments": ROOT / "experiments",
}
EXPECTED_TRAINING_CONFIG_COUNTS = {
    "finding": 176,
    "segmentation": 28,
    "transcript_type": 42,
    "experiments": 12,
}


def test_all_shipped_training_configs_clip_gradients_at_one() -> None:
    actual_counts = {}
    for group, root in TRAINING_CONFIG_ROOTS.items():
        paths = []
        for path in sorted(root.rglob("*.json")):
            cfg = json.loads(path.read_text(encoding="utf-8"))
            training = cfg.get("training")
            if not isinstance(training, dict):
                continue
            paths.append(path)
            assert isinstance(training["max_grad_norm"], float), path
            assert training["max_grad_norm"] == 1.0, path
        actual_counts[group] = len(paths)

    assert actual_counts == EXPECTED_TRAINING_CONFIG_COUNTS
