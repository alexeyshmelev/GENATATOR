"""Reproduce the independent edge/region minima in the supplied score tables."""

import csv
from decimal import Decimal
import json
from pathlib import Path


GROUPS = ("short_human_mrna_lncrna", "long_human_mrna_lncrna", "long_human_mrna")
CHANNELS = {
    "edge": ("TSS+", "TSS-", "PolyA+", "PolyA-"),
    "region": ("intragenic+", "intragenic-"),
}


def select_pair(path):
    candidates = {stage: [] for stage in CHANNELS}
    names = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            name = row["model"]
            if name in names:
                raise ValueError(f"Duplicate model: {name}")
            names.add(name)
            stage = name.split("_", 1)[0]
            if stage not in CHANNELS:
                raise ValueError(f"Unknown model stage: {name}")
            scores = [Decimal(row[f"roc_auc_{channel}"]) for channel in CHANNELS[stage]]
            if any(not x.is_finite() or not 0 <= x <= 1 for x in scores):
                raise ValueError(f"Invalid ROC-AUC values for {name}")
            candidates[stage].append((sum(scores) / len(scores), name, scores))
    result = {}
    for stage, rows in candidates.items():
        if not rows:
            raise ValueError(f"No {stage} models in {path}")
        mean, name, scores = min(rows, key=lambda row: (row[0], row[1]))
        result[stage] = {
            "model": name,
            "mean_roc_auc": str(mean),
            "channel_roc_aucs": dict(zip(CHANNELS[stage], map(str, scores))),
            "candidate_count": len(rows),
        }
    return result


def all_selections():
    source = Path(__file__).resolve().parent / "sources"
    return {group: select_pair(source / f"{group}_highest_roc_aucs.csv") for group in GROUPS}


if __name__ == "__main__":
    print(json.dumps(all_selections(), indent=2))
