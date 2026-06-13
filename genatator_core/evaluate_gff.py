from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import evaluate


def evaluate_annotation(pred_gff: str, true_gff: str, output_json: str, k_values: List[int], use_strand: bool = True) -> Dict:
    metric = evaluate.load("AIRI-Institute/genatator-ab-initio-annotation-leaderboard")
    result = metric.compute(pred_gff=pred_gff, true_gff=true_gff, k_values=k_values, use_strand=use_strand)
    import json
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def evaluate_segmentation(pred_gff: str, true_gff: str, output_json: str) -> Dict:
    metric = evaluate.load("AIRI-Institute/genatator-ab-initio-segmentation-leaderboard", revision="metric-only")
    result = metric.compute_gene_level_gff(pred_gff=pred_gff, true_gff=true_gff, stratifier="type", types=["mRNA", "lnc_RNA"], segments=["exon", "CDS"])
    import json
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
