#!/usr/bin/env python
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from genatator_core.config import load_json
from genatator_core.evaluate_gff import evaluate_segmentation
from genatator_core.gff import labels_to_segmentation_record, write_segmentation_gff
from genatator_core.infer_common import predict_dataset_logits, sigmoid

parser = ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
cfg = load_json(args.config)
rows = predict_dataset_logits(cfg, task="segmentation", device=cfg.get("inference", {}).get("device", "cuda"))
records = []
for r in rows:
    probs = sigmoid(r["logits"])
    rec = labels_to_segmentation_record(r["metadata"], probs, threshold=float(cfg.get("inference", {}).get("threshold", 0.5)))
    rec["start"] = int(r["metadata"].start) + int(r["local_start"])
    rec["end"] = rec["start"] + probs.shape[0]
    records.append(rec)
out_gff = cfg["inference"]["output_gff"]
write_segmentation_gff(records, out_gff)
if cfg["inference"].get("true_gff"):
    evaluate_segmentation(out_gff, cfg["inference"]["true_gff"], cfg["inference"].get("metrics_json", str(Path(out_gff).with_suffix(".metrics.json"))) )
