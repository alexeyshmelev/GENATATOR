#!/usr/bin/env python
from argparse import ArgumentParser
import csv
import json
from pathlib import Path

import evaluate
import numpy as np

from genatator_core.config import load_json
from genatator_core.infer_common import predict_dataset_logits, sigmoid

parser = ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
cfg = load_json(args.config)
rows = predict_dataset_logits(cfg, task="transcript_type", device=cfg.get("inference", {}).get("device", "cuda"))
out_tsv = Path(cfg["inference"].get("output_tsv", "transcript_type_predictions.tsv"))
out_tsv.parent.mkdir(parents=True, exist_ok=True)
y_true, y_pred = [], []
with open(out_tsv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["transcript_id", "gene_id", "genome", "chrom", "start", "end", "true_type", "lnc_probability", "pred_type"])
    for r in rows:
        meta = r["metadata"]
        prob = float(sigmoid(np.asarray(r["logits"]).reshape(-1))[0])
        pred = "lnc_RNA" if prob >= float(cfg.get("inference", {}).get("threshold", 0.5)) else "mRNA"
        true = "lnc_RNA" if meta.transcript_type.lower() in {"lnc_rna", "lncrna"} else "mRNA"
        y_true.append(1 if true == "lnc_RNA" else 0)
        y_pred.append(1 if pred == "lnc_RNA" else 0)
        w.writerow([meta.transcript_id, meta.gene_id, meta.genome, meta.chrom, meta.start, meta.end, true, prob, pred])
metrics = {
    "accuracy": evaluate.load("accuracy").compute(predictions=y_pred, references=y_true)["accuracy"],
    "f1": evaluate.load("f1").compute(predictions=y_pred, references=y_true)["f1"],
    "precision": evaluate.load("precision").compute(predictions=y_pred, references=y_true)["precision"],
    "recall": evaluate.load("recall").compute(predictions=y_pred, references=y_true)["recall"],
}
metrics_path = Path(cfg["inference"].get("metrics_json", out_tsv.with_suffix(".metrics.json")))
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
