#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path
import logging

from genatator_core.config import load_json
from genatator_core.evaluate_gff import evaluate_segmentation
from genatator_core.gff import labels_to_segmentation_record, write_segmentation_gff
from genatator_core.infer_common import predict_dataset_logits, sigmoid

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
cfg = load_json(args.config)
rows = predict_dataset_logits(cfg, task="segmentation", device=cfg.get("inference", {}).get("device", "cuda"))
records = []
force_nonempty = cfg.get("inference", {}).get("empty_segment_policy", "error") == "best_interval"
for r in rows:
    probs = sigmoid(r["logits"])
    rec = labels_to_segmentation_record(
        r["metadata"],
        probs,
        threshold=float(cfg.get("inference", {}).get("threshold", 0.5)),
        force_nonempty=force_nonempty,
    )
    # For the official segmentation metric, prediction GFF coordinates are
    # transcript-relative and seqid is transcript_id. local_start is still kept
    # in case users choose coordinate_mode='genome', but the default is
    # coordinate_mode='transcript'.
    rec["local_start"] = int(r.get("local_start", 0))
    if cfg.get("inference", {}).get("coordinate_mode", "transcript") == "genome":
        rec["start"] = int(r["metadata"].start) + int(r["local_start"])
        rec["end"] = rec["start"] + probs.shape[0]
    else:
        rec["start"] = 0
        rec["end"] = probs.shape[0]
        rec["transcript_length"] = probs.shape[0]
    if force_nonempty and not rec.get("exons"):
        raise RuntimeError(f"empty_segment_policy=best_interval failed to create an exon for transcript_id={rec.get('transcript_id')}")
    records.append(rec)

if not records:
    raise RuntimeError("Segmentation inference produced zero transcript records; cannot run official metric on an empty GFF.")

out_gff = cfg["inference"]["output_gff"]
write_segmentation_gff(records, out_gff, coordinate_mode=cfg.get("inference", {}).get("coordinate_mode", "transcript"))
if cfg["inference"].get("true_gff"):
    evaluate_segmentation(out_gff, cfg["inference"]["true_gff"], cfg["inference"].get("metrics_json", str(Path(out_gff).with_suffix(".metrics.json"))))
