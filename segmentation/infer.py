#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path
import logging

from genatator_core.cds_heuristic import infer_cds_with_benchmark_heuristic
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
coordinate_mode = cfg.get("inference", {}).get("coordinate_mode", "transcript")
use_cds_heuristic = bool(cfg.get("inference", {}).get("use_cds_heuristic", True))
logger.info(
    "Segmentation decoding | coordinate_mode=%s use_cds_heuristic=%s",
    coordinate_mode,
    use_cds_heuristic,
)
for r in rows:
    probs = sigmoid(r["logits"])
    rec = labels_to_segmentation_record(
        r["metadata"],
        probs,
        threshold=float(cfg.get("inference", {}).get("threshold", 0.5)),
        force_nonempty=force_nonempty,
    )
    if use_cds_heuristic and rec["transcript_type"] == "mRNA":
        # ``rec`` intervals are local to this exact model input. Restrict the
        # sequence to the covered prediction span before applying the same
        # exon-splicing/ORF heuristic as GENATATOR-PIPELINE.
        sequence = str(r["dna_sequence"])[: probs.shape[0]]
        rec["cds"] = infer_cds_with_benchmark_heuristic(
            sequence=sequence,
            interval_start=0,
            exons=rec["exons"],
            strand=rec["strand"],
        )

    # For the official segmentation metric, prediction GFF coordinates are
    # transcript-relative and seqid is transcript_id. Predictions are decoded
    # in crop-local coordinates, so a nonzero crop start must be restored in
    # either coordinate mode before the GFF is written.
    local_start = int(r.get("local_start", 0))
    rec["local_start"] = local_start
    if coordinate_mode == "genome":
        rec["start"] = int(r["metadata"].start) + local_start
        rec["end"] = rec["start"] + probs.shape[0]
    elif coordinate_mode == "transcript":
        if local_start:
            rec["exons"] = [(start + local_start, end + local_start) for start, end in rec["exons"]]
            rec["cds"] = [(start + local_start, end + local_start) for start, end in rec["cds"]]
        rec["start"] = local_start
        rec["end"] = local_start + probs.shape[0]
        metadata_length = max(0, int(r["metadata"].end) - int(r["metadata"].start))
        rec["transcript_length"] = max(metadata_length, rec["end"])
    else:
        raise RuntimeError(f"Unsupported segmentation GFF coordinate_mode={coordinate_mode!r}")
    if force_nonempty and not rec.get("exons"):
        raise RuntimeError(f"empty_segment_policy=best_interval failed to create an exon for transcript_id={rec.get('transcript_id')}")
    records.append(rec)

if not records:
    raise RuntimeError("Segmentation inference produced zero transcript records; cannot run official metric on an empty GFF.")

out_gff = cfg["inference"]["output_gff"]
write_segmentation_gff(records, out_gff, coordinate_mode=coordinate_mode)
if cfg["inference"].get("true_gff"):
    evaluate_segmentation(out_gff, cfg["inference"]["true_gff"], cfg["inference"].get("metrics_json", str(Path(out_gff).with_suffix(".metrics.json"))))
