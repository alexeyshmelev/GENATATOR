#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path
import logging
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from genatator_core.cds_heuristic import infer_cds_with_benchmark_heuristic
from genatator_core.config import load_json
from genatator_core.evaluate_gff import evaluate_segmentation
from genatator_core.gff import labels_to_segmentation_record, write_segmentation_gff
from genatator_core.infer_common import (
    merge_rank_strided_results,
    predict_dataset_logits,
    sigmoid,
)
from genatator_core.inference_policy import (
    gpt_inference_num_transcripts,
    is_gpt_segmentation,
    segmentation_uses_cds_heuristic,
)

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)


def _distributed_runtime(cfg):
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = str(cfg.get("inference", {}).get("device", "cuda"))
    is_gpt = is_gpt_segmentation(cfg.get("model", {}), "segmentation")

    if world_size > 1:
        if not is_gpt:
            raise RuntimeError(
                "Distributed standalone segmentation inference is supported only "
                "for GPT-head models"
            )
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError(
                "Distributed GPT inference requires inference.device='cuda' and "
                "CUDA-capable PyTorch"
            )
        visible_gpus = int(torch.cuda.device_count())
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
        if local_world_size != visible_gpus:
            raise RuntimeError(
                "GPT inference must launch one torchrun process for every visible "
                "GPU: "
                f"LOCAL_WORLD_SIZE={local_world_size} visible_gpus={visible_gpus}"
            )
        if local_rank < 0 or local_rank >= visible_gpus:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is outside visible_gpus={visible_gpus}"
            )
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", timeout=timedelta(hours=48))
    elif is_gpt and device.startswith("cuda") and torch.cuda.device_count() > 1:
        raise RuntimeError(
            "GPT inference detected multiple visible GPUs but was launched as one "
            "process. Use torchrun --standalone --nproc_per_node=gpu "
            "segmentation/infer.py --config <evaluation_config.json>."
        )

    logger.info(
        "[segmentation.inference.runtime] rank=%d local_rank=%d world_size=%d device=%s",
        rank,
        local_rank,
        world_size,
        device,
    )
    return rank, world_size, device


def _decode_records(cfg, rows):
    records = []
    force_nonempty = cfg.get("inference", {}).get("empty_segment_policy", "error") == "best_interval"
    coordinate_mode = cfg.get("inference", {}).get("coordinate_mode", "transcript")
    use_cds_heuristic = segmentation_uses_cds_heuristic(cfg)
    logger.info(
        "Segmentation decoding | coordinate_mode=%s use_cds_heuristic=%s",
        coordinate_mode,
        use_cds_heuristic,
    )
    for row in rows:
        probs = sigmoid(row["logits"])
        rec = labels_to_segmentation_record(
            row["metadata"],
            probs,
            force_nonempty=force_nonempty,
        )
        if "_inference_transcript_ordinal" in row:
            rec["_inference_transcript_ordinal"] = int(
                row["_inference_transcript_ordinal"]
            )
            rec["_inference_selected_transcripts"] = int(
                row["_inference_selected_transcripts"]
            )
        if use_cds_heuristic and rec["transcript_type"] == "mRNA":
            # ``rec`` intervals are local to this exact model input. Restrict the
            # sequence to the covered prediction span before applying the same
            # exon-splicing/ORF heuristic as GENATATOR-PIPELINE.
            sequence = str(row["dna_sequence"])[: probs.shape[0]]
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
        local_start = int(row.get("local_start", 0))
        rec["local_start"] = local_start
        if coordinate_mode == "genome":
            rec["start"] = int(row["metadata"].start) + local_start
            rec["end"] = rec["start"] + probs.shape[0]
        elif coordinate_mode == "transcript":
            if local_start:
                rec["exons"] = [(start + local_start, end + local_start) for start, end in rec["exons"]]
                rec["cds"] = [(start + local_start, end + local_start) for start, end in rec["cds"]]
            rec["start"] = local_start
            rec["end"] = local_start + probs.shape[0]
            metadata_length = max(0, int(row["metadata"].end) - int(row["metadata"].start))
            rec["transcript_length"] = max(metadata_length, rec["end"])
        else:
            raise RuntimeError(f"Unsupported segmentation GFF coordinate_mode={coordinate_mode!r}")
        if force_nonempty and not rec.get("exons"):
            raise RuntimeError(
                "empty_segment_policy=best_interval failed to create an exon for "
                f"transcript_id={rec.get('transcript_id')}"
            )
        records.append(rec)
    return records


def _gather_records(records, rank: int, world_size: int):
    if world_size == 1:
        return merge_rank_strided_results([records])
    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(records, gathered, dst=0)
    if rank != 0:
        return []
    return merge_rank_strided_results(gathered)


def _write_outputs(cfg, records):
    if not records:
        raise RuntimeError(
            "Segmentation inference produced zero transcript records; cannot run "
            "official metric on an empty GFF."
        )
    requested = gpt_inference_num_transcripts(cfg, "segmentation")
    if requested is not None and requested > 0 and len(records) != requested:
        raise RuntimeError(
            "Distributed GPT inference returned the wrong number of transcripts: "
            f"requested={requested} produced={len(records)}"
        )

    inference_cfg = cfg["inference"]
    coordinate_mode = inference_cfg.get("coordinate_mode", "transcript")
    out_gff = inference_cfg["output_gff"]
    write_segmentation_gff(records, out_gff, coordinate_mode=coordinate_mode)
    if inference_cfg.get("true_gff"):
        if requested is not None and requested > 0:
            logger.warning(
                "[gpt.inference.metrics] num_transcripts=%d is a subset, but "
                "true_gff is configured; full-reference metrics will include "
                "transcripts that were intentionally not predicted",
                requested,
            )
        evaluate_segmentation(
            out_gff,
            inference_cfg["true_gff"],
            inference_cfg.get(
                "metrics_json",
                str(Path(out_gff).with_suffix(".metrics.json")),
            ),
        )


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    rank, world_size, device = _distributed_runtime(cfg)
    process_group_destroyed = False
    try:
        rows = predict_dataset_logits(
            cfg,
            task="segmentation",
            device=device,
            rank=rank,
            world_size=world_size,
        )
        records = _decode_records(cfg, rows)
        del rows
        records = _gather_records(records, rank, world_size)
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
            process_group_destroyed = True
        if rank == 0:
            _write_outputs(cfg, records)
    finally:
        if (
            world_size > 1
            and not process_group_destroyed
            and dist.is_initialized()
        ):
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
