from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import GenatatorCollator, GenatatorDataset, make_tokenizer
from .inference_policy import (
    gpt_inference_num_transcripts,
    inference_uses_reverse_complement,
    is_gpt_segmentation,
)
from .model_builders import build_model, load_finetuned_weights
from .train_common import dataset_family_from_model, prepare_nucleotide_tokenizer

logger = logging.getLogger(__name__)


@contextmanager
def suppress_repeated_rmt_inference_logs(model_cfg: Dict[str, Any]):
    """Hide per-segment embedding diagnostics during RMT inference only."""
    is_rmt = model_cfg.get("family") == "rmt" or (
        model_cfg.get("family") == "gpt" and "rmt" in model_cfg
    )
    if not is_rmt:
        yield
        return
    backbone_logger = logging.getLogger("genatator_core.backbones")
    previous_level = backbone_logger.level
    backbone_logger.setLevel(max(previous_level, logging.WARNING))
    try:
        yield
    finally:
        backbone_logger.setLevel(previous_level)


def sigmoid(x):
    values = np.asarray(x, dtype=np.float64)
    positive = values >= 0
    output = np.empty_like(values, dtype=np.float64)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def prepare_tokenizers(model_cfg: Dict[str, Any], task: str | None = None):
    tokenizer = make_tokenizer(model_cfg["tokenizer_path"], trust_remote_code=bool(model_cfg.get("trust_remote_code", True)))
    if model_cfg.get("padding_side"):
        tokenizer.padding_side = model_cfg["padding_side"]
    elif model_cfg.get("backbone_kind") == "caduceus":
        tokenizer.padding_side = "left"
        logger.info("[infer.tokenizer] using Caduceus default padding_side=left")
    nucleotide_tokenizer = prepare_nucleotide_tokenizer(model_cfg, tokenizer, task=task)
    logger.info("[infer.tokenizer] main pad=%s cls=%s sep=%s side=%s", tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.padding_side)
    if nucleotide_tokenizer is not None:
        logger.info("[infer.tokenizer] nucleotide ids source=main path=%s vocab_size=%s", model_cfg["tokenizer_path"], model_cfg.get("vocab_size"))
    return tokenizer, nucleotide_tokenizer


def prepare_model(cfg: Dict[str, Any], task: str, device: str):
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
    model_checkpoint = cfg.get("model", {}).get("checkpoint_path")
    inference_checkpoint = cfg.get("inference", {}).get("checkpoint_path")
    if model_checkpoint and inference_checkpoint:
        raise RuntimeError(
            "Set only inference.checkpoint_path for evaluation. Defining both "
            "model.checkpoint_path and inference.checkpoint_path would load two finetuned "
            "checkpoints into the same model."
        )
    tokenizer, nucleotide_tokenizer = prepare_tokenizers(cfg["model"], task=task)
    cfg["_tokenizer"] = tokenizer
    model = build_model(cfg, task=task)
    checkpoint = cfg.get("inference", {}).get("checkpoint_path")
    if checkpoint:
        load_finetuned_weights(model, checkpoint)
    model.to(device)
    model.eval()
    return model, tokenizer, nucleotide_tokenizer


def undo_reverse_complement_logits(logits: np.ndarray, task: str) -> np.ndarray:
    if task == "finding_edge":
        # channels: TSS+, TSS-, PolyA+, PolyA-
        return logits[::-1][:, [1, 0, 3, 2]]
    if task == "finding_region":
        # channels: intragenic+, intragenic-
        return logits[::-1][:, [1, 0]]
    if task == "segmentation":
        # classes: 5UTR, exon, intron, 3UTR, CDS
        return logits[::-1][:, [3, 1, 2, 0, 4]]
    if task == "transcript_type":
        return logits
    raise RuntimeError(task)


def project_masked_letter_logits_to_nucleotides(
    logits: np.ndarray,
    mask: np.ndarray,
    dna_length: int,
) -> np.ndarray:
    """Place retained letter logits on a full crop; uncovered positions stay NaN."""
    logits = np.asarray(logits)
    mask = np.asarray(mask, dtype=bool)
    retained = logits[mask]
    out = np.full((int(dna_length), logits.shape[-1]), np.nan, dtype=np.float32)
    n = min(len(out), retained.shape[0])
    out[:n] = retained[:n]
    return out


def project_bpe_token_logits_to_nucleotides(
    logits: np.ndarray,
    offset_mapping,
    attention_mask: np.ndarray,
    dna_length: int,
) -> np.ndarray:
    """Expand BPE-token logits to nucleotide coordinates without inventing zeros."""
    logits = np.asarray(logits)
    dna_length = int(dna_length)
    tmp = np.zeros((dna_length, logits.shape[-1]), dtype=np.float32)
    counts = np.zeros(dna_length, dtype=np.float32)
    for token_i, ((start, end), attended) in enumerate(zip(offset_mapping, attention_mask)):
        if not int(attended) or int(end) <= int(start):
            continue
        start = max(0, min(dna_length, int(start)))
        end = max(0, min(dna_length, int(end)))
        if end <= start:
            continue
        tmp[start:end] += logits[token_i]
        counts[start:end] += 1.0
    out = np.full((dna_length, logits.shape[-1]), np.nan, dtype=np.float32)
    covered = counts > 0
    out[covered] = tmp[covered] / counts[covered, None]
    return out




def _transcript_row_key(row: Dict[str, Any]) -> tuple:
    meta = row["metadata"]
    return (
        meta.transcript_id,
        meta.gene_id,
        meta.genome,
        meta.chrom,
        int(meta.start),
        int(meta.end),
        meta.strand,
    )


def aggregate_full_segmentation_chunks(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Gather non-overlapping segmentation chunks into one full transcript row."""
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    order: List[tuple] = []
    for row in rows:
        key = _transcript_row_key(row)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)

    gathered: List[Dict[str, Any]] = []
    for key in order:
        parts = sorted(grouped[key], key=lambda item: int(item["local_start"]))
        full_length = max(int(part["local_start"]) + int(np.asarray(part["logits"]).shape[0]) for part in parts)
        channels = int(np.asarray(parts[0]["logits"]).shape[-1])
        full_logits = np.full((full_length, channels), np.nan, dtype=np.float32)
        covered = np.zeros(full_length, dtype=bool)
        sequence = [""] * full_length
        for part in parts:
            start = int(part["local_start"])
            logits = np.asarray(part["logits"], dtype=np.float32)
            end = start + int(logits.shape[0])
            if np.any(covered[start:end]):
                raise RuntimeError(
                    f"Full-transcript segmentation chunks overlap for transcript={key[0]!r} at [{start}, {end})"
                )
            full_logits[start:end] = logits
            covered[start:end] = True
            dna = str(part["dna_sequence"])[: logits.shape[0]]
            if bool(part.get("reverse_complement", False)):
                from .utils import reverse_complement
                dna = reverse_complement(dna)
            sequence[start:end] = list(dna)
        if not bool(covered.all()):
            missing = int((~covered).sum())
            raise RuntimeError(
                f"Full-transcript segmentation gathering left {missing} nucleotide positions uncovered "
                f"for transcript={key[0]!r}"
            )
        if any(base == "" for base in sequence):
            raise RuntimeError(f"Full-transcript DNA gathering failed for transcript={key[0]!r}")
        first = parts[0]
        gathered.append({
            "metadata": first["metadata"],
            "dna_sequence": "".join(sequence),
            "local_start": 0,
            "offset_mapping": [],
            "model_family": first["model_family"],
            "reverse_complement": bool(first.get("reverse_complement", False)),
            "logits": full_logits,
        })
    return gathered


def model_logits_for_inference(
    model,
    tensor_batch: Dict[str, torch.Tensor],
    *,
    task: str,
    model_cfg: Dict[str, Any],
) -> torch.Tensor:
    """Run the inference-only model path and return its logits tensor."""

    if task == "segmentation" and model_cfg.get("family") == "gpt":
        return model.generate(**tensor_batch)
    output = model(**tensor_batch)
    return output["logits"] if isinstance(output, dict) else output.logits


def _predict_once(
    cfg: Dict[str, Any],
    task: str,
    device: str,
    reverse_complement: bool,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> List[Dict[str, Any]]:
    if reverse_complement and is_gpt_segmentation(cfg.get("model", {}), task):
        raise RuntimeError(
            "Reverse-complement inference is forbidden for GPT segmentation models"
        )
    model, tokenizer, nucleotide_tokenizer = prepare_model(cfg, task, device)
    data_cfg = dict(cfg["dataset"])
    data_cfg["model_family"] = dataset_family_from_model(cfg["model"], task=task)
    data_cfg["reverse_complement"] = reverse_complement
    transcript_count = gpt_inference_num_transcripts(cfg, task)
    if transcript_count is not None:
        conflicting_limits = [
            key
            for key in (
                "max_rows",
                "max_windows",
                "streaming_max_rows",
                "streaming_max_scanned_rows",
                "streaming_trim_rows",
            )
            if data_cfg.get(key) not in (None, 0)
        ]
        if conflicting_limits:
            raise RuntimeError(
                "GPT inference.num_transcripts must be validated against the "
                "complete filtered chromosome and cannot be combined with "
                f"dataset debug limits: {conflicting_limits}"
            )
        data_cfg["_gpt_inference_num_transcripts"] = transcript_count
        data_cfg["_gpt_inference_rank"] = int(rank)
        data_cfg["_gpt_inference_world_size"] = int(world_size)
    dataset = GenatatorDataset(data_cfg, task=task, tokenizer=tokenizer, nucleotide_tokenizer=nucleotide_tokenizer, for_inference=True)
    configured_batch_size = int(cfg.get("inference", {}).get("batch_size", 1))
    if configured_batch_size != 1:
        raise RuntimeError("GENATATOR inference batch_size must be 1 for every task/model")
    loader = DataLoader(dataset, batch_size=1, collate_fn=GenatatorCollator())
    rows = []
    with suppress_repeated_rmt_inference_logs(cfg["model"]):
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"infer:{task}:rc={reverse_complement}"):
                meta = batch.pop("metadata")
                dna = batch.pop("dna_sequence")
                local_start = batch.pop("local_start")
                offset_mapping = batch.pop("offset_mapping")
                batch.pop("reverse_complement")
                tensor_batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                # GPT validation and inference are strictly autoregressive. The
                # materialized benchmark batch still contains reference labels,
                # so dispatch through generate() to keep them out of decoder
                # context.
                logits = model_logits_for_inference(
                    model,
                    tensor_batch,
                    task=task,
                    model_cfg=cfg["model"],
                )
                logits = logits.detach().cpu().numpy()
                family = data_cfg["model_family"]
                if task == "transcript_type":
                    masks = None
                elif family in {"nucleotide", "bpe_unet", "rmt_unet", "amt_unet", "bpe_gpt"}:
                    masks = batch["letter_level_labels_mask"].detach().cpu().numpy().astype(bool)
                else:
                    masks = batch.get("labels_mask")
                    masks = masks.detach().cpu().numpy().astype(bool) if masks is not None else None
                for i in range(logits.shape[0]):
                    if task == "transcript_type":
                        row_logits = logits[i]
                    elif family in {"nucleotide", "bpe_unet", "rmt_unet", "amt_unet", "bpe_gpt"}:
                        row_logits = project_masked_letter_logits_to_nucleotides(
                            logits[i],
                            masks[i],
                            len(dna[i]),
                        )
                    else:
                        row_logits = logits[i][
                            masks[i] if masks is not None else np.ones(logits.shape[1], dtype=bool)
                        ]
                    if reverse_complement:
                        row_logits = undo_reverse_complement_logits(row_logits, task)
                    rows.append({
                        "metadata": meta[i],
                        "dna_sequence": dna[i],
                        "local_start": int(local_start[i]),
                        "offset_mapping": offset_mapping[i],
                        "model_family": family,
                        "reverse_complement": bool(reverse_complement),
                        "logits": row_logits,
                    })
    if task == "segmentation" and bool(data_cfg.get("full_transcript_chunks", False)):
        gathered = aggregate_full_segmentation_chunks(rows)
        if transcript_count is not None:
            ordinals = dataset.inference_assigned_ordinals or []
            selected_total = dataset.inference_selected_transcripts
            if selected_total is None or len(gathered) != len(ordinals):
                raise RuntimeError(
                    "GPT inference transcript aggregation changed the assigned "
                    "transcript count: "
                    f"assigned={len(ordinals)} gathered={len(gathered)}"
                )
            for row, ordinal in zip(gathered, ordinals):
                row["_inference_transcript_ordinal"] = int(ordinal)
                row["_inference_selected_transcripts"] = int(selected_total)
        return gathered
    return rows


def merge_rank_strided_results(
    rank_results: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Restore deterministic dataset order after rank-strided transcript work."""

    if not rank_results:
        return []
    world_size = len(rank_results)
    flattened = [row for results in rank_results for row in results]
    tagged = ["_inference_transcript_ordinal" in row for row in flattened]
    if any(tagged) and not all(tagged):
        raise RuntimeError(
            "Distributed GPT inference results mixed tagged and untagged transcripts"
        )

    if flattened and all(tagged):
        selected_totals = {
            int(row["_inference_selected_transcripts"]) for row in flattened
        }
        if len(selected_totals) != 1:
            raise RuntimeError(
                "Distributed GPT inference ranks disagree on the selected "
                f"transcript count: {sorted(selected_totals)}"
            )
        total = selected_totals.pop()
        for rank, results in enumerate(rank_results):
            expected_ordinals = list(range(rank, total, world_size))
            actual_ordinals = [
                int(row["_inference_transcript_ordinal"]) for row in results
            ]
            if actual_ordinals != expected_ordinals:
                raise RuntimeError(
                    "Distributed GPT inference rank shard is incomplete or out "
                    f"of order: rank={rank} expected={expected_ordinals} "
                    f"actual={actual_ordinals}"
                )
        merged = sorted(
            flattened,
            key=lambda row: int(row["_inference_transcript_ordinal"]),
        )
        cleaned: List[Dict[str, Any]] = []
        for row in merged:
            item = dict(row)
            item.pop("_inference_transcript_ordinal", None)
            item.pop("_inference_selected_transcripts", None)
            cleaned.append(item)
        return cleaned

    total = len(flattened)
    expected_lengths = [len(range(rank, total, world_size)) for rank in range(world_size)]
    actual_lengths = [len(results) for results in rank_results]
    if actual_lengths != expected_lengths:
        raise RuntimeError(
            "Distributed GPT inference rank shards are not a complete strided "
            f"partition: expected_lengths={expected_lengths} actual_lengths={actual_lengths}"
        )

    merged: List[Dict[str, Any]] = []
    for position in range(max(actual_lengths, default=0)):
        for results in rank_results:
            if position < len(results):
                merged.append(results[position])
    if len(merged) != total:
        raise RuntimeError(
            "Distributed GPT inference result merge changed the transcript count: "
            f"before={total} after={len(merged)}"
        )
    return merged


def predict_dataset_logits(
    cfg: Dict[str, Any],
    task: str,
    device: str = "cuda",
    *,
    rank: int = 0,
    world_size: int = 1,
) -> List[Dict[str, Any]]:
    rank = int(rank)
    world_size = int(world_size)
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise RuntimeError(
            f"Invalid inference rank topology: rank={rank} world_size={world_size}"
        )
    if world_size > 1 and not is_gpt_segmentation(cfg.get("model", {}), task):
        raise RuntimeError(
            "Distributed standalone inference is supported only for GPT "
            "segmentation models"
        )
    # Validate the GPT-only public option before loading a checkpoint.
    gpt_inference_num_transcripts(cfg, task)
    use_rc = inference_uses_reverse_complement(cfg, task)
    rows = _predict_once(
        copy.deepcopy(cfg),
        task,
        device,
        reverse_complement=False,
        rank=rank,
        world_size=world_size,
    )
    if not use_rc:
        return rows
    rc_rows = _predict_once(
        copy.deepcopy(cfg),
        task,
        device,
        reverse_complement=True,
        rank=rank,
        world_size=world_size,
    )
    if len(rows) != len(rc_rows):
        raise RuntimeError(f"RC row count mismatch: forward={len(rows)} rc={len(rc_rows)}")
    merged = []
    for a, b in zip(rows, rc_rows):
        if a["metadata"] != b["metadata"] or a["local_start"] != b["local_start"]:
            raise RuntimeError("RC rows are not aligned with forward rows")
        if np.asarray(a["logits"]).shape != np.asarray(b["logits"]).shape:
            raise RuntimeError(f"RC logits shape mismatch: {np.asarray(a['logits']).shape} vs {np.asarray(b['logits']).shape}")
        m = dict(a)
        stacked = np.stack(
            [np.asarray(a["logits"], dtype=np.float32), np.asarray(b["logits"], dtype=np.float32)],
            axis=0,
        )
        finite = np.isfinite(stacked)
        totals = np.where(finite, stacked, 0.0).sum(axis=0)
        counts = finite.sum(axis=0)
        averaged = np.full_like(totals, np.nan, dtype=np.float32)
        np.divide(totals, counts, out=averaged, where=counts > 0)
        m["logits"] = averaged
        merged.append(m)
    return merged
