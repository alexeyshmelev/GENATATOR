from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import GenatatorCollator, GenatatorDataset, make_tokenizer
from .model_builders import build_model, load_finetuned_weights
from .train_common import dataset_family_from_model, prepare_nucleotide_tokenizer

logger = logging.getLogger(__name__)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def prepare_tokenizers(model_cfg: Dict[str, Any]):
    tokenizer = make_tokenizer(model_cfg["tokenizer_path"], trust_remote_code=bool(model_cfg.get("trust_remote_code", True)))
    if model_cfg.get("padding_side"):
        tokenizer.padding_side = model_cfg["padding_side"]
    elif model_cfg.get("backbone_kind") == "caduceus":
        tokenizer.padding_side = "left"
        logger.info("[infer.tokenizer] using Caduceus default padding_side=left")
    nucleotide_tokenizer = prepare_nucleotide_tokenizer(model_cfg, tokenizer)
    logger.info("[infer.tokenizer] main pad=%s cls=%s sep=%s side=%s", tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.padding_side)
    if nucleotide_tokenizer is not None:
        logger.info("[infer.tokenizer] nucleotide ids source=main path=%s vocab_size=%s", model_cfg["tokenizer_path"], model_cfg.get("nucleotide_vocab_size"))
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
    tokenizer, nucleotide_tokenizer = prepare_tokenizers(cfg["model"])
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

def _predict_once(cfg: Dict[str, Any], task: str, device: str, reverse_complement: bool) -> List[Dict[str, Any]]:
    model, tokenizer, nucleotide_tokenizer = prepare_model(cfg, task, device)
    data_cfg = dict(cfg["dataset"])
    data_cfg["model_family"] = dataset_family_from_model(cfg["model"])
    data_cfg["reverse_complement"] = reverse_complement
    dataset = GenatatorDataset(data_cfg, task=task, tokenizer=tokenizer, nucleotide_tokenizer=nucleotide_tokenizer, for_inference=True)
    loader = DataLoader(dataset, batch_size=int(cfg.get("inference", {}).get("batch_size", 1)), collate_fn=GenatatorCollator())
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"infer:{task}:rc={reverse_complement}"):
            meta = batch.pop("metadata")
            dna = batch.pop("dna_sequence")
            local_start = batch.pop("local_start")
            offset_mapping = batch.pop("offset_mapping")
            batch.pop("reverse_complement")
            tensor_batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            out = model(**tensor_batch)
            logits = out["logits"] if isinstance(out, dict) else out.logits
            logits = logits.detach().cpu().numpy()
            family = data_cfg["model_family"]
            if task == "transcript_type":
                masks = None
            elif family in {"nucleotide", "bpe_unet", "rmt_unet", "amt_unet"}:
                masks = batch["letter_level_labels_mask"].detach().cpu().numpy().astype(bool)
            else:
                masks = batch.get("labels_mask")
                masks = masks.detach().cpu().numpy().astype(bool) if masks is not None else None
            for i in range(logits.shape[0]):
                if task == "transcript_type":
                    row_logits = logits[i]
                elif family in {"nucleotide", "bpe_unet", "rmt_unet", "amt_unet"}:
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
        return aggregate_full_segmentation_chunks(rows)
    return rows


def predict_dataset_logits(cfg: Dict[str, Any], task: str, device: str = "cuda") -> List[Dict[str, Any]]:
    use_rc = bool(cfg.get("inference", {}).get("use_reverse_complement", False))
    rows = _predict_once(copy.deepcopy(cfg), task, device, reverse_complement=False)
    if not use_rc:
        return rows
    rc_rows = _predict_once(copy.deepcopy(cfg), task, device, reverse_complement=True)
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
