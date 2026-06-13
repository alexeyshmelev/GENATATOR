from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import GenatatorCollator, GenatatorDataset, make_tokenizer
from .model_builders import build_model, load_finetuned_weights
from .utils import ensure_dir


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def prepare_model(cfg: Dict[str, Any], task: str, device: str):
    tokenizer = make_tokenizer(cfg["model"]["tokenizer_path"], trust_remote_code=bool(cfg["model"].get("trust_remote_code", True)))
    if cfg["model"].get("padding_side"):
        tokenizer.padding_side = cfg["model"]["padding_side"]
    nucleotide_tokenizer = None
    if cfg["model"].get("nucleotide_tokenizer_path"):
        nucleotide_tokenizer = make_tokenizer(cfg["model"]["nucleotide_tokenizer_path"], trust_remote_code=bool(cfg["model"].get("trust_remote_code", True)))
        if cfg["model"].get("nucleotide_padding_side"):
            nucleotide_tokenizer.padding_side = cfg["model"]["nucleotide_padding_side"]
    cfg["_tokenizer"] = tokenizer
    model = build_model(cfg, task=task)
    checkpoint = cfg.get("inference", {}).get("checkpoint_path")
    if checkpoint:
        load_finetuned_weights(model, checkpoint)
    model.to(device)
    model.eval()
    return model, tokenizer, nucleotide_tokenizer


def predict_dataset_logits(cfg: Dict[str, Any], task: str, device: str = "cuda") -> List[Dict[str, Any]]:
    model, tokenizer, nucleotide_tokenizer = prepare_model(cfg, task, device)
    data_cfg = dict(cfg["dataset"])
    data_cfg["model_family"] = "nucleotide" if cfg["model"]["family"] == "caduceus" else "bpe"
    dataset = GenatatorDataset(data_cfg, task=task, tokenizer=tokenizer, nucleotide_tokenizer=nucleotide_tokenizer, for_inference=True)
    loader = DataLoader(dataset, batch_size=int(cfg.get("inference", {}).get("batch_size", 1)), collate_fn=GenatatorCollator())
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"infer:{task}"):
            meta = batch.pop("metadata")
            dna = batch.pop("dna_sequence")
            local_start = batch.pop("local_start")
            batch.pop("offset_mapping", None)
            tensor_batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            out = model(**tensor_batch)
            logits = out["logits"] if isinstance(out, dict) else out.logits
            logits = logits.detach().cpu().numpy()
            masks = None
            if "letter_level_labels_mask" in batch:
                masks = batch["letter_level_labels_mask"].detach().cpu().numpy().astype(bool)
            elif "labels_mask" in batch:
                masks = batch["labels_mask"].detach().cpu().numpy().astype(bool)
            for i in range(logits.shape[0]):
                m = masks[i] if masks is not None else np.ones(logits.shape[1], dtype=bool)
                rows.append({"metadata": meta[i], "dna_sequence": dna[i], "local_start": int(local_start[i]), "logits": logits[i][m]})
    return rows
