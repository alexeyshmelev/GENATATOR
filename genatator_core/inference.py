from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .data import GenatatorDataset
from .modeling import build_model
from .utils import parse_fasta, sliding_windows, dna_reverse_complement


def load_model_for_inference(cfg: Dict[str, Any], task: str, checkpoint: str, device: str = "cuda"):
    model = build_model(cfg, task)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def _features_for_sequence(cfg: Dict[str, Any], seq: str, task: str) -> Dict[str, torch.Tensor]:
    fake_cfg = dict(cfg)
    fake_cfg["window"] = dict(cfg["window"])
    # Build a lightweight dataset object without reading HF data.
    ds = object.__new__(GenatatorDataset)
    ds.cfg = cfg
    ds.task = task
    ds.split = "inference"
    ds.tokenizer_cfg = cfg["tokenizer"]
    ds.tokenizer = AutoTokenizer.from_pretrained(
        ds.tokenizer_cfg["path"],
        trust_remote_code=ds.tokenizer_cfg.get("trust_remote_code", True),
        local_files_only=ds.tokenizer_cfg.get("local_files_only", False),
    )
    ds.tokenizer_kind = ds.tokenizer_cfg["kind"]
    ds.max_tokens = int(ds.tokenizer_cfg.get("max_tokens", cfg["window"]["nucleotide_length"]))
    ds.pad_token_id = int(ds.tokenizer_cfg.get("pad_token_id", ds.tokenizer.pad_token_id))
    ds.eos_token_id = ds.tokenizer_cfg.get("eos_token_id")
    if ds.eos_token_id is not None:
        ds.eos_token_id = int(ds.eos_token_id)
    ds.pad_side = ds.tokenizer_cfg.get("pad_side", "right")
    ds.add_eos = bool(ds.tokenizer_cfg.get("add_eos", False))
    ds.label_mode = cfg["model"].get("label_mode", "token")
    ds.window = type("W", (), cfg["window"])
    num_labels = cfg["model"].get("num_labels", len(cfg["task"].get("label_names", [])))
    labels = np.zeros((len(seq), num_labels), dtype=np.float32)
    if ds.tokenizer_kind == "bpe":
        out = ds._encode_bpe(seq, labels)
    else:
        out = ds._encode_nucleotide(seq, labels)
    return {k: v.unsqueeze(0) for k, v in out.items() if isinstance(v, torch.Tensor)}


def _probs_to_nt(cfg: Dict[str, Any], seq: str, out_logits: torch.Tensor, features: Dict[str, torch.Tensor]) -> np.ndarray:
    logits = out_logits.squeeze(0).detach().cpu()
    probs = torch.sigmoid(logits).numpy()
    if cfg["model"].get("label_mode") == "nucleotide_unet" or cfg["tokenizer"]["kind"] == "nucleotide":
        n = min(len(seq), probs.shape[0])
        return probs[:n]
    token_to_nt = features.get("token_to_nt")
    if token_to_nt is None:
        # For token-level gene-finding models: expand each token score to its offset span.
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["tokenizer"]["path"],
            trust_remote_code=cfg["tokenizer"].get("trust_remote_code", True),
            local_files_only=cfg["tokenizer"].get("local_files_only", False),
        )
        enc = tokenizer(seq, add_special_tokens=False, truncation=True, max_length=cfg["tokenizer"].get("max_tokens"), return_offsets_mapping=True)
        nt = np.zeros((len(seq), probs.shape[-1]), dtype=np.float32)
        counts = np.zeros(len(seq), dtype=np.float32)
        for i, (s, e) in enumerate(enc["offset_mapping"]):
            if e > s:
                nt[s:e] += probs[i]
                counts[s:e] += 1
        counts[counts == 0] = 1
        return nt / counts[:, None]
    ttn = token_to_nt.squeeze(0).cpu().numpy()
    mask = ttn != -100
    return probs[mask]


def _predict_sequence_one_orientation(model, cfg: Dict[str, Any], task: str, seq: str, device: str) -> np.ndarray:
    window = int(cfg["window"]["nucleotide_length"])
    overlap = float(cfg["window"].get("overlap", 0.5))
    label_n = int(cfg["model"].get("num_labels", len(cfg["task"].get("label_names", []))))
    acc = np.zeros((len(seq), label_n), dtype=np.float32)
    counts = np.zeros(len(seq), dtype=np.float32)
    for s, e in sliding_windows(len(seq), window, overlap):
        sub = seq[s:e]
        features = _features_for_sequence(cfg, sub, task)
        features = {k: v.to(device) for k, v in features.items()}
        with torch.no_grad():
            out = model(**features)
        nt = _probs_to_nt(cfg, sub, out.logits, features)
        acc[s:s + len(nt)] += nt
        counts[s:s + len(nt)] += 1
    counts[counts == 0] = 1
    return acc / counts[:, None]


def _swap_rc_channels(probs: np.ndarray, task: str) -> np.ndarray:
    if task == "finding" and probs.shape[1] == 4:
        return probs[:, [1, 0, 3, 2]]
    if task == "finding" and probs.shape[1] == 2:
        return probs[:, [1, 0]]
    if task == "finding" and probs.shape[1] == 6:
        return probs[:, [1, 0, 3, 2, 5, 4]]
    return probs


def predict_sequence(model, cfg: Dict[str, Any], task: str, seq: str, device: str) -> np.ndarray:
    fwd = _predict_sequence_one_orientation(model, cfg, task, seq, device)
    if not cfg.get("inference", {}).get("use_reverse_complement", False):
        return fwd
    rc = dna_reverse_complement(seq)
    rev = _predict_sequence_one_orientation(model, cfg, task, rc, device)[::-1]
    rev = _swap_rc_channels(rev, task)
    return 0.5 * (fwd + rev)


def predict_fasta_to_npz(cfg: Dict[str, Any], task: str, checkpoint: str, fasta: str, output_dir: str, device: str = "cuda"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = load_model_for_inference(cfg, task, checkpoint, device=device)
    for name, seq in tqdm(list(parse_fasta(fasta)), desc=f"predict/{task}"):
        probs = predict_sequence(model, cfg, task, seq, device=device)
        np.savez_compressed(output / f"{name}.npz", name=name, probs=probs, length=len(seq))
