import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import gather_object
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import get_scheduler

from .config import save_config
from .data import build_dataset, collate_fn
from .metrics import token_metrics, segmentation_interval_metrics, gene_level_segmentation_metrics
from .modeling import build_model
from .utils import seed_everything


def _batch_to_model(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}


def _collect_records(batch: Dict[str, Any], out, cfg: Dict[str, Any], task: str):
    probs = torch.sigmoid(out.logits.detach()).cpu().numpy()
    label_names = cfg["task"].get("label_names")
    records = []
    if task == "transcript_type":
        y = batch["transcript_type"].detach().cpu().numpy().reshape(-1)
        p = probs.reshape(-1)
        return [{"labels": np.asarray([[yy]], dtype=np.float32), "probs": np.asarray([[pp]], dtype=np.float32)} for yy, pp in zip(y, p)]
    if cfg["model"].get("label_mode") == "nucleotide_unet":
        y = batch["nt_labels"].detach().cpu().numpy()
        mask = batch["nt_labels_mask"].detach().cpu().numpy().astype(bool)
    else:
        y = batch["labels"].detach().cpu().numpy()
        mask = batch["labels_mask"].detach().cpu().numpy().astype(bool)
    for i in range(len(probs)):
        records.append({
            "labels": y[i][mask[i]],
            "probs": probs[i][mask[i]],
            "metadata": batch["metadata"][i],
            "label_names": label_names,
        })
    return records


def _metrics(records: list[dict], cfg: Dict[str, Any], task: str) -> Dict[str, float]:
    if task == "transcript_type":
        y = np.concatenate([r["labels"] for r in records], axis=0).reshape(-1)
        p = np.concatenate([r["probs"] for r in records], axis=0).reshape(-1)
        pred = p >= 0.5
        tp = np.logical_and(pred, y == 1).sum()
        fp = np.logical_and(pred, y == 0).sum()
        fn = np.logical_and(~pred, y == 1).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"transcript_type/precision": float(prec), "transcript_type/recall": float(rec), "opt/f1": float(f1)}
    label_names = cfg["task"]["label_names"]
    if task == "segmentation":
        m = token_metrics(records, label_names)
        m.update(segmentation_interval_metrics(records, label_names, thresholds=tuple(cfg["eval"].get("thresholds", [0.5]))))
        if cfg["eval"].get("gene_level", False):
            m.update(gene_level_segmentation_metrics(records, threshold=cfg["eval"].get("gene_level_threshold", 0.5)))
        return m
    return token_metrics(records, label_names)


def evaluate(model, loader, accelerator, cfg, task):
    model.eval()
    total_loss = []
    records = []
    iterator = tqdm(loader, desc="eval", disable=not accelerator.is_main_process)
    with torch.no_grad():
        for batch in iterator:
            model_batch = _batch_to_model(batch, accelerator.device)
            out = model(**model_batch)
            if out.loss is not None:
                total_loss.append(accelerator.gather(out.loss.detach()).mean().item())
            local_records = _collect_records(batch, out, cfg, task)
            gathered = gather_object(local_records)
            if accelerator.is_main_process:
                records.extend(gathered)
    metrics = {"loss": float(np.mean(total_loss)) if total_loss else 0.0}
    if accelerator.is_main_process:
        metrics.update(_metrics(records, cfg, task))
    return metrics


def save_checkpoint(accelerator: Accelerator, model, output_dir: Path, name: str, config: Dict[str, Any], metrics: Dict[str, float], step: int):
    ckpt = output_dir / name
    ckpt.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    accelerator.save(unwrapped.state_dict(), ckpt / "pytorch_model.bin")
    if accelerator.is_main_process:
        save_config(config, ckpt / "config.json")
        with open(ckpt / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"step": step, "metrics": metrics}, f, indent=2)


def train_from_config(cfg: Dict[str, Any], task: str):
    seed_everything(int(cfg["training"].get("seed", 42)))
    ddp = DistributedDataParallelKwargs(find_unused_parameters=cfg["training"].get("find_unused_parameters", False))
    accelerator = Accelerator(
        mixed_precision=cfg["training"].get("mixed_precision", "bf16"),
        gradient_accumulation_steps=int(cfg["training"].get("gradient_accumulation_steps", 1)),
        kwargs_handlers=[ddp],
    )
    output_dir = Path(cfg["training"]["output_dir"])
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, output_dir / "config.json")
    tokenizer_batch = int(cfg["training"].get("batch_size", 1))
    workers = int(cfg["training"].get("num_workers", 4))
    train_ds = build_dataset(cfg, "train", task)
    valid_ds = build_dataset(cfg, "validation", task)
    train_sampler = DistributedSampler(train_ds, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=True, seed=cfg["training"].get("seed", 42))
    valid_sampler = DistributedSampler(valid_ds, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=tokenizer_batch, sampler=train_sampler, collate_fn=collate_fn, num_workers=workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=tokenizer_batch, sampler=valid_sampler, collate_fn=collate_fn, num_workers=workers, pin_memory=True)
    model = build_model(cfg, task)
    if not cfg["training"].get("backbone_trainable", True):
        for name, p in model.named_parameters():
            if "classifier" not in name and "unet" not in name and "nt_embedding" not in name:
                p.requires_grad = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"].get("lr", 5e-5)), weight_decay=float(cfg["training"].get("weight_decay", 1e-4)))
    scheduler = get_scheduler(
        cfg["training"].get("scheduler", "constant_with_warmup"),
        optimizer,
        num_warmup_steps=int(cfg["training"].get("warmup_steps", 1000)),
        num_training_steps=int(cfg["training"].get("max_steps", 500000)),
    )
    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, valid_loader, scheduler)
    writer = SummaryWriter(output_dir / "tb") if accelerator.is_main_process else None
    max_steps = int(cfg["training"].get("max_steps", 500000))
    log_every = int(cfg["training"].get("log_every", 100))
    eval_every = int(cfg["training"].get("eval_every", 1000))
    save_every = int(cfg["training"].get("save_every", 0))
    optimize_metric = cfg["training"].get("optimize_metric", "loss")
    optimize_mode = cfg["training"].get("optimize_mode", "min")
    best = float("inf") if optimize_mode == "min" else -float("inf")
    step = 0
    recent_loss = []
    pbar = tqdm(total=max_steps, desc=f"train/{task}", disable=not accelerator.is_main_process)
    while step < max_steps:
        train_sampler.set_epoch(step)
        for batch in train_loader:
            with accelerator.accumulate(model):
                out = model(**_batch_to_model(batch, accelerator.device))
                accelerator.backward(out.loss)
                if cfg["training"].get("clip_grad_norm"):
                    accelerator.clip_grad_norm_(model.parameters(), float(cfg["training"]["clip_grad_norm"]))
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
            step += 1
            recent_loss.append(out.loss.detach().float().item())
            if accelerator.is_main_process and step % log_every == 0:
                loss_value = float(np.mean(recent_loss)); recent_loss = []
                writer.add_scalar("loss/train", loss_value, step)
                writer.add_scalar("lr", scheduler.get_last_lr()[0], step)
                pbar.set_postfix(loss=f"{loss_value:.4f}")
            if step % eval_every == 0:
                metrics = evaluate(model, valid_loader, accelerator, cfg, task)
                if accelerator.is_main_process:
                    for k, v in metrics.items():
                        writer.add_scalar(k.replace("/", "_") + "/validation", v, step)
                    current = metrics.get(optimize_metric, metrics.get("opt/" + optimize_metric, metrics["loss"]))
                    improved = current < best if optimize_mode == "min" else current > best
                    if improved:
                        best = current
                        save_checkpoint(accelerator, model, output_dir, "best", cfg, metrics, step)
                    with open(output_dir / "last_metrics.json", "w", encoding="utf-8") as f:
                        json.dump({"step": step, "metrics": metrics, "best": best}, f, indent=2)
                accelerator.wait_for_everyone()
            if save_every and step % save_every == 0:
                save_checkpoint(accelerator, model, output_dir, f"step_{step}", cfg, {"loss/train": float(np.mean(recent_loss)) if recent_loss else 0.0}, step)
            pbar.update(1)
            if step >= max_steps:
                break
    pbar.close()
    if accelerator.is_main_process:
        writer.flush(); writer.close()
