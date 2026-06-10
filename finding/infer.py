import argparse
from pathlib import Path
import numpy as np

from genatator_core.config import load_config
from genatator_core.inference import load_model_for_inference, predict_sequence
from genatator_core.postprocess import build_intervals, write_intervals_tsv
from genatator_core.utils import parse_fasta
from tqdm.auto import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="JSON with edge_config, edge_checkpoint, region_config, region_checkpoint, postprocess")
    p.add_argument("--fasta", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    cfg = load_config(a.config)
    edge_cfg = load_config(cfg["edge_config"])
    region_cfg = load_config(cfg["region_config"])
    edge = load_model_for_inference(edge_cfg, "finding", cfg["edge_checkpoint"], a.device)
    region = load_model_for_inference(region_cfg, "finding", cfg["region_checkpoint"], a.device)
    outdir = Path(a.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    for name, seq in tqdm(list(parse_fasta(a.fasta)), desc="finding"):
        edge_probs = predict_sequence(edge, edge_cfg, "finding", seq, a.device)
        region_probs = predict_sequence(region, region_cfg, "finding", seq, a.device)
        np.savez_compressed(outdir / f"{name}.tracks.npz", edge=edge_probs, region=region_probs, length=len(seq))
        intervals = build_intervals(edge_probs, region_probs, cfg.get("postprocess", {}))
        write_intervals_tsv(intervals, str(outdir / f"{name}.intervals.tsv"), chrom=name)


if __name__ == "__main__":
    main()
