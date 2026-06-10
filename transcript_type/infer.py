import argparse
import numpy as np
from genatator_core.config import load_config
from genatator_core.inference import load_model_for_inference, _features_for_sequence
from genatator_core.utils import parse_fasta
import torch
from tqdm.auto import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--fasta", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    cfg = load_config(a.config)
    model = load_model_for_inference(cfg, "transcript_type", a.checkpoint, a.device)
    with open(a.output, "w", encoding="utf-8") as f:
        f.write("name\tlncRNA_probability\tpredicted_type\n")
        for name, seq in tqdm(list(parse_fasta(a.fasta)), desc="predict/transcript_type"):
            features = _features_for_sequence(cfg, seq[: cfg["window"]["nucleotide_length"]], "transcript_type")
            features = {k: v.to(a.device) for k, v in features.items()}
            with torch.no_grad():
                out = model(**features)
            prob = torch.sigmoid(out.logits).item()
            label = "lnc_RNA" if prob >= 0.5 else "mRNA"
            f.write(f"{name}\t{prob:.6f}\t{label}\n")


if __name__ == "__main__":
    main()
