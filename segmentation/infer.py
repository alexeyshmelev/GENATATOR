import argparse
from genatator_core.config import load_config
from genatator_core.inference import predict_fasta_to_npz


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--fasta", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    predict_fasta_to_npz(load_config(a.config), "segmentation", a.checkpoint, a.fasta, a.output_dir, a.device)


if __name__ == "__main__":
    main()
