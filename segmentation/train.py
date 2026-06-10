import argparse
from genatator_core.config import load_config
from genatator_core.trainer import train_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train_from_config(load_config(args.config), task="segmentation")


if __name__ == "__main__":
    main()
