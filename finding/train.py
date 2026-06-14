#!/usr/bin/env python
from argparse import ArgumentParser
from genatator_core.train_common import train_from_config

parser = ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--task", choices=["edge", "region"], required=True)
args = parser.parse_args()
train_from_config(args.config, task=f"finding_{args.task}")
