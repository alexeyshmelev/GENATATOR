#!/bin/bash
#SBATCH --job-name=scbest105971
#SBATCH --partition=rnd
#SBATCH --account=airi
#SBATCH --qos=airi-high
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G

set -eo pipefail

date
source "$HOME/envs/genatator_pipeline/bin/activate"
set -u
cd "$HOME/DNALM/GENATATOR"

export HF_HOME="$HOME/.hf"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=0
export PYTHONPATH="$PWD"

python segmentation/infer.py --config "runs/segmentation_gena_base_amt_unet/20260728_110409_981923/evaluation_config_best_105971_20260729_095758.json"

date
echo "Done!"
