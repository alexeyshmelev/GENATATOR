#!/bin/bash
#SBATCH --job-name=findrmlruc3896
#SBATCH --partition=rnd

#SBATCH --account=shared
#SBATCH --qos=shared-high
#SBATCH --nodes=1
#SBATCH --time=480:00:00
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=16

set -eo pipefail

date
source $HOME/envs/genatator_pipeline/bin/activate
set -u
cd $HOME/DNALM/GENATATOR/

export HF_HOME=$HOME/.hf
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=0
export PYTHONPATH=$PWD

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 finding/train.py \
  --config finding/configs/region_moderngena_large_rmt_unet.json

date
echo "Done!"
