#!/bin/bash
#SBATCH --job-name=edgeMG32k
#SBATCH --partition=rnd

# #SBATCH --account=shared
# #SBATCH --qos=shared-high

#SBATCH --account=airi
#SBATCH --qos=airi-high
#SBATCH --nodes=1
#SBATCH --time=480:00:00 
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=16


date
source $HOME/envs/genatator_pipeline/bin/activate

set -euo pipefail

cd $HOME/DNALM/GENATATOR/

export HF_HOME=$HOME/.hf
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=0

export PYTHONPATH=$PWD

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 -m finding.train \
                                --task edge \
                                --config finding/configs/edge_moderngena_base_plain_32k.json
date
echo "Done!"
