#!/bin/bash
#SBATCH --job-name=sclive105971
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

python3 - <<'PYCFG'
import json, os, re
source = 'runs/segmentation_gena_base_amt_unet/20260728_110409_981923/evaluation_config.json'
run_dir = 'runs/segmentation_gena_base_amt_unet/20260728_110409_981923'
runtime_cfg = 'runs/segmentation_gena_base_amt_unet/20260728_110409_981923/evaluation_config_live_105971_20260729_095400.json'
out_dir = 'runs/segmentation_gena_base_amt_unet/20260728_110409_981923/evaluation_live_105971_20260729_095400'
with open(source) as f:
    cfg = json.load(f)
ckpts = []
for name in os.listdir(run_dir):
    m = re.fullmatch(r'checkpoint-(\d+)', name)
    if m and os.path.isdir(os.path.join(run_dir, name)):
        ckpts.append((int(m.group(1)), os.path.abspath(os.path.join(run_dir, name))))
if not ckpts:
    raise SystemExit(f'No checkpoints found in {run_dir}')
step, checkpoint = max(ckpts)
os.makedirs(out_dir, exist_ok=True)
cfg.setdefault('_generated', {})
cfg['_generated']['checkpoint_selection'] = 'latest_at_eval_job_start'
cfg['_generated']['source_evaluation_config'] = os.path.abspath(source)
cfg['_generated']['selected_checkpoint_step'] = step
cfg['_generated']['selected_checkpoint'] = checkpoint
cfg['inference']['checkpoint_path'] = checkpoint
cfg['inference']['metrics_json'] = os.path.abspath(os.path.join(out_dir, 'segmentation_metrics.json'))
cfg['inference']['output_gff'] = os.path.abspath(os.path.join(out_dir, 'segmentation_predictions.gff'))
with open(runtime_cfg, 'w') as f:
    json.dump(cfg, f, indent=2)
print(f'Runtime eval config: {runtime_cfg}')
print(f'Selected checkpoint: {checkpoint}')
print(f'Output dir: {out_dir}')
PYCFG

python segmentation/infer.py --config "runs/segmentation_gena_base_amt_unet/20260728_110409_981923/evaluation_config_live_105971_20260729_095400.json"

date
echo "Done!"
