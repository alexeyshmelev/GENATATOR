# Clean GENATATOR fine-tuning repository

This repository is a simplified first-pass training and inference codebase for GENATATOR-style ab initio gene annotation.
It is only for **fine-tuning**, not pretraining.

## Layout

```text
finding/              edge and region models for transcript interval discovery
segmentation/         exon / intron / UTR / CDS segmentation models
transcript_type/      mRNA vs lnc_RNA interval classifier
genatator_core/       shared loaders, model builders, metrics, GFF writers
smoke_tests/          tiny local datasets and a GPU-assignable smoke-test matrix
```

## Main design decisions

There is no `source` field. Dataset paths, model backbone paths, tokenizer paths, and checkpoint paths are detected automatically:

- if the path exists locally, it is used as a local path;
- otherwise it is passed to Hugging Face as a repository id.

Backbones are loaded from local/HF paths and the fine-tuning heads are created in this repository. Full GENATATOR checkpoints can be loaded through `model.checkpoint_path` or `inference.checkpoint_path`.

ModernGENA fine-tuning uses `transformers.ModernBertForTokenClassification` in the task wrappers. This is intentionally different from the older `AnnotationModel` class that used `ModernBertModel` plus a separate classifier.

The provided RMT and Caduceus class logic is preserved as much as possible:

- `genatator_core/legacy_rmt.py` is copied from the supplied RMT/UNet code.
- `genatator_core/legacy_caduceus.py` contains the selected Caduceus classes with only small marked edits: lazy output widths and Trainer-friendly transcript-type logits.
- RMT repeater models with cycles=3 are hard-locked to `per_device_train_batch_size = 1` and `per_device_eval_batch_size = 1`.

## Training

Run one of the task trainers with a JSON config:

```bash
python finding/train.py --task edge --config finding/configs/edge_moderngena_base.json
python finding/train.py --task region --config finding/configs/region_moderngena_base.json
python segmentation/train.py --config segmentation/configs/caduceus_ps.json
python transcript_type/train.py --config transcript_type/configs/caduceus_ps.json
```

For multiple GPUs, use Hugging Face Accelerate or torchrun. The scripts use `transformers.Trainer`, so checkpointing, TensorBoard logging, tqdm progress bars, and resume are handled by Trainer.

```bash
accelerate launch --num_processes 4 segmentation/train.py --config segmentation/configs/caduceus_ps.json
```

Resume is controlled by the JSON field:

```json
"resume_from_checkpoint": "runs/segmentation_caduceus_ps/checkpoint-10000"
```

Use an empty string to start from scratch.

## Training-time validation metrics

The training scripts do not run a test phase.

- `finding`: ROC-AUC per output channel and mean ROC-AUC.
- `segmentation`: exact interval-level F1 for exon and CDS.
- `transcript_type`: accuracy, F1, precision, recall.

## Inference and final metrics

Inference is separate:

```bash
python finding/infer.py --config finding/configs/infer_moderngena_base.json
python segmentation/infer.py --config segmentation/configs/infer_caduceus_ps.json
python transcript_type/infer.py --config transcript_type/configs/infer_caduceus_ps.json
```

The GFF-based evaluators use Hugging Face Evaluate:

- `AIRI-Institute/genatator-ab-initio-annotation-leaderboard`
- `AIRI-Institute/genatator-ab-initio-segmentation-leaderboard`, revision `metric-only`

Set `true_gff` in an inference config to compute and store final metrics.

## Dataset filtering

Every train/eval/inference dataset block supports:

```json
"genomes": ["GCF_009914755.1"],
"chromosomes": ["NC_060944.1"]
```

Empty lists mean no filtering.

## Smoke tests

Create tiny local datasets and run a matrix of training + inference jobs:

```bash
python smoke_tests/run_smoke.py --matrix smoke_tests/smoke_matrix.json
```

Each job has a `gpus` field, so tasks can be pinned to different GPUs:

```json
{
  "name": "segmentation_caduceus_ps_train",
  "gpus": "0",
  "num_processes": 1,
  "command": "python segmentation/train.py --config smoke_tests/segmentation_caduceus_ps.json"
}
```

To test only selected jobs:

```bash
python smoke_tests/run_smoke.py --only segmentation_caduceus_ps_train segmentation_infer_metrics
```

The smoke matrix includes a disabled RMT cycles=3 job; enable it when the large GENA backbone is available locally or you want to download it.

## Known first-pass limitations

- The code is organized for correctness and readability, but it has not been run against the full 30+ GB datasets in this environment.
- ARMT support is wired through a class path and backend config, but should be validated with the exact ARMT remote-code version you use.
- The finding inference GFF writer emits a one-exon transcript interval for boundary evaluation. If you want full GFFs after segmentation, run the segmentation inference stage.
