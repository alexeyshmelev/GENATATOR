# GENATATOR fine-tuning repository

This repository contains a unified fine-tuning and inference pipeline for ab initio gene annotation models.  It is organized by task:

```text
finding/             # transcript boundary and intragenic-region prediction
segmentation/        # exon / intron / UTR / CDS segmentation inside transcripts
transcript_type/     # mRNA vs lncRNA transcript classification
genatator_core/      # shared datasets, model wrappers, trainers, metrics, GFF writers
smoke_tests/         # end-to-end real-data smoke/overfit tests
```

The repository is for **fine-tuning**, not pretraining.  All model parameters are trainable; there is no freezing option.

## Installation

Install the package in an environment that already contains the desired PyTorch build:

```bash
pip install -e .
pip install -r requirements.txt
```

`requirements.txt` intentionally does not install or upgrade PyTorch.  The code supports older trusted environments such as `torch==2.2.2+cu121` by enabling an explicit Transformers checkpoint-load compatibility patch for trusted GENA / ModernGENA / AMT checkpoints.

## Local path vs Hugging Face repository

There is no `source` field.  Any dataset, tokenizer, backbone, or checkpoint value is interpreted as local if the path exists; otherwise it is passed to Hugging Face as a repo ID/path.  This applies to:

```json
"path": "AIRI-Institute/genatator-gene-finding-dataset"
"backbone_path": "AIRI-Institute/moderngena-base"
"tokenizer_path": "/local/tokenizer/or/hf/repo"
"checkpoint_path": null
```

## Supported model families

`model.family` selects the fine-tuning wrapper:

- `caduceus`: nucleotide-resolution Caduceus PH/PS with middle-loss token head.
- `plain`: BPE-token head on GENA/ModernGENA hidden states.
- `unet`: BPE hidden states expanded to nucleotide positions, then passed through the GENATATOR U-Net head.
- `rmt`: RMT encoder with `RMTEncoderForLetterLevelTokenClassificationUNETsegmentedRepeater` for GENA/ModernGENA only.
- `amt`: AMT/ARMT-style associative-memory wrapper for GENA/ModernGENA only.  If `use_unet=true`, BPE states are expanded to nucleotide positions and passed through the same U-Net head.

Rules enforced by the code:

- RMT and AMT are only valid for GENA/ModernGENA, never for Caduceus.
- Caduceus is always nucleotide-level and uses the middle-loss wrapper.
- GENA/ModernGENA segmentation must be nucleotide-resolution: use `unet`, `rmt`, or `amt` with `use_unet=true`.
- Any U-Net/RMT/AMT+U-Net path requires batch size 1 because nucleotide expansion is sample-specific.
- All parameters must remain trainable.

## JSON config structure

Every train config has the same top-level structure:

```json
{
  "seed": 42,
  "model": {},
  "train_dataset": {},
  "eval_dataset": {},
  "training": {}
}
```

### `model`

Common fields:

```json
{
  "family": "plain",
  "backbone_kind": "moderngena",
  "backbone_path": "AIRI-Institute/moderngena-base",
  "tokenizer_path": "AIRI-Institute/moderngena-base",
  "trust_remote_code": true,
  "checkpoint_path": null,
  "allow_unsafe_torch_load_with_torch_lt_2_6": true
}
```

For U-Net/RMT/AMT+U-Net models add:

```json
{
  "nucleotide_tokenizer_path": "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
  "nucleotide_vocab_size": 1000
}
```

For RMT:

```json
{
  "family": "rmt",
  "cycles": 3,
  "rmt": {
    "input_size": 512,
    "max_n_segments": 10000,
    "num_mem_tokens": 10,
    "bptt_depth": -1,
    "unet_sub_model_input_size": 8192
  }
}
```

For AMT:

```json
{
  "family": "amt",
  "use_unet": false,
  "amt": {
    "amt_repo_id": "irodkin/armt-neox-tiny",
    "num_mem_tokens": 5,
    "d_mem": 64,
    "segment_size": 1019,
    "segment_alignment": "left",
    "sliding_window": false,
    "layers_attr": "layers",
    "wrap_pos": false,
    "correction": true,
    "n_heads": 1,
    "use_denom": true,
    "gating": false,
    "act_on": false
  }
}
```

For old GENA backbones the AMT layer path is normally `bert.encoder.layer`; for ModernGENA it is normally `layers`.

### `training`

Important fields:

```json
{
  "output_dir": "runs/my_run",
  "num_train_epochs": 4,
  "max_steps": -1,
  "per_device_train_batch_size": 1,
  "per_device_eval_batch_size": 1,
  "gradient_accumulation_steps": 1,
  "learning_rate": 5e-5,
  "weight_decay": 1e-4,
  "warmup_steps": 1000,
  "lr_scheduler_type": "constant_with_warmup",
  "logging_interval": 100,
  "eval_interval": 1000,
  "save_interval": 1000,
  "logging_strategy": "steps",
  "evaluation_strategy": "steps",
  "save_strategy": "steps",
  "load_best_model_at_end": true,
  "metric_for_best_model": "loss",
  "greater_is_better": false,
  "save_safetensors": false,
  "resume_from_checkpoint": null,
  "report_to": "tensorboard"
}
```

`logging_interval` is the primary interval field.  It is mapped to `TrainingArguments.logging_steps`.  If `eval_interval` or `save_interval` are omitted, they default to `logging_interval` and `eval_interval`, respectively.

Training does not run a test phase.  Test/inference commands are separate.

## Gene-finding task

### Dataset

Use `AIRI-Institute/genatator-gene-finding-dataset`.  Each row is one genomic block from a chromosome.  The dataset is chromosome-first: a full chromosome may be split across multiple non-overlapping parquet rows.  The pipeline groups rows by `(genome, chrom)`, sorts them by `metadata.start`, and creates one chromosome assembly per chromosome.  It is strictly forbidden to concatenate two different chromosomes; the code never joins the end of one chromosome with the start of another.

Every nucleotide has 12 target channels.  Use `target_group` to choose which half is used:

```json
"target_group": "primary"  // channels 0..5, mRNA + lncRNA
"target_group": "mrna"     // channels 6..11, mRNA/protein-coding only
```

For an edge model, the selected target group is mapped to four output channels:

```text
TSS+, TSS-, PolyA+, PolyA-
```

For a region model, the selected target group is mapped to two output channels:

```text
intragenic+, intragenic-
```

### Sampling and BPE handling

Training windows are produced per chromosome with 50% overlap.  For BPE models, targets are projected to token resolution by taking the maximum label value over the nucleotide span covered by each token.  For U-Net/RMT/AMT+U-Net models, BPE hidden states are expanded back to nucleotide positions with `embedding_repeater`; positions truncated away by the BPE tokenizer are masked out.

Before window generation, the requested dataset subset is loaded into CPU RAM.  The code logs:

- number of selected rows/blocks;
- number of selected chromosomes/genomes;
- selected disk size;
- expected sequence and target RAM;
- selected target indices and target names;
- assembled chromosome length per chromosome.

### Training metrics

During training/validation only PR-AUC is computed:

- edge: `pr_auc_TSS+`, `pr_auc_TSS-`, `pr_auc_PolyA+`, `pr_auc_PolyA-`
- region: `pr_auc_intragenic+`, `pr_auc_intragenic-`

Gene-level, MI, boundary-tolerance, and GFF-based metrics are computed only by `finding/infer.py`.

### Inference and GFF

`finding/infer.py` runs edge and region models over chromosome windows, gathers full-chromosome nucleotide-resolution predictions, computes whole-chromosome PR-AUC per class, applies FFT low-pass smoothing and peak calling, pairs TSS and PolyA peaks, filters intervals with the region model, and writes genome-coordinate GFF.

Because gene-finding models do not predict exon-intron structure, each predicted transcript interval is written with one exon spanning the whole interval:

```text
gene
mRNA or lnc_RNA
exon covering the full predicted interval
```

This allows the annotation leaderboard metric to assess TSS/PolyA boundary recovery, while internal transcript segmentation is evaluated separately in the segmentation task.

## Segmentation task

### Dataset

Use `AIRI-Institute/genatator-gene-segmentation-dataset`.  Choose the dataset configuration with `config_name`:

```json
"config_name": "train-human"
"config_name": "train-multi-specie"
"config_name": "val-human"
```

Each sample is one complete transcript sequence.  The labels are nucleotide-resolution tracks in this order:

```text
5UTR, exon, intron, 3UTR, CDS
```

The dataset retains all annotated transcripts.  You may filter representative transcripts by `statuses: [1]`; for final chromosome-20 evaluation, leave `statuses` unset when all isoforms are desired.

### Sampling and BPE handling

The model input is only the transcript DNA sequence.  It must not include intergenic regions, neighboring genes, or other chromosome context.  Training crops transcripts to `max_nucleotides`; by default the crop starts at least `crop_margin=500` bp away from transcript ends when possible.  For inference, transcript-coordinate predictions are written.

GENA/ModernGENA segmentation must use U-Net/RMT/AMT+U-Net so that BPE states are expanded to nucleotide resolution before loss and metrics are computed.

### Training metrics

Only exact interval-level F1 is computed during training/validation:

```text
interval_f1_exon
interval_f1_CDS
```

UTR and intron training-time metrics are intentionally excluded.  Gene-level and MI metrics are computed only by `segmentation/infer.py` through the official Evaluate metric.

### Prediction GFF

Segmentation prediction GFF is transcript-coordinate:

```text
seqid = transcript_id
```

It is not a chromosome-track GFF and is not intended for IGV visualization.

## Transcript-type task

Transcript type uses the same transcript rows as segmentation, but the model predicts a single binary label:

```text
0 = mRNA / protein-coding
1 = lncRNA
```

Training/validation/inference report only accuracy.

## Smoke tests

Smoke tests use real data and are designed to verify that every task/model path can train, validate, infer, write metrics, and overfit a small held-out subset.

Run:

```bash
python smoke_tests/run_smoke.py \
  --num-gpus 2 \
  --reference-gff /path/to/chr20.gff \
  --work-dir smoke_tests/runs \
  --smoke-cache-dir /path/to/.smoke_real_data_cache \
  --smoke-epochs 4
```

The smoke runner:

1. locates the Hugging Face cache or local dataset path;
2. lists only the relevant parquet files for the requested dataset/configuration;
3. iterates through metadata with tqdm;
4. selects only samples from the requested chromosome;
5. saves indexes under `smoke_tests/indexes` for future runs;
6. materializes selected smoke data under the selected-data directory;
7. trains, validates, and tests on the same selected subset to check visible overfitting;
8. assigns one GPU per active task/model job;
9. writes `summary.md` with durations and metrics.

For gene finding, smoke tests scan the real T2T `test` split and then train/validate/test on the first 10% of one selected chromosome parquet block.  This keeps the smoke run short while using real nucleotide/target arrays.  For segmentation and transcript type, smoke tests use the selected chromosome transcripts from `val-human`.

Use `--refresh-index` to rescan metadata.  Without it, stored indexes are reused.

