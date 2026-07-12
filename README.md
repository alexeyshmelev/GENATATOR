# GENATATOR fine-tuning repository

This repository contains a unified fine-tuning and inference pipeline for ab initio gene annotation models.  It is organized by task:

```text
finding/             # transcript boundary and intragenic-region prediction
segmentation/        # exon / intron / UTR / CDS segmentation inside transcripts
transcript_type/     # mRNA vs lncRNA transcript classification
genatator_core/      # shared datasets, model wrappers, trainers, metrics, GFF writers
smoke_tests/         # end-to-end real-data smoke tests
```

The repository is for **fine-tuning**, not pretraining.  All model parameters are trainable; there is no freezing option.

See [`AUDIT_REPORT.md`](AUDIT_REPORT.md) for the requirement-by-requirement
change map, additional defects found, and verification limits.

## Installation

Install the package in an environment that already contains the desired PyTorch build:

```bash
pip install -e .
pip install -r requirements.txt
```

`requirements.txt` intentionally does not install or upgrade PyTorch.  The code supports older trusted environments such as `torch==2.2.2+cu121` by enabling an explicit Transformers checkpoint-load compatibility patch for trusted GENA / ModernGENA / AMT checkpoints.

## Local paths and Hugging Face repositories

There is no `source` field. Dataset, tokenizer, and backbone values are interpreted as local when the path exists; otherwise they are passed to Hugging Face as repository IDs. Fine-tuned checkpoint values used by the evaluation scripts must currently point to a local checkpoint directory or weight file.

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
- Every model that contains a U-Net exposes `model.unet_chunk_size`, in nucleotides, with a default of `8192`.
- Transformer batches may contain multiple samples, but padding left by the Transformer is removed before the U-Net. Each unpadded sample is sent through the U-Net separately with U-Net batch size 1, then the per-sample results are assembled back into the model batch.
- All parameters must remain trainable.

The supplied task/model combinations are:

| Task | Supplied model variants |
| --- | --- |
| Gene-finding edge and region | Caduceus PH/PS; GENA base/large plain and U-Net; GENA large RMT+U-Net; ModernGENA base plain, U-Net, RMT+U-Net, AMT plain, and AMT+U-Net; ModernGENA large plain and U-Net |
| Segmentation | Caduceus PH/PS; GENA base/large U-Net; GENA large RMT+U-Net; ModernGENA base U-Net, RMT+U-Net, and AMT+U-Net; ModernGENA large U-Net |
| Transcript type | Caduceus PH/PS; GENA base/large plain; ModernGENA base/large plain |

Use the matching JSON under the task's `configs/` directory. The generated evaluation config copies the complete model block, so evaluation uses the same backbone, tokenizer, memory wrapper, U-Net settings, and head shape as training.

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

For U-Net/RMT/AMT+U-Net models add a nucleotide tokenizer.  For GENA and ModernGENA this must normally be the **same tokenizer as the backbone tokenizer**, because those tokenizers already contain single-nucleotide tokens (`A`, `C`, `G`, `T`).  Do not use the Caduceus tokenizer for GENA/ModernGENA nucleotide expansion.

```json
{
  "tokenizer_path": "AIRI-Institute/moderngena-base",
  "nucleotide_tokenizer_path": "AIRI-Institute/moderngena-base",
  "nucleotide_vocab_size": null
}
```

`nucleotide_vocab_size: null` means that the code infers the vocabulary size from the nucleotide tokenizer before constructing the nucleotide embedding table.  The same pattern is used for GENA:

```json
{
  "tokenizer_path": "AIRI-Institute/gena-lm-bert-base-lastln-t2t",
  "nucleotide_tokenizer_path": "AIRI-Institute/gena-lm-bert-base-lastln-t2t",
  "nucleotide_vocab_size": null
}
```

For RMT:

```json
{
  "family": "rmt",
  "cycles": 3,
  "unet_chunk_size": 8192,
  "rmt": {
    "input_size": 512,
    "max_n_segments": 10000,
    "num_mem_tokens": 10,
    "bptt_depth": -1
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

For `family: "unet"` and for AMT with `use_unet: true`, also set:

```json
"unet_chunk_size": 8192
```

### Sequence-length fields

Caduceus is nucleotide-tokenized, so its dataset length is configured directly in nucleotides:

```json
"max_nucleotides": 131000
```

GENA and ModernGENA are BPE-tokenized. Their configs must not define a fixed nucleotide context as the primary length. Instead they define both the maximum BPE-token count and the tokenizer's average nucleotide span per BPE token:

```json
"max_bpe_tokens": 32768,
"average_bpe_token_length": 9.0
```

The nucleotide slice length is derived from these two values. Gene-finding overlap is then calculated in nucleotide coordinates from that derived length. Tokenization may produce fewer tokens, in which case the model input is padded, or more tokens, in which case it is truncated to `max_bpe_tokens`.

Direct GENA plain/U-Net models use absolute position embeddings and the released
backbones support at most 512 BPE positions. Their shipped configs therefore use
`max_bpe_tokens: 512`, and the code raises a clear error if a custom direct-GENA
config exceeds the backbone limit. RMT and AMT may accept longer outer sequences
because they segment tokens before each GENA backbone call. ModernGENA uses its
own long-context mechanism and is configured separately.

### `training`

Important fields:

```json
{
  "output_dir": "runs/segmentation_moderngena_base_unet",
  "custom_prefix": "experiment_a",
  "num_train_epochs": 4,
  "max_steps": -1,
  "per_device_train_batch_size": 1,
  "per_device_eval_batch_size": 1,
  "eval_accumulation_steps": 1,
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

Every new training invocation treats `output_dir` as the base/run-family directory and creates a new timestamped child directory beneath it; it does not reuse or overwrite an earlier run. `custom_prefix` is an optional string prepended to the timestamped run name. Leave it empty when no extra label is needed. Resume paths still point to the exact checkpoint directory from the run being resumed.

`logging_interval` is the primary interval field.  It is mapped to `TrainingArguments.logging_steps`.  If `eval_interval` or `save_interval` are omitted, they default to `logging_interval` and `eval_interval`, respectively.

Training does not run a test phase. Validation during training and post-training evaluation are separate operations.


### Direct parquet loading for transcript datasets

The segmentation and transcript-type tasks use `AIRI-Institute/genatator-gene-segmentation-dataset`.  These parquet files contain long nested nucleotide-label arrays.  The training code does **not** use `datasets.load_dataset(...)` for this dataset, because preparing the Hugging Face Arrow cache can fail with PyArrow `List index overflow` on large nested lists.  Instead, configs use:

```json
"loader": "direct_parquet",
"parquet_batch_size": 64
```

The direct loader resolves the requested `config_name` (`train-human`, `train-multi-specie`, or `val-human`), downloads or reuses the exact parquet files from the Hugging Face cache, scans them in bounded PyArrow batches, applies the requested genome, chromosome, and status filters, and then materializes only the selected rows into CPU RAM before training starts. For local parquet files or local dataset roots, the same direct loader is used when `loader` is set to `direct_parquet`.

## Normal training

Run training commands from the repository root.  Either install the repository in editable mode or set `PYTHONPATH` to the repository root.  The most robust form is to run task scripts as modules.

```bash
cd /path/to/GENATATOR
export PYTHONPATH=$PWD:$PYTHONPATH
```

### Two-GPU ModernGENA segmentation on the human dataset

ModernGENA is BPE-based, while segmentation is evaluated at nucleotide resolution.  Therefore ModernGENA segmentation without RMT/AMT must use the U-Net wrapper:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node=2 \
  -m segmentation.train \
  --config segmentation/configs/moderngena_base_unet.json
```

The human segmentation config uses:

```json
"train_dataset": {
  "path": "AIRI-Institute/genatator-gene-segmentation-dataset",
  "config_name": "train-human",
  "split": "train"
},
"eval_dataset": {
  "path": "AIRI-Institute/genatator-gene-segmentation-dataset",
  "config_name": "val-human",
  "split": "validation"
}
```

To train on multispecies data, set `train_dataset.config_name` to `train-multi-specie`.  To restrict training or validation to particular chromosomes, set `chromosomes`, for example:

```json
"chromosomes": ["NC_060944.1"]
```

For all chromosomes in the selected dataset/configuration, leave `chromosomes` as an empty list.

### Gene-finding training

Edge and region models are trained separately:

```bash
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  -m finding.train \
  --task edge \
  --config finding/configs/edge_moderngena_base_plain.json

PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  -m finding.train \
  --task region \
  --config finding/configs/region_moderngena_base_plain.json
```

Use `target_group` in the dataset config to choose the gene-finding target half:

```json
"target_group": "primary"  // combined mRNA + lncRNA channels
"target_group": "mrna"     // mRNA/protein-coding-only channels
```

### Transcript-type training

```bash
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  -m transcript_type.train \
  --config transcript_type/configs/moderngena_base_plain.json
```

### Resume training

Set `training.resume_from_checkpoint` in the JSON config:

```json
"resume_from_checkpoint": "runs/segmentation_moderngena_base_unet/<timestamped-run>/checkpoint-10000"
```

Leave it as `null` to start from the backbone checkpoint.

### Logging

All training configs expose `logging_interval`, which maps to TensorBoard logging steps. `eval_interval` and `save_interval` control validation and checkpointing. TensorBoard logs are written inside the timestamped run directory.

### Validation memory

Validation uses a rank-0 streaming loop for all tasks. Rank 0 evaluates the selected validation set sequentially with the unwrapped model, moves each batch of logits and labels to CPU immediately, updates task-specific metric accumulators, writes the small metric dictionary under the run's `rank0_eval_metrics/` directory, and the other ranks wait for that file without NCCL collectives.

This avoids the default Transformers evaluation behavior where logits and labels are all-gathered across ranks.  That default behavior can exhaust GPU memory or trigger NCCL timeouts for long nucleotide-resolution validation examples.  The model forward pass, labels, loss, and metric definitions are unchanged; only validation accumulation is moved from distributed GPU all-gather to rank-0 CPU streaming.

## Post-training evaluation

### Generated evaluation configs

Every training run creates:

```text
<run-directory>/evaluation_config.json
```

The file contains the complete model definition and the held-out dataset definition needed by that task. Its checkpoint path is updated automatically whenever the best checkpoint changes. A copy is also stored in checkpoint directories, so the architecture and evaluation settings remain next to the saved weights. Do not replace the copied model block with a generic inference example: U-Net, RMT, AMT, tokenizer, and Caduceus settings are architecture-critical.

Run post-training evaluation on one GPU from the repository root. It does not require `torchrun`.

### Evaluate one gene-finding stage

Edge and region models are trained and scored independently. Run the generated config from either run with:

```bash
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 \
  python -m finding.evaluate \
  --config runs/<edge-or-region-run>/evaluation_config.json
```

This evaluates the trained stage on its held-out chromosome data and writes its PR-AUC metrics. It does not create transcript intervals because interval construction requires both a trained edge model and a trained region model.

### Run complete gene-finding inference

To evaluate a complete gene model, put the matching trained edge and region checkpoints into the two stage blocks of a paired inference config, then run:

```bash
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 \
  python -m finding.infer \
  --config finding/configs/infer_moderngena_base_plain.json
```

The paired models must use compatible backbones/tokenizers, the same target group, and the same chromosome subset. The command writes a genome-coordinate GFF and whole-chromosome PR-AUC for all edge and region channels. When `inference.true_gff` is set, it also runs the official annotation metric with the configured boundary tolerances. The reference GFF must cover the same assembly and chromosome subset as the prediction; passing a whole-genome reference while predicting one chromosome gives a misleading score.

### Evaluate segmentation

Run the generated config from a segmentation run with:

```bash
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 \
  python -m segmentation.infer \
  --config runs/<segmentation-run>/evaluation_config.json
```

Training-time human validation deliberately keeps representative rows with status 1. On `NC_060944.1`, that is 963 transcript rows representing 963 genes; the dataset contains 980 genes in total, but 17 have no status-1 row. This is expected and is not a loader loss. Separate final segmentation evaluation removes the status filter and evaluates all 3,998 transcripts from all 980 genes on that chromosome.

Segmentation inference writes a transcript-coordinate GFF (`seqid = transcript_id`). If a model-length crop starts after transcript position zero, exon and CDS intervals are shifted back by the crop's `local_start` before writing, so they remain aligned to the original transcript coordinate system. A transcript longer than the selected model context contributes one deterministic slice; segmentation and transcript-type tasks do not create overlapping windows or stitch multiple predictions.

Set `inference.true_gff` to run the official gene-level segmentation metric and write `inference.metrics_json`. If it is `null`, inference still writes predictions but does not run the remote official metric.

Segmentation evaluation has a switchable CDS decoder:

```json
"use_cds_heuristic": true
```

- `true` matches GENATATOR-PIPELINE: predicted exons are spliced, all three reading frames are translated, and the longest complete methionine-to-stop ORF is mapped back to the exon intervals. Only mRNA records receive CDS intervals; no partial ORF is invented when a complete ORF is absent.
- `false` keeps CDS intervals decoded directly from the model's CDS channel.

The default is `true`, matching the pipeline. The benchmark heuristic has no minimum-length or partial-ORF tuning knobs.

### Evaluate transcript type

Run the generated config from a transcript-type run with:

```bash
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 \
  python -m transcript_type.infer \
  --config runs/<transcript-type-run>/evaluation_config.json
```

The command writes one TSV row per selected transcript with the true type, lncRNA probability, and predicted type, plus a JSON file containing accuracy. Its threshold is applied to the lncRNA probability. As with segmentation, each long transcript contributes one deterministic, non-overlapping model-length slice.

### Evaluation checklist

Before launching a long evaluation, verify:

1. `checkpoint_path` points to the automatically selected best checkpoint and exists locally.
2. The dataset split, assembly, chromosomes, and transcript-status policy match the intended benchmark.
3. BPE models retain the training run's `max_bpe_tokens` and `average_bpe_token_length`; nucleotide models retain their nucleotide context length.
4. Models containing a U-Net retain `model.unet_chunk_size` and their nucleotide tokenizer settings.
5. Prediction and metric output paths are unique if several evaluations will run concurrently.
6. A reference GFF is restricted to the same sequences being predicted.



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

Gene finding is the only task that uses overlapping training windows. Windows are produced independently within each chromosome, normally with 50% overlap measured in nucleotides. For BPE models, the nucleotide window length is derived from `max_bpe_tokens` and `average_bpe_token_length`, and targets are projected to token resolution by taking the maximum label value over the nucleotide span covered by each token. For U-Net/RMT/AMT+U-Net models, BPE hidden states are expanded back to nucleotide positions with `embedding_repeater`; positions truncated away by the BPE tokenizer are masked out.

Before window generation, the requested dataset subset is loaded into CPU RAM.  The code logs:

- number of selected rows/blocks;
- number of selected chromosomes/genomes;
- selected disk size;
- expected sequence and target RAM;
- selected target indices and target names;
- assembled chromosome length per chromosome.

### Training metrics

During training/validation only PR-AUC is computed.  The order is fixed and matches the HF dataset target layout:

- edge: `pr_auc_TSS+`, `pr_auc_TSS-`, `pr_auc_PolyA+`, `pr_auc_PolyA-`
- region: `pr_auc_intragenic+`, `pr_auc_intragenic-`

Each class also logs `*_defined`, `*_positives`, `*_negatives`, and `*_dropped_nonfinite`.  Non-finite model scores are dropped before calling sklearn; an undefined channel is reported as `0.0` with `*_defined=0.0` so validation never crashes because a short smoke run emitted NaN/Inf logits.

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

The dataset retains all annotated transcripts. Training-time validation uses status-1 representative rows. Final segmentation evaluation intentionally removes the status filter so all isoforms in the selected validation subset are scored.

### Sampling and BPE handling

The model input is only the transcript DNA sequence. It must not include intergenic regions, neighboring genes, or other chromosome context. Each selected transcript contributes exactly one model-length slice. The crop keeps the configured left margin (normally about 500 nt) when the transcript is long enough; it is never expanded into overlapping windows. Caduceus slice length is defined in nucleotides, while GENA/ModernGENA slice length is derived from the BPE fields described above. Inference writes transcript-coordinate predictions.

GENA/ModernGENA segmentation must use U-Net/RMT/AMT+U-Net so that BPE states are expanded to nucleotide resolution before loss and metrics are computed.

### Training metrics

Only exact interval-level F1 is computed during training/validation:

```text
interval_f1_exon
interval_f1_CDS
```

UTR and intron training-time metrics are intentionally excluded.  Gene-level and MI metrics are computed only by `segmentation/infer.py` through the official Evaluate metric.

### Validation and long transcripts

Training-time validation uses the same one-slice model-context construction as training, with deterministic cropping. If a transcript is longer than the model context, validation and standalone inference evaluate one deterministic model-length slice rather than feeding the whole transcript or reconstructing it from overlapping windows. Standalone inference restores the crop offset in the transcript-coordinate GFF before the official metric is called.

For memory safety, validation uses the rank-0 streaming loop described in the training section.  No validation logits or labels are gathered across GPUs.  Each evaluated batch is converted to CPU NumPy arrays immediately and only task-specific counters or CPU arrays required for the metric are retained.  This same evaluation rule is applied to gene finding, segmentation, and transcript-type training.

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

Run the focused regression suite after installing the project dependencies:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

The focused tests cover config contracts, BPE length derivation, deterministic
transcript slicing, reverse-complement alignment, U-Net sample/chunk semantics,
RMT batch identity, run/checkpoint preservation, and strict checkpoint loading.

Smoke tests use real data and are designed to verify that every task/model path can train, validate, infer, and write metrics on the selected held-out subset. The runner does not enforce any metric-based success criterion; inspect the metrics manually.

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
7. trains, validates, and tests on the same selected subset so metric changes are easy to inspect manually;
8. assigns one GPU per active task/model job;
9. writes `summary.md` with durations and metrics.

For gene finding, smoke tests scan the real T2T `test` split and then train/validate/test on the first 10% of one selected chromosome parquet block.  This keeps the smoke run short while using real nucleotide/target arrays.  For segmentation and transcript type, smoke tests use the selected chromosome transcripts from `val-human`.

Use `--refresh-index` to rescan metadata.  Without it, stored indexes are reused.
