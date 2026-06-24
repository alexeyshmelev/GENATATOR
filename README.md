# GENATATOR fine-tuning and evaluation

This repository trains and evaluates the four model stages used by the GENATATOR ab initio annotation pipeline:

```text
finding/            edge and region models for transcript-boundary discovery
segmentation/       nucleotide-resolution exon/CDS segmentation
transcript_type/    mRNA versus lnc_RNA classification
genatator_core/     shared loaders, model wrappers, training, inference, metrics, and GFF utilities
smoke_tests/        real-data, one-chromosome end-to-end validation
```

The repository is for **fine-tuning pretrained DNA backbones**. It does not implement language-model pretraining. Training performs train/validation only. Final test inference and benchmark metrics are always started separately.

## Installation

Keep the existing PyTorch installation, then install the repository dependencies:

```bash
python -c "import torch; print(torch.__version__)"  # expected: 2.2.2+cu121
python -m pip install -e .
python -m pip install -r requirements.txt
```

`requirements.txt` intentionally does not install or upgrade PyTorch. The remaining stack includes Transformers, Datasets, Evaluate, Accelerate, scikit-learn, PyArrow, TensorBoard, tqdm, SciPy, and safetensors. Caduceus additionally requires the dependencies used by its Hugging Face remote model code.

### PyTorch 2.2.2 compatibility

The code supports a runtime pinned to:

```text
torch==2.2.2+cu121
```

Recent Transformers releases block `torch.load` on torch versions below 2.6. Several trusted GENA checkpoints and all Trainer checkpoints in this repository use `pytorch_model.bin`, because tied model parameters can make direct safetensors serialization invalid. When:

```json
"allow_unsafe_torch_load_with_torch_lt_2_6": true
```

is present in the model configuration, the repository explicitly patches all three Transformers checkpoint-loading locations used by:

- model/backbone loading;
- resume-from-checkpoint;
- `Trainer._load_best_model()` at the end of training.

The patch is logged with the exact context and module names. It must be used only with trusted checkpoints. Training configs set:

```json
"save_safetensors": false
```

and save model weights as `pytorch_model.bin`.

## Local and Hugging Face paths

There is no `source` field. Every dataset, tokenizer, backbone, and fine-tuned checkpoint string is resolved in the same way:

1. if the path exists locally, it is loaded locally;
2. otherwise, the string is passed to Hugging Face as a repository ID.

Examples:

```json
"backbone_path": "AIRI-Institute/moderngena-base"
```

```json
"backbone_path": "/models/moderngena-base"
```

```json
"path": "AIRI-Institute/genatator-gene-finding-dataset"
```

```json
"path": "/datasets/genatator-gene-finding-dataset"
```

Tokenizers are configured separately with `model.tokenizer_path`. UNET/RMT/AMT+UNET models also require `model.nucleotide_tokenizer_path`.

## Supported backbones and model families

| `model.family` | Supported backbones | Output resolution | Active implementation |
|---|---|---:|---|
| `plain` | GENA, ModernGENA | BPE/token | `PlainTokenClassifier`; `TranscriptTypeClassifier` |
| `unet` | GENA, ModernGENA | nucleotide | `TokenClassifierWithUNet` |
| `rmt` | GENA, ModernGENA | nucleotide | `RMTEncoderForLetterLevelTokenClassificationUNETsegmentedRepeater` |
| `amt` | GENA, ModernGENA | BPE/token, or nucleotide with `use_unet=true` | `AMTTokenClassifier` |
| `caduceus` | Caduceus PS/PH | nucleotide | middle-loss token or transcript-type classifier |

Supported public backbones include:

```text
AIRI-Institute/moderngena-base
AIRI-Institute/moderngena-large
AIRI-Institute/gena-lm-bert-base-lastln-t2t
AIRI-Institute/gena-lm-bert-large-t2t
kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16
kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16
```

### Enforced model rules

- Every parameter is trainable. There is no freezing option.
- RMT and AMT are restricted to GENA and ModernGENA.
- RMT is never applied to Caduceus.
- Caduceus always uses the middle-loss implementation.
- GENA/ModernGENA segmentation must be nucleotide-resolution: `unet`, `rmt`, or `amt` with `use_unet=true`.
- `unet`, `rmt`, and `amt` with UNET require train/eval batch size 1.
- Caduceus supports batch size greater than 1 through tensorized batch operations.
- RMT defaults to three UNET refinement cycles where the configuration sets `cycles=3`.
- Hidden size, embedding width, classifier width, and UNET input width are detected from the loaded backbone and logged.
- Caduceus PS/PH hidden-state widths are inferred and then verified on the first forward pass.

## Repository entry points

### Training

```bash
python finding/train.py --task edge --config finding/configs/edge_moderngena_base_plain.json
python finding/train.py --task region --config finding/configs/region_moderngena_base_plain.json
python segmentation/train.py --config segmentation/configs/moderngena_base_unet.json
python transcript_type/train.py --config transcript_type/configs/moderngena_base_plain.json
```

### Inference and final evaluation

```bash
python finding/infer.py --config finding/configs/infer_moderngena_base_plain.json
python segmentation/infer.py --config segmentation/configs/infer_caduceus_ps.json
python transcript_type/infer.py --config transcript_type/configs/infer_moderngena_base.json
```

No training entry point automatically runs a test phase.

# Configuration reference

All configuration files are JSON. Hydra and YAML are not used.

## Training configuration

```json
{
  "seed": 42,
  "model": {},
  "train_dataset": {},
  "eval_dataset": {},
  "training": {}
}
```

## Inference configuration

Segmentation and transcript type use:

```json
{
  "model": {},
  "dataset": {},
  "inference": {}
}
```

Gene finding uses separate edge and region stages:

```json
{
  "edge": {"model": {}, "dataset": {}, "inference": {}},
  "region": {"model": {}, "dataset": {}, "inference": {}},
  "postprocess": {},
  "inference": {}
}
```

## `model` fields

| Field | Meaning |
|---|---|
| `family` | `plain`, `unet`, `rmt`, `amt`, or `caduceus`. |
| `backbone_kind` | `gena`, `moderngena`, or `caduceus`. |
| `backbone_path` | Local backbone directory or HF model ID. Only the backbone is loaded before the local fine-tuning wrapper is constructed. |
| `tokenizer_path` | Local tokenizer directory or HF tokenizer ID. Special and padding IDs are read from the tokenizer. |
| `trust_remote_code` | Passed to HF model/tokenizer loading. |
| `checkpoint_path` | Optional local fine-tuned checkpoint loaded into the constructed wrapper. `null` starts from the pretrained backbone. |
| `allow_unsafe_torch_load_with_torch_lt_2_6` | Explicit trusted-checkpoint compatibility switch for torch 2.2.2. |
| `padding_side` | Optional main-tokenizer padding side. Caduceus defaults to `left`. |
| `nucleotide_tokenizer_path` | Required by `unet`, `rmt`, and `amt` with UNET. |
| `nucleotide_padding_side` | Optional nucleotide-tokenizer padding side. |
| `nucleotide_vocab_size` | Nucleotide embedding-table size for UNET paths. |
| `unet_cycles` | Number of UNET refinement cycles for the direct UNET and AMT+UNET paths. |
| `unet_channels` | Optional explicit 1D-UNET channel list. Omit to use the implementation defaults. |
| `cycles` | RMT UNET refinement cycles. |
| `bidirectional_weight_tie` | Caduceus configuration field. Public training configs use `false`, matching the middle-loss setup. |
| `hidden_size` | Optional explicit Caduceus hidden width. Normally inferred and checked. |
| `rmt` | RMT-specific settings described below. |
| `amt` | AMT-specific settings described below. |

### `model.rmt`

| Field | Meaning |
|---|---|
| `input_size` | Token length of one recurrent segment, including memory/special tokens. |
| `max_n_segments` | Maximum number of recurrent segments. |
| `num_mem_tokens` | Number of recurrent memory tokens. |
| `bptt_depth` | Truncated-BPTT depth; `-1` keeps the full recurrent graph. |
| `unet_sub_model_input_size` | Nucleotide chunk length processed by the segmented UNET head. |

The RMT wrapper automatically obtains the loaded backbone hidden size and embedding table. It maps BPE hidden states to nucleotide positions with `embedding_repeater`, concatenates nucleotide embeddings, and applies the same UNET head used by the direct UNET path.

### `model.amt`

| Field | Meaning |
|---|---|
| `amt_repo_id` | HF repository that exposes `AssociativeMemoryCell` and `AssociativeRecurrentWrapper`. |
| `num_mem_tokens` | Number of associative memory tokens. |
| `d_mem` | Associative memory dimension. |
| `segment_size` | AMT segment length. |
| `segment_alignment` | Segment alignment, normally `left`. |
| `sliding_window` | Enables AMT wrapper sliding-window behavior. |
| `layers_attr` | Layer path inside the wrapped backbone. Defaults are selected for GENA and ModernGENA. |
| `wrap_pos` | AMT positional wrapping switch. |
| `correction` | AMT correction switch. |
| `n_heads` | Number of associative memory heads. |
| `use_denom` | AMT denominator switch. |
| `gating` | AMT gating switch. |
| `act_on` | Adaptive-computation switch. |
| `max_hop`, `act_type`, `constant_depth`, `act_format`, `noisy_halting`, `attend_to_previous_input`, `use_sink`, `time_penalty` | Forwarded unchanged to the AMT implementation when present. |

## Dataset fields

| Field | Meaning |
|---|---|
| `path` | Local dataset/file path or HF dataset ID. |
| `config_name` | HF configuration name, such as `train-human`, `train-multi-specie`, or `val-human`. |
| `split` | Split name. |
| `data_files` | Optional local file or file-pattern specification. |
| `genomes` | Optional assembly whitelist. `[]` or `null` means no genome filter. |
| `chromosomes` | Optional chromosome/contig whitelist. Simple `chr20`/`20` aliases are recognized. |
| `statuses` | Optional transcript `status` whitelist. Use `[1]` for representative transcripts. |
| `max_rows` | Optional row cap. Omit for full data. |
| `max_nucleotides` | Nucleotide context/output length. |
| `max_tokens` | BPE token context length. |
| `overlap` | Sliding-window overlap; gene-finding defaults use `0.5`. |
| `target_group` | `primary` for combined mRNA+lncRNA targets or `mrna` for mRNA-only finding channels. |
| `crop_margin` | Minimum preferred transcript crop offset from each transcript boundary. Default `500`. |
| `random_crop` | Randomizes transcript crops during training. |
| `streaming` | Streams a remote dataset and materializes only matching rows. Normal cluster training can use it when appropriate. |
| `streaming_max_scanned_rows` | Maximum rows scanned by streaming selection. |
| `streaming_trim_rows` | Optional debugging-only trimming of selected streamed rows. |
| `prewindowed` | Treats each row as an already prepared model window. Standard gene-finding runs use chromosome assembly and set this to `false` or omit it. |
| `max_windows` | Optional debug cap applied after window generation. Omit for complete data. |

## `training` fields

| Field | Meaning |
|---|---|
| `output_dir` | Run directory. |
| `overwrite_output_dir` | Allows replacement of an existing run directory. |
| `max_steps` | Optimizer-step limit. `-1` uses `num_train_epochs`. |
| `num_train_epochs` | Number of complete dataset passes when `max_steps=-1`. |
| `per_device_train_batch_size`, `per_device_eval_batch_size` | Per-GPU batch sizes. |
| `gradient_accumulation_steps` | Microbatch accumulation count. |
| `learning_rate`, `weight_decay` | AdamW hyperparameters. |
| `warmup_steps`, `lr_scheduler_type` | Scheduler settings. |
| `logging_strategy`, `logging_steps` | Trainer/TensorBoard logging schedule. |
| `evaluation_strategy` or `eval_strategy`, `eval_steps` | Validation schedule. The installed Transformers spelling is detected automatically. |
| `save_strategy`, `save_steps`, `save_total_limit` | Checkpoint schedule and retention. |
| `save_safetensors` | Keep `false` for tied/shared GENATATOR wrappers. |
| `load_best_model_at_end` | Restores the best checkpoint after training. Supported with torch 2.2.2 by the explicit trusted-checkpoint compatibility path. |
| `metric_for_best_model` | Default configs use `loss` for finding, `interval_f1_exon` for segmentation, and `accuracy` for transcript type. |
| `greater_is_better` | Direction for the selected metric. |
| `dataloader_num_workers` | DataLoader worker count. Smoke tests use `0`; cluster configs use more workers. |
| `bf16`, `fp16` | Mixed-precision switches. |
| `resume_from_checkpoint` | Checkpoint directory to resume optimizer, scheduler, model, and Trainer state. `null` starts a new fine-tuning run. |
| `sequential_train` | Uses a sequential sampler. Smoke gene-finding runs enable it to traverse chromosome blocks/windows in genomic order. |

All runs log to TensorBoard under:

```text
<training.output_dir>/tensorboard
```

and display train/evaluation progress with tqdm.

# Dataset semantics

## Gene finding

Dataset:

```text
AIRI-Institute/genatator-gene-finding-dataset
```

Each Parquet row is one contiguous genomic block, up to 10 Mb. The loader groups selected rows by `(genome, chromosome)`, sorts them by `metadata.start`, constructs a chromosome assembly, and traverses fixed-length windows with the configured overlap. A requested window can cross a Parquet block boundary. Selected Parquet blocks are loaded lazily, one block at a time.

The combined (`target_group="primary"`) channel order is fixed:

```text
0  primary_tss_+            -> TSS+
1  primary_tss_-            -> TSS-
2  primary_polya_+          -> PolyA+
3  primary_polya_-          -> PolyA-
4  intragenic_regions_+     -> intragenic+
5  intragenic_regions_-     -> intragenic-
```

The mRNA-only channels occupy indexes 6–11 in the same order.

Edge boundary tracks are smooth nucleotide-level targets. Training-time PR-AUC treats every target value greater than zero as positive. Intragenic tracks are binary.

## Segmentation and transcript type

Dataset:

```text
AIRI-Institute/genatator-gene-segmentation-dataset
```

Each row is one complete transcript with fields:

```text
dna_sequence
labels
metadata
status
```

The nucleotide label order is fixed:

```text
0  5UTR
1  exon
2  intron
3  3UTR
4  CDS
```

Transcript metadata encode transcript ID, gene ID, transcript type, strand, assembly, chromosome, and genomic interval. Transcript type maps mRNA to class `0` and lnc_RNA aliases to class `1`.

Long transcript training samples are cropped to the configured context. With random cropping, the loader respects `crop_margin` when the transcript is long enough.

# Training and validation metrics

Only lightweight development metrics are computed during validation inside training. Benchmark, gene-level, multi-isoform, MI, stratified, and detailed transcript metrics are **not** run by the training loop.

## Gene-finding edge model

PR-AUC is reported independently in this exact order:

```text
eval_pr_auc_TSS+
eval_pr_auc_TSS-
eval_pr_auc_PolyA+
eval_pr_auc_PolyA-
```

No ROC-AUC, anonymous channel names, or aggregate gene-level metrics are calculated during training.

## Gene-finding region model

PR-AUC is reported independently in this exact order:

```text
eval_pr_auc_intragenic+
eval_pr_auc_intragenic-
```

## Segmentation model

Only exact interval-level F1 for exon and CDS is reported:

```text
eval_interval_f1_exon
eval_interval_f1_CDS
```

TP, FP, and FN are accumulated across the complete validation set before F1 is calculated. No training-time metric is calculated for 5UTR, 3UTR, or intron.

## Transcript-type model

Only accuracy is reported:

```text
eval_accuracy
```

# Inference and final test metrics

## Gene finding

```bash
python finding/infer.py --config finding/configs/infer_moderngena_base_plain.json
```

Inference:

1. predicts edge and region tracks over the requested assembled chromosome;
2. expands BPE predictions to nucleotide resolution;
3. optionally averages forward and reverse-complement passes;
4. gathers the complete chromosome prediction and truth tracks;
5. computes whole-chromosome PR-AUC for `TSS+`, `TSS-`, `PolyA+`, `PolyA-`, `intragenic+`, and `intragenic-`;
6. applies FFT low-pass filtering and peak calling to edge tracks;
7. pairs strand-compatible TSS/polyA peaks;
8. filters candidate intervals with region-model intragenic tracks;
9. writes a genome-coordinate GFF;
10. optionally invokes the official annotation leaderboard metric through `evaluate`.

Gene-finding models do not predict internal exon/intron structure. Each predicted transcript interval is therefore represented in GFF as one transcript with one exon spanning the full interval. Internal segmentation is evaluated only by the segmentation task.

### FFT and interval post-processing

| `postprocess` field | Meaning | Default |
|---|---|---:|
| `lp_frac` | Fraction of Fourier coefficients retained by low-pass filtering. | `0.05` |
| `pk_prom` | `scipy.signal.find_peaks` prominence. | `0.1` |
| `pk_dist` | Minimum peak distance in bp. | `50` |
| `pk_height` | Optional minimum peak height. | `null` |
| `interval_window_size` | Maximum TSS/polyA pairing distance. | `2000000` |
| `max_pairs_per_seed` | Maximum nearest partners retained per seed peak. | `10` |
| `prob_threshold` | Region-track threshold. | `0.5` |
| `zero_fraction_drop_threshold` | Maximum fraction of non-intragenic bases allowed inside a candidate. | `0.01` |
| `pairing_progress_every` | Optional explicit pairing progress-log interval. | `null` |

Final GFF evaluation uses:

```python
import evaluate
metric = evaluate.load("AIRI-Institute/genatator-ab-initio-annotation-leaderboard")
```

with configurable `k_values`, `use_strand`, and optional transcript-type filtering supported by the metric.

## Segmentation

```bash
python segmentation/infer.py --config segmentation/configs/infer_caduceus_ps.json
```

The model receives the DNA sequence of one individual transcript, without intergenic context or neighboring genes. Prediction GFF is transcript-coordinate:

```text
seqid = transcript_id
```

Exon/CDS coordinates are relative to that transcript. This is intentionally different from a genome-coordinate browser track.

Final evaluation uses:

```python
import evaluate
metric = evaluate.load(
    "AIRI-Institute/genatator-ab-initio-segmentation-leaderboard",
    revision="metric-only",
)
result = metric.compute_gene_level_gff(
    pred_gff="predictions.gff",
    true_gff="reference.gff",
    stratifier="type",
    types=["mRNA", "lnc_RNA"],
    segments=["exon", "CDS"],
)
```

The reference is a standard genome-oriented GFF containing gene rows, mRNA/lnc_RNA transcript rows, and exon/CDS children.

## Transcript type

```bash
python transcript_type/infer.py --config transcript_type/configs/infer_moderngena_base.json
```

Inference writes a TSV containing transcript metadata, lncRNA probability, and predicted class. The metrics JSON contains accuracy only.

## Reverse-complement averaging

Every inference task supports:

```json
"use_reverse_complement": true
```

or:

```json
"use_reverse_complement": false
```

Finding swaps strand-specific channels before averaging. Segmentation reverses the sequence axis and swaps 5UTR with 3UTR. Transcript type averages scalar probabilities.

# Smoke tests

Smoke tests use real held-out data from one requested chromosome. They deliberately use the same selected test subset for training, validation, and separate inference so loss reduction/overfitting verifies that model construction, gradients, checkpointing, loading, metrics, and final inference all work.

## Data selection

Before launching GPUs, the smoke runner:

1. resolves the local HF system cache or an explicitly supplied local dataset root;
2. lists only the gene-finding `test` Parquet files;
3. scans every test sample’s metadata with tqdm;
4. retains every block whose metadata match the requested chromosome;
5. stores the selected block index under `smoke_tests/indexes/`;
6. lists only the segmentation `val-human` Parquet source;
7. scans every transcript metadata row with tqdm;
8. copies every requested-chromosome transcript to a compact selected Parquet file;
9. persists both indexes so later runs do not repeat metadata scans.

Rejected samples are discarded immediately. Gene-finding keeps only one selected large block in RAM at a time. Segmentation/transcript-type selected rows are stored in one local compact Parquet file.

All selected samples are used. There is no smoke row/window sampling control. The only training-size control is the number of complete epochs.

## Scheduling

Each active task/model job owns one GPU. The launcher runs up to `--num-gpus` independent jobs concurrently. An inference job waits for its matching training job. Any subprocess failure stops all active jobs and prints the command plus log tail.

## Run

```bash
python smoke_tests/run_smoke.py \
  --num-gpus 2 \
  --reference-gff /path/to/chr20.gff3 \
  --requested-chromosome NC_060944.1 \
  --work-dir smoke_tests/runs \
  --smoke-cache-dir /path/to/persistent/selected_data \
  --smoke-epochs 4
```

Specific GPUs:

```bash
python smoke_tests/run_smoke.py \
  --num-gpus 2 \
  --gpus 0,2 \
  --reference-gff /path/to/chr20.gff3 \
  --smoke-epochs 4
```

Reuse existing indexes by omitting `--refresh-index`. Force a complete metadata rescan with:

```bash
--refresh-index
```

Use only already cached HF files with:

```bash
--hf-local-files-only
```

Use explicit local dataset roots with:

```bash
--gene-finding-dataset-path /path/to/gene-finding-dataset
--segmentation-dataset-path /path/to/gene-segmentation-dataset
```

The runner writes:

```text
smoke_tests/runs/configs/
smoke_tests/runs/logs/
smoke_tests/runs/jobs.json
smoke_tests/runs/summary.md
```

`summary.md` records selected data counts, chromosome length, windows per epoch, job durations, train/eval loss changes, test metric files, and log paths.
