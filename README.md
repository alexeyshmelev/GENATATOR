# GENATATOR fine-tuning repository

This repository is a cleaned first-pass training and inference stack for the GENATATOR collection. It is intentionally organized around the three biological tasks in the annotation pipeline:

```text
finding/            # edge and region models for transcript interval discovery
segmentation/       # nucleotide-level exon/intron/UTR/CDS segmentation
transcript_type/    # mRNA versus lnc_RNA classification
genatator_core/     # shared data, model, training, inference, metrics, and GFF utilities
smoke_tests/        # real-data smoke-test launcher
```

The code is for **fine-tuning only**. It does not implement pretraining.

## Supported backbones and active wrappers

The repository automatically treats every dataset, tokenizer, model, and checkpoint path as local when the path exists; otherwise the value is passed directly to Hugging Face.

| Config `model.family` | Backbones | Output resolution | Active class / wrapper |
|---|---|---|---|
| `plain` | GENA, ModernGENA | BPE/token | `PlainTokenClassifier` or `TranscriptTypeClassifier` |
| `unet` | GENA, ModernGENA | nucleotide | `TokenClassifierWithUNet` |
| `rmt` | GENA, ModernGENA only | nucleotide | `RMTEncoderForLetterLevelTokenClassificationUNETsegmentedRepeater` |
| `amt` | GENA, ModernGENA only | BPE/token or nucleotide with `use_unet=true` | `AMTTokenClassifier` |
| `caduceus` | Caduceus PS/PH | nucleotide | `CaduceusMiddleLossTokenClassifier` or `CaduceusTranscriptTypeMiddleLossClassifier` |

Important constraints are enforced at startup:

```text
RMT is never adapted to Caduceus.
AMT is never adapted to Caduceus.
GENA/ModernGENA segmentation must use nucleotide output: family=unet, family=rmt, or family=amt with use_unet=true.
RMT, AMT+UNET, and plain UNET require per-device train/eval batch size 1.
Caduceus supports batch size greater than 1 for token tasks.
All model parameters are trainable; there is no freezing option in configs.
```

ModernGENA plain fine-tuning loads the backbone through `transformers.ModernBertForTokenClassification`. GENA uses `AutoModel`. Caduceus uses `AutoModel` with `trust_remote_code=true` and middle-loss heads only.

## Installation

```bash
pip install -e .
pip install -r requirements.txt
```

Caduceus requires its remote-code dependencies, including Mamba-related packages. The smoke tests and real runs use CUDA GPUs.

## JSON config structure

Every training config has four top-level sections:

```json
{
  "seed": 42,
  "model": {},
  "train_dataset": {},
  "eval_dataset": {},
  "training": {}
}
```

Every inference config has:

```json
{
  "model": {},
  "dataset": {},
  "inference": {}
}
```

The finding inference config has separate `edge` and `region` stage configs plus global `postprocess` and `inference` sections.

### `model` parameters

| Parameter | Meaning |
|---|---|
| `family` | One of `plain`, `unet`, `rmt`, `amt`, `caduceus`. |
| `backbone_kind` | One of `gena`, `moderngena`, `caduceus`. |
| `backbone_path` | Local path or HF repo ID for the backbone only. |
| `tokenizer_path` | Local path or HF repo ID for the main tokenizer. |
| `nucleotide_tokenizer_path` | Required for `unet`, `rmt`, and `amt` with `use_unet=true`; used to map A/C/G/T to nucleotide IDs. |
| `trust_remote_code` | Passed to HF loading calls. Usually `true` for GENA/Caduceus. |
| `checkpoint_path` | Optional fine-tuned model checkpoint loaded into the local wrapper before training or inference. |
| `bidirectional_weight_tie` | Caduceus setting. Default configs use `false`, matching the working middle-loss setup. |
| `unet_cycles` | Number of recurrent UNET refinement cycles for `unet` and `amt` with UNET. |
| `cycles` | Number of RMT UNET refinement cycles. Default RMT configs use `3`. |
| `nucleotide_vocab_size` | Size of nucleotide embedding table for UNET models. |
| `rmt` | RMT settings: `input_size`, `max_n_segments`, `num_mem_tokens`, `bptt_depth`, `unet_sub_model_input_size`. |
| `amt` | AMT settings: `amt_repo_id`, `num_mem_tokens`, `d_mem`, `segment_size`, and optional AMT wrapper parameters. |

The code logs detected hidden sizes, embedding-table shapes, memory-token settings, UNET input dimensions, tokenizer IDs, model family, and parameter counts. Caduceus no longer uses lazy heads: PS hidden width is inferred as `2 * d_model`, PH hidden width as `d_model`, and the first forward pass verifies the emitted hidden-state shape explicitly.

### Dataset parameters

The repository supports both HF datasets and local mirrors. There is no source selector.

| Parameter | Meaning |
|---|---|
| `path` | Local dataset directory/file or HF dataset repo ID. |
| `config_name` | HF dataset configuration, used by the segmentation dataset: `train-human`, `train-multi-specie`, `val-human`. |
| `split` | HF/local split name. |
| `data_files` | Optional local file pattern for parquet/json loading. |
| `genomes` | Optional list of genome/assembly IDs. Empty list means no genome filter. |
| `chromosomes` | Optional list of chromosome/contig IDs. Empty list means no chromosome filter. |
| `statuses` | Optional list of representative-transcript status values, for example `[1]`. If requested, the dataset must have a `status` column. |
| `max_rows` | Optional row cap, useful for smoke tests. |
| `streaming` | Optional HF streaming mode. When `true`, the loader scans remote rows, applies `genomes`/`chromosomes`/`statuses`, materializes only matching rows, and then trains normally on that small real-data subset. |
| `streaming_max_scanned_rows` | Maximum number of streamed rows to scan while looking for rows that match filters. |
| `streaming_trim_rows` | Optional debug option. When `true`, a streamed row is trimmed to the span required by `max_nucleotides`, `overlap`, and `max_windows` before it is kept in memory. Smoke tests do not use this option; they build persistent chromosome indexes and compact selected-data files before any model starts. |
| `max_windows` | Optional window cap after dataset windowing. |
| `prewindowed` | When `true`, each gene-finding row is already one model-sized chromosome window and is used directly without reassembling/generating windows. Used by persistent smoke subsets. |
| `max_nucleotides` | Nucleotide context length used for nucleotide models and UNET output. |
| `max_tokens` | Token context length used for BPE models. |
| `overlap` | Sliding-window overlap. The default is `0.5`. |
| `target_group` | `primary`/`combined` for all selected transcript isoforms or `mrna` for mRNA-only channels in gene finding. |
| `crop_margin` | Segmentation/transcript-type training crop margin. Default is `500` bp. |
| `random_crop` | When `true`, transcript tasks choose random crops whose start is at least `crop_margin` from transcript boundaries when possible. |

### Gene-finding dataset logic

The gene-finding HF dataset stores large genomic blocks rather than model-sized samples. The code first groups rows by `(genome, chrom)`, sorts blocks by genomic start, builds a chromosome assembly abstraction, and then creates sliding model windows over the assembled chromosome with the requested overlap. Slices can cross parquet-row/block boundaries.

For edge models, the default target channels are:

```text
0 primary_tss_+
1 primary_tss_-
2 primary_polya_+
3 primary_polya_-
```

For region models:

```text
4 intragenic_regions_+
5 intragenic_regions_-
```

With `target_group="mrna"`, the corresponding mRNA-only channels are used.

### Segmentation and transcript-type dataset logic

Segmentation and transcript-type tasks use the transcript-level dataset. Each row is a full transcript. During training, long transcripts are cropped with `crop_margin` (default `500`) so the random crop does not start too close to the transcript edge when the transcript is long enough. During evaluation and inference, the dataset uses deterministic leading windows unless the inference script performs its own interval/chromosome logic.

Segmentation class order is fixed:

```text
0 5UTR
1 exon
2 intron
3 3UTR
4 CDS
```

Transcript type uses metadata: `lnc_RNA` maps to label 1, `mRNA` maps to label 0.

## Training

### Finding edge model

```bash
python finding/train.py --task edge --config finding/configs/edge_moderngena_base_plain.json
```

### Finding region model

```bash
python finding/train.py --task region --config finding/configs/region_moderngena_base_plain.json
```

### Segmentation

```bash
python segmentation/train.py --config segmentation/configs/caduceus_ps_middle_loss.json
```

### Transcript type

```bash
python transcript_type/train.py --config transcript_type/configs/moderngena_base_plain.json
```

Training uses Hugging Face `Trainer`, TensorBoard logging, tqdm progress bars, validation during training, and checkpoint saving according to the `training` section. There is no automatic test phase. Use separate inference scripts for final metrics.

### Resume training

Set:

```json
"resume_from_checkpoint": "runs/my_run/checkpoint-10000"
```

If the field is `null` or empty, training starts from the model/backbone checkpoint specified in `model.backbone_path` and optional `model.checkpoint_path`.

## Training-time metrics

| Task | Validation metric during training |
|---|---|
| Gene finding edge/region | ROC-AUC only (`auc_channel_*`, `auc_mean`) |
| Segmentation | Exact interval-level F1 only (`interval_f1_exon`, `interval_f1_cds`, `interval_f1_mean`) |
| Transcript type | `accuracy`, `f1`, `precision`, `recall` |

Final GFF-based metrics are computed only by the inference scripts.

## Inference and final metrics

### Finding

```bash
python finding/infer.py --config finding/configs/infer_moderngena_base_plain.json
```

The finding inference path builds edge and region tracks, expands BPE outputs to nucleotide coordinates when needed, optionally applies reverse-complement averaging, smooths edge tracks with FFT low-pass filtering, calls TSS/PolyA peaks, pairs strand-compatible TSS/PolyA candidates, filters candidates with region-model intragenic signal, writes GFF, and optionally runs:

```python
evaluate.load("AIRI-Institute/genatator-ab-initio-annotation-leaderboard")
```

Important finding inference parameters:

| Parameter | Meaning |
|---|---|
| `use_reverse_complement` | Enables or disables RC averaging. |
| `lp_frac` | Fraction of Fourier coefficients retained. |
| `pk_prom` | Peak prominence threshold. |
| `pk_dist` | Minimum distance between peaks. |
| `pk_height` | Optional peak-height threshold. |
| `interval_window_size` | Maximum distance for TSS/PolyA pairing. |
| `max_pairs_per_seed` | Number of nearest PolyA partners per TSS seed. |
| `prob_threshold` | Region-model threshold used for intragenic masks. |
| `zero_fraction_drop_threshold` | Maximum allowed non-intragenic fraction inside a candidate interval. |
| `true_gff` | Optional reference GFF/GFF3 path. |
| `k_values` | Boundary tolerances for final metrics. |
| `use_strand` | Whether final annotation metrics require strand matching. |

### Segmentation

```bash
python segmentation/infer.py --config segmentation/configs/infer_caduceus_ps.json
```

The script writes exon/CDS GFF records and optionally runs:

```python
evaluate.load("AIRI-Institute/genatator-ab-initio-segmentation-leaderboard", revision="metric-only")
```

Reverse-complement averaging is controlled by `inference.use_reverse_complement`. For segmentation, RC averaging reverses the sequence axis and swaps `5UTR` with `3UTR`; exon, intron, and CDS stay in their original classes.

### Transcript type

```bash
python transcript_type/infer.py --config transcript_type/configs/infer_moderngena_base.json
```

The script writes a TSV with transcript IDs, reference type, predicted type, and lncRNA probability, then writes accuracy/F1/precision/recall JSON metrics. Reverse-complement averaging is controlled by `inference.use_reverse_complement`.

## Smoke tests on one real held-out chromosome

Smoke tests use no dummy DNA, labels, or GFF. The default chromosome is T2T `NC_060944.1`. The supplied reference GFF is used both to determine the exact chromosome alias and to run the final GFF metrics.

The two public datasets have different held-out layouts:

- gene finding uses the official `test` split, which is the complete T2T genome stored as genomic parquet blocks;
- segmentation and transcript type use `val-human/validation`, because that repository has no split named `test`; chromosome 20 in `val-human` is the released held-out final-evaluation set.

### Dataset discovery and persistent indexes

Before any GPU job starts, `smoke_tests/run_smoke.py`:

1. prints the local dataset location it detected: an explicitly supplied local path or the resolved Hugging Face cache snapshot/file;
2. iterates through every gene-finding `test` parquet sample with a tqdm bar, reads only its metadata column, selects only files whose metadata confirms the requested chromosome, and verifies the assembled chromosome span;
3. iterates through every `val-human` transcript metadata row with a tqdm bar and stores every row index whose metadata chromosome matches the request;
4. writes persistent per-sample metadata and chromosome-selection indexes to `smoke_tests/indexes/`;
5. writes only the selected, compact smoke data to `smoke_tests/selected_data/`.

For an uncached remote gene-finding sample, the one-time indexer uses seekable Hugging Face filesystem range reads to fetch the Parquet footer and metadata column without materializing DNA or target arrays. Its progress is checkpointed in `smoke_tests/indexes/gene_finding_test_sample_metadata.json`, so an interrupted scan resumes from stored metadata. The completed chromosome indexes are reused on later runs. Pass `--refresh-index` to repeat metadata discovery. Both generated directories are ignored by Git.

For gene finding, every selected chromosome block contributes informative edge and region windows. The extraction scans one block at a time, keeps no rejected chromosome block in RAM, and writes model-sized windows to local JSONL. The default is four windows per chromosome block. For segmentation and transcript type, all chromosome-matching transcript rows are copied to one local parquet file in bounded batches; complete source transcripts are retained on disk, while model training takes one deterministic context crop from every transcript.

### Deliberate overfit protocol

The same chromosome-selected held-out samples are used for smoke training, validation, and separate inference. This is intentional: a visible fall in both training and validation loss confirms that the backbone, task head, labels, optimizer, validation path, checkpoint, and inference path are connected correctly.

Defaults:

```text
4 epochs
constant learning rate 1e-4
evaluation once per epoch
checkpoint once per epoch
logging every optimizer step
all selected samples visited each epoch
no dataloader worker copies
```

After every training job, the launcher reads `trainer_state.json`. By default it stops the whole smoke run if the final training loss and final validation loss do not both improve over their first recorded values. Disable only for diagnosis with `--no-require-overfit`, or require a larger relative loss drop with `--overfit-min-relative-loss-drop`.

### Run

```bash
python smoke_tests/run_smoke.py \
  --num-gpus 4 \
  --reference-gff /path/to/human_T2T_chr20_reference.gff3 \
  --work-dir smoke_tests/runs
```

Useful controls:

```bash
python smoke_tests/run_smoke.py \
  --num-gpus 4 \
  --reference-gff /path/to/human_T2T_chr20_reference.gff3 \
  --requested-chromosome NC_060944.1 \
  --gene-finding-windows-per-block 4 \
  --smoke-epochs 4 \
  --smoke-learning-rate 1e-4 \
  --metadata-batch-size 16
```

Use already downloaded local datasets:

```bash
python smoke_tests/run_smoke.py \
  --num-gpus 4 \
  --reference-gff /path/to/human_T2T_chr20_reference.gff3 \
  --gene-finding-dataset-path /path/to/genatator-gene-finding-dataset \
  --segmentation-dataset-path /path/to/val-human/data.parquet
```

Use specific GPUs:

```bash
python smoke_tests/run_smoke.py \
  --gpus 0,2,3 \
  --num-gpus 3 \
  --reference-gff /path/to/human_T2T_chr20_reference.gff3
```

Each active job owns exactly one GPU. Independent task/model jobs run concurrently up to the GPU count, while inference waits for its matching training job. Any failed command, missing metric, or failed overfit check terminates the other active jobs and reports the command plus log tail.

The final `smoke_tests/runs/summary.md` includes selected-data statistics, job durations, first→last training and validation losses, overfit decisions, logs, and metric-file previews.

## Repository hygiene

The active model files are:

```text
genatator_core/backbones.py
genatator_core/token_models.py
genatator_core/legacy_rmt.py
genatator_core/legacy_caduceus.py
genatator_core/amt_models.py
genatator_core/model_builders.py
```

The only memory-wrapper spelling used in configs and code is `amt` / `AMT`. Smoke tests use real HF datasets and a user-provided reference GFF.
