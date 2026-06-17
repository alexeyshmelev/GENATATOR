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
| `streaming_trim_rows` | Optional debug option. When `true`, a streamed row is trimmed to the span required by `max_nucleotides`, `overlap`, and `max_windows` before it is kept in memory. Current smoke tests instead build a local real-data cache from a direct HF test slice. |
| `max_windows` | Optional window cap after dataset windowing, useful for smoke tests. |
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

## Smoke tests on real HF data

Smoke tests do **not** generate dummy data and do **not** use a dummy GFF. They use real data from the HF repositories and require a user-provided human T2T chromosome 20 reference GFF/GFF3. For gene finding, smoke tests use **only** a real HF `test` parquet shard from `NC_060944.1`; they never open the huge gene-finding `train` split or the gene-finding `validation` split. For segmentation and transcript type, the runner uses real `val-human` chromosome-20 transcript rows. Before launching any per-model job, the runner creates tiny persistent JSONL caches. The per-model train/validation/inference jobs then read only these local JSONL caches and never call the remote HF datasets.

Run:

```bash
python smoke_tests/run_smoke.py \
  --num-gpus 4 \
  --reference-gff /path/to/human_T2T_chr20_reference.gff3 \
  --work-dir smoke_tests/runs
```

Optional real-data cache controls. The segmentation cache uses `val-human/data.parquet` by default and reads it in small batches:

```bash
python smoke_tests/run_smoke.py \
  --num-gpus 4 \
  --reference-gff /path/to/human_T2T_chr20_reference.gff3 \
  --gene-finding-cache-len 1536 \
  --segmentation-cache-len 768 \
  --segmentation-cache-rows 2 \
  --segmentation-parquet-batch-size 16
```

To bypass HF completely, pass local parquet files:

```bash
python smoke_tests/run_smoke.py \
  --num-gpus 4 \
  --reference-gff /path/to/human_T2T_chr20_reference.gff3 \
  --gene-finding-local-parquet /path/to/gene_finding_chr20.parquet \
  --segmentation-local-parquet /path/to/segmentation_val_human.parquet
```

Or choose GPU IDs explicitly:

```bash
python smoke_tests/run_smoke.py \
  --gpus 0,2,3 \
  --num-gpus 3 \
  --reference-gff /path/to/human_T2T_chr20_reference.gff3
```

The smoke runner:

1. builds small JSON configs using real HF datasets;
2. trains for two optimization steps;
3. validates after the first step;
4. saves checkpoints;
5. runs inference on small human T2T chr20 subsets;
6. computes final metrics with the configured metric packages;
7. writes `summary.md` with job durations, logs, and metric previews.

Each smoke job uses one GPU. Jobs are launched concurrently up to the number of available GPUs while respecting train→inference dependencies. If any job fails, the runner terminates active jobs and raises an error with the failed job name, command, GPU, log file, and the last log lines.

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


### Smoke-test dataset filtering note

The smoke runner uses only real Hugging Face data, but it does **not** call `datasets.load_dataset(...)` while preparing smoke caches. For gene finding it downloads or reuses exactly one known T2T chr20 test parquet shard. For transcript-level tasks it downloads or reuses **only** `val-human/data.parquet` by default, then iterates over small Parquet record batches and keeps only real rows whose metadata points to `NC_060944.1` / chr20. It never auto-downloads `train-human` or `train-multi-specie` shards during smoke-cache preparation. The resulting local JSONL files are tiny and reused by every model. If filtering selects zero rows, the runner stops and tells you which parquet file was inspected; pass `--segmentation-local-parquet` or `--segmentation-remote-parquet val-human/data.parquet` to make the source explicit.

## Smoke-test real-data cache behavior

Smoke tests use real Hugging Face data only, but they now write tiny persistent JSONL caches before launching per-model jobs. This is necessary because the raw HF datasets are huge and repeated `load_dataset(...)` calls may re-check or re-prepare remote files even when each model only needs a few real rows.

By default the cache directory is:

```bash
~/.cache/genatator_smoke
```

or the value of:

```bash
GENATATOR_SMOKE_CACHE_DIR
```

You can also set it explicitly:

```bash
python smoke_tests/run_smoke.py \
  --num-gpus 2 \
  --reference-gff /path/to/chr20.gff \
  --work-dir smoke_tests/runs \
  --smoke-cache-dir /disk/10tb/home/shmelev/GENATATOR/.smoke_real_data_cache
```

The smoke runner creates two persistent caches:

- a gene-finding cache from a real `AIRI-Institute/genatator-gene-finding-dataset` chr20 test parquet shard;
- a segmentation/transcript-type cache from real `AIRI-Institute/genatator-gene-segmentation-dataset` `val-human/data.parquet` only, filtered to human T2T chr20 without materializing the full HF dataset and without downloading multispecies data.

After these files exist, deleting `smoke_tests/runs` will not trigger dataset preparation again. All train, validation, and inference jobs read the tiny local JSONL caches instead of touching the remote HF datasets.

## v13 inference GFF behavior

The inference writers now follow the two official evaluate interfaces used by the repository.

### Gene finding GFF

`finding/infer.py` writes a genome-oriented GFF3 file. Each predicted interval becomes:

- one `gene` row on the chromosome coordinate system;
- one transcript row of type `mRNA` by default;
- one `exon` row spanning the full predicted transcript interval.

The official annotation metric rejects empty prediction GFF files. Normal inference keeps the strict default behavior and raises an error if no intervals pass post-processing. Smoke-test inference explicitly sets:

```json
"empty_gff_policy": "best_interval"
```

With this setting, if a two-step smoke model produces no intervals, the code writes one best-scoring interval derived from the edge and region tracks. This is only to verify the full train-infer-evaluate path under tiny smoke training.

### Segmentation GFF

`segmentation/infer.py` writes prediction GFFs in the format expected by `AIRI-Institute/genatator-ab-initio-segmentation-leaderboard`:

- prediction `seqid` is the reference `transcript_id`;
- coordinates are transcript-relative by default;
- rows include `gene`, transcript (`mRNA` or `lnc_RNA`), `exon`, and, for mRNA, `CDS` features.

This differs from standard genome-oriented GFF used by the reference annotation. Use:

```json
"coordinate_mode": "transcript"
```

for the official segmentation metric. Smoke-test segmentation also sets:

```json
"empty_segment_policy": "best_interval"
```

so that a tiny randomly initialized or barely trained model still produces at least one exon/CDS feature and the evaluator can run.

## Gene-finding inference metrics and post-processing

`finding/infer.py` now performs two independent evaluations during inference.

First, it gathers the edge and region probability tracks for the selected chromosome(s) and computes nucleotide-level PR-AUC on the whole chromosome at once. The reported blocks are:

```json
{
  "pr_auc": {
    "edge": {
      "pooled": {"TSS+": ..., "TSS-": ..., "PolyA+": ..., "PolyA-": ...},
      "per_chromosome": {"NC_060944.1": {...}},
      "mean": ...
    },
    "region": {
      "pooled": {"Intragenic+": ..., "Intragenic-": ...},
      "per_chromosome": {"NC_060944.1": {...}},
      "mean": ...
    }
  }
}
```

Second, when `inference.true_gff` is provided, the script writes a genome-oriented prediction GFF and evaluates it through `AIRI-Institute/genatator-ab-initio-annotation-leaderboard`. The JSON output then contains both `pr_auc` and `annotation`.

Gene-finding GFF construction uses the same FFT peak-calling and TSS/PolyA pairing logic as the public GENATATOR pipeline. The model edge-channel order is `TSS+, TSS-, PolyA+, PolyA-`; internally the post-processing code reorders this to the pipeline peak-calling order `TSS+, PolyA+, TSS-, PolyA-`.

The `postprocess` JSON block supports:

```json
{
  "lp_frac": 0.05,
  "pk_prom": 0.1,
  "pk_dist": 50,
  "pk_height": null,
  "interval_window_size": 2000000,
  "max_pairs_per_seed": 10,
  "prob_threshold": 0.5,
  "zero_fraction_drop_threshold": 0.01,
  "pairing_progress_every": null
}
```

These parameters correspond to the FFT low-pass fraction, peak prominence, peak distance, optional peak height, maximum TSS/PolyA pairing distance, maximum nearest PolyA partners per TSS seed, intragenic probability threshold, maximum allowed non-intragenic fraction inside a candidate interval, and optional logging interval for peak pairing.

## v15 notes: torch 2.2.2, chromosome-only smoke data, and GFF conventions

### Loading GENA checkpoints with torch 2.2.2

Several GENA checkpoints are distributed as legacy PyTorch `.bin` weights. Newer
`transformers` releases block those files when `torch < 2.6` is installed. This
repository keeps compatibility with `torch==2.2.2+cu121` by explicitly patching
Transformers' safety gate before `from_pretrained(...)` when the model config has:

```json
"allow_unsafe_torch_load_with_torch_lt_2_6": true
```

This is enabled in the shipped configs because the requested runtime is pinned to
torch 2.2.2. The code logs a warning every time this compatibility path is active.
Use only trusted backbone repositories when this option is enabled.


### GENA hidden-state extraction

Some GENA checkpoints load as `BertForMaskedLM` through the remote-code `AutoModel` mapping.
For those checkpoints, `out.logits` has shape `(batch, tokens, vocab_size)` and must not be
used as the downstream hidden representation. The backbone adapter now explicitly uses
`last_hidden_state` when available, otherwise `hidden_states[-1]`, and logs a detailed shape
error if neither matches the configured hidden size.

### Gene-finding GFF output

Gene-finding edge/region models predict transcript boundaries and intragenic
coverage, not exon-intron structure. For GFF-based boundary evaluation, every
predicted TSS/PolyA interval is therefore written as:

```text
gene
mRNA or lnc_RNA
one exon spanning the full predicted transcript interval
```

This is intentional. The gene-finding leaderboard can then assess TSS/PolyA
boundary localization through the transcript interval, while true exon/CDS
segmentation is evaluated separately by the segmentation task.

Gene-finding inference also computes whole-chromosome PR-AUC at nucleotide
resolution before GFF post-processing. Edge PR-AUC is reported for TSS+, TSS-,
PolyA+, and PolyA-. Region PR-AUC is reported for Intragenic+ and Intragenic-.

### Segmentation GFF output

Segmentation prediction GFFs are transcript-coordinate files. The `seqid` column
is the transcript ID, not a chromosome. This follows the official segmentation
metric assumption: each model receives only the DNA sequence of an individual
transcript, without intergenic sequence or neighboring genes.

### Smoke-test data extraction

Smoke tests must stay on T2T human chromosome 20 only. The smoke runner now:

1. downloads or reuses exactly seven gene-finding chr20 parquet blocks for `NC_060944.1`; the final block is explicitly `000060000000_000066210255`, not `000060000000_000070000000`;
2. downloads or reuses only `val-human/data.parquet` for segmentation/transcript
   smoke rows;
3. scans the segmentation parquet in small PyArrow batches with a tqdm progress
   bar;
4. selects only rows whose metadata chromosome matches `NC_060944.1`, `chr20`, or
   `20`;
5. immediately trims each selected row before writing the persistent JSONL smoke
   cache;
6. prints the number of selected chr20 transcripts and, during dataset loading,
   prints transcript metadata spans and finding assembled chromosome lengths.

The smoke path never auto-downloads `train-human` or `train-multi-specie`.

## v17 smoke-test notes

Gene-finding smoke cache now indexes the full T2T `NC_060944.1` chromosome as seven exact local parquet blocks. The JSONL cache stores only `parquet_path` and metadata for each block, so the full 66 Mb chromosome is assembled from coordinates without storing the whole DNA sequence or `targets` matrix in JSONL or RAM. Dataset logs should show `assembled_total_length` around `66210255` and `chrom_length_metadata=66210255`.

For smoke runtime, training uses only a few windows (`max_windows=2`) even though the chromosome assembly is full. Gene-finding smoke inference uses `GENATATOR_SMOKE_GF_INFER_MAX_WINDOWS=4` by default to remain quick; set `GENATATOR_SMOKE_GF_INFER_MAX_WINDOWS=full` to run inference windows across the full chromosome. Normal non-smoke inference configs can set `max_windows: null` to compute whole-chromosome PR-AUC and GFF metrics.

Trainer checkpoint saving defaults to `save_safetensors=false` to avoid shared-tensor save failures with GENA/ModernGENA wrappers and PyTorch 2.2.2. You can override this in JSON, but `.bin` checkpoints are the safest default in this environment.

## v18 smoke-test fix

The gene-finding chr20 block list is now hard-coded to the released file names. The final block ends at `000066210255`; the code no longer generates a non-existent `000070000000` endpoint. If a stale partial index exists, the smoke runner detects `blocks != 7` or `assembled_total_length < 66210255`, deletes the stale index, and rebuilds it from the exact block list.
