# GENATATOR

Training and inference code for four related training tasks:

1. **Gene finding — edge model (`finding_edge`)**: predicts strand-specific transcript starts and ends: `TSS+`, `TSS-`, `PolyA+`, and `PolyA-`.
2. **Gene finding — region model (`finding_region`)**: predicts strand-specific intragenic coverage: `intragenic+` and `intragenic-`.
3. **Gene segmentation (`segmentation`)**: predicts nucleotide-level `5UTR`, `exon`, `intron`, `3UTR`, and `CDS` tracks inside transcript sequences.
4. **Transcript type (`transcript_type`)**: classifies a transcript as mRNA or lncRNA.

The edge and region models are trained separately. Their chromosome-level tracks are combined by the finding inference pipeline to produce stranded transcript intervals.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

For multi-GPU training, launch with `torchrun`. On systems without IPv6 support, explicitly use IPv4 to avoid the harmless `[::]:29500` socket warnings:

```bash
torchrun \
  --master_addr 127.0.0.1 \
  --master_port 29500 \
  --nproc_per_node 8 \
  finding/train.py \
  --config finding/configs/edge_moderngena_base_plain.json
```

`OMP_NUM_THREADS=1` is the default selected by `torchrun`. Set it before launch only when CPU tokenization or data preparation requires a different value.

## Repository structure

```text
GENATATOR-main/
├── genatator_core/          shared models, data loading, metrics, inference and run management
├── finding/
│   ├── configs/             edge, region and complete-pipeline inference configs
│   ├── train.py             training entry point; task comes from the JSON config
│   └── infer.py             single-stage or complete-pipeline inference and evaluation
├── segmentation/
│   ├── configs/
│   ├── train.py
│   └── infer.py             inference, GFF writing and optional official evaluation
├── transcript_type/
│   ├── configs/
│   ├── train.py
│   └── infer.py             inference, TSV writing and accuracy calculation
├── smoke_tests/
└── tests/
```

There is no separate task-level `evaluate.py`. Every `infer.py` entry point performs the evaluation available for that task.

## Task workflows

### Gene-finding edge model

Input is a reconstructed chromosome window. Output is four nucleotide tracks in model order `TSS+`, `TSS-`, `PolyA+`, `PolyA-`. Training uses `task="finding_edge"`. Standalone inference calculates whole-chromosome PR-AUC; complete-pipeline inference supplies these tracks to boundary peak calling and interval pairing.

### Gene-finding region model

Input is the same kind of reconstructed chromosome window. Output is two nucleotide tracks, `intragenic+` and `intragenic-`. Training uses `task="finding_region"`. The complete finding pipeline uses thresholded region tracks to remove candidate TSS–PolyA pairs whose interiors lack the expected strand-specific intragenic signal.

### Gene segmentation

Input is one transcript sequence or one non-overlapping chunk of a complete transcript. Output is five nucleotide tracks ordered `5UTR`, `exon`, `intron`, `3UTR`, `CDS`. Training uses exact interval F1 for exon and CDS checkpoint selection. Inference gathers every chunk back into one transcript prediction, optionally averages its reverse complement, writes GFF3, and evaluates against `true_gff` when supplied.

### Transcript type

Input is a transcript sequence. Output is one binary logit for mRNA versus lncRNA. Inference writes one TSV row per transcript and an accuracy JSON.

## Supported backbones and heads

| Backbone | Input resolution | Gene finding | Segmentation | Transcript type |
|---|---|---|---|---|
| Caduceus PH | nucleotide | nucleotide backbone + linear head | nucleotide backbone + linear head | pooled nucleotide backbone + binary head |
| Caduceus PS | nucleotide, RC-equivariant architecture | nucleotide backbone + linear head | nucleotide backbone + linear head | pooled nucleotide backbone + binary head |
| GENA Base / Large | BPE | plain linear head; U-Net; RMT + U-Net; AMT plain; AMT + U-Net | U-Net; RMT + U-Net; AMT + U-Net | plain pooled head |
| ModernGENA Base / Large | BPE | plain linear head; U-Net; RMT + U-Net; AMT plain; AMT + U-Net | U-Net; RMT + U-Net; AMT + U-Net | plain pooled head |

Important architecture rules:

- Caduceus always uses `bidirectional_weight_tie=false`; the loader forces it regardless of the downloaded checkpoint config.
- Plain/direct GENA accepts at most 512 BPE positions. It never elongates the backbone by independently chunking and concatenating hidden states. Use RMT or AMT for longer GENA inputs.
- RMT uses 10 memory tokens for GENA and 20 for ModernGENA. Its full segment defaults remain 512 and 1,024 BPE positions because RMT reserves memory positions internally.
- AMT uses the same 10/20 memory-token rule, but its data-token segment must reserve those positions explicitly. The shipped GENA and ModernGENA defaults are therefore 502 and 1,004. ModernGENA contexts may still be extended beyond the 1,024-position default.
- Every U-Net uses one cycle by default. The shipped configs use `unet_cycles=1` or `cycles=1` and an 8,192-nucleotide `unet_chunk_size`.
- For BPE + U-Net models, each retained BPE hidden state is repeated over the nucleotide offsets covered by that token. A learned embedding of the actual nucleotide is concatenated with the repeated hidden state before U-Net processing.

## Configuration layout

Every training config contains these top-level fields:

```json
{
  "seed": 42,
  "task": "finding_edge",
  "model": {},
  "train_dataset": {},
  "eval_dataset": {},
  "true_gff": null,
  "training": {}
}
```

### `task`

Allowed values are:

```text
finding_edge
finding_region
segmentation
transcript_type
```

The training entry point reads this value directly. Finding training no longer accepts or needs `--task edge` or `--task region`.

### `model`

Common fields:

| Field | Meaning |
|---|---|
| `family` | `caduceus`, `plain`, `unet`, `rmt`, or `amt` |
| `backbone_kind` | `caduceus`, `gena`, or `moderngena` |
| `backbone_path` | local path or Hugging Face model ID |
| `tokenizer_path` | local path or Hugging Face tokenizer ID |
| `trust_remote_code` | passed to Hugging Face loaders |
| `checkpoint_path` | optional model weights loaded before training; normally `null` |
| `vocab_size` | full main-tokenizer vocabulary size, inferred when `null` for U-Net paths |
| `unet_chunk_size` | independent nucleotide chunk processed by the U-Net |
| `unet_cycles` / `cycles` | U-Net cycle count; shipped default is one |
| `rmt` | RMT memory-token, segment-size and maximum-segment settings |
| `amt` | AMT repository, memory and segment settings |
| `use_unet` | selects plain AMT versus AMT + U-Net |

### `training`

All shipped configurations and the runtime validator enforce:

```json
{
  "per_device_train_batch_size": 1,
  "per_device_eval_batch_size": 1
}
```

This invariant applies to every task and every model. `gradient_accumulation_steps` may still be used to change the optimizer-step batch. Finding additionally uses `dataloader_num_workers=0`, because worker subprocesses would otherwise create independent chromosome-sized RAM caches.

Every shipped config uses `max_steps=500000`, `eval_steps=1000`, `save_steps=1000`, and `patience=100`. Patience counts consecutive evaluations without improvement in the selected best-model metric before early stopping. Other important fields include `num_train_epochs`, `learning_rate`, `weight_decay`, `warmup_steps`, mixed precision, checkpoint retention, and `resume_from_checkpoint`.

Reverse-complement processing is intentionally absent from training configs. Training and training-time validation always use one orientation only.

## Dataset configuration and filtering

A dataset can be a local path or a Hugging Face dataset ID. Filters are applied from metadata:

```json
{
  "genomes": ["GCF_009914755.1"],
  "chromosomes": ["NC_060944.1"]
}
```

- Empty lists select every available value.
- `genomes` is the assembly identifier.
- `chromosomes` is the sequence/chromosome identifier.
- Chromosomes are always keyed internally as `(genome, chromosome)`. Therefore, `chr1` from two species is never assembled or sampled as one chromosome.

To train on one species but all of its chromosomes:

```json
"genomes": ["GCF_009914755.1"],
"chromosomes": []
```

To train on one chromosome from that assembly:

```json
"genomes": ["GCF_009914755.1"],
"chromosomes": ["NC_060944.1"]
```

### Gene-finding dataset

The default dataset is:

```text
AIRI-Institute/genatator-gene-finding-dataset
```

The repository layout is `data/<split>/*.parquet`, where `<split>` is `train`, `validation`, or `test`. Each Parquet file contains one genomic block with:

| Column | Type | Meaning |
|---|---|---|
| `dna_sequence` | string | DNA for one contiguous chromosome block |
| `targets` | matrix `[block_length, 12]` | nucleotide-level finding targets |
| `metadata` | JSON object/string | at least `genome`, `chrom`, `start`, `end`, and usually `chrom_length` |

The 12 target channels are:

```text
0 primary_tss_+                 6 mrna_tss_+
1 primary_tss_-                 7 mrna_tss_-
2 primary_polya_+               8 mrna_polya_+
3 primary_polya_-               9 mrna_polya_-
4 intragenic_regions_+         10 mrna_intragenic_regions_+
5 intragenic_regions_-         11 mrna_intragenic_regions_-
```

`target_group="primary"` selects combined mRNA + lncRNA targets. `target_group="mrna"` selects the mRNA-only channels.

#### Direct Parquet indexing and chromosome reconstruction

Finding data never passes through `datasets.load_dataset()` and never creates a Hugging Face Arrow cache. This avoids Arrow's 2³¹−1 nested-array limit.

The loading sequence is:

```text
list data/<split>/*.parquet
        ↓
resolve one shared local file manifest for all DDP ranks
        ↓
read only the metadata column from every block
        ↓
cache a small JSON block index
        ↓
group block descriptors by (genome, chromosome)
        ↓
create overlapping window coordinates for each chromosome span
        ↓
when a rank reaches a chromosome:
    load one block directly with PyArrow
    convert it to {
        dna_sequence: Python string,
        targets: selected-channel float32 NumPy matrix,
        metadata: Python dictionary
    }
    copy it into one chromosome byte buffer and one target matrix
    release the block
        ↓
cache only the currently used reconstructed chromosome
```

The final chromosome sequence is one Python string; the final target object is one contiguous NumPy matrix containing only the channels needed by the current task—four for edge or two for region. Moving to another chromosome releases the previous chromosome cache.

Blocks must be sorted, contiguous, non-overlapping, and metadata-consistent. Gaps, overlaps, changed metadata, or DNA/target-length mismatches raise an error instead of silently creating invalid training samples.

#### Overlapping finding windows

For chromosome length `L`, model nucleotide window `W`, and overlap fraction `o`:

```text
step = max(1, floor(W × (1 − o)))
windows = [0:W], [step:step+W], ...
```

The last window ends exactly at the end of the reconstructed chromosome. With `overlap=0.5`, adjacent windows overlap by approximately half their length.

#### Multi-GPU finding sampling without duplicates

The finding sampler keeps window indices grouped by chromosome, shuffles chromosome groups and windows per epoch, and then builds equal non-overlapping GPU lanes. The lanes are interleaved only so Accelerate can shard them correctly.

Consequences:

- a window index is assigned to at most one GPU in an epoch;
- Accelerate does not pad the epoch by repeating windows;
- at most `world_size − 1` windows are dropped when the total is not divisible by the GPU count;
- each GPU processes chromosome-grouped windows, so it normally assembles a chromosome once and reuses it for its assigned windows;
- training-time validation is run sequentially on rank 0 over every validation window once.

### Segmentation dataset

The default dataset is:

```text
AIRI-Institute/genatator-gene-segmentation-dataset
```

Select a dataset configuration:

```text
train-human
train-multi-specie
val-human
```

Each transcript row contains:

| Column | Meaning |
|---|---|
| `dna_sequence` | transcript/genomic sequence string |
| `labels` | float matrix `[length, 5]` ordered as `5UTR, exon, intron, 3UTR, CDS` |
| `metadata` | transcript ID, gene ID, type, strand, genome, chromosome and coordinates |
| `status` | representative-transcript marker used by training filters |

Shipped training configs use `statuses=[1]`. Automatically generated inference configs remove this filter and evaluate **all transcripts/isoforms** on `val-human`, restricted to `GCF_009914755.1 / NC_060944.1`.

During complete segmentation inference, every transcript is processed from beginning to end in non-overlapping model-sized chunks. The reverse-complement pass uses the same chunking, its channels and coordinates are restored, and the forward/RC logits are averaged.

### Transcript-type dataset

Transcript-type classification reuses the segmentation dataset's DNA and metadata. The target is derived from `metadata.transcript_type` (`mRNA` versus `lnc_RNA`). Training may use the representative-transcript status filter; generated inference removes it and evaluates every selected transcript.

## Training

Task selection is entirely inside each JSON file.

### Edge model

```bash
torchrun --master_addr 127.0.0.1 --nproc_per_node 8 \
  finding/train.py \
  --config finding/configs/edge_moderngena_base_plain.json
```

### Region model

```bash
torchrun --master_addr 127.0.0.1 --nproc_per_node 8 \
  finding/train.py \
  --config finding/configs/region_moderngena_base_plain.json
```

### Segmentation

```bash
torchrun --master_addr 127.0.0.1 --nproc_per_node 8 \
  segmentation/train.py \
  --config segmentation/configs/moderngena_base_unet.json
```

### Transcript type

```bash
torchrun --master_addr 127.0.0.1 --nproc_per_node 8 \
  transcript_type/train.py \
  --config transcript_type/configs/moderngena_base_plain.json
```

Each launch creates a timestamped child under `training.output_dir`. The run contains:

```text
training_config.json
evaluation_config.json
checkpoint-*/
final_model/
train_metrics.json
trainer_state.json
```

`evaluation_config.json` is created immediately and updated to point to the selected best checkpoint or final model.

## Inference and evaluation

All inference configs enforce:

```json
"batch_size": 1
```

They also expose:

```json
"use_reverse_complement": true
```

Reverse-complement averaging is inference-only and defaults to on. Set it to `false` in an inference config for a forward-only run.

### Generated complete gene-finding evaluation

Training either an edge or region model now generates a complete edge + region pipeline config. The trained stage and shared benchmark dataset fields are filled automatically. For the opposite stage, replace the complete `model` placeholder, replace the model-dependent dataset-length marker (including its field name) with either `max_nucleotides` or the `max_bpe_tokens`/`average_bpe_token_length` pair, fill its checkpoint, and fill `inference.true_gff`. Then run:

```bash
python finding/infer.py --config runs/.../evaluation_config.json
```

It writes `finding_predictions.gff` and a combined metrics JSON containing both stages' whole-chromosome PR-AUC and the official Hugging Face annotation metrics.

### Complete gene-finding pipeline

Use a config containing both `edge` and `region` stages, such as:

```bash
python finding/infer.py --config finding/configs/infer_moderngena_base_plain.json
```

Set the edge and region checkpoint paths in their respective `inference.checkpoint_path` fields. The script:

1. predicts and averages overlapping edge tracks;
2. predicts and averages overlapping region tracks;
3. optionally averages forward and reverse-complement passes;
4. denoises/calls TSS and PolyA peaks;
5. pairs strand-compatible boundaries;
6. filters candidates with intragenic tracks;
7. writes GFF3;
8. computes whole-chromosome PR-AUC;
9. when `true_gff` is set, computes the official boundary/interval metrics as well.

### Segmentation

```bash
python segmentation/infer.py --config runs/.../evaluation_config.json
```

Important inference switches:

```json
{
  "use_reverse_complement": true,
  "use_cds_heuristic": true
}
```

`use_cds_heuristic` defaults to on. It replaces the directly decoded mRNA CDS with the benchmark-compatible longest complete ORF inferred from predicted exons. Set it to `false` to keep the model's direct CDS track.

The script writes a GFF3 file and runs the official segmentation metric when `true_gff` is non-null.

### Transcript type

```bash
python transcript_type/infer.py --config runs/.../evaluation_config.json
```

The script writes a TSV containing probabilities and classes and always writes an accuracy JSON for the selected dataset.

## Metrics

### Training-time metrics

| Task | Metric |
|---|---|
| `finding_edge` | PR-AUC for TSS+/TSS-/PolyA+/PolyA- and their mean |
| `finding_region` | PR-AUC for intragenic+/intragenic- and their mean |
| `segmentation` | exact interval-level F1 for exon and CDS |
| `transcript_type` | accuracy |

Segmentation interval decoding uses raw-score argmax, not independent 0.5 thresholds:

```text
exon prediction: argmax among [exon, 5UTR, 3UTR]; positive only when exon wins
CDS prediction:  argmax among [CDS, intron]; positive only when CDS wins
```

Contiguous positive bases become half-open intervals. A predicted interval is a true positive only when it exactly equals a reference interval. Counts are pooled across validation samples before F1 is computed.

Training-time validation never applies reverse-complement averaging.

### Inference metrics

- **Finding single-stage:** whole-chromosome per-channel and pooled PR-AUC.
- **Finding complete pipeline:** the same PR-AUC plus official gene-boundary metrics when a reference GFF is provided.
- **Segmentation:** official gene/interval metrics from prediction and reference GFF files when `true_gff` is provided.
- **Transcript type:** accuracy over every selected transcript.

## Configuration matrix

The shipped JSON files cover the logical model/task combinations:

- Finding edge and region: Caduceus PH/PS; GENA and ModernGENA Base/Large with plain, U-Net, RMT + U-Net, AMT plain, and AMT + U-Net variants.
- Segmentation: Caduceus PH/PS; GENA and ModernGENA Base/Large with U-Net, RMT + U-Net, and AMT + U-Net.
- Transcript type: Caduceus PH/PS; plain GENA and ModernGENA Base/Large.

The configs represent model choices, not a hyperparameter grid. Copy the closest model config when changing dataset filters, optimization settings, overlap, context length, RMT/AMT segment settings, inference RC averaging, or segmentation CDS postprocessing.
