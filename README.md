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

ModernGENA with AMT requires PyTorch 2.4 or newer. All other shipped
model/task combinations work with the currently provided environment.

For multi-GPU training, launch with `torchrun`. On systems without IPv6 support, explicitly use IPv4 to avoid the harmless `[::]:29500` socket warnings:

```bash
torchrun \
  --master_addr 127.0.0.1 \
  --master_port 29500 \
  --nproc_per_node 8 \
  finding/train.py \
  --config finding/configs/long_human_mrna_lncrna/edge_moderngena_base_plain.json
```

`OMP_NUM_THREADS=1` is the default selected by `torchrun`. Set it before launch only when CPU tokenization or data preparation requires a different value.

## Repository structure

```text
GENATATOR-main/
├── genatator_core/          shared models, data loading, metrics, inference and run management
├── experiments/
│   ├── first_gpt_model_experiments/  12 human-data GPT training configs
│   ├── small_finding_evaluation_v1/
│   │   └── evaluation_config.json
│   └── massive_gene_finding_evaluation/
│       └── <training-setup>/        all 484 edge × region pairs for that setup
├── finding/
│   ├── configs/
│   │   └── <training-setup>/        four setup folders; 22 edge + 22 region configs each
│   ├── train.py             training entry point; task comes from the JSON config
│   └── infer.py             single-stage or complete-pipeline inference and evaluation
├── segmentation/
│   ├── configs/             existing heads plus 14 segmentation-only GPT variants
│   ├── train.py
│   └── infer.py             inference, GFF writing and optional official evaluation
├── transcript_type/
│   ├── configs/
│   │   ├── short_human/
│   │   ├── long_human/
│   │   └── long_multispecies/
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

#### GPT generative segmentation head

The segmentation-only GPT family uses the decoder half of T5Gemma as a shallow
categorical nucleotide decoder. T5Gemma is initialized from scratch; no
pretrained Gemma weights or text encoder are loaded. The selected GENA,
ModernGENA, RMT, AMT, or Caduceus encoder supplies contextual nucleotide
representations through cross-attention.

For BPE encoders, special tokens and padding are removed, every retained BPE
state is expanded to its covered nucleotides with embedding_repeater, and a
learned A/C/G/T nucleotide embedding is concatenated to that state. Caduceus
already supplies nucleotide-resolution states. Each sample is then processed
without padding. Direct GENA and ModernGENA GPT models encode long BPE inputs
in non-overlapping chunks no larger than the backbone's native context and
scatter the hidden states back before nucleotide expansion.

The decoder models exactly two mutually exclusive internal tokens:

| Class ID | Internal token | Public segmentation slot |
|---:|---|---:|
| 0 | intron | 2 |
| 1 | exon | 1 |

Training labels must contain exactly one active exon/intron class at every
valid nucleotide. The GPT head does not predict 5' UTR, 3' UTR, or CDS. Its
public output still has shape [B, N, 5] in the repository order 5UTR, exon,
intron, 3UTR, CDS so the shared validation, reverse-complement, exon selection,
GFF, and CDS-heuristic pipeline remains usable. Unavailable tracks receive a
large finite negative logit; padding positions remain zero.

Both training and validation use teacher forcing. For an exact, unpadded chunk
with targets y0 ... y(n-1), the decoder inputs are BOS, y0 ... y(n-2), so one
forward pass returns probabilities for every nucleotide. Validation therefore
does not generate an internal transcript one token at a time. Only label-free
inference uses autoregressive argmax feedback.

The decoder context C and encoder lookahead A are independent configurable
values. Teacher-forced samples are split into exact chunks of at most C
nucleotides; the final short chunk is never padded. Each decoder chunk
cross-attends to its current C encoder states plus up to A following states.
During inference, cached self-attention retains at most C preceding decoder
inputs while the visible encoder interval advances one nucleotide at a time
until it reaches its final right-edge position:

```text
moving_start = max(0, p - C + 1)
last_full_start = max(0, N - (C + A))
start = min(moving_start, last_full_start)
visible encoder = [start, min(N, start + C + A))
```

The implementation physically slices this encoder interval; it does not attend
to the complete N-state encoder behind a mask. When the interval moves, cached
cross-attention keys/values for the overlap are retained, exited states are
dropped, and only the newly visible right-edge state is projected. Once the
right boundary reaches N, the left boundary stops advancing: the final full
encoder window remains anchored at the sequence end for every remaining
prediction. A fixed-size T5Gemma sliding cache retains the latest C decoder
inputs. In addition to the backbone's N encoder outputs, the decoder's
cross-attention working set is bounded by C+A states and its autoregressive self
cache by C states.

The following example uses N=10, C=3, and A=3. y values are previously known
argmax decisions; e values are encoder nucleotide states.

| Next position p | Decoder history visible before prediction | Encoder states visible | Output |
|---:|---|---|---|
| 0 | BOS | e0 ... e5 | y0 |
| 1 | BOS, y0 | e0 ... e5 | y1 |
| 2 | BOS, y0, y1 | e0 ... e5 | y2 |
| 3 | y0, y1, y2 | e1 ... e6 | y3 |
| 7 | y4, y5, y6 | e4 ... e9 | y7 |
| 9 | y6, y7, y8 | e4 ... e9 | y9 |

This table also shows the right-edge behavior: once the full encoder window is
right-aligned at N, it no longer moves or shrinks from the left, and generation
stops immediately after y(N-1). If `N <= C+A`, the complete sequence remains visible
throughout generation. If A=0, cross-attention sees only the moving C-state
encoder interval.

##### Multi-token prediction

The GPT-only field multi_token_prediction=K creates K independent linear
classifiers over the same final decoder state. Head 1 predicts the next
structure token, head 2 predicts the second future token, and so on through
head K. For a sequence of length N, the heads contribute N, N-1, ...,
max(N-K+1, 0) valid targets. Targets that would fall past the right edge are
excluded before cross-entropy, and the summed loss is divided by the total
number of valid token-offset pairs. A head with no valid target contributes
neither loss nor denominator. This prevents short sequences and final chunks
from being biased by fabricated targets or padding.

The auxiliary future-token heads are a training and validation objective only.
Autoregressive inference uses head 1 exclusively, feeds back its argmax class,
and advances by one nucleotide.

The decoder remains intentionally shallow: two layers and width 256 by default,
with two through four layers permitted. Context size, lookahead, width,
intermediate size, attention heads, dropout, and multi-token prediction depth
are all configurable inside the model.gpt object. The implementation uses the
internal T5GemmaDecoder API and therefore requires transformers>=4.53.0.
Teacher-forced decoder layers use memory-efficient SDPA and activation
checkpointing during training so they do not retain dense C-by-(C+A) attention
weights for every chunk. Activation checkpointing recomputes decoder layers
during backward, exchanging additional compute for lower peak training memory.

Autoregressive inference remains substantially slower than linear or U-Net
inference despite key/value caching, because its N decisions are sequential.
Each decision still attends to as many as C+A encoder keys, and the already
materialized backbone output plus returned N-by-5 logits scale with N. The
rolling K/V update uses semi-internal T5Gemma/DynamicCache attributes; if their
layout changes, a guarded compatibility path rebuilds only the current bounded
encoder slice instead of risking stale or misaligned keys.
Training and validation are parallel teacher-forced passes and do not pay this
one-token-at-a-time cost.

##### First GPT model experiments

`experiments/first_gpt_model_experiments/` contains the requested Caduceus PS
and ModernGENA Base GPT sweep. Every config uses human training data, a
four-layer decoder, and one of `multi_token_prediction=1,2,5,10,50,100`.
Caduceus uses 32,768 nucleotides; ModernGENA uses 8,192 BPE tokens. Each config
has its own output directory so all 12 experiments can run concurrently without
sharing an automatic-restart manifest.

### Transcript type

Input is one transcript sequence. Every model returns exactly one binary logit
per sample, shaped `[batch, 1]`; it does not return a token-classification track.
The sigmoid of that logit is the lncRNA probability. Training applies binary
cross-entropy to the single transcript label, and inference predicts lncRNA at
probability `>=0.5` and mRNA otherwise. Inference writes one TSV row per
transcript and an accuracy JSON.

GENA and ModernGENA always classify the contextual state of the tokenizer's
exact `CLS` token. Caduceus classifies the exact `SEP` state. The implementation
obtains those IDs from the active tokenizer and asserts that every sample has
exactly one attended copy of the required token before pooling. It never silently
substitutes the last non-padding position. Caduceus keeps its middle-loss design:
the final and middle hidden layers each classify the same `SEP` position, their
binary losses are averaged during training, and the final-layer logit is returned.

The GENA/ModernGENA checkpoint loaders may use
`BertForTokenClassification`/`ModernBertForTokenClassification` as
checkpoint-compatible containers for contextual token states. Their token-level
task heads are not used for transcript prediction. GENATATOR pools `CLS` and
applies its own one-logit sequence-classification head.

Long-context transcript classification is available through RMT and AMT:

- RMT builds every recurrent segment as `CLS, memory, SEP, content, SEP` and
  right-aligns segments across the batch. All samples therefore participate in
  the final recurrent step. Only the final segment's asserted `CLS` state is sent
  to the binary head, after that state has received the preceding recurrent
  memory and the final content segment. The transcript loss is calculated once,
  not independently for every segment.
- AMT processes each transcript sample separately. Earlier content segments
  update associative memory; the final segment is constructed as
  `CLS, final content, SEP`. Its asserted `CLS` state therefore sees both the
  accumulated associative memory and the final content before the one-logit head.

## Supported backbones and heads

| Backbone | Input resolution | Gene finding | Segmentation | Transcript type |
|---|---|---|---|---|
| Caduceus PH | nucleotide | nucleotide backbone + linear head | linear or GPT head | final + middle `SEP` binary heads |
| Caduceus PS | nucleotide, RC-equivariant architecture | nucleotide backbone + linear head | linear or GPT head | final + middle `SEP` binary heads |
| GENA Base / Large | BPE | plain linear head; U-Net; RMT + U-Net; AMT plain; AMT + U-Net | U-Net; RMT + U-Net; AMT + U-Net; direct/RMT/AMT + GPT | plain `CLS`; RMT `CLS`; AMT `CLS` |
| ModernGENA Base / Large | BPE | plain linear head; U-Net; RMT + U-Net; AMT plain; AMT + U-Net | U-Net; RMT + U-Net; AMT + U-Net; direct/RMT/AMT + GPT | plain `CLS`; RMT `CLS`; AMT `CLS` |

Important architecture rules:

- Caduceus always uses `bidirectional_weight_tie=false`; the loader forces it regardless of the downloaded checkpoint config.
- Ordinary plain/direct GENA accepts at most 512 BPE positions. Segmentation U-Net and GPT adapters support the shipped long inputs by running the direct backbone on non-overlapping native-context chunks, concatenating/scattering those hidden states, and only then applying the nucleotide head. RMT or AMT remains required when cross-chunk memory is part of the intended encoder architecture.
- RMT uses 10 memory tokens for GENA and 20 for ModernGENA. Its full segment defaults remain 512 and 1,024 BPE positions because RMT reserves memory positions internally.
- AMT uses the same 10/20 memory-token rule, but its data-token segment must reserve those positions explicitly. The shipped GENA and ModernGENA defaults are therefore 502 and 1,004. Before recurrence, each sample is compacted to attended tokens so a 30k-padded batch cannot create PAD-only memory updates. The small associative-memory projection, DPFP normalization, denominator, and recurrent state updates always run in FP32. Every shipped ModernGENA+AMT training config—including AMT+GPT—also disables both bf16 and fp16, so the complete model trains in FP32.
- Transcript-type RMT and AMT are sequence classifiers and do not use a U-Net,
  nucleotide repeater, nucleotide vocabulary, or `unet_chunk_size`.
- `family="gpt"` is valid only for segmentation. `backbone_kind` selects the
  direct GENA, ModernGENA, or Caduceus encoder; the presence of the existing
  `rmt` or `amt` block selects that memory wrapper without another selector.
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
  "true_gff": "datasets/chr20.gff",
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
| `family` | `caduceus`, `plain`, `unet`, `rmt`, `amt`, or segmentation-only `gpt` |
| `backbone_kind` | `caduceus`, `gena`, or `moderngena` |
| `backbone_path` | local path or Hugging Face model ID |
| `tokenizer_path` | local path or Hugging Face tokenizer ID |
| `trust_remote_code` | passed to Hugging Face loaders |
| `checkpoint_path` | optional model weights loaded before training; normally `null` |
| `vocab_size` | full main-tokenizer vocabulary size, inferred when `null` for U-Net/GPT nucleotide embeddings |
| `gpt` | shallow decoder dimensions, configurable context/lookahead, and `multi_token_prediction` |
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

Every shipped training config uses `max_steps=500000`, `eval_steps=5000`, and
`save_steps=5000`. Patience is 25 for every model and task except the finding
setups `long_human_mrna`, `long_human_mrna_lncrna`, and
`short_human_mrna_lncrna`, where it is 10. Patience counts consecutive
evaluations without improvement in the selected best-model metric before early
stopping. Other important fields include `num_train_epochs`, `learning_rate`,
`weight_decay`, `warmup_steps`, mixed precision, checkpoint retention, and
`resume_from_checkpoint`.

Every shipped training config also sets:

```json
"automatic_restart": true
```

Every invocation still creates a new timestamped run directory. With automatic
restart enabled and no explicit `resume_from_checkpoint`, the launcher checks
only the immediately previous run recorded for that configured output directory.
If its effective JSON configuration is identical, training resumes in the new
run from the newest complete checkpoint available from that previous run. It
never searches older sibling runs for another match. A changed or invalid
previous configuration starts fresh. An explicit `resume_from_checkpoint` takes
precedence; set `automatic_restart=false` to start fresh when no explicit
checkpoint is supplied.

Finding and transcript-type configs use globally unique `custom_prefix` values
derived from their task, setup folder, and config name. This keeps timestamped
run names unambiguous during massive parallel launches. The prefix is naming
metadata and is excluded from automatic-restart compatibility matching.

Reverse-complement processing is intentionally absent from training configs. Training and training-time validation always use one orientation only.

Training disables Transformers' `logging_nan_inf_filter`: a non-finite loss is
never replaced by the current logging accumulator and displayed as a misleading
zero. `GenatatorTrainer` checks the scalar loss before backward and the BF16/FP32
gradient norm immediately after backward, raising at the first NaN or infinity
before clipping or the optimizer step. FP16 keeps GradScaler's standard overflow
recovery instead of treating a temporary scaled-gradient overflow as fatal.

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
This field is finding-specific and is absent from every segmentation and
transcript-type training/evaluation config.

Finding training configurations are grouped as follows. Every setup contains
the same complete matrix of 22 edge and 22 region models (44 JSON files), and
validation is explicitly restricted to `hg38` with combined mRNA+lncRNA
targets in all four setups.

| Config subfolder | Training genomes | Training targets | Context class |
|---|---|---|---|
| `short_human_mrna_lncrna` | `hg38` only | combined mRNA + lncRNA | short |
| `long_human_mrna_lncrna` | `hg38` only | combined mRNA + lncRNA | long |
| `long_human_mrna` | `hg38` only | mRNA-only channels | long |
| `long_multispecies_mrna_lncrna` | all training genomes | combined mRNA + lncRNA | long |

Short means at most 1,024 BPE positions or 8,192 nucleotide positions; long
means at most 4,096 BPE positions or 32,768 nucleotide positions. Direct GENA
and GENA + U-Net remain capped at the backbone's 512 BPE positions in both
classes. GENA RMT/AMT use 1,024 or 4,096; ModernGENA uses the setup BPE cap;
Caduceus uses the exact nucleotide cap. BPE configs retain the repository's
empirical `average_bpe_token_length=9.0`, so the derived nucleotide crop is an
estimate rather than a second simultaneous hard nucleotide limit.

The finding test split uses the exact metadata value
`GCF_009914755.1_T2T-CHM13v2.0` with chromosome `NC_060944.1`. The shorter
`GCF_009914755.1` identifier belongs to the segmentation/transcript dataset and
does not match finding-test rows.

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

All 28 shipped segmentation training configs use `train-multi-specie`; their
validation dataset is always `val-human`. The configured limits are exact and
apply to both the training and validation dataset blocks:

| Backbone input | Configured limit |
|---|---:|
| Caduceus nucleotide input | 250,000 nucleotides |
| GENA / ModernGENA BPE input | 30,000 BPE tokens |

The BPE configs retain `average_bpe_token_length=9.0` for nucleotide crop
estimation. Token padding is distinct from model execution: AMT compacts each
sample before recurrence, and direct U-Net/GPT encoders process only native-size
backbone chunks.

During complete segmentation inference, every transcript is processed from beginning to end in non-overlapping model-sized chunks. The reverse-complement pass uses the same chunking, its channels and coordinates are restored, and the forward/RC logits are averaged.

### Transcript-type dataset

Transcript-type classification reuses the segmentation dataset's DNA and metadata. The target is derived from `metadata.transcript_type` (`mRNA` versus `lnc_RNA`). Training may use the representative-transcript status filter; generated inference removes it and evaluates every selected transcript.

Training configurations are grouped by dataset and context setup:

| Config subfolder | Training data | Validation data | Maximum context |
|---|---|---|---|
| `short_human` | human training split | always `val-human` | at most 1,024 BPE tokens or 8,192 nucleotides |
| `long_human` | human training split | always `val-human` | at most 4,096 BPE tokens or 32,768 nucleotides |
| `long_multispecies` | multi-species training split | always `val-human` | at most 4,096 BPE tokens or 32,768 nucleotides |

For GENA, the plain backbone's hard 512-position limit overrides the general
1,024/4,096-BPE setup cap. Long GENA configurations therefore use RMT or AMT to
consume more than 512 BPE positions. ModernGENA can use the setup cap directly.
Caduceus uses the nucleotide caps. The human setups select only the human
training dataset; `long_multispecies` uses the multi-species training dataset,
while validation remains human in every setup.

## Training

Task selection is entirely inside each JSON file.

### Edge model

```bash
torchrun --master_addr 127.0.0.1 --nproc_per_node 8 \
  finding/train.py \
  --config finding/configs/long_human_mrna_lncrna/edge_moderngena_base_plain.json
```

### Region model

```bash
torchrun --master_addr 127.0.0.1 --nproc_per_node 8 \
  finding/train.py \
  --config finding/configs/long_human_mrna_lncrna/region_moderngena_base_plain.json
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
  --config transcript_type/configs/short_human/moderngena_base_plain.json
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

All checked-in inference/evaluation JSON templates live under `experiments/`.
The task-level `configs/` trees contain training configs only. A training run
still creates its own runtime `evaluation_config.json` beside its checkpoints;
that generated run artifact is not a checked-in template.

### Generated complete gene-finding evaluation

Training either an edge or region model now generates a complete edge + region pipeline config. The trained stage and shared benchmark dataset fields are filled automatically. For the opposite stage, replace the complete `model` placeholder, replace the model-dependent dataset-length marker (including its field name) with either `max_nucleotides` or the `max_bpe_tokens`/`average_bpe_token_length` pair, fill its checkpoint, and verify the default `inference.true_gff` path. Then run:

```bash
python finding/infer.py --config runs/.../evaluation_config.json
```

It writes `finding_predictions.gff`, a combined metrics JSON containing both
stages' whole-chromosome PR-AUC and the official Hugging Face annotation
metrics, and `finding_auc_metrics.csv` containing per-class PR-AUC and ROC-AUC
for the paired edge and region models.

### Massive gene-finding pair evaluation

`experiments/massive_gene_finding_evaluation/` contains one subfolder for each
finding training setup. Each has every Cartesian pair of the 22 edge and 22
region models: 484 configs per setup and 1,936 in total. A pair config copies
the complete model/wrapper block and the setup-specific context, overlap and
target group from both source training configs. Optimizer and scheduler fields
are intentionally absent because inference never consumes them.

Before running a pair, replace both stage checkpoint placeholders and the
reference-GFF placeholder. Result GFF, JSON and CSV paths are unique to that
pair. For example:

```bash
python finding/infer.py \
  --config experiments/massive_gene_finding_evaluation/long_human_mrna_lncrna/edge_moderngena_base_plain__region_moderngena_base_plain.json
```

These configs preserve the requested T2T test benchmark for post-training
edge-region pair comparison. Even in the `long_human_mrna` setup, validation
and pair evaluation use the combined mRNA+lncRNA targets and reference; only
that setup's training targets are mRNA-only.

### Complete gene-finding pipeline

Use a config containing both `edge` and `region` stages, such as:

```bash
python finding/infer.py \
  --config experiments/small_finding_evaluation_v1/evaluation_config.json
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
9. writes per-class edge/region PR-AUC and ROC-AUC to
   `inference.metrics_csv`;
10. when `true_gff` is set, computes the official boundary/interval metrics as well.

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

`use_cds_heuristic` defaults to on. It replaces the model's directly decoded
mRNA CDS with the benchmark-compatible longest complete ORF inferred from
predicted exons. Set it to `false` to keep the model's direct CDS track. GPT
models do not emit a CDS class, so their CDS annotations require this heuristic.

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
| `finding_edge` | Per-class PR-AUC and ROC-AUC for TSS+/TSS-/PolyA+/PolyA- |
| `finding_region` | Per-class PR-AUC and ROC-AUC for intragenic+/intragenic- |
| `segmentation` | exact interval-level F1 for exon and CDS |
| `transcript_type` | accuracy |

Training validation and direct segmentation GFF inference use the same
raw-score competition sets, not independent 0.5 thresholds:

```text
exon prediction: argmax among [exon, intron]; positive only when exon wins
CDS prediction:  argmax among [CDS, intron, 5UTR, 3UTR]; positive only when CDS wins
```

The target class wins ties. Contiguous positive bases become half-open
intervals. A predicted interval is a true positive only when it exactly equals a
reference interval. Counts are pooled across validation samples before F1 is
computed.

Training-time validation never applies reverse-complement averaging.

### Inference metrics

- **Finding single-stage:** whole-chromosome per-channel and pooled PR-AUC.
- **Finding complete pipeline:** per-class PR-AUC and ROC-AUC in a separate CSV,
  plus the existing JSON and official gene-boundary metrics when a reference GFF
  is provided.
- **Segmentation:** official gene/interval metrics from prediction and reference GFF files when `true_gff` is provided.
- **Transcript type:** accuracy over every selected transcript.

## Configuration matrix

The shipped JSON files cover the logical model/task combinations:

- Finding edge and region: each of the four dataset/context setup folders has
  22 edge and 22 region configs—Caduceus PH/PS plus GENA and ModernGENA
  Base/Large with plain, U-Net, RMT + U-Net, AMT plain, and AMT + U-Net.
- Segmentation: the existing Caduceus PH/PS and GENA/ModernGENA U-Net,
  RMT + U-Net and AMT + U-Net configs remain; 14 additional GPT configs cover
  Caduceus and direct/RMT/AMT GENA/ModernGENA encoders.
- Transcript type: Caduceus PH/PS; GENA and ModernGENA Base/Large with plain,
  RMT, and AMT sequence-classification heads, organized under `short_human`,
  `long_human`, and `long_multispecies` (14 configs per setup, 42 total).

The configs represent model choices, not a hyperparameter grid. Copy the closest model config when changing dataset filters, optimization settings, overlap, context length, RMT/AMT segment settings, inference RC averaging, or segmentation CDS postprocessing.

## Sampling and shuffling examples

### Gene finding across two GPUs

Assume three chromosomes, each with four precomputed window indices:

```text
chrA: A1 A2 A3 A4
chrB: B1 B2 B3 B4
chrC: C1 C2 C3 C4
```

At the beginning of one epoch, suppose the sampler produces:

```text
shuffled chromosomes: chrB → chrC → chrA

shuffled windows inside each chromosome:
chrB: B3 B1 B4 B2
chrC: C2 C3 C1 C4
chrA: A4 A1 A3 A2
```

The chromosome-grouped order is therefore:

```text
B3 B1 B4 B2 | C2 C3 C1 C4 | A4 A1 A3 A2
```

For two GPUs, the sampler divides that order into two equal contiguous lanes:

```text
GPU 0: B3 B1 B4 B2 C2 C3
GPU 1: C1 C4 A4 A1 A3 A2
```

Training then proceeds synchronously:

| DDP step | GPU 0 sample | GPU 0 chromosome cache | GPU 1 sample | GPU 1 chromosome cache |
|---:|---|---|---|---|
| 1 | B3 | assemble chrB | C1 | assemble chrC |
| 2 | B1 | reuse chrB | C4 | reuse chrC |
| 3 | B4 | reuse chrB | A4 | release chrC; assemble chrA |
| 4 | B2 | reuse chrB | A1 | reuse chrA |
| 5 | C2 | release chrB; assemble chrC | A3 | reuse chrA |
| 6 | C3 | reuse chrC | A2 | reuse chrA |

The lane boundary falls inside chrC, so each GPU assembles its own chrC copy once. Within each GPU, however, that GPU's chrC windows remain consecutive. The sampler internally interleaves the lanes so Accelerate can shard them:

```text
B3 C1 B1 C4 B4 A4 B2 A1 C2 A3 C3 A2
```

GPU 0 receives the even positions and GPU 1 receives the odd positions. At the next epoch, both chromosome order and the window order inside each chromosome are reshuffled.

### Segmentation across two GPUs

Suppose the source Parquet rows contain several isoforms per gene:

| Gene | Transcript | Status | Training filter result |
|---|---|---:|---|
| gene A | A1 | 1 | keep |
| gene A | A2 | 0 | drop |
| gene B | B1 | 0 | drop |
| gene B | B2 | 1 | keep |
| gene B | B3 | 0 | drop |
| gene C | C1 | 1 | keep |
| gene D | D1 | 0 | drop |
| gene D | D2 | 1 | keep |
| gene E | E1 | 1 | keep |
| gene E | E2 | 0 | drop |
| gene F | F1 | 0 | drop |
| gene F | F2 | 1 | keep |

With `statuses=[1]`, the loader streams the source Parquet files once in bounded
batches, filters the rows, and writes the selected transcripts into an automatic
row-addressable Parquet sidecar. Each sidecar shard contains at most 64 selected
transcripts, with exactly one transcript per row group. A completed sidecar is
reused by later runs and by DDP ranks that share the same cache directory.

After that one-time build, the persistent in-memory dataset state contains only
compact sidecar locations. The transcript names below are explanatory labels;
DNA, labels, and per-row metadata are not retained in the index:

```text
dataset index 0 → (cache shard 0, row group 0, row 0) → A1
dataset index 1 → (cache shard 0, row group 1, row 0) → B2
dataset index 2 → (cache shard 0, row group 2, row 0) → C1
dataset index 3 → (cache shard 0, row group 3, row 0) → D2
dataset index 4 → (cache shard 0, row group 4, row 0) → E1
dataset index 5 → (cache shard 0, row group 5, row 0) → F2
```

The loader does not independently choose an isoform per gene. It keeps every row matching the configured status; the dataset is responsible for marking its representative transcript with `status=1`. If two rows for one gene have status 1, both are kept; if none do, that gene is absent.

Suppose the standard random sampler creates this epoch permutation:

```text
indices:     4, 1, 5, 0, 3, 2
transcripts: E1 B2 F2 A1 D2 C1
```

With two GPUs and `per_device_train_batch_size=1`, alternating samples are assigned to the ranks:

```text
GPU 0: E1 F2 D2
GPU 1: B2 A1 C1
```

| DDP step | GPU 0 | GPU 1 |
|---:|---|---|
| 1 | E1 | B2 |
| 2 | F2 | A1 |
| 3 | D2 | C1 |

For each selected index, that GPU opens the referenced sidecar shard and reads
its one-row Parquet row group from disk on demand, then applies the unchanged
processing sequence:

```text
read one transcript row
→ choose the transcript crop
→ slice DNA and the [length, 5] labels with identical coordinates
→ tokenize and pad
→ create tensors
→ run forward/backward
```

No application-level transcript payload or open Parquet handle is retained
between samples. Each rank's persistent dataset state holds only its compact
locations; cropping, tokenization, padding, and GPU transfer happen after
shuffling when a sample is requested. DataLoader prefetching may temporarily
hold a bounded number of in-flight samples, and the operating system may keep
disk pages in its own cache. A new deterministic permutation is generated for
the next epoch. With an odd number of retained transcripts, the standard
distributed loader repeats one sample so both GPUs execute the same number of
steps.
