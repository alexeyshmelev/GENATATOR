# GENATATOR clean training repository — first pass

This repository is a simplified, unified rewrite of the current GENATATOR finetuning code. It keeps the public task layout explicit:

```text
finding/          edge and region models for transcript-interval discovery
segmentation/     exon / intron / UTR / CDS segmentation models
transcript_type/  mRNA vs lnc_RNA transcript classifier
genatator_core/   shared dataset, model, training, inference, metrics code
```

The extra `genatator_core/` folder is the only shared layer. Without it, the same tokenizer alignment, DDP training, checkpointing, and metric code would be duplicated three times.

## What is unified

The old pipeline mixed shell launchers, Hydra YAML, task-specific datasets, model wrappers, and repeated Accelerate trainers. This version uses one JSON-driven path for all tasks:

```bash
accelerate launch finding/train.py --config finding/configs/edge_moderngena_base.json
accelerate launch finding/train.py --config finding/configs/region_moderngena_base.json
accelerate launch segmentation/train.py --config segmentation/configs/caduceus_ps.json
accelerate launch segmentation/train.py --config segmentation/configs/gena_large_rmt_unet.json
accelerate launch transcript_type/train.py --config transcript_type/configs/caduceus_ps.json
```

Every config has the same sections:

```json
{
  "task": {},
  "data": {},
  "tokenizer": {},
  "window": {},
  "model": {},
  "training": {},
  "eval": {}
}
```

No Hydra is used. Local and Hugging Face data are configured by changing only `data.source` and paths.

## Supported data layouts

### Hugging Face datasets

Gene finding:

```json
"data": {
  "source": "hf",
  "repo": "AIRI-Institute/genatator-gene-finding-dataset",
  "train_split": "train",
  "validation_split": "validation"
}
```

Segmentation and transcript type:

```json
"data": {
  "source": "hf",
  "repo": "AIRI-Institute/genatator-gene-segmentation-dataset",
  "train_name": "train-multi-specie",
  "validation_name": "val-human",
  "train_split": "train",
  "validation_split": "validation",
  "representative_only": true
}
```

### Local parquet mirror

```json
"data": {
  "source": "local_parquet",
  "data_files": {
    "train": "data/train/part-*/*.parquet",
    "validation": "data/validation/part-*/*.parquet"
  }
}
```

Set `tokenizer.local_files_only` and `model.local_files_only` to `true` when running without internet.

## Model families

### ModernGENA / ModernBERT

Use:

```json
"model": {
  "family": "modernbert_token_classifier",
  "pretrained_model_name_or_path": "AIRI-Institute/moderngena-base",
  "num_labels": 4,
  "label_mode": "token"
}
```

This intentionally uses `transformers.ModernBertForTokenClassification` and computes masked BCE outside the HF loss so edge, region, and multilabel tracks work correctly.

### Caduceus PS / PH

Use nucleotide tokenization and a linear token head:

```json
"tokenizer": {
  "kind": "nucleotide",
  "path": "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
  "pad_token_id": 4,
  "eos_token_id": 1,
  "add_eos": true,
  "pad_side": "left"
}
```

### GENA base / large with RMT + U-Net

Use BPE tokenization, the RMT adapter, and `label_mode = nucleotide_unet`:

```json
"model": {
  "family": "auto_encoder",
  "pretrained_model_name_or_path": "AIRI-Institute/gena-lm-bert-large-t2t",
  "adapter": {
    "type": "rmt",
    "num_mem_tokens": 10,
    "segment_size": 512,
    "cls_token_id": 1,
    "sep_token_id": 2,
    "pad_token_id": 3
  },
  "label_mode": "nucleotide_unet"
}
```

For BPE segmentation, token hidden states are repeated back to nucleotide resolution using tokenizer offsets, concatenated with nucleotide embeddings, and passed through a compact 1D U-Net.

### ARMT

Set:

```json
"adapter": {
  "type": "armt",
  "armt_repo_id": "irodkin/armt-neox-tiny",
  "segment_size": 512,
  "num_mem_tokens": 16
}
```

This is wired in `genatator_core/modeling.py`, but it is the least tested part because the exact external ARMT wrapper API may differ between checkpoints.

## Inference

### Finding pipeline

Train edge and region separately, then run the interval pipeline:

```bash
python finding/infer.py \
  --config finding/configs/infer_pipeline.json \
  --fasta genome_or_chromosome.fa \
  --output-dir predictions/finding \
  --device cuda
```

For each FASTA record it writes:

```text
<record>.tracks.npz       edge and region nucleotide probability tracks
<record>.intervals.tsv    postprocessed stranded transcript intervals
```

The post-processing implements FFT low-pass smoothing, peak calling, TSS/PolyA pairing, and region-model filtering.

### Segmentation

```bash
python segmentation/infer.py \
  --config segmentation/configs/caduceus_ps.json \
  --checkpoint runs/segmentation/caduceus_ps/best/pytorch_model.bin \
  --fasta transcripts.fa \
  --output-dir predictions/segmentation
```

### Transcript type

```bash
python transcript_type/infer.py \
  --config transcript_type/configs/caduceus_ps.json \
  --checkpoint runs/transcript_type/caduceus_ps/best/pytorch_model.bin \
  --fasta transcript_candidates.fa \
  --output predictions/transcript_type.tsv
```

## Logging and checkpoints

TensorBoard logs are written to:

```text
<training.output_dir>/tb
```

Best checkpoints are written to:

```text
<training.output_dir>/best/pytorch_model.bin
<training.output_dir>/best/config.json
<training.output_dir>/best/metrics.json
```

The training loop uses `accelerate`, `DistributedSampler`, gradient accumulation, tqdm progress bars, AdamW, warmup scheduling, and best-checkpoint selection by any metric in the JSON config.

## Metrics included

`genatator_core/metrics.py` includes:

- masked PR-AUC and ROC-AUC for token / nucleotide labels;
- exact interval precision, recall, and F1 for segmentation classes;
- a first gene-level segmentation utility that groups rows by `gene_id`;
- kX-style interval boundary metrics for transcript discovery.

The full chromosome-level gene-finding benchmark and full isoform-aware gene-level segmentation evaluation should be run as separate evaluation jobs after model inference, because they require complete chromosome/transcript reconstruction rather than mini-batch validation windows.

## Problems and known first-pass gaps

1. I did not download or run the HF datasets or model checkpoints. The repository compiles, but the tokenizer details for Caduceus and ARMT still need real-cluster smoke tests.
2. The old code uses several experimental middle-loss and all-layer-loss variants. I kept the clean final-head path only; middle-loss can be added as a small optional model flag later.
3. The full official gene-level benchmark needs a second pass: `segmentation/infer.py` produces per-sequence predictions, and the metric primitives are present, but a complete chromosome-20 evaluator that exactly reproduces the paper table is not yet implemented.
4. ARMT support is config-wired but not validated against the exact checkpoint class API.
5. Reverse-complement test-time augmentation is implemented behind `inference.use_reverse_complement`, but I did not smoke-test it with real GENATATOR checkpoints.
6. The repository assumes that BPE tokenizers provide offset mappings. That is true for the intended ModernGENA/GENA fast-tokenizer path, but should be checked for every local tokenizer directory.
