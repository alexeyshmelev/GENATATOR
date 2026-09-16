# Lowest-scoring finding models: chromosome-20 BigWig export

Runs one edge model and one region model from each of the three supplied score
tables. All experiment-specific code, configs, source tables, and tests are in
this folder. The runner imports the existing GENATATOR model builders and data
loader; it does not modify their implementations or require new dependencies.

## Selected models

"Lowest" means the smallest arithmetic mean of the four edge ROC-AUC columns
or the two region ROC-AUC columns. Edge and region are selected independently
within each table, from 22 candidates each. Ties are broken by model name.

| Experiment | Edge model | Mean ROC-AUC | Region model | Mean ROC-AUC |
| --- | --- | ---: | --- | ---: |
| `short_human_mrna_lncrna` | `edge_moderngena_large_rmt_unet` | 0.6692375275 | `region_moderngena_large_rmt_unet` | 0.54807162 |
| `long_human_mrna_lncrna` | `edge_gena_large_amt_unet` | 0.62119363 | `region_gena_large_amt_unet` | 0.57065919 |
| `long_human_mrna` | `edge_gena_large_amt_unet` | 0.615109995 | `region_moderngena_large_rmt_unet` | 0.53258994 |

These are minima **among the per-model highest scores in the attached CSVs**.
They are not the lowest training checkpoints, and do not represent measured
joint edge/region pipeline performance. The CSVs have no checkpoint/run/step
identifiers. Channel maxima may come from different evaluations. Therefore they
cannot identify one checkpoint that reproduces all listed values.

To reproduce the selection without running inference:

```bash
python experiments/lowest_scoring_finding_bigwig/selection.py
```

## Checkpoints: six required values

In each of these three files, replace both `null` checkpoint values:

- `configs/short_human_mrna_lncrna.json`
- `configs/long_human_mrna_lncrna.json`
- `configs/long_human_mrna.json`

The fields are `edge.inference.checkpoint_path` and
`region.inference.checkpoint_path`. Use an absolute local checkpoint directory
containing `pytorch_model.bin` or `model.safetensors`, or a supported weight file.
The existing loader does not support sharded checkpoint index files. Relative paths
are interpreted from the GENATATOR repository root.

Each stage includes `training_run_root` to show where its training config would
normally save runs. It is a hint, not an automatic checkpoint selector. Choose
the checkpoint from the correct experiment and model; do not reuse a long model
for a short model, or an mRNA-only checkpoint for the combined target group.
Leave `model.checkpoint_path` null so weights are loaded only once.

The six trained weights and their exact paths were not supplied. They are not
bundled, and the launcher never silently substitutes an unfine-tuned backbone.
Do not alter the architecture to make an incompatible checkpoint load.

## Launch

Run from your existing GENATATOR environment and repository root. No package
installation, `pyBigWig`, UCSC converter, or other external binary is needed.
The usual model dependencies (PyTorch, Transformers, NumPy, etc.) are still
required; removing those would make running the existing models impossible.

First validate all six paths without loading models or downloading data:

```bash
python experiments/lowest_scoring_finding_bigwig/run.py --all --check
```

Then run all six models sequentially on one visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -u experiments/lowest_scoring_finding_bigwig/run.py --all
```

Or run one pair:

```bash
CUDA_VISIBLE_DEVICES=0 python -u experiments/lowest_scoring_finding_bigwig/run.py \
  --config experiments/lowest_scoring_finding_bigwig/configs/long_human_mrna.json
```

Use `--stage edge` or `--stage region` for just one model, `--device cpu` for CPU,
or `--output-root /path/to/output` to change the output location. This launcher
is not distributed: do not start it with `torchrun`, which would duplicate work.

## Chromosome and inference behavior

- Dataset: `AIRI-Institute/genatator-gene-finding-dataset`, split `test`.
- Genome: `GCF_009914755.1_T2T-CHM13v2.0`.
- Chromosome: `NC_060944.1` (T2T human chromosome 20), as in the existing finding
  evaluation configs. This is not GRCh38/hg38 chromosome 20.
- The default Hugging Face dataset/backbone identifiers can require downloads,
  using the repository's existing dependencies. Local dataset/backbone paths
  can be substituted in the configs where supported by the existing loader.
- Batch size 1, overlap 0.5, reverse-complement inference enabled, existing
  model/chunk sizes unchanged: 1,024 BPE tokens for short, 4,096 for long.
- Export `sigmoid(logits)` probabilities, averaged over all finite contributions
  from overlapping windows and forward/reverse-complement passes, matching the
  existing finding inference order. RC coordinates and output channels are
  restored before accumulation. No thresholding, smoothing, peak detection,
  interval pairing, GFF generation, or evaluation metrics are run.
- The chromosome length comes from dataset metadata, not the last covered base.
  Coordinates are zero-based, half-open. Use `NC_060944.1` in a matching T2T
  genome browser/reference; the script does not silently relabel it `chr20`.
- The existing dataset loader reads its annotated dataset, but target arrays
  are never sent into the models or used to select output positions. Alignment
  masks/repeaters are rebuilt from DNA length and tokenizer offsets. The
  `labels_mask` keyword required by legacy RMT/AMT interfaces contains only this
  input-derived content mask. No losses or reference-based metrics are computed.
- Uncovered positions are omitted from the BigWig rather than written as zero.
  Non-finite logits on covered nucleotides cause an error. Internal uncovered
  gaps retain their coordinates; later predictions are not shifted left.

The mRNA-only configs use `target_group: mrna`, matching their training configs.
The existing joint-evaluation templates use `primary` even there; that controls
which reference channels are loaded, not the trained output head. Since this
runner does not use reference targets, this does not alter model predictions.

## Outputs: 18 BigWigs from six models

A BigWig represents one numeric signal, not a multi-channel model. All channels
are preserved in separate files:

| Model role | Files per checkpoint |
| --- | --- |
| Edge | `TSS_plus.bw`, `TSS_minus.bw`, `PolyA_plus.bw`, `PolyA_minus.bw` |
| Region | `intragenic_plus.bw`, `intragenic_minus.bw` |

Default layout:

```text
experiments/lowest_scoring_finding_bigwig/outputs/<UTC_timestamp>/
    <experiment>/<model_name>/<channel>.bw
    <experiment>/<model_name>/prediction_manifest.json
```

The manifests record model, checkpoint, coordinate system, and output filenames
only, not metrics. Runs have new timestamped directories. Existing files are
never overwritten. If a later model fails, already completed models remain;
failed BigWig writes do not leave a final `.bw` that looks complete.

Disk-backed float32 accumulators bound prediction-memory usage and are removed
when each stage exits. Allow roughly `8 × chromosome_length × channel_count`
bytes of temporary disk space (about 1.9 GiB for a 66-million-base edge model),
plus final BigWigs. The existing loader and neural models have their own RAM/GPU
requirements. Models are processed one at a time.

`bigwig.py` implements compressed BigWig v4 using only Python's standard library.
It writes full-resolution float32 data, a chromosome B+ tree, a multi-level
R-tree, and the required file metadata/signatures. No zoom-summary levels are
written; whole-chromosome browsing can be slower than with a zoom-enabled file.
The total signal summary required for this format is not an accuracy metric.

Format references:
[UCSC BigWig documentation](https://genome.ucsc.edu/goldenPath/help/bigWig.html),
[UCSC writer source](https://github.com/ucscGenomeBrowser/kent/blob/master/src/lib/bwgCreate.c).

## Tests

```bash
python -m unittest discover -s experiments/lowest_scoring_finding_bigwig/tests -v
```

Tests cover the six CSV minima, matching model configs, input-only alignment,
strand swaps, overlap averaging, retained missing positions, all 18 channels,
BigWig header/index/data/summary consistency, indexed interval queries, and
exclusive/atomic output behavior. Binary-format tests use an independent
test reader, not an external library.

Full neural inference requires your six checkpoints and the existing model
environment. It was not run in the development workspace, where PyTorch and
those checkpoints were unavailable. Browser interoperability was not tested.
