# Gemma and GPT-2 segmentation experiments

The existing `*_gpt_mtp_*.json` files remain the Gemma experiments. The new
`*_gpt2_mtp_*.json` files mirror both encoders (Caduceus PS and ModernGENA base)
at MTP horizons **1, 2, 5, 10, 50, 100**, for 12 additional configurations.
There are also 14 GPT-2 counterparts of the standard Gemma configurations in
`segmentation/configs/*_gpt2.json`, including the RMT and AMT encoder variants.

Each counterpart changes only `model.gpt.decoder_type`, the explicit GPT-2
`model.gpt.max_position_embeddings`, and output-directory/prefix names to avoid
collisions. Dataset lengths, encoders, MTP horizons, shifted categorical losses,
loss weighting, and training/inference policies are copied unchanged. Omitted
`decoder_type` continues to select T5Gemma, preserving existing checkpoints.
GPT-2 weights are freshly initialized, just as the Gemma decoder weights are;
Gemma decoder checkpoints cannot be loaded into a GPT-2 decoder.

## Positional and memory incompatibilities: read before launching

GPT-2 uses its stock learned absolute position embeddings and native
cross-attention. `context_size` bounds the retained self-attention history;
`max_position_embeddings` separately bounds the **entire nucleotide sample**.
BPE encoder tokens are expanded to nucleotides before decoding. The mirrors
therefore set capacity from the existing dataset's resolved nucleotide limit,
not its BPE-token count. Samples exceeding that capacity fail before decoder
generation. Positions are never wrapped, clamped, or reset during generation.

Two constraints cannot be removed while preserving the requested behavior:

1. Existing Gemma teacher forcing starts each independent training chunk with
   BOS and positions `0..context_size-1`. This is preserved, but GPT-2 position
   embeddings beyond that range receive no training signal. Continuous
   inference beyond the first chunk consequently uses untrained position rows.
   A global-position training policy would address this particular mismatch,
   but would change the requested training behavior and is not applied here.
2. Stock Transformers 4.55.4 GPT-2 allocates a square Boolean causal-mask buffer
   per self- and cross-attention module, even with SDPA. Its approximate buffer
   storage is `2 * num_decoder_layers * max_position_embeddings**2` bytes:

   | Config group | Position capacity | Layers | Mask buffers alone |
   |---|---:|---:|---:|
   | Caduceus MTP mirrors | 32,768 | 4 | 8 GiB |
   | ModernGENA MTP mirrors | 73,728 | 4 | 40.5 GiB |
   | Standard Caduceus mirrors | 250,000 | 2 | 232.8 GiB |
   | Standard BPE mirrors | 270,000 | 2 | 271.6 GiB |

   These estimates exclude weights, activations, optimizer state, and temporary
   allocations. DDP replicates these buffers on each GPU; it does not divide
   them among GPUs. The standard long-context mirrors are therefore not
   practical on ordinary GPUs under the stock-internals constraint. They are
   faithful configuration mirrors, **not validated production-scale runs**.
   No dataset lengths or GPT-2 internals are silently changed to bypass this.

## Decoder behavior and testing

Generation starts with BOS once per sample, then feeds back only the offset-1
classifier's argmax. The recent decoder K/V history rolls continuously across
context boundaries; explicit absolute position IDs keep increasing. The same
moving encoder-window bounds as Gemma are used, including the final anchored
window. Cross-attention cache entries are invalidated when that window moves.
Auxiliary MTP heads participate in training only. The existing shared loss and
five-track output mapping are retained.

The targeted CPU tests use tiny real stock GPT-2 models with
`transformers==4.55.4` and PyTorch 2.6.0. Use the same Transformers version for
these experiments; the repository-wide dependency range is unchanged. These
tests exercise cache rollover, native cross-attention, position limits, MTP,
and the configuration matrix without allocating the full experiment models.

Once the above capacity and training-position limitations have been accounted
for, the existing entry point accepts the new files, for example:

```bash
python segmentation/train.py --config experiments/first_gpt_model_experiments/caduceus_ps_gpt2_mtp_2.json
```

As with the Gemma configurations, this entry point may relaunch under torchrun
when multiple GPUs are visible. This change does not launch any training jobs.
