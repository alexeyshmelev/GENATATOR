# Gemma and GPT-2 segmentation experiments

The existing `*_gpt_mtp_*.json` files remain the Gemma experiments. The new
`*_gpt2_mtp_*.json` files mirror both encoders (Caduceus PS and ModernGENA base)
at MTP horizons **1, 2, 5, 10, 50, 100**, for 12 additional configurations.
There are also 14 GPT-2 counterparts of the standard Gemma configurations in
`segmentation/configs/*_gpt2.json`, including the RMT and AMT encoder variants.

Each counterpart changes only `model.gpt.decoder_type`, the explicit GPT-2
`model.gpt.max_position_embeddings`, and output-directory/prefix names to avoid
collisions. Dataset lengths, encoders, MTP horizons, shifted categorical losses,
loss weighting, and non-overlapping training chunks are copied unchanged. Omitted
`decoder_type` continues to select T5Gemma, preserving existing checkpoints.
GPT-2 weights are freshly initialized, just as the Gemma decoder weights are;
Gemma decoder checkpoints cannot be loaded into a GPT-2 decoder.

## Training and validation: non-overlapping chunks

GPT-2 inherits Gemma's teacher-forcing implementation. Training and validation
split the nucleotide sequence into non-overlapping chunks of up to 8,192
positions. Each chunk receives BOS followed by its previous ground-truth
classes, uses local positions starting at zero, and runs with `use_cache=False`.
Causal attention prevents each next-token prediction from seeing its target.
Every decoder position produces logits, rather than only the last position.

Each chunk attends to its encoder chunk plus the configured lookahead, so the
encoder windows may overlap. Optional encoder additions to decoder inputs stay
aligned to the preceding class's nucleotide. All MTP heads, future-target
offsets, valid-target masks and cross-entropy normalization remain unchanged.

## Generation: continuous sliding windows with local positions

GPT-2 uses the unmodified stock model with native cross-attention and exactly
8,192 learned position embeddings. `max_position_embeddings` must equal
`context_size`; an omitted value defaults to `context_size`. This bounds the
decoder input window, **not the full sample length**. BPE encoder states are
expanded to nucleotide states before decoding. Dataset lengths are unchanged.

Generation inserts BOS once per sample and feeds back only the offset-1
classifier's argmax. Before the window fills, its cache can be extended with
one input per step. Once the window slides, the oldest input is dropped, the
latest 8,192 inputs receive positions **0..8191**, and the entire window is
recomputed with fresh self- and cross-attention KV caches. No new BOS is added,
no ground-truth classes become decoder inputs, and generation is not restarted.
Positions are relative to each current window, not cyclic transcript positions.

For a four-position example, the decoder inputs predicting successive classes
are `[BOS]`, `[BOS,y1]`, `[BOS,y1,y2]`, `[BOS,y1,y2,y3]`, then
`[y1,y2,y3,y4]` and `[y2,y3,y4,y5]`, where `y` denotes generated classes.
The last two windows are both processed at positions `[0,1,2,3]`. Predictions
already emitted are retained; only the last state produces the next class.

Each recomputed window cross-attends to the current moving encoder slice,
using the same bounds and final anchored encoder window as Gemma. Cached states
from old decoder positions or an old encoder slice are never reused after a
shift. The five-track output mapping and offset-1-only feedback are retained.
Gemma generation itself is unchanged.

## Computation and compatibility

After the first 8,192 inputs, every new generated class requires a full
8,192-query decoder pass. For context C and encoder span E, self-attention work
per step scales as O(C^2) and cross-attention as O(C*E), versus O(C) and O(E)
for a single cached query (excluding projection costs). This is much slower,
although parallel processing means it is not necessarily 8,192 times slower
in wall-clock time. Training and validation need one forward pass per non-overlapping
chunk and do not incur this per-prediction recomputation.

Stock Transformers 4.55.4 allocates square Boolean causal buffers in each
self- and cross-attention module. At 8,192 positions these consume about
256 MiB for two layers or 512 MiB for four layers, excluding weights, activations
and optimizer state. They no longer scale with the full transcript length.
GPT-2 checkpoints from the earlier full-transcript position-table implementation
have a different positional-embedding shape and are not automatically converted.

## Testing and entry point

The targeted CPU tests use tiny real stock GPT-2 models with
`transformers==4.55.4` and PyTorch 2.6.0. Use the same Transformers version for
these experiments; the repository-wide dependency range is unchanged. These
tests compare generation with a cache-free window-local reference, check cache
recomputation and feedback, preserve non-overlapping training and MTP gradients,
and cover all configurations. Full-size GPU training/inference was not run.

The existing entry point accepts these files, for example:

```bash
python segmentation/train.py --config experiments/first_gpt_model_experiments/caduceus_ps_gpt2_mtp_2.json
```

As with the Gemma configurations, this entry point may relaunch under torchrun
when multiple GPUs are visible. This change does not launch any training jobs.
