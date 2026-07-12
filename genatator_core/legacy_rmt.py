from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import TokenClassifierOutput

from .backbones import get_word_embeddings, infer_hidden_size, infer_vocab_size_from_embeddings
from .unet import DEFAULT_UNET_CHUNK_SIZE, UNET1DSegmentationHead, run_samplewise_chunked_unet

logger = logging.getLogger(__name__)


def scatter_active_rows(compact: torch.Tensor, non_empty_mask, batch_size: int) -> torch.Tensor:
    """Restore compact RMT segment rows to their original batch identities."""

    active_indices = torch.tensor(
        [i for i, is_active in enumerate(non_empty_mask) if is_active],
        dtype=torch.long,
        device=compact.device,
    )
    if int(compact.shape[0]) != int(active_indices.numel()):
        raise RuntimeError(
            f"RMT active-row mismatch: compact_rows={compact.shape[0]} active_indices={active_indices.numel()}"
        )
    full = compact.new_zeros((int(batch_size), *compact.shape[1:]))
    full.index_copy_(0, active_indices, compact)
    return full


class RMTEncoderForLetterLevelTokenClassificationUNETsegmentedRepeater(nn.Module):
    """Active RMT repeater model.

    This is the cleaned version of the provided
    `RMTEncoderForLetterLevelTokenClassificationUNETsegmentedRepeater` class.

    Preserved logic:
    - memory tokens are inserted into each BPE segment;
    - segments are processed recurrently;
    - BPE hidden states are mapped to nucleotide positions with `embedding_repeater`;
    - nucleotide embeddings are concatenated with repeated BPE hidden states;
    - the same 1D UNET head refines nucleotide-level logits;
    - cycles=3 is supported and defaults to 3.

    Deliberate changes, all shape-related:
    - hidden size is detected automatically from the loaded backbone config and embedding table;
    - UNET input width is computed as 2 * hidden_size, so GENA/ModernGENA base/large work;
    - output labels are configurable by `num_labels`;
    - RMT token encoding may be batched, while UNET is always called with one
      unpadded nucleotide sample at a time.
    """

    def __init__(self, base_model, **rmt_kwargs):
        super().__init__()
        self.model = base_model
        self.num_labels = int(rmt_kwargs.pop("num_labels"))
        self.cycles = int(rmt_kwargs.pop("cycles", 3))
        configured_chunk = rmt_kwargs.pop("unet_chunk_size", None)
        legacy_chunk = rmt_kwargs.pop("unet_sub_model_input_size", None)
        if configured_chunk is not None and legacy_chunk is not None and int(configured_chunk) != int(legacy_chunk):
            raise RuntimeError(
                "Conflicting RMT UNET chunk sizes: model.unet_chunk_size="
                f"{configured_chunk} and model.rmt.unet_sub_model_input_size={legacy_chunk}"
            )
        self.unet_chunk_size = int(
            configured_chunk if configured_chunk is not None else (
                legacy_chunk if legacy_chunk is not None else DEFAULT_UNET_CHUNK_SIZE
            )
        )
        if self.unet_chunk_size <= 0:
            raise RuntimeError("unet_chunk_size must be positive")
        self.hidden_size = infer_hidden_size(self.model.config, context="RMT.backbone")
        word_embeddings = get_word_embeddings(self.model, context="RMT.backbone")
        _, emb_hidden = infer_vocab_size_from_embeddings(word_embeddings, context="RMT.backbone")
        if emb_hidden != self.hidden_size:
            raise RuntimeError(f"RMT backbone config hidden_size={self.hidden_size}, embedding dim={emb_hidden}")
        self.nucleotide_embedding = nn.Embedding(int(rmt_kwargs.pop("nucleotide_vocab_size", 1000)), self.hidden_size)
        self.unet_input_dim = self.hidden_size * 2
        channels = rmt_kwargs.pop("unet_channels", None)
        self.sub_model = UNET1DSegmentationHead(
            embed_dim=self.unet_input_dim,
            num_classes=self.unet_input_dim,
            output_channels_list=channels,
            num_conv_layers_per_block=2,
        )
        self.activation_fn = nn.SiLU()
        self.fc = nn.Linear(self.unet_input_dim, self.num_labels)
        logger.info(
            "[RMTEncoderForLetterLevelTokenClassificationUNETsegmentedRepeater] hidden_size=%d num_labels=%d unet_input_dim=%d cycles=%d unet_chunk=%d",
            self.hidden_size, self.num_labels, self.unet_input_dim, self.cycles, self.unet_chunk_size,
        )
        self.set_params(**rmt_kwargs)
        self.rmt_config["sum_loss"] = True

    def set_params(self, num_mem_tokens, tokenizer, **rmt_config):
        self.rmt_config = rmt_config
        self.extract_special_tokens(tokenizer)
        self.extend_word_embeddings(int(num_mem_tokens))
        self.rmt_segment_size = int(rmt_config["segment_size"])
        self.max_n_segments = int(rmt_config["max_n_segments"])
        if self.rmt_segment_size <= 0:
            raise RuntimeError(f"RMT segment_size must be positive, got {self.rmt_segment_size}")
        if self.max_n_segments <= 0:
            raise RuntimeError(f"RMT max_n_segments must be positive, got {self.max_n_segments}")
        self.content_segment_size = self.rmt_segment_size - self.num_mem_tokens - 3
        if self.content_segment_size <= 0:
            raise RuntimeError(
                "RMT segment_size must exceed num_mem_tokens + 3 special positions: "
                f"segment_size={self.rmt_segment_size} num_mem_tokens={self.num_mem_tokens}"
            )
        logger.info(
            "[RMT] token ids: pad=%s cls=%s sep=%s mem_tokens=%d segment_size=%d content_tokens_per_segment=%d max_n_segments=%d bptt_depth=%s",
            self.pad_token_id, int(self.cls_token.item()), int(self.sep_token.item()), self.num_mem_tokens,
            self.rmt_segment_size, self.content_segment_size, self.max_n_segments, rmt_config.get("bptt_depth", -1),
        )

    def extract_special_tokens(self, tokenizer):
        if tokenizer.pad_token_id is None or tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
            raise RuntimeError(f"Tokenizer must define pad/cls/sep token ids. Got pad={tokenizer.pad_token_id}, cls={tokenizer.cls_token_id}, sep={tokenizer.sep_token_id}")
        self.pad_token_id = int(tokenizer.pad_token_id)
        self.register_buffer("cls_token", torch.tensor([int(tokenizer.cls_token_id)], dtype=torch.long))
        self.register_buffer("sep_token", torch.tensor([int(tokenizer.sep_token_id)], dtype=torch.long))

    def extend_word_embeddings(self, num_mem_tokens: int):
        emb = get_word_embeddings(self.model, context="RMT.extend_word_embeddings.before")
        vocab_size, hidden = infer_vocab_size_from_embeddings(emb, context="RMT.extend_word_embeddings.before")
        if hidden != self.hidden_size:
            raise RuntimeError(f"RMT hidden mismatch before extending embeddings: hidden={hidden}, expected={self.hidden_size}")
        extended_vocab_size = vocab_size + num_mem_tokens
        self.num_mem_tokens = num_mem_tokens
        self.register_buffer("mem_token_ids", torch.arange(vocab_size, extended_vocab_size, dtype=torch.long))
        self.model.resize_token_embeddings(extended_vocab_size)
        _ = get_word_embeddings(self.model, context="RMT.extend_word_embeddings.after")
        self.memory_position = range(1, 1 + num_mem_tokens)
        logger.info("[RMT] extended embeddings: old_vocab=%d new_vocab=%d hidden=%d", vocab_size, extended_vocab_size, hidden)

    def set_memory(self, memory=None):
        if memory is None:
            memory = get_word_embeddings(self.model, context="RMT.memory_embeddings")(self.mem_token_ids)
        return memory

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask

    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)

    def pad_add_special_tokens(self, tensor, segment_size, add_to="inputs"):
        if add_to == "inputs":
            elements = [self.cls_token, self.mem_token_ids, self.sep_token, tensor, self.sep_token]
            out = torch.cat(elements)
            pad_size = segment_size - out.shape[0]
            if pad_size > 0:
                out = F.pad(out, (0, pad_size), value=self.pad_token_id)
            return out
        if add_to == "labels":
            masked = torch.zeros((1, tensor.shape[-1]), device=tensor.device, dtype=tensor.dtype)
            out = torch.cat([masked, masked.repeat(self.num_mem_tokens, 1), masked, tensor, masked])
            pad_size = segment_size - out.shape[0]
            if pad_size > 0:
                out = F.pad(out, (0, 0, 0, pad_size), value=0)
            return out
        if add_to == "labels_mask":
            masked = torch.zeros((1,), device=tensor.device, dtype=tensor.dtype)
            out = torch.cat([masked, masked.repeat(self.num_mem_tokens), masked, tensor, masked])
            pad_size = segment_size - out.shape[0]
            if pad_size > 0:
                out = F.pad(out, (0, pad_size), value=0)
            return out
        raise RuntimeError(f"Unknown add_to={add_to}")

    def pad_and_segment(self, input_ids, labels=None, labels_mask=None):
        segmented_batch = []
        segmented_batch_labels = []
        segmented_batch_labels_mask = []
        if labels is None:
            labels = [None] * input_ids.shape[0]
        if labels_mask is None:
            labels_mask = [None] * input_ids.shape[0]
        for seq, y, ym in zip(input_ids, labels, labels_mask):
            content_tokens_mask = (seq != self.pad_token_id) & (seq != self.cls_token.item()) & (seq != self.sep_token.item())
            seq = seq[content_tokens_mask]
            seq = seq[: self.content_segment_size * self.max_n_segments]
            if seq.numel() == 0:
                raise RuntimeError("RMT received an empty content-token sequence after removing special tokens.")
            if y is not None:
                y = y[content_tokens_mask][: self.content_segment_size * self.max_n_segments]
            if ym is not None:
                ym = ym[content_tokens_mask][: self.content_segment_size * self.max_n_segments]
            n_seg = math.ceil(len(seq) / self.content_segment_size)
            input_segments = torch.chunk(seq, n_seg)
            segmented_batch.append([self.pad_add_special_tokens(t, self.rmt_segment_size) for t in input_segments])
            if y is not None:
                y_segments = torch.chunk(y, n_seg)
                segmented_batch_labels.append([self.pad_add_special_tokens(t, self.rmt_segment_size, add_to="labels") for t in y_segments])
            if ym is not None:
                ym_segments = torch.chunk(ym, n_seg)
                segmented_batch_labels_mask.append([self.pad_add_special_tokens(t, self.rmt_segment_size, add_to="labels_mask") for t in ym_segments])
        actual_n_segments = max(len(s) for s in segmented_batch)
        segmented_batch = [[s[::-1][i] if len(s) > i else None for s in segmented_batch] for i in range(actual_n_segments)][::-1]
        if segmented_batch_labels:
            segmented_batch_labels = [[s[::-1][i] if len(s) > i else None for s in segmented_batch_labels] for i in range(actual_n_segments)][::-1]
        if segmented_batch_labels_mask:
            segmented_batch_labels_mask = [[s[::-1][i] if len(s) > i else None for s in segmented_batch_labels_mask] for i in range(actual_n_segments)][::-1]
        return segmented_batch, segmented_batch_labels, segmented_batch_labels_mask

    def _encode_rmt_tokens(self, input_ids, labels=None, labels_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        bs = input_ids.shape[0]
        memory = self.set_memory().repeat(bs, 1, 1)
        segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(input_ids, labels, labels_mask)
        logits = []
        logits_masks = []
        labels_segm = []
        out = None
        for seg_num, (segment_input_ids, segment_labels, segment_labels_mask) in enumerate(zip(segmented, segmented_labels, segmented_labels_mask)):
            if (int(self.rmt_config.get("bptt_depth", -1)) > -1) and (len(segmented) - seg_num > int(self.rmt_config["bptt_depth"])):
                memory = memory.detach()
            non_empty_mask = [s is not None for s in segment_input_ids]
            if sum(non_empty_mask) == 0:
                continue
            seg_input_ids = torch.stack([s for s in segment_input_ids if s is not None])
            seg_attention_mask = self.get_attention_mask(seg_input_ids)
            seg_token_type_ids = self.get_token_type_ids(seg_input_ids)
            seg_inputs_embeds = get_word_embeddings(self.model, context="RMT.forward_embeddings")(seg_input_ids)
            seg_inputs_embeds[:, self.memory_position] = memory[non_empty_mask]
            out = self.model(input_ids=None, inputs_embeds=seg_inputs_embeds, attention_mask=seg_attention_mask, token_type_ids=seg_token_type_ids, output_hidden_states=True, return_dict=True)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]
            # Compacting active rows and then padding at the end changes sample
            # identity when, for example, non_empty_mask=[False, True]. Scatter
            # every segment back to its original batch indices immediately.
            logits.append(scatter_active_rows(out.logits, non_empty_mask, bs))
            if segment_labels is not None:
                compact_labels = torch.stack([el for el, m in zip(segment_labels, non_empty_mask) if m])
                labels_segm.append(scatter_active_rows(compact_labels, non_empty_mask, bs))
            if segment_labels_mask is not None:
                compact_mask = torch.stack([el for el, m in zip(segment_labels_mask, non_empty_mask) if m])
                logits_masks.append(scatter_active_rows(compact_mask, non_empty_mask, bs))
        if out is None:
            raise RuntimeError("RMT produced no segment outputs.")
        token_logits = torch.cat(logits, dim=1)
        token_mask = torch.cat(logits_masks, dim=1) if logits_masks else None
        return token_logits, token_mask

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, labels_mask=None, pos_weight=None, output_attentions=None, output_hidden_states=None, return_dict=None, embedding_repeater=None, letter_level_tokens=None, letter_level_labels=None, letter_level_labels_mask=None, letter_level_token_types_ids=None, letter_level_attention_mask=None):
        if embedding_repeater is None or letter_level_tokens is None:
            raise RuntimeError("RMT repeater requires embedding_repeater and letter_level_tokens.")
        token_logits, token_mask = self._encode_rmt_tokens(input_ids, labels=labels, labels_mask=labels_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if token_mask is None:
            raise RuntimeError("RMT token mask was not produced. labels_mask is required for repeater alignment.")
        loss, logits = run_samplewise_chunked_unet(
            token_hidden=token_logits,
            token_content_mask=token_mask,
            embedding_repeater=embedding_repeater,
            letter_level_tokens=letter_level_tokens,
            letter_level_attention_mask=letter_level_attention_mask,
            letter_level_labels=letter_level_labels,
            letter_level_labels_mask=letter_level_labels_mask,
            pos_weight=pos_weight,
            nucleotide_embedding=self.nucleotide_embedding,
            unet=self.sub_model,
            activation_fn=self.activation_fn,
            classifier=self.fc,
            cycles=self.cycles,
            chunk_size=self.unet_chunk_size,
            context="RMTRepeater",
        )
        return TokenClassifierOutput(loss=loss, logits=logits)
