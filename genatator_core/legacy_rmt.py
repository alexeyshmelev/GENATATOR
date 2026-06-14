from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from transformers.modeling_outputs import TokenClassifierOutput

from .backbones import get_word_embeddings, infer_hidden_size, infer_vocab_size_from_embeddings
from .unet import UNET1DSegmentationHead

logger = logging.getLogger(__name__)


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
    - batch size is explicitly restricted to 1.
    """

    def __init__(self, base_model, **rmt_kwargs):
        super().__init__()
        self.model = base_model
        self.num_labels = int(rmt_kwargs.pop("num_labels"))
        self.cycles = int(rmt_kwargs.pop("cycles", 3))
        self.unet_sub_model_input_size = int(rmt_kwargs.get("unet_sub_model_input_size", 8192))
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
            self.hidden_size, self.num_labels, self.unet_input_dim, self.cycles, self.unet_sub_model_input_size,
        )
        self.set_params(**rmt_kwargs)
        self.rmt_config["sum_loss"] = True

    def set_params(self, num_mem_tokens, tokenizer, **rmt_config):
        self.rmt_config = rmt_config
        self.extract_special_tokens(tokenizer)
        self.extend_word_embeddings(int(num_mem_tokens))
        self.segment_size = int(rmt_config["input_size"]) - self.num_mem_tokens - 3
        if self.segment_size <= 0:
            raise RuntimeError(f"RMT segment_size must be positive: input_size={rmt_config['input_size']} num_mem_tokens={self.num_mem_tokens}")
        logger.info(
            "[RMT] token ids: pad=%s cls=%s sep=%s mem_tokens=%d input_size=%d segment_size=%d max_n_segments=%d bptt_depth=%s",
            self.pad_token_id, int(self.cls_token.item()), int(self.sep_token.item()), self.num_mem_tokens,
            int(rmt_config["input_size"]), self.segment_size, int(rmt_config["max_n_segments"]), rmt_config.get("bptt_depth", -1),
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
        self.model.embeddings = get_word_embeddings(self.model, context="RMT.extend_word_embeddings.after")
        self.memory_position = range(1, 1 + num_mem_tokens)
        logger.info("[RMT] extended embeddings: old_vocab=%d new_vocab=%d hidden=%d", vocab_size, extended_vocab_size, hidden)

    def set_memory(self, memory=None):
        if memory is None:
            memory = self.model.embeddings(self.mem_token_ids)
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
            seq = seq[: self.segment_size * int(self.rmt_config["max_n_segments"])]
            if seq.numel() == 0:
                raise RuntimeError("RMT received an empty content-token sequence after removing special tokens.")
            if y is not None:
                y = y[content_tokens_mask][: self.segment_size * int(self.rmt_config["max_n_segments"])]
            if ym is not None:
                ym = ym[content_tokens_mask][: self.segment_size * int(self.rmt_config["max_n_segments"])]
            n_seg = math.ceil(len(seq) / self.segment_size)
            input_segments = torch.chunk(seq, n_seg)
            segmented_batch.append([self.pad_add_special_tokens(t, int(self.rmt_config["input_size"])) for t in input_segments])
            if y is not None:
                y_segments = torch.chunk(y, n_seg)
                segmented_batch_labels.append([self.pad_add_special_tokens(t, int(self.rmt_config["input_size"]), add_to="labels") for t in y_segments])
            if ym is not None:
                ym_segments = torch.chunk(ym, n_seg)
                segmented_batch_labels_mask.append([self.pad_add_special_tokens(t, int(self.rmt_config["input_size"]), add_to="labels_mask") for t in ym_segments])
        max_n_segments = int(self.rmt_config["max_n_segments"])
        segmented_batch = [[s[::-1][i] if len(s) > i else None for s in segmented_batch] for i in range(max_n_segments)][::-1]
        if segmented_batch_labels:
            segmented_batch_labels = [[s[::-1][i] if len(s) > i else None for s in segmented_batch_labels] for i in range(max_n_segments)][::-1]
        if segmented_batch_labels_mask:
            segmented_batch_labels_mask = [[s[::-1][i] if len(s) > i else None for s in segmented_batch_labels_mask] for i in range(max_n_segments)][::-1]
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
            seg_inputs_embeds = self.model.embeddings(seg_input_ids)
            seg_inputs_embeds[:, self.memory_position] = memory[non_empty_mask]
            out = self.model(input_ids=None, inputs_embeds=seg_inputs_embeds, attention_mask=seg_attention_mask, token_type_ids=seg_token_type_ids, output_hidden_states=True, return_dict=True)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]
            logits.append(out.logits)
            if segment_labels is not None:
                labels_segm.append(torch.stack([el for el, m in zip(segment_labels, non_empty_mask) if m]))
            if segment_labels_mask is not None:
                logits_masks.append(torch.stack([el for el, m in zip(segment_labels_mask, non_empty_mask) if m]))
        if out is None:
            raise RuntimeError("RMT produced no segment outputs.")
        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            if labels_segm:
                labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
            if logits_masks:
                logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))
        token_logits = torch.cat(logits, dim=1)
        token_mask = torch.cat(logits_masks, dim=1) if logits_masks else None
        return token_logits, token_mask

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, labels_mask=None, pos_weight=None, output_attentions=None, output_hidden_states=None, return_dict=None, embedding_repeater=None, letter_level_tokens=None, letter_level_labels=None, letter_level_labels_mask=None, letter_level_token_types_ids=None, letter_level_attention_mask=None):
        if input_ids.shape[0] != 1:
            raise RuntimeError("RMTEncoderForLetterLevelTokenClassificationUNETsegmentedRepeater requires batch size 1.")
        if embedding_repeater is None or letter_level_tokens is None or letter_level_labels_mask is None:
            raise RuntimeError("RMT repeater requires embedding_repeater, letter_level_tokens and letter_level_labels_mask.")
        token_logits, token_mask = self._encode_rmt_tokens(input_ids, labels=labels, labels_mask=labels_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if token_mask is None:
            raise RuntimeError("RMT token mask was not produced. labels_mask is required for repeater alignment.")
        curr_logits = token_logits[0, token_mask[0].bool(), :].unsqueeze(0)
        lmask = letter_level_labels_mask[0].bool()
        curr_repeater = embedding_repeater[0][lmask].long()
        if curr_repeater.numel() == 0:
            raise RuntimeError("RMT embedding_repeater is empty after masking.")
        if curr_repeater.min().item() < 0 or curr_repeater.max().item() >= curr_logits.shape[1]:
            raise RuntimeError(f"RMT repeater range [{curr_repeater.min().item()}, {curr_repeater.max().item()}] incompatible with token length {curr_logits.shape[1]}")
        nt_emb = self.nucleotide_embedding(letter_level_tokens[0][lmask].unsqueeze(0))
        repeated = torch.cat((nt_emb, curr_logits[:, curr_repeater, :]), dim=-1)
        target = letter_level_labels[0][lmask].unsqueeze(0) if letter_level_labels is not None else None
        weight = pos_weight[0, 0, :].to(repeated.device).float() if pos_weight is not None else None
        loss_fct = BCEWithLogitsLoss(pos_weight=weight)
        loss = 0.0
        collected_logits = None
        x = repeated
        num_chunks = math.ceil(x.shape[1] / self.unet_sub_model_input_size)
        for _ in range(self.cycles):
            logits_chunks = []
            embedding_chunks = []
            for i in range(num_chunks):
                chunk = x[:, i * self.unet_sub_model_input_size : (i + 1) * self.unet_sub_model_input_size, :]
                z = self.activation_fn(self.sub_model(chunk.transpose(1, 2))).transpose(1, 2)
                embedding_chunks.append(z)
                logits_chunks.append(self.fc(z))
            collected_logits = torch.cat(logits_chunks, dim=1)
            if target is not None:
                loss = loss + loss_fct(collected_logits.float(), target.float())
            x = x + torch.cat(embedding_chunks, dim=1)
        if collected_logits.shape[1] != letter_level_tokens.shape[1]:
            collected_logits = F.pad(collected_logits, (0, 0, 0, letter_level_tokens.shape[1] - collected_logits.shape[1]))
        return TokenClassifierOutput(loss=(loss / self.cycles if target is not None else None), logits=collected_logits)
