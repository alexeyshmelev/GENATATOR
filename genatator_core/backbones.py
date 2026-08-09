from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, ModernBertForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput

from .config import local_or_remote
from .torch_compat import allow_transformers_torch_load_on_legacy_torch

logger = logging.getLogger(__name__)


def infer_hidden_size(config: Any, *, context: str) -> int:
    for name in ("hidden_size", "d_model", "n_embd", "embed_dim"):
        value = getattr(config, name, None)
        if value is not None:
            hidden = int(value)
            logger.info("[%s] hidden_size detected from config.%s=%d", context, name, hidden)
            return hidden
    raise RuntimeError(f"Could not infer hidden size for {context}. Config={config}")


def infer_vocab_size_from_embeddings(emb: nn.Module, *, context: str) -> tuple[int, int]:
    if not hasattr(emb, "weight"):
        raise RuntimeError(f"Embedding module for {context} has no weight: {type(emb).__name__}")
    shape = tuple(emb.weight.shape)
    if len(shape) != 2:
        raise RuntimeError(f"Embedding weight for {context} must be 2D, got {shape}")
    logger.info("[%s] embedding table detected: vocab_size=%d hidden_size=%d", context, shape[0], shape[1])
    return int(shape[0]), int(shape[1])


def get_word_embeddings(model: nn.Module, *, context: str) -> nn.Embedding:
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None:
            infer_vocab_size_from_embeddings(emb, context=context)
            return emb
    for path in (
        ("base_model", "embeddings", "word_embeddings"),
        ("model", "embeddings", "tok_embeddings"),
        ("model", "embeddings", "word_embeddings"),
        ("bert", "embeddings", "word_embeddings"),
        ("encoder", "embeddings", "word_embeddings"),
    ):
        obj: Any = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            infer_vocab_size_from_embeddings(obj, context=context)
            return obj
    raise RuntimeError(f"Could not detect word embeddings for {context}; model class={type(model).__name__}")


class HiddenStateBackbone(nn.Module):
    """GENA/ModernGENA hidden-state adapter.

    Active class choices:
    - ModernGENA loads through `transformers.ModernBertForTokenClassification.from_pretrained`.
      Only one registered module reference is kept, so Trainer/safetensors does not see
      duplicated shared tensors.
    - GENA loads through `AutoModel.from_pretrained`. If the checkpoint is exposed as
      `BertForMaskedLM`, we keep only its internal `.bert` encoder as the trainable module;
      the LM head is intentionally dropped because the fine-tuning head is defined here.

    All retained parameters stay trainable. This class never freezes anything.
    """

    def __init__(self, backbone_path: str, backbone_kind: str, trust_remote_code: bool = True, modernbert_num_labels: int = 2, allow_unsafe_torch_load: bool = True):
        super().__init__()
        self.backbone_kind = backbone_kind
        self.backbone_path = local_or_remote(backbone_path)
        self.trust_remote_code = trust_remote_code
        self.uses_owner = False
        self.encoder_attr = None
        allow_transformers_torch_load_on_legacy_torch(allow_unsafe_torch_load, context=f"HiddenStateBackbone:{backbone_kind}:{self.backbone_path}")

        if backbone_kind == "moderngena":
            logger.info("[backbone] loading ModernGENA through ModernBertForTokenClassification: %s", self.backbone_path)
            owner = ModernBertForTokenClassification.from_pretrained(
                self.backbone_path,
                num_labels=int(modernbert_num_labels),
                trust_remote_code=trust_remote_code,
            )
            # HiddenStateBackbone calls the encoder directly and supplies its own
            # task head in the enclosing GENATATOR model.  Keeping ModernBERT's
            # pretrained token-classification head here would register trainable
            # parameters that never participate in forward/backward, which can
            # break DDP when unused-parameter discovery is disabled.
            if hasattr(owner, "classifier"):
                owner.classifier = nn.Identity()
                logger.info("[backbone] replaced unused ModernBERT classifier with Identity")
            # Keep exactly one registered module. Do not additionally assign
            # `self.encoder = owner.model`, because that creates duplicated named
            # parameters and safetensors refuses to save them.
            self.owner = owner
            self.uses_owner = True
            for attr in ("model", "modernbert", "bert"):
                if hasattr(owner, attr):
                    self.encoder_attr = attr
                    break
            if self.encoder_attr is None:
                raise RuntimeError(f"ModernBertForTokenClassification has no known encoder attribute: children={list(dict(owner.named_children()).keys())}")
            self.config = owner.config
            encoder_for_shape = getattr(owner, self.encoder_attr)
            logger.info("[backbone] ModernGENA owner class=%s encoder_attr=%s encoder_class=%s", type(owner).__name__, self.encoder_attr, type(encoder_for_shape).__name__)

        elif backbone_kind == "gena":
            logger.info("[backbone] loading GENA AutoModel: %s", self.backbone_path)
            raw = AutoModel.from_pretrained(self.backbone_path, trust_remote_code=trust_remote_code)
            self.config = raw.config
            # Some released GENA checkpoints expose a masked-language-model class through
            # AutoModel. Its first output is vocabulary logits [B, T, vocab_size], not
            # hidden states. For fine-tuning we keep only the internal encoder as a
            # registered module. This also avoids duplicate shared tensors during save.
            if hasattr(raw, "bert"):
                self.encoder = raw.bert
                logger.info("[backbone] GENA AutoModel class=%s contains `.bert`; registering only internal BertModel encoder for hidden states", type(raw).__name__)
                del raw
            else:
                self.encoder = raw
                logger.info("[backbone] GENA AutoModel class=%s registered directly as hidden-state encoder", type(raw).__name__)
        else:
            raise RuntimeError(f"HiddenStateBackbone supports only backbone_kind='gena' or 'moderngena', got {backbone_kind}")

        self.hidden_size = infer_hidden_size(self.config, context=f"HiddenStateBackbone:{backbone_kind}")
        emb = get_word_embeddings(self._embedding_source(), context=f"HiddenStateBackbone:{backbone_kind}")
        _, emb_hidden = infer_vocab_size_from_embeddings(emb, context=f"HiddenStateBackbone:{backbone_kind}")
        if emb_hidden != self.hidden_size:
            raise RuntimeError(f"Backbone hidden mismatch: config hidden_size={self.hidden_size}, embedding dim={emb_hidden}")
        logger.info("[backbone] loaded kind=%s hidden_size=%d class=%s", backbone_kind, self.hidden_size, type(self._encoder()).__name__)

    def _encoder(self) -> nn.Module:
        if self.uses_owner:
            return getattr(self.owner, self.encoder_attr)
        return self.encoder

    def _embedding_source(self) -> nn.Module:
        # For ModernBERT token-classification wrapper, input embeddings are on the owner.
        # For GENA we registered only the encoder.
        return self.owner if self.uses_owner else self.encoder

    def get_input_embeddings(self):
        return get_word_embeddings(self._embedding_source(), context="HiddenStateBackbone.get_input_embeddings")

    def resize_token_embeddings(self, new_num_tokens: int):
        logger.info("[backbone] resize token embeddings to %d", new_num_tokens)
        source = self._embedding_source()
        if not hasattr(source, "resize_token_embeddings"):
            raise RuntimeError(f"Registered backbone source {type(source).__name__} does not support resize_token_embeddings")
        resized = source.resize_token_embeddings(new_num_tokens)
        _ = get_word_embeddings(source, context="HiddenStateBackbone.after_resize")
        return resized

    @property
    def embeddings(self):
        # Kept as a compatibility property for RMT code. It is intentionally
        # not a registered submodule, avoiding duplicate shared tensors during save.
        return get_word_embeddings(self._embedding_source(), context="HiddenStateBackbone.embeddings")

    def _extract_hidden(self, out):
        hidden_states = getattr(out, "hidden_states", None)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            first = out[0] if isinstance(out, (tuple, list)) or hasattr(out, "__getitem__") else None
            if first is not None and getattr(first, "shape", None) is not None and first.shape[-1] == self.hidden_size:
                hidden = first
            elif hidden_states is not None and len(hidden_states) > 0:
                hidden = hidden_states[-1]
                logger.info("[backbone.forward] using hidden_states[-1] because first output is not hidden-sized")
            else:
                raise RuntimeError(
                    f"Backbone did not return hidden states with hidden_size={self.hidden_size}. "
                    f"Output type={type(out).__name__}"
                )
        if hidden.shape[-1] != self.hidden_size:
            if hidden_states is not None and len(hidden_states) > 0 and hidden_states[-1].shape[-1] == self.hidden_size:
                logger.info(
                    "[backbone.forward] first hidden candidate had width %d; using hidden_states[-1] width %d instead",
                    hidden.shape[-1], hidden_states[-1].shape[-1],
                )
                hidden = hidden_states[-1]
            else:
                raise RuntimeError(f"Backbone emitted hidden width {hidden.shape[-1]}, expected {self.hidden_size}")
        return hidden, hidden_states

    def context_limit(self) -> int:
        """Return the maximum number of BPE positions accepted per encoder call."""

        for name in (
            "max_position_embeddings",
            "max_sequence_length",
            "max_seq_len",
            "n_positions",
        ):
            value = getattr(self.config, name, None)
            if value is not None and int(value) > 0:
                return int(value)
        # Released GENA checkpoints use 512 positions.  ModernGENA uses an
        # 8K training context, but keep this only as a compatibility fallback
        # for remote configs that omit the otherwise standard field.
        return 512 if self.backbone_kind == "gena" else 8192

    def forward_chunked(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        *,
        chunk_size: int | None = None,
        **kwargs,
    ):
        """Encode long padded inputs as non-overlapping, unpadded calls.

        Segmentation may expose as many as 30,000 BPE positions while the
        underlying GENA/ModernGENA encoder retains its native context limit.
        Each sample is compacted to its attended span before it is split.  In
        particular, fully padded tail chunks never enter attention, which also
        keeps the long-input execution numerically well-defined in BF16.

        The chunks do not overlap and no hidden state is carried between them;
        this changes only how an over-length outer input is executed, not the
        encoder architecture.  Returned logits retain the original padded
        ``[batch, sequence, hidden]`` shape for the existing repeater maps.
        """

        source = input_ids if input_ids is not None else inputs_embeds
        if source is None:
            raise RuntimeError("forward_chunked requires input_ids or inputs_embeds")
        if source.ndim < 2:
            raise RuntimeError(
                f"forward_chunked expects a batched sequence, got {tuple(source.shape)}"
            )
        batch_size, padded_length = int(source.shape[0]), int(source.shape[1])
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, padded_length),
                dtype=torch.long,
                device=source.device,
            )
        if tuple(attention_mask.shape) != (batch_size, padded_length):
            raise RuntimeError(
                "attention_mask must align with the chunked encoder input: "
                f"mask={tuple(attention_mask.shape)} input={tuple(source.shape[:2])}"
            )
        limit = int(chunk_size if chunk_size is not None else self.context_limit())
        if limit <= 0:
            raise RuntimeError(f"chunk_size must be positive, got {limit}")

        sample_outputs = []
        for sample_index in range(batch_size):
            attended = attention_mask[sample_index].bool()
            positions = attended.nonzero(as_tuple=False).flatten()
            if positions.numel() == 0:
                raise RuntimeError(
                    f"Chunked encoder sample {sample_index} contains no attended tokens"
                )
            start = int(positions[0].item())
            stop = int(positions[-1].item()) + 1
            compact_chunks = []
            for chunk_start in range(start, stop, limit):
                chunk_stop = min(stop, chunk_start + limit)
                chunk_kwargs = dict(kwargs)
                chunk_kwargs.update(
                    attention_mask=attention_mask[
                        sample_index : sample_index + 1, chunk_start:chunk_stop
                    ]
                )
                if input_ids is not None:
                    chunk_kwargs["input_ids"] = input_ids[
                        sample_index : sample_index + 1, chunk_start:chunk_stop
                    ]
                if inputs_embeds is not None:
                    chunk_kwargs["inputs_embeds"] = inputs_embeds[
                        sample_index : sample_index + 1, chunk_start:chunk_stop, :
                    ]
                if token_type_ids is not None:
                    chunk_kwargs["token_type_ids"] = token_type_ids[
                        sample_index : sample_index + 1, chunk_start:chunk_stop
                    ]
                compact_chunks.append(self.forward(**chunk_kwargs).logits)
            compact = torch.cat(compact_chunks, dim=1)
            if int(compact.shape[1]) != stop - start:
                raise RuntimeError(
                    "Chunked encoder did not preserve sequence length: "
                    f"expected={stop - start} actual={compact.shape[1]}"
                )
            padded = compact.new_zeros((1, padded_length, self.hidden_size))
            padded[:, start:stop, :] = compact
            sample_outputs.append(padded)
        return TokenClassifierOutput(loss=None, logits=torch.cat(sample_outputs, dim=0))

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, inputs_embeds=None, output_hidden_states=True, return_dict=True, **kwargs):
        if self.backbone_kind == "gena":
            sequence_length = int(
                input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
            )
            position_limit = self.context_limit()
            if sequence_length > position_limit:
                raise RuntimeError(
                    "Direct/plain GENA does not support outer-input elongation: "
                    f"received {sequence_length} BPE positions, maximum is {position_limit}. "
                    "Set max_bpe_tokens to at most 512 or use RMT/AMT. Independent "
                    "backbone chunking and hidden-state concatenation are intentionally disabled."
                )
        common = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        if self.backbone_kind == "gena":
            common["token_type_ids"] = token_type_ids
        out = self._encoder()(**common)
        hidden, hidden_states = self._extract_hidden(out)
        return TokenClassifierOutput(loss=None, logits=hidden, hidden_states=hidden_states, attentions=getattr(out, "attentions", None))
