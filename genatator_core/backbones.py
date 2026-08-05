from __future__ import annotations

import importlib
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


def _validate_gena_token_classifier_transfer(missing, unexpected) -> None:
    """Accept only the task-head differences expected from an MLM checkpoint."""

    compatibility_suffixes = ("position_ids", "token_type_ids")
    allowed_pretraining_heads = ("cls.", "lm_head.", "predictions.")

    def compatibility_buffer(name: str) -> bool:
        return any(
            name == suffix or name.endswith(f".{suffix}")
            for suffix in compatibility_suffixes
        )

    disallowed_missing = [
        key
        for key in missing
        if not key.startswith("classifier.") and not compatibility_buffer(key)
    ]
    disallowed_unexpected = [
        key
        for key in unexpected
        if not key.startswith(allowed_pretraining_heads)
        and not compatibility_buffer(key)
    ]
    if disallowed_missing or disallowed_unexpected:
        raise RuntimeError(
            "GENA BertForTokenClassification transfer is incomplete; refusing to "
            "continue with randomly initialized encoder parameters. "
            f"missing={disallowed_missing[:20]} (total={len(disallowed_missing)}), "
            f"unexpected={disallowed_unexpected[:20]} "
            f"(total={len(disallowed_unexpected)})."
        )


def _load_gena_token_classifier_owner(
    backbone_path: str,
    *,
    trust_remote_code: bool,
    num_labels: int,
) -> nn.Module:
    """Instantiate the checkpoint's concrete remote BertForTokenClassification."""

    raw = AutoModel.from_pretrained(
        backbone_path,
        trust_remote_code=trust_remote_code,
    )
    config = raw.config
    config.num_labels = int(num_labels)
    module_name = raw.__class__.__module__
    gena_module = importlib.import_module(module_name)
    owner_class = getattr(gena_module, "BertForTokenClassification", None)
    if owner_class is None:
        raise RuntimeError(
            f"GENA remote module {module_name} has no BertForTokenClassification"
        )
    owner = owner_class(config)
    if owner.__class__.__name__ != "BertForTokenClassification":
        raise RuntimeError(
            "GENA transcript classification must use BertForTokenClassification, "
            f"got {owner.__class__.__name__}"
        )
    missing, unexpected = owner.load_state_dict(raw.state_dict(), strict=False)
    _validate_gena_token_classifier_transfer(missing, unexpected)
    logger.info(
        "[transcript.backbone] GENA transfer accepted task-head/buffer differences "
        "missing=%s unexpected=%s",
        missing,
        unexpected,
    )
    del raw
    return owner


class TranscriptTokenClassificationBackbone(nn.Module):
    """Token-classification container used by transcript sequence classifiers.

    The concrete owner is always ``BertForTokenClassification`` for GENA or
    ``ModernBertForTokenClassification`` for ModernGENA.  Its encoder produces
    contextual token states, and its own classification projection is applied
    once to the asserted/pooled CLS state by :meth:`classify`.
    """

    def __init__(
        self,
        backbone_path: str,
        backbone_kind: str,
        *,
        trust_remote_code: bool = True,
        allow_unsafe_torch_load: bool = True,
        num_labels: int = 1,
        attn_implementation: str | None = None,
    ):
        super().__init__()
        if int(num_labels) != 1:
            raise RuntimeError(
                "Transcript sequence classification requires exactly one binary logit"
            )
        if backbone_kind not in {"gena", "moderngena"}:
            raise RuntimeError(
                "TranscriptTokenClassificationBackbone supports only GENA/ModernGENA, "
                f"got {backbone_kind!r}"
            )
        self.backbone_kind = str(backbone_kind)
        self.backbone_path = local_or_remote(backbone_path)
        allow_transformers_torch_load_on_legacy_torch(
            allow_unsafe_torch_load,
            context=(
                "TranscriptTokenClassificationBackbone:"
                f"{self.backbone_kind}:{self.backbone_path}"
            ),
        )

        if self.backbone_kind == "gena":
            owner = _load_gena_token_classifier_owner(
                self.backbone_path,
                trust_remote_code=trust_remote_code,
                num_labels=1,
            )
            expected_class = "BertForTokenClassification"
            encoder_candidates = ("bert",)
        else:
            load_kwargs = {
                "num_labels": 1,
                "trust_remote_code": trust_remote_code,
            }
            if attn_implementation is not None:
                load_kwargs["attn_implementation"] = str(attn_implementation)
            owner = ModernBertForTokenClassification.from_pretrained(
                self.backbone_path,
                **load_kwargs,
            )
            expected_class = "ModernBertForTokenClassification"
            encoder_candidates = ("model", "modernbert", "bert")
        if owner.__class__.__name__ != expected_class:
            raise RuntimeError(
                f"{self.backbone_kind} transcript classification must use "
                f"{expected_class}, got {owner.__class__.__name__}"
            )
        if not hasattr(owner, "classifier"):
            raise RuntimeError(
                f"{expected_class} has no classifier projection for transcript output"
            )

        self.owner = owner
        self.encoder_attr = next(
            (name for name in encoder_candidates if hasattr(owner, name)),
            None,
        )
        if self.encoder_attr is None:
            raise RuntimeError(
                f"{expected_class} has no known encoder attribute; "
                f"children={list(dict(owner.named_children()).keys())}"
            )
        self.config = owner.config
        self.hidden_size = infer_hidden_size(
            self.config,
            context=f"TranscriptTokenClassificationBackbone:{self.backbone_kind}",
        )
        embeddings = get_word_embeddings(
            self.owner,
            context=f"TranscriptTokenClassificationBackbone:{self.backbone_kind}",
        )
        _, embedding_hidden = infer_vocab_size_from_embeddings(
            embeddings,
            context=f"TranscriptTokenClassificationBackbone:{self.backbone_kind}",
        )
        if embedding_hidden != self.hidden_size:
            raise RuntimeError(
                "Transcript token-classification embedding width mismatch: "
                f"embedding={embedding_hidden} hidden_size={self.hidden_size}"
            )
        if self.backbone_kind == "moderngena" and hasattr(self.config, "deterministic_flash_attn"):
            self.config.deterministic_flash_attn = True
        if self.backbone_kind == "moderngena" and hasattr(self.config, "use_sdpa_attn_mask"):
            self.config.use_sdpa_attn_mask = True
        logger.info(
            "[transcript.backbone] kind=%s owner=%s encoder_attr=%s hidden_size=%d",
            self.backbone_kind,
            type(self.owner).__name__,
            self.encoder_attr,
            self.hidden_size,
        )

    @property
    def layers_attr(self) -> str:
        if self.backbone_kind == "gena":
            return f"owner.{self.encoder_attr}.encoder.layer"
        return f"owner.{self.encoder_attr}.layers"

    def _encoder(self) -> nn.Module:
        return getattr(self.owner, self.encoder_attr)

    def get_input_embeddings(self):
        return get_word_embeddings(
            self.owner,
            context="TranscriptTokenClassificationBackbone.get_input_embeddings",
        )

    def resize_token_embeddings(self, new_num_tokens: int):
        if not hasattr(self.owner, "resize_token_embeddings"):
            raise RuntimeError(
                f"{type(self.owner).__name__} cannot resize token embeddings"
            )
        resized = self.owner.resize_token_embeddings(int(new_num_tokens))
        _ = self.get_input_embeddings()
        return resized

    def classify(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        if pooled_hidden.ndim != 2 or pooled_hidden.shape[-1] != self.hidden_size:
            raise RuntimeError(
                "Transcript classifier expects [batch, hidden] pooled states, got "
                f"{tuple(pooled_hidden.shape)}"
            )
        value = pooled_hidden
        # ModernBertForTokenClassification includes a prediction head before its
        # dropout/classifier projection; BertForTokenClassification does not.
        prediction_head = getattr(self.owner, "head", None)
        if isinstance(prediction_head, nn.Module):
            value = prediction_head(value)
        dropout = getattr(self.owner, "drop", None)
        if not isinstance(dropout, nn.Module):
            dropout = getattr(self.owner, "dropout", None)
        if isinstance(dropout, nn.Module):
            value = dropout(value)
        logits = self.owner.classifier(value)
        if logits.ndim != 2 or tuple(logits.shape) != (pooled_hidden.shape[0], 1):
            raise RuntimeError(
                "Transcript token-classification projection must return [batch, 1], "
                f"got {tuple(logits.shape)}"
            )
        return logits

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        output_hidden_states=True,
        return_dict=True,
        **kwargs,
    ):
        if self.backbone_kind == "gena":
            sequence_length = int(
                input_ids.shape[1]
                if input_ids is not None
                else inputs_embeds.shape[1]
            )
            position_limit = int(getattr(self.config, "max_position_embeddings", 512))
            if sequence_length > position_limit:
                raise RuntimeError(
                    "GENA token-classification segment exceeds its position limit: "
                    f"received={sequence_length} maximum={position_limit}"
                )
        common = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }
        if self.backbone_kind == "gena":
            common["token_type_ids"] = token_type_ids
        output = self._encoder()(**common)
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            hidden = output[0]
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise RuntimeError(
                "Transcript token-classification encoder returned an invalid hidden tensor: "
                f"{tuple(hidden.shape)}"
            )
        return TokenClassifierOutput(
            loss=None,
            logits=hidden,
            hidden_states=getattr(output, "hidden_states", None),
            attentions=getattr(output, "attentions", None),
        )


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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, inputs_embeds=None, output_hidden_states=True, return_dict=True, **kwargs):
        if self.backbone_kind == "gena":
            sequence_length = int(
                input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
            )
            position_limit = int(getattr(self.config, "max_position_embeddings", 512))
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
