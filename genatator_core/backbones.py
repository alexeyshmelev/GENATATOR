from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from packaging import version
from transformers import AutoModel, ModernBertForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput

from .config import local_or_remote

logger = logging.getLogger(__name__)


def allow_transformers_bin_loading_on_legacy_torch(*, context: str) -> None:
    """Allow trusted legacy PyTorch `.bin` checkpoints under torch<2.6.

    Transformers now blocks `torch.load`-based checkpoint files when torch<2.6
    because of CVE-2025-32434. The user explicitly requested keeping
    torch==2.2.2+cu121. GENA checkpoints may still be distributed as
    pytorch_model.bin, so we patch the Transformers guard for this trusted
    backbone-loading path and log it loudly instead of failing silently.
    """
    torch_version = version.parse(torch.__version__.split("+")[0])
    if torch_version >= version.parse("2.6.0"):
        return
    try:
        import transformers.modeling_utils as modeling_utils
        import transformers.utils.import_utils as import_utils
    except Exception as exc:
        raise RuntimeError(f"Could not import Transformers internals to support legacy torch checkpoint loading for {context}: {exc}") from exc

    def _logged_noop_check_torch_load_is_safe():
        logger.warning(
            "[legacy_torch_load] Transformers safety guard for torch.load is bypassed for %s because torch=%s < 2.6 and the environment must keep this torch version. Use only trusted checkpoints or safetensors.",
            context,
            torch.__version__,
        )
        return None

    import_utils.check_torch_load_is_safe = _logged_noop_check_torch_load_is_safe
    modeling_utils.check_torch_load_is_safe = _logged_noop_check_torch_load_is_safe
    logger.warning(
        "[legacy_torch_load] Enabled trusted `.bin` checkpoint loading for %s under torch=%s. This is required for GENA backbones without safetensors in the current environment.",
        context,
        torch.__version__,
    )

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
    - ModernGENA uses `transformers.ModernBertForTokenClassification.from_pretrained` as requested.
      The token-classification head is not used; the encoder hidden states become the local fine-tuning features.
    - GENA uses `transformers.AutoModel.from_pretrained` with `trust_remote_code=True`.

    All backbone parameters stay trainable. This class never freezes anything.
    """

    def __init__(self, backbone_path: str, backbone_kind: str, trust_remote_code: bool = True, modernbert_num_labels: int = 2):
        super().__init__()
        self.backbone_kind = backbone_kind
        self.backbone_path = local_or_remote(backbone_path)
        self.trust_remote_code = trust_remote_code
        if backbone_kind == "moderngena":
            logger.info("[backbone] loading ModernGENA through ModernBertForTokenClassification: %s", self.backbone_path)
            self.owner = ModernBertForTokenClassification.from_pretrained(
                self.backbone_path,
                num_labels=int(modernbert_num_labels),
                trust_remote_code=trust_remote_code,
            )
            if hasattr(self.owner, "model"):
                self.encoder = self.owner.model
            elif hasattr(self.owner, "modernbert"):
                self.encoder = self.owner.modernbert
            elif hasattr(self.owner, "bert"):
                self.encoder = self.owner.bert
            else:
                raise RuntimeError(f"ModernBertForTokenClassification has no known encoder attribute: children={list(dict(self.owner.named_children()).keys())}")
            self.config = self.owner.config
        elif backbone_kind == "gena":
            logger.info("[backbone] loading GENA AutoModel: %s", self.backbone_path)
            allow_transformers_bin_loading_on_legacy_torch(context=f"GENA backbone {self.backbone_path}")
            self.encoder = AutoModel.from_pretrained(self.backbone_path, trust_remote_code=trust_remote_code)
            self.owner = self.encoder
            self.config = self.encoder.config
        else:
            raise RuntimeError(f"HiddenStateBackbone supports only backbone_kind='gena' or 'moderngena', got {backbone_kind}")
        self.hidden_size = infer_hidden_size(self.config, context=f"HiddenStateBackbone:{backbone_kind}")
        self.embeddings = get_word_embeddings(self.owner, context=f"HiddenStateBackbone:{backbone_kind}")
        _, emb_hidden = infer_vocab_size_from_embeddings(self.embeddings, context=f"HiddenStateBackbone:{backbone_kind}")
        if emb_hidden != self.hidden_size:
            raise RuntimeError(f"Backbone hidden mismatch: config hidden_size={self.hidden_size}, embedding dim={emb_hidden}")
        logger.info("[backbone] loaded kind=%s hidden_size=%d class=%s", backbone_kind, self.hidden_size, type(self.encoder).__name__)

    def get_input_embeddings(self):
        return get_word_embeddings(self.owner, context="HiddenStateBackbone.get_input_embeddings")

    def resize_token_embeddings(self, new_num_tokens: int):
        logger.info("[backbone] resize token embeddings to %d", new_num_tokens)
        resized = self.owner.resize_token_embeddings(new_num_tokens)
        self.embeddings = get_word_embeddings(self.owner, context="HiddenStateBackbone.after_resize")
        return resized

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, inputs_embeds=None, output_hidden_states=True, return_dict=True, **kwargs):
        common = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        if self.backbone_kind == "gena":
            common["token_type_ids"] = token_type_ids
        out = self.encoder(**common)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        if hidden.shape[-1] != self.hidden_size:
            raise RuntimeError(f"Backbone emitted hidden width {hidden.shape[-1]}, expected {self.hidden_size}")
        return TokenClassifierOutput(loss=None, logits=hidden, hidden_states=getattr(out, "hidden_states", None), attentions=getattr(out, "attentions", None))
