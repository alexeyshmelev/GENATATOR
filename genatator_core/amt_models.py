from __future__ import annotations

import importlib
import logging
import types

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers import AutoModel, AutoModelForCausalLM, ModernBertModel
from transformers.modeling_outputs import TokenClassifierOutput

from .backbones import infer_hidden_size, get_word_embeddings
from .config import local_or_remote
from .unet import DEFAULT_UNET_CHUNK_SIZE, UNET1DSegmentationHead, run_samplewise_chunked_unet
from .torch_compat import allow_transformers_torch_load_on_legacy_torch

logger = logging.getLogger(__name__)


def validate_gena_amt_transfer(missing, unexpected) -> None:
    """Fail closed unless GENA transfer differences are known non-encoder keys."""

    compatibility_suffixes = ("position_ids", "token_type_ids")
    allowed_pretraining_heads = ("cls.", "lm_head.", "predictions.")

    def compatibility_buffer(name: str) -> bool:
        return any(name == suffix or name.endswith(f".{suffix}") for suffix in compatibility_suffixes)

    disallowed_missing = [
        key for key in missing
        if not key.startswith("classifier.") and not compatibility_buffer(key)
    ]
    disallowed_unexpected = [
        key for key in unexpected
        if not key.startswith(allowed_pretraining_heads) and not compatibility_buffer(key)
    ]
    if disallowed_missing or disallowed_unexpected:
        raise RuntimeError(
            "GENA AMT backbone transfer is incomplete; refusing to continue with randomly "
            "initialized encoder parameters. "
            f"missing={disallowed_missing[:20]} (total={len(disallowed_missing)}), "
            f"unexpected={disallowed_unexpected[:20]} (total={len(disallowed_unexpected)})."
        )


def _patch_forward_ignore_cache(model: nn.Module) -> None:
    """AMT may pass cache kwargs. BERT/ModernBERT encoders do not need them."""
    orig_forward = model.forward

    def _forward(self_, *args, **kwargs):
        kwargs.pop("use_cache", None)
        kwargs.pop("past_key_values", None)
        return orig_forward(*args, **kwargs)

    model.forward = types.MethodType(_forward, model)


def _patch_forward_return_hidden_as_logits(model: nn.Module) -> None:
    """Make a bare encoder look like a token-classification model for AMT.

    The remote AMT wrapper expects base_model(...).logits. For ModernBERT we keep
    the exact user-provided AMT logic: load ModernBertModel, then return
    last_hidden_state in a `.logits` field.
    """
    orig_forward = model.forward

    def _forward(self_, *args, **kwargs):
        kwargs.pop("use_cache", None)
        kwargs.pop("past_key_values", None)
        out = orig_forward(*args, **kwargs)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None and isinstance(out, (tuple, list)) and len(out) > 0:
            hidden = out[0]
        if hidden is None:
            raise RuntimeError(f"AMT base encoder {type(self_).__name__} did not return last_hidden_state")
        class _Out:
            pass
        wrapped = _Out()
        wrapped.logits = hidden
        wrapped.hidden_states = getattr(out, "hidden_states", None)
        wrapped.attentions = getattr(out, "attentions", None)
        return wrapped

    model.forward = types.MethodType(_forward, model)


def _load_amt_base_model(backbone_path: str, backbone_kind: str, trust_remote_code: bool, allow_unsafe_torch_load: bool) -> tuple[nn.Module, object, int, str]:
    """Load the base model in the same style as the provided AMT code.

    ModernGENA: ModernBertModel -> forward patched to expose hidden states as logits.
    GENA: AutoModel checkpoint -> remote BertForTokenClassification with Identity
    classifier, so AMT receives a model with get_input_embeddings() and logits equal
    to hidden states. This mirrors the supplied associative-memory fine-tuning logic.
    """
    path = local_or_remote(backbone_path)
    allow_transformers_torch_load_on_legacy_torch(allow_unsafe_torch_load, context=f"AMT.base:{backbone_kind}:{path}")

    if backbone_kind == "moderngena":
        logger.info("[AMT.base] loading ModernBertModel path=%s attn_implementation=sdpa", path)
        base_model = ModernBertModel.from_pretrained(
            path,
            trust_remote_code=trust_remote_code,
            attn_implementation="sdpa",
        )
        if hasattr(base_model, "config"):
            if hasattr(base_model.config, "deterministic_flash_attn"):
                base_model.config.deterministic_flash_attn = True
            if hasattr(base_model.config, "use_sdpa_attn_mask"):
                base_model.config.use_sdpa_attn_mask = True
        _patch_forward_return_hidden_as_logits(base_model)
        config = base_model.config
        hidden_size = infer_hidden_size(config, context="AMT.ModernGENA")
        layers_attr = "layers"
        logger.info("[AMT.base] ModernGENA base_class=%s layers_attr=%s hidden_size=%d", type(base_model).__name__, layers_attr, hidden_size)
        return base_model, config, hidden_size, layers_attr

    if backbone_kind == "gena":
        logger.info("[AMT.base] loading GENA AutoModel for state transfer path=%s", path)
        auto_backbone = AutoModel.from_pretrained(path, trust_remote_code=trust_remote_code)
        config = auto_backbone.config
        module_name = auto_backbone.__class__.__module__
        gena_mod = importlib.import_module(module_name)
        if not hasattr(gena_mod, "BertForTokenClassification"):
            raise RuntimeError(f"GENA remote module {module_name} has no BertForTokenClassification required by AMT")
        BertForTokenClassification = getattr(gena_mod, "BertForTokenClassification")
        base_model = BertForTokenClassification(config)
        missing, unexpected = base_model.load_state_dict(auto_backbone.state_dict(), strict=False)
        validate_gena_amt_transfer(missing, unexpected)
        logger.info(
            "[AMT.base] GENA transfer accepted only task-head/buffer differences missing=%s unexpected=%s",
            missing,
            unexpected,
        )
        if not hasattr(base_model, "classifier"):
            raise RuntimeError("GENA BertForTokenClassification has no classifier attribute")
        base_model.classifier = nn.Identity()
        _patch_forward_ignore_cache(base_model)
        hidden_size = infer_hidden_size(config, context="AMT.GENA")
        layers_attr = "bert.encoder.layer"
        del auto_backbone
        logger.info("[AMT.base] GENA base_class=%s classifier=Identity layers_attr=%s hidden_size=%d", type(base_model).__name__, layers_attr, hidden_size)
        return base_model, config, hidden_size, layers_attr

    raise RuntimeError(f"AMT supports only backbone_kind='gena' or 'moderngena', got {backbone_kind}")


class AMTTokenClassifier(nn.Module):
    """AMT memory wrapper for GENA/ModernGENA only.

    Active class choice follows the provided AMT code: the remote implementation
    must expose `AssociativeMemoryCell` and `AssociativeRecurrentWrapper`.
    No parameters are frozen.
    """

    def __init__(self, backbone_path: str, backbone_kind: str, num_labels: int, trust_remote_code: bool = True, amt_repo_id: str = "irodkin/armt-neox-tiny", use_unet: bool = False, nucleotide_vocab_size: int = 1000, unet_cycles: int = 1, unet_channels=None, unet_chunk_size: int = DEFAULT_UNET_CHUNK_SIZE, allow_unsafe_torch_load: bool = True, **amt_kwargs):
        super().__init__()
        if backbone_kind not in {"gena", "moderngena"}:
            raise RuntimeError(f"AMT is allowed only for GENA/ModernGENA, got backbone_kind={backbone_kind}")
        self.num_labels = int(num_labels)
        self.use_unet = bool(use_unet)

        base_model, encoder_cfg, hidden_size, default_layers_attr = _load_amt_base_model(backbone_path, backbone_kind, trust_remote_code, allow_unsafe_torch_load)
        self.hidden_size = int(hidden_size)
        self.encoder_config = encoder_cfg

        # Verify embedding interface before constructing AMT. This is the exact
        # method that the remote AssociativeMemoryCell calls internally.
        emb = base_model.get_input_embeddings()
        if emb is None:
            raise RuntimeError(f"AMT base model {type(base_model).__name__} returned None from get_input_embeddings()")
        _ = get_word_embeddings(base_model, context=f"AMT.{backbone_kind}.base_embeddings")

        allow_transformers_torch_load_on_legacy_torch(allow_unsafe_torch_load, context=f"AMT:{amt_repo_id}")
        loaded = AutoModelForCausalLM.from_pretrained(local_or_remote(amt_repo_id), trust_remote_code=True)
        amt_mod = importlib.import_module(loaded.__class__.__module__)
        AssociativeMemoryCell = getattr(amt_mod, "AssociativeMemoryCell")
        AssociativeRecurrentWrapper = getattr(amt_mod, "AssociativeRecurrentWrapper")
        del loaded

        layers_attr = amt_kwargs.pop("layers_attr", default_layers_attr)
        act_on_value = bool(amt_kwargs.pop("act_on", False))
        attend_prev_value = bool(amt_kwargs.pop("attend_to_previous_input", False))
        segment_size_value = int(amt_kwargs.pop("segment_size", 128))
        segment_alignment_value = amt_kwargs.pop("segment_alignment", "left")
        sliding_window_value = bool(amt_kwargs.pop("sliding_window", False))
        time_penalty_value = float(amt_kwargs.pop("time_penalty", 0.0))
        logger.info(
            "[AMT] repo=%s base_class=%s layers_attr=%s hidden=%d use_unet=%s num_mem=%s d_mem=%s segment_size=%d",
            amt_repo_id, type(base_model).__name__, layers_attr, self.hidden_size, self.use_unet,
            amt_kwargs.get("num_mem_tokens", 16), amt_kwargs.get("d_mem", 32), segment_size_value,
        )
        memory_cell = AssociativeMemoryCell(
            base_model=base_model,
            num_mem_tokens=int(amt_kwargs.pop("num_mem_tokens", 16)),
            d_mem=int(amt_kwargs.pop("d_mem", 32)),
            layers_attr=layers_attr,
            wrap_pos=bool(amt_kwargs.pop("wrap_pos", False)),
            correction=bool(amt_kwargs.pop("correction", True)),
            n_heads=int(amt_kwargs.pop("n_heads", 1)),
            use_denom=bool(amt_kwargs.pop("use_denom", True)),
            gating=bool(amt_kwargs.pop("gating", False)),
            freeze_mem=False,
            act_on=act_on_value,
            max_hop=int(amt_kwargs.pop("max_hop", 4)),
            act_type=amt_kwargs.pop("act_type", "associative"),
            constant_depth=bool(amt_kwargs.pop("constant_depth", False)),
            act_format=amt_kwargs.pop("act_format", "linear"),
            noisy_halting=bool(amt_kwargs.pop("noisy_halting", False)),
            attend_to_previous_input=attend_prev_value,
            use_sink=bool(amt_kwargs.pop("use_sink", False)),
        )
        self.amt = AssociativeRecurrentWrapper(
            memory_cell,
            segment_size=segment_size_value,
            segment_alignment=segment_alignment_value,
            sliding_window=sliding_window_value,
            attend_to_previous_input=attend_prev_value,
            act_on=act_on_value,
            time_penalty=time_penalty_value,
        )
        if amt_kwargs:
            raise RuntimeError(f"Unused AMT parameters: {sorted(amt_kwargs.keys())}")

        if self.use_unet:
            self.unet_cycles = int(unet_cycles)
            if self.unet_cycles < 1:
                raise RuntimeError("unet_cycles must be >= 1")
            self.unet_chunk_size = int(unet_chunk_size)
            if self.unet_chunk_size <= 0:
                raise RuntimeError("unet_chunk_size must be positive")
            self.nucleotide_embedding = nn.Embedding(int(nucleotide_vocab_size), self.hidden_size)
            self.unet_input_dim = self.hidden_size * 2
            self.unet = UNET1DSegmentationHead(self.unet_input_dim, self.unet_input_dim, output_channels_list=unet_channels)
            self.activation_fn = nn.SiLU()
            self.fc = nn.Linear(self.unet_input_dim, self.num_labels)
            logger.info("[AMTTokenClassifier] UNET hidden=%d input=%d labels=%d cycles=%d chunk=%d", self.hidden_size, self.unet_input_dim, self.num_labels, self.unet_cycles, self.unet_chunk_size)
        else:
            self.classifier = nn.Linear(self.hidden_size, self.num_labels)
            logger.info("[AMTTokenClassifier] plain hidden=%d labels=%d", self.hidden_size, self.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, labels_mask=None, pos_weight=None, embedding_repeater=None, letter_level_tokens=None, letter_level_labels=None, letter_level_labels_mask=None, letter_level_attention_mask=None, **kwargs):
        out = self.amt(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.logits
        if hidden.shape[-1] != self.hidden_size:
            raise RuntimeError(f"AMT hidden size mismatch: expected {self.hidden_size}, got {hidden.shape[-1]}")
        if not self.use_unet:
            logits = self.classifier(hidden)
            mask = labels_mask if labels_mask is not None else attention_mask.bool()
            loss = BCEWithLogitsLoss()(logits[mask.bool()].float(), labels[mask.bool()].float()) if labels is not None else None
            return TokenClassifierOutput(loss=loss, logits=logits)
        if labels_mask is None:
            raise RuntimeError("AMT+UNET requires labels_mask to identify retained BPE content tokens")
        loss, logits = run_samplewise_chunked_unet(
            token_hidden=hidden,
            token_content_mask=labels_mask,
            embedding_repeater=embedding_repeater,
            letter_level_tokens=letter_level_tokens,
            letter_level_attention_mask=letter_level_attention_mask,
            letter_level_labels=letter_level_labels,
            letter_level_labels_mask=letter_level_labels_mask,
            pos_weight=pos_weight,
            nucleotide_embedding=self.nucleotide_embedding,
            unet=self.unet,
            activation_fn=self.activation_fn,
            classifier=self.fc,
            cycles=self.unet_cycles,
            chunk_size=self.unet_chunk_size,
            context="AMTTokenClassifier",
        )
        return TokenClassifierOutput(loss=loss, logits=logits)
