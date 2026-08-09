from __future__ import annotations

import importlib
import logging
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from transformers import AutoModel, AutoModelForCausalLM, ModernBertModel
from transformers.modeling_outputs import TokenClassifierOutput

from .backbones import infer_hidden_size, get_word_embeddings
from .config import local_or_remote
from .unet import DEFAULT_UNET_CHUNK_SIZE, UNET1DSegmentationHead, run_samplewise_chunked_unet
from .torch_compat import allow_transformers_torch_load_on_legacy_torch

logger = logging.getLogger(__name__)


def _fp32_linear(layer: nn.Linear, inputs: torch.Tensor) -> torch.Tensor:
    """Apply an AMT projection in fp32 without changing its trainable weights."""

    bias = None if layer.bias is None else layer.bias.float()
    return F.linear(inputs.float(), layer.weight.float(), bias)


def _stable_associate(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Retrieve associative memory in fp32, then restore the encoder dtype.

    ARMT's DPFP features, recurrent state and normalization denominator are a
    poor fit for autocast: a rare, almost-orthogonal query can make the
    denominator and its gradient especially sensitive to bf16 rounding.  The
    surrounding ModernBERT encoder remains under bf16 autocast; only this small
    linear-memory calculation is promoted to fp32.
    """

    output_dtype = hidden_states.dtype
    device_type = hidden_states.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        queries = self._to_heads(_fp32_linear(self.W_mq, hidden_states))
        queries = F.normalize(self.phi(queries), dim=-1, p=2.0)
        memory = self.W_mem.to(device=hidden_states.device, dtype=torch.float32)
        numerator = torch.einsum("bhsk,bhkd->bhsd", queries, memory)
        if self.use_denom:
            denominator_state = self.z.to(
                device=hidden_states.device,
                dtype=torch.float32,
            )
            # DPFP and z are non-negative by construction.  clamp_min only
            # guards against numerical noise and retains the upstream 1e-5
            # denominator floor/equation.
            denominator = torch.einsum(
                "bhk,bhsk->bhs", denominator_state, queries
            ).clamp_min(0.0)[..., None]
            numerator = numerator / (denominator + 1e-5)
        retrieved = self._from_heads(numerator)
    return retrieved.to(dtype=output_dtype)


def _stable_update_mem(self, mem_tokens: torch.Tensor) -> None:
    """Update differentiable associative state in fp32 under bf16 training."""

    device_type = mem_tokens.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        tokens = mem_tokens.float()
        keys = self._to_heads(_fp32_linear(self.W_mk, tokens))
        keys = F.normalize(self.phi(keys), dim=-1, p=2.0)
        new_values = self._to_heads(_fp32_linear(self.W_mv, tokens))

        memory = self.W_mem.to(device=mem_tokens.device, dtype=torch.float32)
        if not self.first_seg:
            numerator = torch.einsum("bhsk,bhkd->bhsd", keys, memory)
            if self.use_denom:
                denominator_state = self.z.to(
                    device=mem_tokens.device,
                    dtype=torch.float32,
                )
                denominator = torch.einsum(
                    "bhk,bhsk->bhs", denominator_state, keys
                ).clamp_min(0.0)[..., None]
                denominator = denominator + 1e-5
                previous_values = numerator / denominator
                if self.correction:
                    key_norm_sq = torch.linalg.vector_norm(
                        keys, dim=-1
                    ).square()[..., None]
                    new_information = (
                        1.0 - denominator / key_norm_sq.clamp_min(1e-12)
                    ).clamp(0.0, 1.0).detach()
                else:
                    new_information = 1.0
            else:
                previous_values = numerator
                new_information = 1.0
        else:
            previous_values = torch.zeros_like(new_values)
            new_information = 1.0

        value_delta = new_values - previous_values
        gates = self._to_heads(torch.sigmoid(_fp32_linear(self.W_mb, tokens)))
        if self.gating:
            associations = torch.einsum(
                "bhsk,bhsd,bhsd->bhkd", keys, value_delta, gates
            )
        else:
            associations = torch.einsum(
                "bhsk,bhsd,bhsx->bhkd", keys, value_delta, gates
            )
        self.W_mem = memory + associations

        if self.use_denom:
            denominator_update = (new_information * keys).sum(dim=-2)
            denominator_state = self.z.to(
                device=mem_tokens.device,
                dtype=torch.float32,
            )
            self.z = denominator_state + denominator_update
        self.seg_num += 1
        self.first_seg = False


def _stable_zero_mem(self) -> None:
    """Reset non-persistent ARMT state in fp32 on the projection device."""

    device = self.W_mq.weight.device
    self.first_seg = True
    self.W_mem = torch.zeros(
        1,
        self.n_heads,
        self.d_key // self.n_heads,
        self.d_model // self.n_heads,
        dtype=torch.float32,
        device=device,
    )
    self.W_mem.requires_grad_(False)
    if self.use_denom:
        self.z = torch.zeros(
            1,
            self.n_heads,
            self.d_key // self.n_heads,
            dtype=torch.float32,
            device=device,
        )
        self.z.requires_grad_(False)
    self.seg_num = 0


def _install_stable_associative_memory(memory_cell: nn.Module) -> None:
    """Patch the loaded ARMT layer primitive with equivalent fp32 math.

    The Hugging Face repository supplies model wrapping/segmentation while the
    numerical primitive is kept local and version-checked here.  Patching the
    primitive's defining class (rather than individual adaptive subclasses)
    preserves optional ACT dispatch through ``super().associate``.
    """

    layers = list(memory_cell.get_layers())
    if not layers:
        raise RuntimeError("AMT memory cell did not expose any associative layers")
    patched_classes = set()
    required = {
        "W_mq",
        "W_mk",
        "W_mv",
        "W_mb",
        "phi",
        "n_heads",
        "d_key",
        "d_model",
        "use_denom",
        "gating",
        "correction",
    }
    for layer in layers:
        missing = sorted(name for name in required if not hasattr(layer, name))
        if missing:
            raise RuntimeError(
                "The loaded AMT implementation is incompatible with GENATATOR's "
                f"stable-memory patch; layer={type(layer).__name__} missing={missing}"
            )
        primitive_class = next(
            (
                cls
                for cls in type(layer).__mro__
                if all(
                    name in cls.__dict__
                    for name in ("associate", "update_mem", "zero_mem")
                )
            ),
            None,
        )
        if primitive_class is None:
            raise RuntimeError(
                "The loaded AMT implementation has no patchable associative "
                f"primitive for layer={type(layer).__name__}"
            )
        if primitive_class in patched_classes:
            continue
        primitive_class.associate = _stable_associate
        primitive_class.update_mem = _stable_update_mem
        primitive_class.zero_mem = _stable_zero_mem
        primitive_class._genatator_fp32_associative_memory = True
        patched_classes.add(primitive_class)
    memory_cell.zero_mem()


def _run_amt_without_padding(
    amt: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    hidden_size: int,
) -> torch.Tensor:
    """Run AMT per sample on attended tokens and scatter back to batch shape.

    Segmentation batches are padded to the configured maximum BPE length.  If
    those tails are sent to recurrent AMT, every short transcript creates many
    all-PAD segments which still append learned memory tokens and update every
    layer's associative state.  Compacting first matches the unpadded ARMT
    execution used by gencoder and makes recurrence depend on biological
    content length rather than configured padding length.
    """

    if input_ids is None or input_ids.ndim != 2:
        raise RuntimeError("AMT requires rank-2 input_ids")
    if attention_mask is None:
        out = amt(input_ids=input_ids, attention_mask=None)
        hidden = out.logits
        if hidden.shape != (*input_ids.shape, int(hidden_size)):
            raise RuntimeError(
                "AMT hidden shape mismatch without an attention mask: "
                f"expected={(*input_ids.shape, int(hidden_size))} got={tuple(hidden.shape)}"
            )
        return hidden
    if attention_mask.shape != input_ids.shape:
        raise RuntimeError(
            "AMT input_ids/attention_mask shape mismatch: "
            f"input_ids={tuple(input_ids.shape)} attention_mask={tuple(attention_mask.shape)}"
        )

    batch_size, padded_length = input_ids.shape
    sample_outputs = []
    for sample_index in range(batch_size):
        attended_positions = attention_mask[sample_index].bool().nonzero(
            as_tuple=False
        ).flatten()
        if attended_positions.numel() == 0:
            raise RuntimeError(
                f"AMT sample {sample_index} has no attended tokens; refusing to run PAD-only recurrence"
            )
        compact_ids = input_ids[sample_index : sample_index + 1].index_select(
            1, attended_positions
        )
        compact_mask = attention_mask[
            sample_index : sample_index + 1
        ].new_ones((1, attended_positions.numel()))
        compact_hidden = amt(
            input_ids=compact_ids,
            attention_mask=compact_mask,
        ).logits
        expected_shape = (1, attended_positions.numel(), int(hidden_size))
        if compact_hidden.shape != expected_shape:
            raise RuntimeError(
                "AMT compact hidden shape mismatch: "
                f"expected={expected_shape} got={tuple(compact_hidden.shape)}"
            )
        sample_hidden = compact_hidden.new_zeros((padded_length, int(hidden_size)))
        sample_hidden = sample_hidden.index_copy(
            0, attended_positions, compact_hidden.squeeze(0)
        )
        sample_outputs.append(sample_hidden)
    return torch.stack(sample_outputs, dim=0)


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

    def __init__(self, backbone_path: str, backbone_kind: str, num_labels: int, trust_remote_code: bool = True, amt_repo_id: str = "irodkin/armt-neox-tiny", use_unet: bool = False, nucleotide_vocab_size: int = 1000, unet_cycles: int = 1, unet_channels=None, unet_chunk_size: int = DEFAULT_UNET_CHUNK_SIZE, allow_unsafe_torch_load: bool = True, encoder_only: bool = False, **amt_kwargs):
        super().__init__()
        if backbone_kind not in {"gena", "moderngena"}:
            raise RuntimeError(f"AMT is allowed only for GENA/ModernGENA, got backbone_kind={backbone_kind}")
        self.num_labels = int(num_labels)
        self.use_unet = bool(use_unet)
        self.encoder_only = bool(encoder_only)
        if self.encoder_only and self.use_unet:
            raise RuntimeError("AMT encoder_only and use_unet cannot both be true")

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
        default_segment_size = 512 if backbone_kind == "gena" else 1024
        segment_size_value = int(amt_kwargs.pop("segment_size", default_segment_size))
        if segment_size_value <= 0:
            raise RuntimeError(f"AMT segment_size must be positive, got {segment_size_value}")
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
        _install_stable_associative_memory(memory_cell)
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

        if self.encoder_only:
            logger.info(
                "[AMTTokenClassifier] encoder_only=true hidden=%d",
                self.hidden_size,
            )
        elif self.use_unet:
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

    def encode_hidden(self, input_ids=None, attention_mask=None) -> torch.Tensor:
        hidden = _run_amt_without_padding(
            self.amt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            hidden_size=self.hidden_size,
        )
        if hidden.shape[-1] != self.hidden_size:
            raise RuntimeError(f"AMT hidden size mismatch: expected {self.hidden_size}, got {hidden.shape[-1]}")
        return hidden

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, labels_mask=None, pos_weight=None, embedding_repeater=None, letter_level_tokens=None, letter_level_labels=None, letter_level_labels_mask=None, letter_level_attention_mask=None, **kwargs):
        if self.encoder_only:
            raise RuntimeError(
                "This AMT instance is encoder-only; call encode_hidden from its enclosing head"
            )
        hidden = self.encode_hidden(input_ids=input_ids, attention_mask=attention_mask)
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
