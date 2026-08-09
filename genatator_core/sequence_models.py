from __future__ import annotations

import importlib
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import SequenceClassifierOutput

from .amt_models import _load_amt_base_model
from .backbones import (
    HiddenStateBackbone,
    get_word_embeddings,
    infer_hidden_size,
    infer_vocab_size_from_embeddings,
)
from .config import local_or_remote
from .torch_compat import allow_transformers_torch_load_on_legacy_torch


def _required_token_id(tokenizer: Any, attribute: str) -> int:
    value = getattr(tokenizer, attribute, None)
    if value is None:
        raise RuntimeError(
            f"Transcript-type classification requires tokenizer.{attribute} to be defined"
        )
    return int(value)


def _special_token_positions(
    input_ids: torch.Tensor,
    token_id: int,
    *,
    attention_mask: torch.Tensor | None,
    token_name: str,
) -> torch.Tensor:
    """Return one attended special-token position per sample.

    Assertions are intentional here.  The transcript classifier's contract is to
    classify one particular special token, so silently choosing a different token
    would train a different model from the one described by the configuration.
    """

    if input_ids.ndim != 2:
        raise RuntimeError(
            f"input_ids must have shape [batch, sequence], got {tuple(input_ids.shape)}"
        )
    matches = input_ids.eq(int(token_id))
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise RuntimeError(
                "attention_mask must have the same shape as input_ids: "
                f"mask={tuple(attention_mask.shape)} ids={tuple(input_ids.shape)}"
            )
        matches = matches & attention_mask.bool()
    counts = matches.sum(dim=1)
    assert bool(torch.all(counts == 1).item()), (
        f"Every sample must contain exactly one attended {token_name} token "
        f"(id={int(token_id)}); counts={counts.detach().cpu().tolist()}"
    )
    positions = matches.long().argmax(dim=1)
    selected_ids = input_ids.gather(1, positions[:, None]).squeeze(1)
    assert bool(torch.all(selected_ids == int(token_id)).item()), (
        f"The pooled token must be {token_name} (id={int(token_id)})"
    )
    return positions


def pool_special_token(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    token_id: int,
    *,
    attention_mask: torch.Tensor | None = None,
    token_name: str = "special",
) -> torch.Tensor:
    """Select the requested special-token hidden state from every batch row."""

    if hidden_states.ndim != 3:
        raise RuntimeError(
            "hidden_states must have shape [batch, sequence, hidden], got "
            f"{tuple(hidden_states.shape)}"
        )
    if hidden_states.shape[:2] != input_ids.shape:
        raise RuntimeError(
            "Hidden states and input ids are not positionally aligned: "
            f"hidden={tuple(hidden_states.shape)} ids={tuple(input_ids.shape)}"
        )
    positions = _special_token_positions(
        input_ids,
        token_id,
        attention_mask=attention_mask,
        token_name=token_name,
    )
    batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_indices, positions.to(hidden_states.device)]


def _sequence_labels(
    transcript_type: torch.Tensor | None,
    labels: torch.Tensor | None,
) -> torch.Tensor | None:
    if transcript_type is not None and labels is not None:
        raise RuntimeError("Pass transcript_type or labels, not both")
    return transcript_type if transcript_type is not None else labels


def _binary_sequence_loss(
    logits: torch.Tensor,
    labels: torch.Tensor | None,
) -> torch.Tensor | None:
    if labels is None:
        return None
    if labels.numel() != logits.shape[0]:
        raise RuntimeError(
            "Transcript-type labels must contain one value per sample: "
            f"labels={tuple(labels.shape)} logits={tuple(logits.shape)}"
        )
    targets = labels.to(device=logits.device, dtype=torch.float32).reshape_as(logits)
    return F.binary_cross_entropy_with_logits(logits.float(), targets)


def _hidden_from_output(output: Any, *, context: str) -> torch.Tensor:
    hidden = getattr(output, "logits", None)
    if hidden is None and isinstance(output, dict):
        hidden = output.get("logits")
    if hidden is None:
        raise RuntimeError(f"{context} did not return hidden states in .logits")
    if hidden.ndim != 3:
        raise RuntimeError(
            f"{context} must return [batch, sequence, hidden], got {tuple(hidden.shape)}"
        )
    return hidden


class GenaModernTranscriptTypeClassifier(nn.Module):
    """Binary transcript classifier for GENA and ModernGENA.

    ``BertForTokenClassification`` and ``ModernBertForTokenClassification`` are
    useful checkpoint-compatible containers, but their task heads are token-level
    and therefore are not the transcript classifier.  The established hidden-state
    adapter discards/bypasses that unused head.  This sequence-level head consumes
    exactly the tokenizer's CLS state and emits one logit per transcript.
    """

    def __init__(
        self,
        backbone_path: str,
        backbone_kind: str,
        tokenizer: Any,
        *,
        trust_remote_code: bool = True,
        allow_unsafe_torch_load: bool = True,
    ):
        super().__init__()
        if backbone_kind not in {"gena", "moderngena"}:
            raise RuntimeError(
                "GenaModernTranscriptTypeClassifier supports only GENA/ModernGENA, "
                f"got backbone_kind={backbone_kind!r}"
            )
        self.hidden_backbone = HiddenStateBackbone(
            backbone_path,
            backbone_kind,
            trust_remote_code=trust_remote_code,
            modernbert_num_labels=1,
            allow_unsafe_torch_load=allow_unsafe_torch_load,
        )
        self.hidden_size = int(self.hidden_backbone.hidden_size)
        self.register_buffer(
            "cls_token_id",
            torch.tensor(_required_token_id(tokenizer, "cls_token_id"), dtype=torch.long),
            persistent=False,
        )
        self.classifier = nn.Linear(self.hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        transcript_type: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        if input_ids is None:
            raise RuntimeError("Transcript-type classification requires input_ids")
        output = self.hidden_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = _hidden_from_output(output, context="GENA/ModernGENA backbone")
        pooled = pool_special_token(
            hidden,
            input_ids,
            int(self.cls_token_id.item()),
            attention_mask=attention_mask,
            token_name="CLS",
        )
        logits = self.classifier(pooled)
        target = _sequence_labels(transcript_type, labels)
        return SequenceClassifierOutput(
            loss=_binary_sequence_loss(logits, target),
            logits=logits,
        )


class CaduceusTranscriptTypeMiddleLossSequenceClassifier(nn.Module):
    """Caduceus sequence classifier pooled at SEP with final and middle loss."""

    def __init__(self, caduceus_model: nn.Module, hidden_size: int, tokenizer: Any):
        super().__init__()
        self.caduceus_model = caduceus_model
        self.hidden_size = int(hidden_size)
        self.register_buffer(
            "sep_token_id",
            torch.tensor(_required_token_id(tokenizer, "sep_token_id"), dtype=torch.long),
            persistent=False,
        )
        self.classifier = nn.Linear(self.hidden_size, 1)
        self.middle_classifier = nn.Linear(self.hidden_size, 1)

    @staticmethod
    def _output_value(output: Any, name: str):
        if isinstance(output, dict):
            return output.get(name)
        return getattr(output, name, None)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        transcript_type: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        if input_ids is None:
            raise RuntimeError("Transcript-type classification requires input_ids")
        # Caduceus' remote implementation does not consume an attention mask.  The
        # project left-pads nucleotide batches and pools the terminal SEP state.
        output = self.caduceus_model(input_ids=input_ids, output_hidden_states=True)
        hidden = self._output_value(output, "last_hidden_state")
        if hidden is None:
            try:
                hidden = output[0]
            except (KeyError, IndexError, TypeError):
                hidden = None
        hidden_states = self._output_value(output, "hidden_states")
        if hidden is None or hidden_states is None or len(hidden_states) == 0:
            raise RuntimeError(
                "Caduceus must return last_hidden_state and hidden_states when "
                "output_hidden_states=True"
            )
        middle = hidden_states[len(hidden_states) // 2]
        if hidden.shape[-1] != self.hidden_size or middle.shape[-1] != self.hidden_size:
            raise RuntimeError(
                "Caduceus hidden-size mismatch: "
                f"expected={self.hidden_size} final={tuple(hidden.shape)} "
                f"middle={tuple(middle.shape)}"
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        sep_token_id = int(self.sep_token_id.item())
        final_pooled = pool_special_token(
            hidden,
            input_ids,
            sep_token_id,
            attention_mask=attention_mask,
            token_name="SEP",
        )
        middle_pooled = pool_special_token(
            middle,
            input_ids,
            sep_token_id,
            attention_mask=attention_mask,
            token_name="SEP",
        )
        logits = self.classifier(final_pooled)
        middle_logits = self.middle_classifier(middle_pooled)
        target = _sequence_labels(transcript_type, labels)
        loss = None
        if target is not None:
            loss = 0.5 * (
                _binary_sequence_loss(logits, target)
                + _binary_sequence_loss(middle_logits, target)
            )
        return SequenceClassifierOutput(loss=loss, logits=logits)


class RMTTranscriptTypeClassifier(nn.Module):
    """Recurrent-memory sequence classifier pooled from the last segment's CLS.

    Every segment is ``CLS, memory, SEP, content, SEP``.  Segments are aligned to
    the right across a batch, so every sample participates in the last recurrent
    step.  That final CLS state has access to the recurrent memory and the final
    content segment and is the only state passed to the binary classifier.
    """

    def __init__(
        self,
        base_model: nn.Module,
        tokenizer: Any,
        *,
        num_mem_tokens: int,
        segment_size: int | None = None,
        input_size: int | None = None,
        max_n_segments: int = 10000,
        bptt_depth: int = -1,
    ):
        super().__init__()
        if segment_size is None:
            segment_size = input_size
        elif input_size is not None and int(segment_size) != int(input_size):
            raise RuntimeError(
                f"Conflicting RMT segment_size={segment_size} and input_size={input_size}"
            )
        if segment_size is None:
            raise RuntimeError("RMT sequence classification requires segment_size")
        self.model = base_model
        self.hidden_size = infer_hidden_size(self.model.config, context="RMT.sequence")
        self.segment_size = int(segment_size)
        self.num_mem_tokens = int(num_mem_tokens)
        self.max_n_segments = int(max_n_segments)
        self.bptt_depth = int(bptt_depth)
        if self.num_mem_tokens <= 0:
            raise RuntimeError("num_mem_tokens must be positive")
        if self.max_n_segments <= 0:
            raise RuntimeError("max_n_segments must be positive")
        self.content_segment_size = self.segment_size - self.num_mem_tokens - 3
        if self.content_segment_size <= 0:
            raise RuntimeError(
                "RMT segment_size must exceed num_mem_tokens + 3: "
                f"segment_size={self.segment_size} num_mem_tokens={self.num_mem_tokens}"
            )
        position_limit = getattr(self.model.config, "max_position_embeddings", None)
        if position_limit is not None and self.segment_size > int(position_limit):
            raise RuntimeError(
                "RMT segment exceeds the backbone position limit: "
                f"segment_size={self.segment_size} limit={int(position_limit)}"
            )

        self.pad_token_id = _required_token_id(tokenizer, "pad_token_id")
        self.register_buffer(
            "cls_token",
            torch.tensor([_required_token_id(tokenizer, "cls_token_id")], dtype=torch.long),
        )
        self.register_buffer(
            "sep_token",
            torch.tensor([_required_token_id(tokenizer, "sep_token_id")], dtype=torch.long),
        )
        embeddings = get_word_embeddings(self.model, context="RMT.sequence.before_resize")
        vocab_size, embedding_hidden = infer_vocab_size_from_embeddings(
            embeddings, context="RMT.sequence.before_resize"
        )
        if embedding_hidden != self.hidden_size:
            raise RuntimeError(
                f"RMT embedding width={embedding_hidden}, expected hidden_size={self.hidden_size}"
            )
        self.register_buffer(
            "mem_token_ids",
            torch.arange(vocab_size, vocab_size + self.num_mem_tokens, dtype=torch.long),
        )
        if not hasattr(self.model, "resize_token_embeddings"):
            raise RuntimeError(
                f"RMT base model {type(self.model).__name__} cannot resize token embeddings"
            )
        self.model.resize_token_embeddings(vocab_size + self.num_mem_tokens)
        resized = get_word_embeddings(self.model, context="RMT.sequence.after_resize")
        resized_vocab, resized_hidden = infer_vocab_size_from_embeddings(
            resized, context="RMT.sequence.after_resize"
        )
        if resized_vocab != vocab_size + self.num_mem_tokens or resized_hidden != self.hidden_size:
            raise RuntimeError(
                "RMT resized embedding table has the wrong shape: "
                f"got=({resized_vocab}, {resized_hidden}) "
                f"expected=({vocab_size + self.num_mem_tokens}, {self.hidden_size})"
            )
        self.classifier = nn.Linear(self.hidden_size, 1)

    def _content_sequences(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id).long()
        _special_token_positions(
            input_ids,
            int(self.cls_token.item()),
            attention_mask=attention_mask,
            token_name="CLS",
        )
        _special_token_positions(
            input_ids,
            int(self.sep_token.item()),
            attention_mask=attention_mask,
            token_name="SEP",
        )
        active = attention_mask.bool()
        content_mask = (
            active
            & input_ids.ne(self.pad_token_id)
            & input_ids.ne(int(self.cls_token.item()))
            & input_ids.ne(int(self.sep_token.item()))
        )
        limit = self.content_segment_size * self.max_n_segments
        rows = [row[mask][:limit] for row, mask in zip(input_ids, content_mask)]
        if any(row.numel() == 0 for row in rows):
            raise RuntimeError("RMT received an empty transcript after removing special tokens")
        return rows

    def _make_segment(self, content: torch.Tensor) -> torch.Tensor:
        segment = torch.cat(
            [self.cls_token, self.mem_token_ids, self.sep_token, content, self.sep_token]
        )
        if segment.numel() > self.segment_size:
            raise RuntimeError(
                f"Internal RMT segment overflow: {segment.numel()} > {self.segment_size}"
            )
        if segment.numel() < self.segment_size:
            segment = F.pad(
                segment,
                (0, self.segment_size - segment.numel()),
                value=self.pad_token_id,
            )
        return segment

    def _segments(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> list[list[torch.Tensor | None]]:
        content_rows = self._content_sequences(input_ids, attention_mask)
        rows = [
            [self._make_segment(chunk) for chunk in torch.split(row, self.content_segment_size)]
            for row in content_rows
        ]
        n_segments = max(len(row) for row in rows)
        # Right alignment guarantees that every sample is active in the final step.
        return [
            [
                row[step - (n_segments - len(row))]
                if step >= n_segments - len(row)
                else None
                for row in rows
            ]
            for step in range(n_segments)
        ]

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        transcript_type: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        if input_ids is None:
            raise RuntimeError("RMT transcript classification requires input_ids")
        batch_size = int(input_ids.shape[0])
        segments = self._segments(input_ids, attention_mask)
        embeddings = get_word_embeddings(self.model, context="RMT.sequence.forward")
        memory = embeddings(self.mem_token_ids).unsqueeze(0).expand(batch_size, -1, -1)
        final_pooled = None
        final_active_indices = None

        for segment_index, batch_segments in enumerate(segments):
            if self.bptt_depth > -1 and len(segments) - segment_index > self.bptt_depth:
                memory = memory.detach()
            active_indices = torch.tensor(
                [i for i, segment in enumerate(batch_segments) if segment is not None],
                device=input_ids.device,
                dtype=torch.long,
            )
            segment_ids = torch.stack(
                [segment for segment in batch_segments if segment is not None]
            )
            segment_mask = segment_ids.ne(self.pad_token_id).long()
            segment_embeddings = embeddings(segment_ids)
            active_memory = memory.index_select(0, active_indices)
            segment_embeddings = torch.cat(
                [
                    segment_embeddings[:, :1],
                    active_memory,
                    segment_embeddings[:, 1 + self.num_mem_tokens :],
                ],
                dim=1,
            )
            output = self.model(
                input_ids=None,
                inputs_embeds=segment_embeddings,
                attention_mask=segment_mask,
                token_type_ids=torch.zeros_like(segment_ids),
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = _hidden_from_output(output, context="RMT base model")
            if hidden.shape[-1] != self.hidden_size:
                raise RuntimeError(
                    f"RMT hidden width={hidden.shape[-1]}, expected={self.hidden_size}"
                )
            memory = memory.index_copy(
                0,
                active_indices,
                hidden[:, 1 : 1 + self.num_mem_tokens],
            )
            if segment_index == len(segments) - 1:
                final_pooled = pool_special_token(
                    hidden,
                    segment_ids,
                    int(self.cls_token.item()),
                    attention_mask=segment_mask,
                    token_name="CLS",
                )
                final_active_indices = active_indices

        if final_pooled is None or final_active_indices is None:
            raise RuntimeError("RMT produced no final segment")
        expected_indices = torch.arange(batch_size, device=input_ids.device)
        assert torch.equal(final_active_indices, expected_indices), (
            "Every sample must participate in the final right-aligned RMT segment"
        )
        logits = self.classifier(final_pooled)
        target = _sequence_labels(transcript_type, labels)
        return SequenceClassifierOutput(
            loss=_binary_sequence_loss(logits, target),
            logits=logits,
        )


class AMTTranscriptTypeClassifier(nn.Module):
    """Associative-memory transcript classifier pooled from a final-segment CLS.

    The ARMT wrapper splits an input tensor at fixed boundaries.  For each sample,
    this adapter makes the last segment ``CLS, final content, SEP`` and pads only
    earlier incomplete segments.  Consequently the pooled CLS is processed after
    all earlier segments have updated associative memory.
    """

    def __init__(
        self,
        amt_model: nn.Module,
        hidden_size: int,
        tokenizer: Any,
        *,
        segment_size: int,
        segment_alignment: str = "left",
    ):
        super().__init__()
        if segment_alignment != "left":
            raise RuntimeError(
                "AMT transcript classification requires segment_alignment='left' so "
                "the classification CLS starts the final segment"
            )
        self.amt = amt_model
        self.hidden_size = int(hidden_size)
        self.segment_size = int(segment_size)
        if self.segment_size <= 2:
            raise RuntimeError("AMT segment_size must exceed two special-token positions")
        self.pad_token_id = _required_token_id(tokenizer, "pad_token_id")
        self.register_buffer(
            "cls_token",
            torch.tensor([_required_token_id(tokenizer, "cls_token_id")], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "sep_token",
            torch.tensor([_required_token_id(tokenizer, "sep_token_id")], dtype=torch.long),
            persistent=False,
        )
        self.classifier = nn.Linear(self.hidden_size, 1)

    @classmethod
    def from_pretrained(
        cls,
        backbone_path: str,
        backbone_kind: str,
        tokenizer: Any,
        *,
        trust_remote_code: bool = True,
        amt_repo_id: str = "irodkin/armt-neox-tiny",
        allow_unsafe_torch_load: bool = True,
        **amt_kwargs,
    ) -> "AMTTranscriptTypeClassifier":
        if backbone_kind not in {"gena", "moderngena"}:
            raise RuntimeError(
                f"AMT is allowed only for GENA/ModernGENA, got {backbone_kind!r}"
            )
        base_model, _encoder_config, hidden_size, default_layers_attr = _load_amt_base_model(
            backbone_path,
            backbone_kind,
            trust_remote_code,
            allow_unsafe_torch_load,
        )
        if base_model.get_input_embeddings() is None:
            raise RuntimeError(
                f"AMT base model {type(base_model).__name__} has no input embeddings"
            )
        allow_transformers_torch_load_on_legacy_torch(
            allow_unsafe_torch_load, context=f"AMT.sequence:{amt_repo_id}"
        )
        loaded = AutoModelForCausalLM.from_pretrained(
            local_or_remote(amt_repo_id), trust_remote_code=True
        )
        amt_module = importlib.import_module(loaded.__class__.__module__)
        AssociativeMemoryCell = getattr(amt_module, "AssociativeMemoryCell")
        AssociativeRecurrentWrapper = getattr(amt_module, "AssociativeRecurrentWrapper")
        del loaded

        layers_attr = amt_kwargs.pop("layers_attr", default_layers_attr)
        act_on = bool(amt_kwargs.pop("act_on", False))
        attend_to_previous_input = bool(
            amt_kwargs.pop("attend_to_previous_input", False)
        )
        num_mem_tokens = int(amt_kwargs.pop("num_mem_tokens", 16))
        backbone_position_limit = 512 if backbone_kind == "gena" else 1024
        default_segment_size = backbone_position_limit - num_mem_tokens
        if default_segment_size <= 2:
            raise RuntimeError(
                "AMT memory tokens must leave room for CLS, content, and SEP: "
                f"position_limit={backbone_position_limit} "
                f"num_mem_tokens={num_mem_tokens}"
            )
        segment_size = int(amt_kwargs.pop("segment_size", default_segment_size))
        if segment_size + num_mem_tokens > backbone_position_limit:
            raise RuntimeError(
                "AMT data positions plus memory tokens exceed the backbone context: "
                f"segment_size={segment_size} num_mem_tokens={num_mem_tokens} "
                f"limit={backbone_position_limit}"
            )
        segment_alignment = str(amt_kwargs.pop("segment_alignment", "left"))
        sliding_window = bool(amt_kwargs.pop("sliding_window", False))
        time_penalty = float(amt_kwargs.pop("time_penalty", 0.0))
        memory_cell = AssociativeMemoryCell(
            base_model=base_model,
            num_mem_tokens=num_mem_tokens,
            d_mem=int(amt_kwargs.pop("d_mem", 32)),
            layers_attr=layers_attr,
            wrap_pos=bool(amt_kwargs.pop("wrap_pos", False)),
            correction=bool(amt_kwargs.pop("correction", True)),
            n_heads=int(amt_kwargs.pop("n_heads", 1)),
            use_denom=bool(amt_kwargs.pop("use_denom", True)),
            gating=bool(amt_kwargs.pop("gating", False)),
            freeze_mem=False,
            act_on=act_on,
            max_hop=int(amt_kwargs.pop("max_hop", 4)),
            act_type=amt_kwargs.pop("act_type", "associative"),
            constant_depth=bool(amt_kwargs.pop("constant_depth", False)),
            act_format=amt_kwargs.pop("act_format", "linear"),
            noisy_halting=bool(amt_kwargs.pop("noisy_halting", False)),
            attend_to_previous_input=attend_to_previous_input,
            use_sink=bool(amt_kwargs.pop("use_sink", False)),
        )
        amt_model = AssociativeRecurrentWrapper(
            memory_cell,
            segment_size=segment_size,
            segment_alignment=segment_alignment,
            sliding_window=sliding_window,
            attend_to_previous_input=attend_to_previous_input,
            act_on=act_on,
            time_penalty=time_penalty,
        )
        if amt_kwargs:
            raise RuntimeError(f"Unused AMT parameters: {sorted(amt_kwargs)}")
        return cls(
            amt_model,
            hidden_size,
            tokenizer,
            segment_size=segment_size,
            segment_alignment=segment_alignment,
        )

    def _prepare_sample(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row_ids = input_ids.unsqueeze(0)
        row_mask = (
            attention_mask.unsqueeze(0)
            if attention_mask is not None
            else input_ids.ne(self.pad_token_id).long().unsqueeze(0)
        )
        _special_token_positions(
            row_ids,
            int(self.cls_token.item()),
            attention_mask=row_mask,
            token_name="CLS",
        )
        _special_token_positions(
            row_ids,
            int(self.sep_token.item()),
            attention_mask=row_mask,
            token_name="SEP",
        )
        active = row_mask[0].bool()
        content = input_ids[
            active
            & input_ids.ne(self.pad_token_id)
            & input_ids.ne(int(self.cls_token.item()))
            & input_ids.ne(int(self.sep_token.item()))
        ]
        final_capacity = self.segment_size - 2
        final_length = min(int(content.numel()), final_capacity)
        if final_length:
            prefix = content[:-final_length]
            final_content = content[-final_length:]
        else:
            prefix = content
            final_content = content

        prepared_segments: list[torch.Tensor] = []
        prepared_masks: list[torch.Tensor] = []
        if prefix.numel() > 0:
            for chunk in torch.split(prefix, self.segment_size):
                chunk_mask = torch.ones_like(chunk)
                if chunk.numel() < self.segment_size:
                    pad = self.segment_size - chunk.numel()
                    chunk = F.pad(chunk, (0, pad), value=self.pad_token_id)
                    chunk_mask = F.pad(chunk_mask, (0, pad), value=0)
                prepared_segments.append(chunk)
                prepared_masks.append(chunk_mask)

        final_segment = torch.cat([self.cls_token, final_content, self.sep_token])
        final_mask = torch.ones_like(final_segment)
        prepared_segments.append(final_segment)
        prepared_masks.append(final_mask)
        prepared_ids = torch.cat(prepared_segments)
        prepared_attention = torch.cat(prepared_masks)
        cls_position = _special_token_positions(
            prepared_ids.unsqueeze(0),
            int(self.cls_token.item()),
            attention_mask=prepared_attention.unsqueeze(0),
            token_name="CLS",
        )[0]
        assert int(cls_position.item()) % self.segment_size == 0, (
            "The AMT classification CLS must be the first token of the final segment"
        )
        return prepared_ids, prepared_attention

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        transcript_type: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        if input_ids is None:
            raise RuntimeError("AMT transcript classification requires input_ids")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise RuntimeError("AMT attention_mask must have the same shape as input_ids")
        pooled_rows = []
        for sample_index in range(input_ids.shape[0]):
            sample_mask = None if attention_mask is None else attention_mask[sample_index]
            prepared_ids, prepared_mask = self._prepare_sample(
                input_ids[sample_index], sample_mask
            )
            output = self.amt(
                input_ids=prepared_ids.unsqueeze(0),
                attention_mask=prepared_mask.unsqueeze(0),
            )
            hidden = _hidden_from_output(output, context="AMT base model")
            if hidden.shape[-1] != self.hidden_size:
                raise RuntimeError(
                    f"AMT hidden width={hidden.shape[-1]}, expected={self.hidden_size}"
                )
            if hidden.shape[1] != prepared_ids.numel():
                raise RuntimeError(
                    "AMT output is not aligned with the prepared sequence: "
                    f"hidden_length={hidden.shape[1]} input_length={prepared_ids.numel()}"
                )
            pooled_rows.append(
                pool_special_token(
                    hidden,
                    prepared_ids.unsqueeze(0),
                    int(self.cls_token.item()),
                    attention_mask=prepared_mask.unsqueeze(0),
                    token_name="CLS",
                )
            )
        pooled = torch.cat(pooled_rows, dim=0)
        logits = self.classifier(pooled)
        target = _sequence_labels(transcript_type, labels)
        return SequenceClassifierOutput(
            loss=_binary_sequence_loss(logits, target),
            logits=logits,
        )


__all__ = [
    "AMTTranscriptTypeClassifier",
    "CaduceusTranscriptTypeMiddleLossSequenceClassifier",
    "GenaModernTranscriptTypeClassifier",
    "RMTTranscriptTypeClassifier",
    "pool_special_token",
]
