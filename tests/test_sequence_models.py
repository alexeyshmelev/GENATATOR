from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    import torch.nn as nn
    from transformers.modeling_outputs import TokenClassifierOutput

    from genatator_core import amt_models, model_builders
    from genatator_core.backbones import TranscriptTokenClassificationBackbone
    from genatator_core.sequence_models import (
        AMTTranscriptTypeClassifier,
        CaduceusTranscriptTypeMiddleLossSequenceClassifier,
        GenaModernTranscriptTypeClassifier,
        RMTTranscriptTypeClassifier,
        pool_special_token,
    )

    _HAS_MODEL_DEPENDENCIES = True
except ImportError:
    torch = None
    TokenClassifierOutput = None
    amt_models = None
    model_builders = None
    TranscriptTokenClassificationBackbone = None
    AMTTranscriptTypeClassifier = None
    CaduceusTranscriptTypeMiddleLossSequenceClassifier = None
    GenaModernTranscriptTypeClassifier = None
    RMTTranscriptTypeClassifier = None
    pool_special_token = None
    _HAS_MODEL_DEPENDENCIES = False

    class _MissingTorchModule:
        pass

    class _MissingNN:
        Module = _MissingTorchModule

    nn = _MissingNN()


class _Tokenizer:
    pad_token_id = 0
    cls_token_id = 1
    sep_token_id = 2


class _EchoBackbone(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 3, position_limit: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            max_position_embeddings=position_limit,
        )
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, 1)
        with torch.no_grad():
            values = torch.arange(vocab_size, dtype=torch.float32)
            self.embeddings.weight.copy_(values[:, None].repeat(1, hidden_size))
        self.forward_batch_sizes = []

    def get_input_embeddings(self):
        return self.embeddings

    def resize_token_embeddings(self, size: int):
        old = self.embeddings
        resized = nn.Embedding(size, self.hidden_size)
        with torch.no_grad():
            resized.weight.zero_()
            resized.weight[: old.num_embeddings].copy_(old.weight)
        self.embeddings = resized
        return resized

    def classify(self, pooled_hidden):
        return self.classifier(pooled_hidden)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        token_type_ids=None,
        output_hidden_states=True,
        return_dict=True,
        **kwargs,
    ):
        hidden = self.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        self.forward_batch_sizes.append(int(hidden.shape[0]))
        return TokenClassifierOutput(logits=hidden, hidden_states=(hidden,))


class _FakeCaduceus(nn.Module):
    def forward(self, input_ids=None, output_hidden_states=False):
        base = input_ids.float().unsqueeze(-1).repeat(1, 1, 2)
        middle = base * 2.0
        final = base * 3.0
        return SimpleNamespace(
            last_hidden_state=final,
            hidden_states=(base, middle, final),
        )


class _FakeAMT(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.calls = []

    def forward(self, input_ids=None, attention_mask=None):
        self.calls.append((input_ids.detach().clone(), attention_mask.detach().clone()))
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, self.hidden_size)
        return SimpleNamespace(logits=hidden)


class _OwnerEncoder(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        token_type_ids=None,
        output_hidden_states=True,
        return_dict=True,
        **kwargs,
    ):
        hidden = self.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        return SimpleNamespace(
            last_hidden_state=hidden,
            hidden_states=(hidden,),
            attentions=None,
        )


class _TokenOwnerBase(nn.Module):
    def __init__(self, *, encoder_attr: str, hidden_size: int = 3):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            max_position_embeddings=64,
        )
        self.embeddings = nn.Embedding(32, hidden_size)
        setattr(self, encoder_attr, _OwnerEncoder(self.embeddings))
        self.dropout = nn.Identity()
        self.classifier = nn.Linear(hidden_size, 1)

    def get_input_embeddings(self):
        return self.embeddings

    def resize_token_embeddings(self, size: int):
        old = self.embeddings
        resized = nn.Embedding(size, old.embedding_dim)
        with torch.no_grad():
            resized.weight.zero_()
            resized.weight[: old.num_embeddings].copy_(old.weight)
        self.embeddings = resized
        encoder = self.bert if hasattr(self, "bert") else self.model
        encoder.embeddings = resized
        return resized


BertForTokenClassification = type(
    "BertForTokenClassification",
    (_TokenOwnerBase,),
    {"__init__": lambda self: _TokenOwnerBase.__init__(self, encoder_attr="bert")},
)
ModernBertForTokenClassification = type(
    "ModernBertForTokenClassification",
    (_TokenOwnerBase,),
    {"__init__": lambda self: _TokenOwnerBase.__init__(self, encoder_attr="model")},
)


@unittest.skipUnless(_HAS_MODEL_DEPENDENCIES, "torch/transformers are not installed")
class SpecialTokenPoolingTests(unittest.TestCase):
    def test_pool_special_token_selects_exact_id_not_last_attended_position(self):
        input_ids = torch.tensor([[1, 7, 2, 9], [8, 2, 6, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
        hidden = input_ids.float().unsqueeze(-1)
        pooled = pool_special_token(
            hidden,
            input_ids,
            2,
            attention_mask=attention_mask,
            token_name="SEP",
        )
        torch.testing.assert_close(pooled, torch.tensor([[2.0], [2.0]]))

    def test_pool_special_token_asserts_exactly_one_attended_token(self):
        hidden = torch.zeros(2, 4, 2)
        with self.assertRaises(AssertionError):
            pool_special_token(hidden, torch.tensor([[1, 7, 0, 0], [1, 2, 2, 0]]), 2)


@unittest.skipUnless(_HAS_MODEL_DEPENDENCIES, "torch/transformers are not installed")
class TranscriptTokenClassificationBackboneTests(unittest.TestCase):
    def test_gena_owns_and_uses_bert_for_token_classification(self):
        owner = BertForTokenClassification()
        with patch(
            "genatator_core.backbones._load_gena_token_classifier_owner",
            return_value=owner,
        ) as loader:
            backbone = TranscriptTokenClassificationBackbone(
                "unused",
                "gena",
                trust_remote_code=True,
                allow_unsafe_torch_load=False,
                num_labels=1,
            )
        self.assertEqual(type(backbone.owner).__name__, "BertForTokenClassification")
        loader.assert_called_once_with(
            "unused",
            trust_remote_code=True,
            num_labels=1,
        )
        with torch.no_grad():
            backbone.owner.classifier.weight.copy_(torch.tensor([[1.0, 0.0, 0.0]]))
            backbone.owner.classifier.bias.zero_()
            backbone.owner.embeddings.weight[1].copy_(torch.tensor([4.0, 5.0, 6.0]))
        output = backbone(
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
            token_type_ids=torch.zeros(1, 2, dtype=torch.long),
        )
        torch.testing.assert_close(backbone.classify(output.logits[:, 0]), torch.tensor([[4.0]]))

    def test_moderngena_owns_and_uses_modernbert_for_token_classification(self):
        owner = ModernBertForTokenClassification()
        with patch(
            "genatator_core.backbones.ModernBertForTokenClassification.from_pretrained",
            return_value=owner,
        ) as loader:
            backbone = TranscriptTokenClassificationBackbone(
                "unused",
                "moderngena",
                trust_remote_code=True,
                allow_unsafe_torch_load=False,
                num_labels=1,
            )
        self.assertEqual(
            type(backbone.owner).__name__, "ModernBertForTokenClassification"
        )
        loader.assert_called_once_with(
            "unused",
            num_labels=1,
            trust_remote_code=True,
        )
        self.assertEqual(backbone.layers_attr, "owner.model.layers")

    def test_amt_transcript_loader_selects_token_classification_container(self):
        fake = SimpleNamespace(
            owner=ModernBertForTokenClassification(),
            config=SimpleNamespace(hidden_size=3),
            hidden_size=3,
            layers_attr="owner.model.layers",
        )
        with patch.object(
            amt_models,
            "TranscriptTokenClassificationBackbone",
            return_value=fake,
        ) as constructor:
            base_model, config, hidden_size, layers_attr = amt_models._load_amt_base_model(
                "unused",
                "moderngena",
                True,
                False,
                transcript_type=True,
            )
        self.assertIs(base_model, fake)
        self.assertIs(config, fake.config)
        self.assertEqual(hidden_size, 3)
        self.assertEqual(layers_attr, "owner.model.layers")
        constructor.assert_called_once_with(
            "unused",
            "moderngena",
            trust_remote_code=True,
            allow_unsafe_torch_load=False,
            num_labels=1,
            attn_implementation="sdpa",
        )


@unittest.skipUnless(_HAS_MODEL_DEPENDENCIES, "torch/transformers are not installed")
class PlainTranscriptClassifierTests(unittest.TestCase):
    def test_gena_and_moderngena_pool_cls(self):
        input_ids = torch.tensor([[1, 8, 2, 0], [1, 4, 5, 2]])
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
        for backbone_kind in ("gena", "moderngena"):
            with self.subTest(backbone_kind=backbone_kind):
                backbone = _EchoBackbone()
                with patch(
                    "genatator_core.sequence_models.TranscriptTokenClassificationBackbone",
                    return_value=backbone,
                ):
                    model = GenaModernTranscriptTypeClassifier(
                        "unused", backbone_kind, _Tokenizer()
                    )
                with torch.no_grad():
                    model.hidden_backbone.classifier.weight.copy_(
                        torch.tensor([[1.0, 0.0, 0.0]])
                    )
                    model.hidden_backbone.classifier.bias.zero_()
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    transcript_type=torch.tensor([[0.0], [1.0]]),
                )
                torch.testing.assert_close(output.logits, torch.ones(2, 1))
                self.assertEqual(output.loss.ndim, 0)

    def test_plain_classifier_rejects_a_missing_cls(self):
        backbone = _EchoBackbone()
        with patch(
            "genatator_core.sequence_models.TranscriptTokenClassificationBackbone",
            return_value=backbone,
        ):
            model = GenaModernTranscriptTypeClassifier("unused", "gena", _Tokenizer())
        with self.assertRaises(AssertionError):
            model(
                input_ids=torch.tensor([[7, 8, 2, 0]]),
                attention_mask=torch.tensor([[1, 1, 1, 0]]),
            )


@unittest.skipUnless(_HAS_MODEL_DEPENDENCIES, "torch/transformers are not installed")
class CaduceusTranscriptClassifierTests(unittest.TestCase):
    def test_caduceus_pools_sep_for_final_and_middle_heads(self):
        model = CaduceusTranscriptTypeMiddleLossSequenceClassifier(
            _FakeCaduceus(), hidden_size=2, tokenizer=_Tokenizer()
        )
        with torch.no_grad():
            model.classifier.weight.copy_(torch.tensor([[1.0, 0.0]]))
            model.classifier.bias.zero_()
            model.middle_classifier.weight.copy_(torch.tensor([[1.0, 0.0]]))
            model.middle_classifier.bias.zero_()
        # SEP deliberately is not the last attended token.  This isolates the
        # token-identity contract from padding-side assumptions.
        output = model(
            input_ids=torch.tensor([[1, 2, 9, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 0]]),
            transcript_type=torch.tensor([[1.0]]),
        )
        torch.testing.assert_close(output.logits, torch.tensor([[6.0]]))
        expected = 0.5 * (
            torch.nn.functional.binary_cross_entropy_with_logits(
                torch.tensor([[6.0]]), torch.tensor([[1.0]])
            )
            + torch.nn.functional.binary_cross_entropy_with_logits(
                torch.tensor([[4.0]]), torch.tensor([[1.0]])
            )
        )
        torch.testing.assert_close(output.loss, expected)

    def test_caduceus_rejects_a_missing_sep(self):
        model = CaduceusTranscriptTypeMiddleLossSequenceClassifier(
            _FakeCaduceus(), hidden_size=2, tokenizer=_Tokenizer()
        )
        with self.assertRaises(AssertionError):
            model(
                input_ids=torch.tensor([[0, 1, 7, 8]]),
                attention_mask=torch.tensor([[0, 1, 1, 1]]),
            )


@unittest.skipUnless(_HAS_MODEL_DEPENDENCIES, "torch/transformers are not installed")
class RMTTranscriptClassifierTests(unittest.TestCase):
    def test_rmt_right_aligns_rows_and_classifies_final_cls(self):
        backbone = _EchoBackbone(hidden_size=2, position_limit=7)
        model = RMTTranscriptTypeClassifier(
            backbone,
            _Tokenizer(),
            num_mem_tokens=1,
            segment_size=7,
            max_n_segments=4,
        )
        self.assertTrue(model._uses_backbone_classifier)
        output = model(
            input_ids=torch.tensor(
                [
                    [1, 3, 4, 5, 6, 2, 0],
                    [1, 8, 9, 2, 0, 0, 0],
                ]
            ),
            attention_mask=torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0, 0, 0],
                ]
            ),
            transcript_type=torch.tensor([[1.0], [0.0]]),
        )
        self.assertEqual(tuple(output.logits.shape), (2, 1))
        self.assertEqual(output.loss.ndim, 0)
        # The four-content-token row has two segments; the shorter row joins at
        # the final recurrent step instead of being reassigned to another row.
        self.assertEqual(backbone.forward_batch_sizes, [1, 2])

    def test_rmt_rejects_missing_raw_cls(self):
        model = RMTTranscriptTypeClassifier(
            _EchoBackbone(hidden_size=2, position_limit=7),
            _Tokenizer(),
            num_mem_tokens=1,
            segment_size=7,
        )
        with self.assertRaises(AssertionError):
            model(
                input_ids=torch.tensor([[3, 4, 2, 0]]),
                attention_mask=torch.tensor([[1, 1, 1, 0]]),
            )


@unittest.skipUnless(_HAS_MODEL_DEPENDENCIES, "torch/transformers are not installed")
class AMTTranscriptClassifierTests(unittest.TestCase):
    def test_amt_places_cls_at_start_of_final_segment_and_pools_it(self):
        amt = _FakeAMT(hidden_size=2)
        model = AMTTranscriptTypeClassifier(
            amt,
            hidden_size=2,
            tokenizer=_Tokenizer(),
            segment_size=5,
        )
        with torch.no_grad():
            model.classifier.weight.copy_(torch.tensor([[1.0, 0.0]]))
            model.classifier.bias.zero_()
        output = model(
            input_ids=torch.tensor([[1, 3, 4, 5, 6, 7, 8, 2, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0]]),
            transcript_type=torch.tensor([[1.0]]),
        )
        prepared_ids, prepared_mask = amt.calls[0]
        self.assertEqual(prepared_ids[0, 5].item(), _Tokenizer.cls_token_id)
        self.assertEqual(prepared_mask[0, 3].item(), 0)
        self.assertEqual(prepared_mask[0, 4].item(), 0)
        torch.testing.assert_close(output.logits, torch.tensor([[1.0]]))
        self.assertEqual(output.loss.ndim, 0)

    def test_amt_processes_padded_batch_samplewise(self):
        amt = _FakeAMT(hidden_size=2)
        model = AMTTranscriptTypeClassifier(
            amt,
            hidden_size=2,
            tokenizer=_Tokenizer(),
            segment_size=5,
        )
        output = model(
            input_ids=torch.tensor([[1, 3, 2, 0], [1, 4, 5, 2]]),
            attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
        )
        self.assertEqual(tuple(output.logits.shape), (2, 1))
        self.assertEqual(len(amt.calls), 2)
        self.assertEqual(amt.calls[0][0][0, 0].item(), _Tokenizer.cls_token_id)
        self.assertEqual(amt.calls[1][0][0, 0].item(), _Tokenizer.cls_token_id)

    def test_amt_uses_token_classification_owner_projection_when_provided(self):
        amt = _FakeAMT(hidden_size=2)
        owner_projection = nn.Linear(2, 1)
        model = AMTTranscriptTypeClassifier(
            amt,
            hidden_size=2,
            tokenizer=_Tokenizer(),
            segment_size=5,
            sequence_classifier=owner_projection.forward,
        )
        self.assertFalse(hasattr(model, "classifier"))
        with torch.no_grad():
            owner_projection.weight.copy_(torch.tensor([[2.0, 0.0]]))
            owner_projection.bias.zero_()
        output = model(
            input_ids=torch.tensor([[1, 3, 2, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 0]]),
        )
        torch.testing.assert_close(output.logits, torch.tensor([[2.0]]))


@unittest.skipUnless(_HAS_MODEL_DEPENDENCIES, "torch/transformers are not installed")
class TranscriptModelBuilderTests(unittest.TestCase):
    @staticmethod
    def _cfg(family: str, backbone_kind: str, **model_fields):
        model = {
            "family": family,
            "backbone_kind": backbone_kind,
            "backbone_path": "unused-checkpoint",
            "trust_remote_code": True,
            **model_fields,
        }
        return {"model": model, "_tokenizer": _Tokenizer()}

    def test_unet_detection_is_task_and_head_aware(self):
        transcript_rmt = {"family": "rmt", "backbone_kind": "gena", "rmt": {}}
        self.assertFalse(
            model_builders.model_uses_unet(transcript_rmt, task="transcript_type")
        )
        self.assertIsNone(
            model_builders.normalize_unet_chunk_size(
                transcript_rmt, task="transcript_type"
            )
        )
        self.assertNotIn("unet_chunk_size", transcript_rmt)

        gpt_rmt = {"family": "rmt", "head_kind": "gpt"}
        self.assertFalse(model_builders.model_uses_unet(gpt_rmt, task="segmentation"))
        self.assertIsNone(
            model_builders.normalize_unet_chunk_size(gpt_rmt, task="segmentation")
        )

        segmentation_rmt = {"family": "rmt", "rmt": {}}
        self.assertTrue(
            model_builders.model_uses_unet(segmentation_rmt, task="segmentation")
        )
        self.assertEqual(
            model_builders.normalize_unet_chunk_size(
                segmentation_rmt, task="segmentation"
            ),
            8192,
        )

    def test_plain_transcript_builder_uses_sequence_wrapper_and_tokenizer(self):
        cfg = self._cfg("plain", "gena")
        built = nn.Linear(1, 1)
        with patch.object(
            model_builders,
            "allow_transformers_torch_load_on_legacy_torch",
        ), patch.object(
            model_builders,
            "GenaModernTranscriptTypeClassifier",
            return_value=built,
        ) as constructor:
            output = model_builders.build_model(cfg, task="transcript_type")
        self.assertIs(output, built)
        constructor.assert_called_once_with(
            "unused-checkpoint",
            "gena",
            cfg["_tokenizer"],
            trust_remote_code=True,
            allow_unsafe_torch_load=True,
        )

    def test_caduceus_transcript_builder_passes_tokenizer_to_sep_wrapper(self):
        cfg = self._cfg("caduceus", "caduceus")
        backbone = nn.Linear(1, 1)
        built = nn.Linear(1, 1)
        caduceus_config = SimpleNamespace(d_model=2, bidirectional_weight_tie=True)
        with patch.object(
            model_builders,
            "allow_transformers_torch_load_on_legacy_torch",
        ), patch.object(
            model_builders.AutoConfig,
            "from_pretrained",
            return_value=caduceus_config,
        ), patch.object(
            model_builders.AutoModel,
            "from_pretrained",
            return_value=backbone,
        ), patch.object(
            model_builders,
            "CaduceusTranscriptTypeMiddleLossSequenceClassifier",
            return_value=built,
        ) as constructor:
            output = model_builders.build_model(cfg, task="transcript_type")
        self.assertIs(output, built)
        constructor.assert_called_once_with(
            backbone,
            hidden_size=2,
            tokenizer=cfg["_tokenizer"],
        )

    def test_rmt_transcript_builder_does_not_require_unet_fields(self):
        cfg = self._cfg(
            "rmt",
            "gena",
            rmt={
                "num_mem_tokens": 10,
                "segment_size": 512,
                "max_n_segments": 8,
                "bptt_depth": -1,
            },
        )
        base_model = nn.Linear(1, 1)
        built = nn.Linear(1, 1)
        with patch.object(
            model_builders,
            "allow_transformers_torch_load_on_legacy_torch",
        ), patch.object(
            model_builders,
            "TranscriptTokenClassificationBackbone",
            return_value=base_model,
        ) as backbone_constructor, patch.object(
            model_builders,
            "RMTTranscriptTypeClassifier",
            return_value=built,
        ) as constructor:
            output = model_builders.build_model(cfg, task="transcript_type")
        self.assertIs(output, built)
        backbone_constructor.assert_called_once_with(
            "unused-checkpoint",
            "gena",
            trust_remote_code=True,
            allow_unsafe_torch_load=True,
            num_labels=1,
        )
        constructor.assert_called_once_with(
            base_model,
            cfg["_tokenizer"],
            num_mem_tokens=10,
            segment_size=512,
            max_n_segments=8,
            bptt_depth=-1,
        )
        self.assertNotIn("vocab_size", cfg["model"])
        self.assertNotIn("unet_chunk_size", cfg["model"])

    def test_amt_transcript_builder_calls_sequence_factory(self):
        cfg = self._cfg(
            "amt",
            "moderngena",
            use_unet=False,
            amt={
                "amt_repo_id": "test/amt",
                "num_mem_tokens": 20,
                "segment_size": 1004,
                "d_mem": 64,
            },
        )
        built = nn.Linear(1, 1)
        with patch.object(
            model_builders,
            "allow_transformers_torch_load_on_legacy_torch",
        ), patch.object(
            model_builders.AMTTranscriptTypeClassifier,
            "from_pretrained",
            return_value=built,
        ) as constructor:
            output = model_builders.build_model(cfg, task="transcript_type")
        self.assertIs(output, built)
        constructor.assert_called_once_with(
            backbone_path="unused-checkpoint",
            backbone_kind="moderngena",
            tokenizer=cfg["_tokenizer"],
            trust_remote_code=True,
            allow_unsafe_torch_load=True,
            amt_repo_id="test/amt",
            num_mem_tokens=20,
            segment_size=1004,
            d_mem=64,
        )


if __name__ == "__main__":
    unittest.main()
