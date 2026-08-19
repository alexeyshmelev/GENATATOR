from __future__ import annotations

import types
import unittest

try:
    import torch
    import torch.nn as nn

    from genatator_core.gpt_models import (
        AMTGPTSegmentationModel,
        CaduceusGPTSegmentationModel,
        GenaModernGPTSegmentationModel,
        RMTGPTSegmentationModel,
        _content_mask,
        _framing_special_token_ids,
        expand_bpe_states_to_nucleotides,
    )
    from genatator_core.gpt_head import (
        EXON_LABEL_INDEX,
        INTRON_LABEL_INDEX,
        T5GemmaDecoder,
        T5GemmaSegmentationHead,
    )
except ImportError:
    torch = None
    nn = None
    T5GemmaDecoder = None


if nn is not None:
    class RecordingGPTHead(nn.Module):
        def __init__(self, num_labels: int = 5):
            super().__init__()
            self.num_labels = int(num_labels)
            self.calls = []

        def forward(
            self,
            encoder_embeddings,
            *,
            nucleotide_mask,
            labels=None,
            labels_mask=None,
            pos_weight=None,
            autoregressive=None,
        ):
            self.calls.append(
                {
                    "encoder_embeddings": encoder_embeddings.detach().clone(),
                    "nucleotide_mask": nucleotide_mask.detach().clone(),
                    "labels": None if labels is None else labels.detach().clone(),
                    "labels_mask": None
                    if labels_mask is None
                    else labels_mask.detach().clone(),
                    "pos_weight": pos_weight,
                    "autoregressive": autoregressive,
                    "training": self.training,
                }
            )
            logits = encoder_embeddings.new_zeros(
                (*encoder_embeddings.shape[:2], self.num_labels)
            )
            loss = logits.sum() if labels is not None else None
            return loss, logits


    class FakeDirectBackbone(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.hidden = hidden
            self.calls = []
            self.forward_calls = []
            self.chunked_calls = []

        def forward(self, **kwargs):
            self.forward_calls.append(kwargs)
            return types.SimpleNamespace(logits=self.hidden)

        def forward_chunked(self, **kwargs):
            self.calls.append(kwargs)
            self.chunked_calls.append(kwargs)
            return types.SimpleNamespace(logits=self.hidden)


    class FakeRMTEncoder(nn.Module):
        def __init__(self, token_hidden, token_mask):
            super().__init__()
            self.hidden_size = int(token_hidden.shape[-1])
            self.token_hidden = token_hidden
            self.token_mask = token_mask
            self.calls = []

        def _encode_rmt_tokens(self, input_ids, **kwargs):
            self.calls.append((input_ids.detach().clone(), kwargs))
            return self.token_hidden, self.token_mask


    class FakeAMTEncoder(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.hidden_size = int(hidden.shape[-1])
            self.hidden = hidden
            self.calls = []

        def encode_hidden(self, **kwargs):
            self.calls.append(kwargs)
            return self.hidden


    class FakeCaduceusEncoder(nn.Module):
        def __init__(self, hidden, *, tuple_output: bool = True):
            super().__init__()
            self.hidden = hidden
            self.tuple_output = tuple_output
            self.calls = []

        def forward(self, **kwargs):
            self.calls.append(kwargs)
            if self.tuple_output:
                return (self.hidden,)
            return types.SimpleNamespace(last_hidden_state=self.hidden)


def _set_scalar_embedding(embedding) -> None:
    """Make each two-wide embedding row a deterministic function of its id."""

    with torch.no_grad():
        ids = torch.arange(embedding.num_embeddings, dtype=embedding.weight.dtype)
        embedding.weight[:, 0] = ids
        embedding.weight[:, 1] = ids + 0.25


def _five_track_labels(classes):
    classes = torch.as_tensor(classes, dtype=torch.long)
    labels = torch.zeros((*classes.shape, 5), dtype=torch.float32)
    labels[..., INTRON_LABEL_INDEX] = classes.eq(0).float()
    labels[..., EXON_LABEL_INDEX] = classes.eq(1).float()
    return labels


def _new_direct_model(hidden):
    model = GenaModernGPTSegmentationModel.__new__(GenaModernGPTSegmentationModel)
    nn.Module.__init__(model)
    model.hidden_size = 2
    model.hidden_backbone = FakeDirectBackbone(hidden)
    model.special_token_ids = (0, 10, 11)
    model.nucleotide_embedding = nn.Embedding(32, 2)
    _set_scalar_embedding(model.nucleotide_embedding)
    model.gpt_head = RecordingGPTHead()
    return model


@unittest.skipIf(torch is None, "torch/transformers are not installed")
class GPTModelAdapterTests(unittest.TestCase):
    def test_special_and_attention_masks_remove_only_framing_positions(self):
        tokenizer = types.SimpleNamespace(
            pad_token_id=0,
            cls_token_id=10,
            sep_token_id=11,
            bos_token_id=12,
            eos_token_id=13,
            unk_token_id=14,
        )
        special_ids = _framing_special_token_ids(tokenizer)
        self.assertEqual(special_ids, (0, 10, 11, 12, 13))
        input_ids = torch.tensor([[0, 10, 4, 14, 5, 11, 13, 6]])
        attention = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 0]])
        mask = _content_mask(input_ids, attention, special_ids)
        self.assertTrue(
            torch.equal(
                mask,
                torch.tensor([[False, False, True, True, True, False, False, False]]),
            )
        )

    def test_bpe_expansion_compacts_specials_and_leaves_padding_uncovered(self):
        token_hidden = torch.tensor(
            [
                [[90.0, 90.0], [1.0, 10.0], [2.0, 20.0], [91.0, 91.0]],
                [[0.0, 0.0], [92.0, 92.0], [3.0, 30.0], [4.0, 40.0]],
            ]
        )
        token_mask = torch.tensor(
            [[False, True, True, False], [False, False, True, True]]
        )
        repeater = torch.tensor([[0, 0, 1, -100, -100], [0, 1, 1, -100, -100]])
        letter_tokens = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]])
        letter_attention = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
        embedding = nn.Embedding(16, 2)
        _set_scalar_embedding(embedding)

        expanded, nucleotide_mask = expand_bpe_states_to_nucleotides(
            token_hidden=token_hidden,
            token_content_mask=token_mask,
            embedding_repeater=repeater,
            letter_level_tokens=letter_tokens,
            letter_level_attention_mask=letter_attention,
            letter_level_labels_mask=None,
            nucleotide_embedding=embedding,
            context="test",
        )

        self.assertEqual(tuple(expanded.shape), (2, 5, 4))
        self.assertTrue(
            torch.equal(
                nucleotide_mask,
                torch.tensor(
                    [[True, True, True, False, False], [True, True, True, False, False]]
                ),
            )
        )
        self.assertTrue(
            torch.allclose(expanded[0, 0], torch.tensor([1.0, 1.25, 1.0, 10.0]))
        )
        self.assertTrue(
            torch.allclose(expanded[0, 1], torch.tensor([2.0, 2.25, 1.0, 10.0]))
        )
        self.assertTrue(
            torch.allclose(expanded[0, 2], torch.tensor([3.0, 3.25, 2.0, 20.0]))
        )
        self.assertTrue(bool((expanded[~nucleotide_mask] == 0).all()))
        # Special-state sentinel values must never survive compaction.
        self.assertFalse(bool((expanded == 90.0).any()))
        self.assertFalse(bool((expanded == 91.0).any()))
        self.assertFalse(bool((expanded == 92.0).any()))

    def test_bpe_expansion_rejects_misaligned_nucleotide_inputs(self):
        with self.assertRaisesRegex(RuntimeError, "embedding_repeater must match"):
            expand_bpe_states_to_nucleotides(
                token_hidden=torch.randn(1, 2, 2),
                token_content_mask=torch.ones((1, 2), dtype=torch.bool),
                embedding_repeater=torch.zeros((1, 2), dtype=torch.long),
                letter_level_tokens=torch.zeros((1, 3), dtype=torch.long),
                letter_level_attention_mask=torch.ones((1, 3), dtype=torch.bool),
                letter_level_labels_mask=None,
                nucleotide_embedding=nn.Embedding(4, 2),
                context="shape_test",
            )

    def test_direct_adapter_expands_hidden_states_and_routes_all_head_inputs(self):
        hidden = torch.tensor(
            [[[99.0, 99.0], [1.0, 10.0], [2.0, 20.0], [98.0, 98.0], [0.0, 0.0]]]
        )
        model = _new_direct_model(hidden)
        input_ids = torch.tensor([[10, 4, 5, 11, 0]])
        attention = torch.tensor([[1, 1, 1, 1, 0]])
        token_types = torch.zeros_like(input_ids)
        repeater = torch.tensor([[0, 0, 1, -100]])
        letter_tokens = torch.tensor([[6, 7, 8, 0]])
        letter_attention = torch.tensor([[1, 1, 1, 0]])
        letter_labels = torch.randint(0, 2, (1, 4, 5)).float()
        letter_label_mask = torch.tensor([[True, True, True, False]])
        pos_weight = torch.ones((1, 1, 5))

        output = model(
            input_ids=input_ids,
            attention_mask=attention,
            token_type_ids=token_types,
            embedding_repeater=repeater,
            letter_level_tokens=letter_tokens,
            letter_level_attention_mask=letter_attention,
            letter_level_labels=letter_labels,
            letter_level_labels_mask=letter_label_mask,
            pos_weight=pos_weight,
            autoregressive=False,
        )

        self.assertEqual(tuple(output.logits.shape), (1, 4, 5))
        self.assertIsNotNone(output.loss)
        self.assertEqual(len(model.hidden_backbone.calls), 1)
        self.assertEqual(len(model.hidden_backbone.chunked_calls), 1)
        self.assertEqual(len(model.hidden_backbone.forward_calls), 0)
        self.assertTrue(
            torch.equal(model.hidden_backbone.calls[0]["token_type_ids"], token_types)
        )
        call = model.gpt_head.calls[0]
        self.assertTrue(
            torch.equal(call["nucleotide_mask"], torch.tensor([[True, True, True, False]]))
        )
        self.assertTrue(
            torch.allclose(
                call["encoder_embeddings"][0, 0],
                torch.tensor([6.0, 6.25, 1.0, 10.0]),
            )
        )
        self.assertTrue(torch.equal(call["labels"], letter_labels))
        self.assertTrue(torch.equal(call["labels_mask"], letter_label_mask))
        self.assertIs(call["pos_weight"], pos_weight)
        self.assertFalse(call["autoregressive"])

    def test_direct_adapter_cannot_reenable_specials_with_broad_labels_mask(self):
        hidden = torch.tensor(
            [[[99.0, 99.0], [1.0, 10.0], [2.0, 20.0], [98.0, 98.0], [0.0, 0.0]]]
        )
        model = _new_direct_model(hidden)
        model(
            input_ids=torch.tensor([[10, 4, 5, 11, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 1, 0]]),
            # Deliberately malformed: framing positions are true.  The adapter
            # must still intersect this with its independent framing mask.
            labels_mask=torch.ones((1, 5), dtype=torch.bool),
            embedding_repeater=torch.tensor([[0, 1, -100]]),
            letter_level_tokens=torch.tensor([[6, 7, 0]]),
            letter_level_attention_mask=torch.tensor([[1, 1, 0]]),
            letter_level_labels_mask=torch.tensor([[True, True, False]]),
        )
        encoder_embeddings = model.gpt_head.calls[0]["encoder_embeddings"]
        self.assertTrue(
            torch.allclose(
                encoder_embeddings[0, 0],
                torch.tensor([6.0, 6.25, 1.0, 10.0]),
            )
        )
        self.assertFalse(bool((encoder_embeddings == 99.0).any()))
        self.assertFalse(bool((encoder_embeddings == 98.0).any()))

    def test_eval_adapter_rejects_teacher_forced_validation(self):
        hidden = torch.tensor(
            [[[99.0, 99.0], [1.0, 10.0], [2.0, 20.0], [98.0, 98.0], [0.0, 0.0]]]
        )
        model = _new_direct_model(hidden)
        model.eval()
        labels = _five_track_labels([[0, 1, 0, 0]])

        with self.assertRaisesRegex(RuntimeError, "cannot consume labels"):
            model(
                input_ids=torch.tensor([[10, 4, 5, 11, 0]]),
                attention_mask=torch.tensor([[1, 1, 1, 1, 0]]),
                labels_mask=torch.tensor([[False, True, True, False, False]]),
                embedding_repeater=torch.tensor([[0, 0, 1, -100]]),
                letter_level_tokens=torch.tensor([[6, 7, 8, 0]]),
                letter_level_attention_mask=torch.tensor([[1, 1, 1, 0]]),
                letter_level_labels=labels,
                letter_level_labels_mask=torch.tensor([[True, True, True, False]]),
            )

        self.assertEqual(model.gpt_head.calls, [])

    def test_public_generate_forces_eval_autoregression_and_restores_training(self):
        hidden = torch.tensor(
            [[[99.0, 99.0], [1.0, 10.0], [2.0, 20.0], [98.0, 98.0], [0.0, 0.0]]]
        )
        model = _new_direct_model(hidden)
        model.train()
        input_ids = torch.tensor([[10, 4, 5, 11, 0]])
        attention = torch.tensor([[1, 1, 1, 1, 0]])
        label_mask = torch.tensor([[False, True, True, False, False]])
        logits = model.generate(
            input_ids=input_ids,
            attention_mask=attention,
            labels_mask=label_mask,
            embedding_repeater=torch.tensor([[0, 0, 1, -100]]),
            letter_level_tokens=torch.tensor([[6, 7, 8, 0]]),
            letter_level_attention_mask=torch.tensor([[1, 1, 1, 0]]),
            letter_level_labels=torch.ones((1, 4, 5)),
            letter_level_labels_mask=torch.tensor([[True, True, True, False]]),
            pos_weight=torch.ones((1, 1, 5)),
        )

        self.assertTrue(model.training)
        self.assertEqual(tuple(logits.shape), (1, 4, 5))
        call = model.gpt_head.calls[0]
        self.assertIsNone(call["labels"])
        self.assertTrue(call["autoregressive"])
        self.assertFalse(call["training"])

    def test_rmt_encoder_only_adapter_uses_recurrent_content_mask(self):
        model = RMTGPTSegmentationModel.__new__(RMTGPTSegmentationModel)
        nn.Module.__init__(model)
        recurrent_hidden = torch.tensor(
            [[[90.0, 90.0], [1.0, 10.0], [91.0, 91.0], [2.0, 20.0]]]
        )
        recurrent_mask = torch.tensor([[False, True, False, True]])
        model.rmt_encoder = FakeRMTEncoder(recurrent_hidden, recurrent_mask)
        model.hidden_size = 2
        model.special_token_ids = (0, 10, 11)
        model.nucleotide_embedding = nn.Embedding(32, 2)
        _set_scalar_embedding(model.nucleotide_embedding)
        model.gpt_head = RecordingGPTHead()
        input_ids = torch.tensor([[10, 4, 5, 11, 0]])
        original_mask = torch.tensor([[False, True, True, False, False]])

        model(
            input_ids=input_ids,
            attention_mask=torch.tensor([[1, 1, 1, 1, 0]]),
            labels_mask=original_mask,
            embedding_repeater=torch.tensor([[0, 1, -100]]),
            letter_level_tokens=torch.tensor([[6, 7, 0]]),
            letter_level_attention_mask=torch.tensor([[1, 1, 0]]),
            letter_level_labels=torch.zeros((1, 3, 5)),
            letter_level_labels_mask=torch.tensor([[True, True, False]]),
        )

        passed_mask = model.rmt_encoder.calls[0][1]["labels_mask"]
        self.assertTrue(torch.equal(passed_mask, original_mask))
        head_call = model.gpt_head.calls[0]
        self.assertTrue(
            torch.allclose(
                head_call["encoder_embeddings"][0, 1],
                torch.tensor([7.0, 7.25, 2.0, 20.0]),
            )
        )
        self.assertFalse(bool((head_call["encoder_embeddings"] == 90.0).any()))
        self.assertFalse(bool((head_call["encoder_embeddings"] == 91.0).any()))

    def test_amt_encoder_only_adapter_uses_hidden_states_not_classifier_logits(self):
        model = AMTGPTSegmentationModel.__new__(AMTGPTSegmentationModel)
        nn.Module.__init__(model)
        hidden = torch.tensor(
            [[[99.0, 99.0], [1.0, 10.0], [2.0, 20.0], [98.0, 98.0], [0.0, 0.0]]]
        )
        model.amt_encoder = FakeAMTEncoder(hidden)
        model.hidden_size = 2
        model.special_token_ids = (0, 10, 11)
        model.nucleotide_embedding = nn.Embedding(32, 2)
        _set_scalar_embedding(model.nucleotide_embedding)
        model.gpt_head = RecordingGPTHead()
        attention = torch.tensor([[1, 1, 1, 1, 0]])

        model(
            input_ids=torch.tensor([[10, 4, 5, 11, 0]]),
            attention_mask=attention,
            labels_mask=torch.tensor([[False, True, True, False, False]]),
            embedding_repeater=torch.tensor([[0, 1, -100]]),
            letter_level_tokens=torch.tensor([[6, 7, 0]]),
            letter_level_attention_mask=torch.tensor([[1, 1, 0]]),
            letter_level_labels_mask=torch.tensor([[True, True, False]]),
        )

        self.assertTrue(
            torch.equal(model.amt_encoder.calls[0]["attention_mask"], attention)
        )
        self.assertTrue(
            torch.allclose(
                model.gpt_head.calls[0]["encoder_embeddings"][0, 1],
                torch.tensor([7.0, 7.25, 2.0, 20.0]),
            )
        )

    def test_caduceus_tuple_adapter_masks_specials_before_gpt(self):
        model = CaduceusGPTSegmentationModel.__new__(CaduceusGPTSegmentationModel)
        nn.Module.__init__(model)
        hidden = torch.tensor(
            [[[0.0, 0.0], [90.0, 90.0], [1.0, 10.0], [2.0, 20.0], [91.0, 91.0]]]
        )
        model.caduceus_model = FakeCaduceusEncoder(hidden, tuple_output=True)
        model.hidden_size = 2
        model.special_token_ids = (0, 10, 11)
        model.nucleotide_embedding = nn.Embedding(32, 2)
        _set_scalar_embedding(model.nucleotide_embedding)
        model.gpt_head = RecordingGPTHead()
        input_ids = torch.tensor([[0, 10, 4, 5, 11]])
        attention = torch.tensor([[0, 1, 1, 1, 1]])
        expected_mask = torch.tensor([[False, False, True, True, False]])

        model(
            input_ids=input_ids,
            attention_mask=attention,
            letter_level_labels=torch.zeros((1, 5, 5)),
            letter_level_labels_mask=expected_mask,
        )

        self.assertEqual(model.caduceus_model.calls[0], {"input_ids": input_ids})
        head_call = model.gpt_head.calls[0]
        self.assertTrue(torch.equal(head_call["nucleotide_mask"], expected_mask))
        self.assertTrue(
            torch.allclose(
                head_call["encoder_embeddings"][0, 2],
                torch.tensor([4.0, 4.25, 1.0, 10.0]),
            )
        )

    @unittest.skipIf(T5GemmaDecoder is None, "transformers>=4.53 with T5Gemma is not installed")
    def test_direct_adapter_runs_tiny_official_head_train_and_generate(self):
        hidden = torch.tensor(
            [[[99.0, 99.0], [1.0, 10.0], [2.0, 20.0], [98.0, 98.0], [0.0, 0.0]]]
        )
        model = _new_direct_model(hidden)
        model.gpt_head = T5GemmaSegmentationHead(
            encoder_dim=4,
            num_labels=5,
            decoder_hidden_size=16,
            decoder_intermediate_size=32,
            num_decoder_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            context_size=4,
            encoder_lookahead=4,
        )
        inputs = {
            "input_ids": torch.tensor([[10, 4, 5, 11, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
            "labels_mask": torch.tensor([[False, True, True, False, False]]),
            "embedding_repeater": torch.tensor([[0, 0, 1, -100]]),
            "letter_level_tokens": torch.tensor([[6, 7, 8, 0]]),
            "letter_level_attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "letter_level_labels": _five_track_labels([[0, 1, 0, 0]]),
            "letter_level_labels_mask": torch.tensor([[True, True, True, False]]),
        }

        model.train()
        trained = model(**inputs)
        self.assertTrue(torch.isfinite(trained.loss))
        self.assertEqual(tuple(trained.logits.shape), (1, 4, 5))
        model.eval()
        with self.assertRaisesRegex(RuntimeError, "cannot consume labels"):
            model(**inputs)
        generated = model.generate(**inputs)
        self.assertTrue(torch.isfinite(generated).all())
        self.assertEqual(tuple(generated.shape), (1, 4, 5))


if __name__ == "__main__":
    unittest.main()
