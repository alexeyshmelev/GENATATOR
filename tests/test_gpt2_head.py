from __future__ import annotations

import types
import unittest
import warnings
from unittest.mock import patch

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import GPT2Model
    from transformers.cache_utils import DynamicCache, EncoderDecoderCache
    from genatator_core.gpt_head import (
        EXON_LABEL_INDEX,
        INTRON_LABEL_INDEX,
        GPT2SegmentationHead,
        T5GemmaSegmentationHead,
        build_segmentation_head,
    )
    from genatator_core.gpt_models import (
        AMTGPTSegmentationModel,
        CaduceusGPTSegmentationModel,
        GenaModernGPTSegmentationModel,
        RMTGPTSegmentationModel,
    )
except ImportError:
    torch = None


def _config(**overrides):
    config = dict(
        decoder_hidden_size=8,
        decoder_intermediate_size=16,
        num_decoder_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        context_size=3,
        encoder_lookahead=2,
        max_position_embeddings=16,
        add_encoder_to_decoder_input=True,
        multi_token_prediction=3,
    )
    config.update(overrides)
    return config


def _head(**overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return GPT2SegmentationHead(encoder_dim=8, **_config(**overrides))


def _labels(classes):
    labels = torch.zeros((*classes.shape, 5))
    labels[..., INTRON_LABEL_INDEX] = classes.eq(0)
    labels[..., EXON_LABEL_INDEX] = classes.eq(1)
    return labels


@unittest.skipIf(torch is None, "torch/transformers are not installed")
class GPT2HeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def test_stock_decoder_and_position_limit_fail_before_decoding(self):
        head = _head(max_position_embeddings=7)
        self.assertIs(type(head.decoder), GPT2Model)
        self.assertTrue(head.decoder.config.add_cross_attention)
        self.assertEqual(tuple(head.decoder.wpe.weight.shape), (7, 8))
        self.assertEqual(head.decoder.config._attn_implementation, "sdpa")
        with patch.object(head.decoder, "forward", wraps=head.decoder.forward) as forward:
            with self.assertRaisesRegex(RuntimeError, "refusing to wrap positions"):
                head.generate(torch.randn(1, 8, 8), nucleotide_mask=torch.ones(1, 8))
            forward.assert_not_called()
        self.assertEqual(
            head.generate(torch.randn(1, 7, 8), nucleotide_mask=torch.ones(1, 7)).shape,
            (1, 7, 5),
        )

    def test_factory_default_and_incompatible_settings(self):
        config = _config()
        config.pop("max_position_embeddings")
        self.assertIs(type(build_segmentation_head(8, **config)), T5GemmaSegmentationHead)
        with self.assertRaisesRegex(RuntimeError, "explicit positive"):
            build_segmentation_head(8, decoder_type="gpt2", **config)
        with self.assertRaisesRegex(RuntimeError, "Unsupported GPT decoder_type"):
            build_segmentation_head(8, decoder_type="other", **config)
        with self.assertRaisesRegex(RuntimeError, "must be >= context_size"):
            _head(max_position_embeddings=2)
        with self.assertRaisesRegex(RuntimeError, "num_key_value_heads =="):
            _head(num_key_value_heads=1)

    def test_long_position_and_native_allocation_warnings_precede_construction(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with patch("transformers.GPT2Model", side_effect=RuntimeError("construction stopped")):
                with self.assertRaisesRegex(RuntimeError, "construction stopped"):
                    GPT2SegmentationHead(8, **_config(max_position_embeddings=32768))
        messages = [str(item.message) for item in caught]
        self.assertTrue(any("receive no teacher-forcing signal" in message for message in messages))
        self.assertTrue(any("4.00 GiB" in message for message in messages))

    @torch.no_grad() if torch is not None else lambda function: function
    def test_rolling_generation_matches_full_cache_with_sliding_attention_reference(self):
        # Independent reference keeps ALL self K/V and masks old entries.  It
        # re-projects encoder K/V every step, catching stale-window cache errors.
        for context_size in (1, 3):
            with self.subTest(context_size=context_size):
                head = _head(context_size=context_size).eval()
                encoder = torch.randn(1, 11, 8)
                actual = head._generate_single(encoder)
                self_cache = DynamicCache()
                previous = None
                expected = []
                for position in range(encoder.shape[1]):
                    if previous is None:
                        inputs = head.decoder.wte(torch.tensor([[2]]))
                    else:
                        inputs = head.label_embedding(previous) + encoder[:, position - 1:position]
                    start = min(
                        max(0, position - context_size + 1),
                        max(0, encoder.shape[1] - context_size - 2),
                    )
                    window = encoder[:, start:start + context_size + 2]
                    attention_mask = torch.ones(1, position + 1)
                    attention_mask[:, :max(0, position - context_size + 1)] = 0
                    output = head.decoder(
                        inputs_embeds=inputs,
                        position_ids=torch.tensor([[position]]),
                        cache_position=torch.tensor([position]),
                        attention_mask=attention_mask,
                        encoder_hidden_states=window,
                        past_key_values=EncoderDecoderCache(self_cache, DynamicCache()),
                        use_cache=True,
                    )
                    self_cache = output.past_key_values.self_attention_cache
                    logits = head.classifier(output.last_hidden_state)
                    expected.append(logits)
                    previous = logits.argmax(-1)
                torch.testing.assert_close(actual, torch.cat(expected, dim=1), atol=1e-6, rtol=1e-5)

    def test_absolute_positions_feedback_bounded_history_and_sample_isolation(self):
        head = _head().eval()
        encoder = torch.randn(1, 11, 8)
        calls = []

        def record(module, args, kwargs):
            calls.append({
                "position": kwargs["position_ids"].item(),
                "history": kwargs["past_key_values"].self_attention_cache.get_seq_length(),
                "inputs": kwargs["inputs_embeds"].detach().clone(),
                "encoder": kwargs["encoder_hidden_states"].detach().clone(),
            })

        hook = head.decoder.register_forward_pre_hook(record, with_kwargs=True)
        actual = head.generate(encoder, nucleotide_mask=torch.ones(1, 11))
        hook.remove()
        self.assertEqual([call["position"] for call in calls], list(range(11)))
        self.assertEqual([call["history"] for call in calls], [0, 1] + [2] * 9)
        torch.testing.assert_close(calls[0]["inputs"], head.decoder.wte(torch.tensor([[2]])))
        for position in range(1, 11):
            previous = actual[:, position - 1:position, [INTRON_LABEL_INDEX, EXON_LABEL_INDEX]].argmax(-1)
            expected = head.label_embedding(previous) + encoder[:, position - 1:position]
            torch.testing.assert_close(calls[position]["inputs"], expected)
        torch.testing.assert_close(calls[3]["encoder"], encoder[:, 1:6])
        torch.testing.assert_close(calls[-1]["encoder"], encoder[:, -5:])
        # Neither auxiliary MTP heads nor a preceding different sample can
        # affect offset-1 generation.
        with torch.no_grad():
            for classifier in head.classifiers[1:]:
                classifier.weight.fill_(1000)
                classifier.bias.fill_(-1000)
        head.generate(torch.randn_like(encoder), nucleotide_mask=torch.ones(1, 11))
        torch.testing.assert_close(actual, head.generate(encoder, nucleotide_mask=torch.ones(1, 11)))

    def test_mtp_matches_manual_shifted_ce_and_all_heads_receive_gradients(self):
        head = _head(multi_token_prediction=10).train()
        encoder = torch.randn(1, 7, 8, requires_grad=True)
        targets = torch.tensor([[0, 1, 1, 0, 1, 0, 1]])
        valid = torch.tensor([[True, True, False, True, True, True, True]])
        offset_logits = head._teacher_forced_single(encoder, targets)
        numerator = encoder.new_zeros(())
        count = 0
        for offset, logits in enumerate(offset_logits):
            if offset >= targets.shape[1]:
                continue
            mask = valid[:, offset:]
            numerator = numerator + F.cross_entropy(
                logits[:, :targets.shape[1] - offset][mask], targets[:, offset:][mask], reduction="sum"
            )
            count += int(mask.sum())
        loss, logits = head(
            encoder, nucleotide_mask=torch.ones(1, 7), labels=_labels(targets), labels_mask=valid,
        )
        torch.testing.assert_close(loss, numerator / count)
        torch.testing.assert_close(logits[..., INTRON_LABEL_INDEX], offset_logits[0][..., 0])
        loss.backward()
        self.assertIsNotNone(encoder.grad)
        self.assertGreater(head.decoder.h[0].crossattention.c_attn.weight.grad.abs().sum(), 0)
        for offset, classifier in enumerate(head.classifiers):
            self.assertIsNotNone(classifier.weight.grad)
            if offset < targets.shape[1]:
                self.assertGreater(classifier.weight.grad.abs().sum(), 0)
            else:
                self.assertEqual(classifier.weight.grad.abs().sum(), 0)
        # Preserve the existing chunk-local teacher-forcing position policy.
        self.assertEqual(head.decoder.wpe.weight.grad[head.context_size:].abs().sum(), 0)

    def test_all_encoder_adapters_select_gpt2_without_changing_the_encoder(self):
        tokenizer = types.SimpleNamespace(pad_token_id=0, cls_token_id=10, sep_token_id=11)
        config = _config(decoder_type="gpt2", max_position_embeddings=3)

        class Encoder(nn.Module):
            hidden_size = 4

        encoder = Encoder()
        with patch("genatator_core.gpt_models.HiddenStateBackbone", return_value=encoder):
            direct = GenaModernGPTSegmentationModel("test", "moderngena", tokenizer, nucleotide_vocab_size=16, gpt_config=config)
        with patch("genatator_core.gpt_models.AMTTokenClassifier", return_value=encoder):
            amt = AMTGPTSegmentationModel("test", "moderngena", tokenizer, nucleotide_vocab_size=16, amt_config={}, gpt_config=config)
        with patch("genatator_core.gpt_models.RMTEncoderForLetterLevelTokenClassificationUNETsegmentedRepeater", return_value=encoder):
            rmt = RMTGPTSegmentationModel(encoder, tokenizer, nucleotide_vocab_size=16, rmt_config={}, gpt_config=config)
        caduceus = CaduceusGPTSegmentationModel(encoder, 4, tokenizer, nucleotide_vocab_size=16, gpt_config=config)
        self.assertIs(direct.hidden_backbone, encoder)
        self.assertIs(amt.amt_encoder, encoder)
        self.assertIs(rmt.rmt_encoder, encoder)
        self.assertIs(caduceus.caduceus_model, encoder)
        for model in (direct, amt, rmt, caduceus):
            self.assertIs(type(model.gpt_head), GPT2SegmentationHead)

    def test_direct_adapter_trains_and_generates_with_stock_gpt2(self):
        class Encoder(nn.Module):
            hidden_size = 4

            def forward_chunked(self, **kwargs):
                return types.SimpleNamespace(logits=torch.arange(20).float().reshape(1, 5, 4))

        tokenizer = types.SimpleNamespace(pad_token_id=0, cls_token_id=10, sep_token_id=11)
        with patch("genatator_core.gpt_models.HiddenStateBackbone", return_value=Encoder()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                model = GenaModernGPTSegmentationModel(
                    "test", "moderngena", tokenizer, nucleotide_vocab_size=16,
                    gpt_config=_config(decoder_type="gpt2"),
                )
        inputs = dict(
            input_ids=torch.tensor([[10, 4, 5, 11, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 1, 0]]),
            embedding_repeater=torch.tensor([[0, 0, 0, 1, 1, 1, 1]]),
            letter_level_tokens=torch.tensor([[4, 4, 4, 5, 5, 5, 5]]),
            letter_level_attention_mask=torch.ones(1, 7),
            letter_level_labels=_labels(torch.tensor([[0, 1, 0, 0, 1, 1, 0]])),
        )
        output = model(**inputs)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(model.gpt_head.decoder.h[0].crossattention.c_attn.weight.grad)
        model.eval()
        with self.assertRaisesRegex(RuntimeError, "cannot consume labels"):
            model(**inputs)
        generated = model.generate(**inputs)
        inputs["letter_level_labels"] = 1 - inputs["letter_level_labels"]
        torch.testing.assert_close(generated, model.generate(**inputs))
        self.assertEqual(generated.shape, (1, 7, 5))
        self.assertTrue(torch.isfinite(generated).all())


if __name__ == "__main__":
    unittest.main()
