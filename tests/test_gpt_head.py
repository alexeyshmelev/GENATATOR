from __future__ import annotations

import types
import unittest

try:
    import torch
    import torch.nn as nn

    from genatator_core.gpt_head import T5GemmaDecoder, T5GemmaSegmentationHead
except ImportError:
    torch = None
    nn = None
    T5GemmaDecoder = None


if nn is not None:
    class RecordingDecoder(nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.config = types.SimpleNamespace(hidden_size=hidden_size, bos_token_id=2)
            self.embed_tokens = nn.Embedding(3, hidden_size)
            self.calls = []

        def forward(
            self,
            *,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values=None,
            use_cache=False,
            **kwargs,
        ):
            self.calls.append(
                {
                    "inputs_embeds": inputs_embeds.detach().clone(),
                    "encoder_shape": tuple(encoder_hidden_states.shape),
                    "cross_mask": encoder_attention_mask["full_attention"].detach().clone(),
                    "past": past_key_values,
                    "use_cache": use_cache,
                }
            )
            # Keep gradients connected to both decoder and encoder inputs.
            hidden = inputs_embeds + encoder_hidden_states.mean(dim=1, keepdim=True)
            cache = past_key_values if past_key_values is not None else object()
            return types.SimpleNamespace(last_hidden_state=hidden, past_key_values=cache)


    def five_track_labels(class_ids):
        """Build legacy labels with categorical exon=0/intron=1 targets."""

        labels = torch.zeros((*class_ids.shape, 5), dtype=torch.float32)
        labels[..., 1] = class_ids.eq(0).float()
        labels[..., 2] = class_ids.eq(1).float()
        # UTR/CDS tracks may overlap exon but are deliberately ignored by GPT.
        labels[..., 0] = labels[..., 1]
        labels[..., 4] = labels[..., 1]
        return labels


@unittest.skipIf(torch is None, "torch is not installed")
class T5GemmaSegmentationHeadTests(unittest.TestCase):
    def _head(self, encoder_dim=2, hidden_size=4, context=3, lookahead=3):
        decoder = RecordingDecoder(hidden_size)
        head = T5GemmaSegmentationHead(
            encoder_dim=encoder_dim,
            num_labels=2,
            decoder_hidden_size=hidden_size,
            decoder_intermediate_size=8,
            num_decoder_layers=2,
            num_attention_heads=2,
            context_size=context,
            encoder_lookahead=lookahead,
            decoder=decoder,
        )
        return head, decoder

    def test_teacher_forcing_is_samplewise_unpadded_and_chunked(self):
        torch.manual_seed(0)
        head, decoder = self._head()
        head.train()
        encoder = torch.randn(2, 7, 2, requires_grad=True)
        nucleotide_mask = torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0]],
            dtype=torch.bool,
        )
        target_ids = torch.randint(0, 2, (2, 7))
        labels = five_track_labels(target_ids)

        loss, logits = head(
            encoder,
            nucleotide_mask=nucleotide_mask,
            labels=labels,
            labels_mask=nucleotide_mask,
        )

        self.assertIsNotNone(loss)
        self.assertEqual(tuple(logits.shape), (2, 7, 2))
        self.assertTrue(bool((logits[~nucleotide_mask] == 0).all()))
        expected_loss = torch.nn.functional.cross_entropy(
            logits[nucleotide_mask].float(), target_ids[nucleotide_mask]
        )
        self.assertTrue(torch.allclose(loss, expected_loss))
        self.assertEqual(
            [(call["inputs_embeds"].shape[1], call["encoder_shape"][1]) for call in decoder.calls],
            [(3, 4), (1, 1), (3, 5), (2, 2)],
        )
        self.assertTrue(all(not call["use_cache"] for call in decoder.calls))

        # Every independent chunk starts with T5Gemma's BOS embedding.
        bos = head.decoder.embed_tokens(torch.tensor([[head.bos_token_id]]))[0, 0]
        for call in decoder.calls:
            self.assertTrue(torch.allclose(call["inputs_embeds"][0, 0], bos))

        loss.backward()
        self.assertGreater(float(encoder.grad.abs().sum()), 0.0)

    def test_teacher_forcing_shifts_categorical_target_ids(self):
        head, decoder = self._head(context=4, lookahead=4)
        head.train()
        encoder = torch.randn(1, 3, 2)
        target_ids = torch.tensor([[0, 1, 0]])
        labels = five_track_labels(target_ids)
        mask = torch.ones((1, 3), dtype=torch.bool)

        head(encoder, nucleotide_mask=mask, labels=labels, labels_mask=mask)

        seen = decoder.calls[0]["inputs_embeds"][:, 1:, :]
        expected = head.target_embedding(target_ids[:, :-1]).detach()
        self.assertTrue(torch.allclose(seen, expected))

    def test_generate_uses_cache_and_shifts_cross_attention_one_base_at_a_time(self):
        head, decoder = self._head(context=3, lookahead=3)
        with torch.no_grad():
            head.classifier.weight.zero_()
            head.classifier.bias.copy_(torch.tensor([10.0, -10.0]))
        head.train()  # generate must temporarily switch to eval and restore this.
        encoder = torch.randn(1, 7, 2)
        mask = torch.ones((1, 7), dtype=torch.bool)

        logits = head.generate(encoder, nucleotide_mask=mask)

        self.assertTrue(head.training)
        self.assertEqual(tuple(logits.shape), (1, 7, 2))
        self.assertEqual(len(decoder.calls), 7)
        self.assertTrue(all(call["inputs_embeds"].shape[1] == 1 for call in decoder.calls))
        self.assertIsNone(decoder.calls[0]["past"])
        self.assertTrue(all(call["past"] is not None for call in decoder.calls[1:]))
        self.assertTrue(all(call["use_cache"] for call in decoder.calls))

        allowed = []
        for call in decoder.calls:
            row = call["cross_mask"][0, 0, 0]
            allowed.append(torch.nonzero(row == 0, as_tuple=False).flatten().tolist())
        self.assertEqual(
            allowed,
            [
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5, 6],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6],
                [4, 5, 6],
            ],
        )

        expected_prediction = torch.tensor([[0]])
        expected_second_input = head.target_embedding(expected_prediction).detach()
        self.assertTrue(
            torch.allclose(decoder.calls[1]["inputs_embeds"], expected_second_input)
        )

    def test_eval_forward_generates_without_ground_truth_history(self):
        head, decoder = self._head(context=3, lookahead=3)
        head.eval()
        encoder = torch.randn(1, 4, 2)
        mask = torch.ones((1, 4), dtype=torch.bool)
        labels = five_track_labels(torch.randint(0, 2, (1, 4)))

        loss, logits = head(
            encoder,
            nucleotide_mask=mask,
            labels=labels,
            labels_mask=mask,
        )

        self.assertIsNotNone(loss)
        self.assertEqual(tuple(logits.shape), (1, 4, 2))
        self.assertEqual(len(decoder.calls), 4)
        self.assertTrue(all(call["use_cache"] for call in decoder.calls))

    def test_projection_is_learned_only_when_dimensions_differ(self):
        projected, _ = self._head(encoder_dim=2, hidden_size=4)
        identity, _ = self._head(encoder_dim=4, hidden_size=4)
        self.assertIsInstance(projected.encoder_projection, nn.Linear)
        self.assertIsInstance(identity.encoder_projection, nn.Identity)

    def test_target_vocabulary_is_exactly_two_embedding_rows(self):
        head, _ = self._head()
        self.assertIsInstance(head.target_embedding, nn.Embedding)
        self.assertEqual(head.target_embedding.num_embeddings, 2)
        decoder = RecordingDecoder(4)
        with self.assertRaisesRegex(RuntimeError, "exactly two target classes"):
            T5GemmaSegmentationHead(
                encoder_dim=2,
                num_labels=5,
                decoder_hidden_size=4,
                num_decoder_layers=2,
                num_attention_heads=2,
                context_size=3,
                encoder_lookahead=3,
                decoder=decoder,
            )

    def test_nonexclusive_exon_intron_targets_are_rejected(self):
        head, _ = self._head()
        encoder = torch.randn(1, 2, 2)
        mask = torch.ones((1, 2), dtype=torch.bool)
        labels = torch.zeros((1, 2, 5))
        labels[0, 0, 1:3] = torch.tensor([1.0, 1.0])
        labels[0, 1, 1:3] = torch.tensor([0.0, 0.0])
        with self.assertRaisesRegex(RuntimeError, "exactly one exon/intron"):
            head(encoder, nucleotide_mask=mask, labels=labels, labels_mask=mask)

    def test_bce_pos_weight_is_safely_ignored_for_categorical_loss(self):
        head, _ = self._head()
        encoder = torch.randn(1, 2, 2)
        mask = torch.ones((1, 2), dtype=torch.bool)
        labels = five_track_labels(torch.tensor([[0, 1]]))
        loss, _ = head(
            encoder,
            nucleotide_mask=mask,
            labels=labels,
            labels_mask=mask,
            pos_weight=torch.ones((9, 17, 23, 31)),
        )
        self.assertTrue(torch.isfinite(loss))

    def test_decoder_depth_is_restricted_to_two_through_four(self):
        decoder = RecordingDecoder(4)
        with self.assertRaisesRegex(RuntimeError, "must be in \\[2, 4\\]"):
            T5GemmaSegmentationHead(
                encoder_dim=2,
                decoder_hidden_size=4,
                num_decoder_layers=5,
                num_attention_heads=2,
                context_size=3,
                encoder_lookahead=3,
                decoder=decoder,
            )

    @unittest.skipIf(T5GemmaDecoder is None, "transformers>=4.53 with T5Gemma is not installed")
    def test_official_tiny_decoder_runs_without_downloading_weights(self):
        head = T5GemmaSegmentationHead(
            encoder_dim=8,
            num_labels=2,
            decoder_hidden_size=16,
            decoder_intermediate_size=32,
            num_decoder_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            context_size=4,
            encoder_lookahead=4,
        )
        encoder = torch.randn(1, 3, 8)
        labels = five_track_labels(torch.randint(0, 2, (1, 3)))
        mask = torch.ones((1, 3), dtype=torch.bool)
        head.train()
        loss, logits = head(
            encoder,
            nucleotide_mask=mask,
            labels=labels,
            labels_mask=mask,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(logits.shape), (1, 3, 2))
        head.eval()
        generated_logits = head.generate(encoder, nucleotide_mask=mask)
        self.assertTrue(torch.isfinite(generated_logits).all())
        self.assertEqual(tuple(generated_logits.shape), (1, 3, 2))


if __name__ == "__main__":
    unittest.main()
