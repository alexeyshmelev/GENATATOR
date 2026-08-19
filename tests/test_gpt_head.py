from __future__ import annotations

import types
import unittest

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from genatator_core.gpt_head import (
        EXON_CLASS_ID,
        EXON_LABEL_INDEX,
        INTRON_CLASS_ID,
        INTRON_LABEL_INDEX,
        T5GemmaDecoder,
        T5GemmaSegmentationHead,
        UNAVAILABLE_TRACK_LOGIT,
    )
except ImportError:
    torch = None
    nn = None
    F = None
    T5GemmaDecoder = None


if nn is not None:
    class FakeCrossAttention(nn.Module):
        def __init__(self, hidden_size: int, num_key_value_heads: int, head_dim: int):
            super().__init__()
            self.head_dim = int(head_dim)
            output_size = int(num_key_value_heads) * self.head_dim
            self.k_proj = nn.Linear(hidden_size, output_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, output_size, bias=False)


    class FakeDecoderLayer(nn.Module):
        def __init__(self, hidden_size: int, num_key_value_heads: int, head_dim: int):
            super().__init__()
            self.cross_attn = FakeCrossAttention(
                hidden_size,
                num_key_value_heads,
                head_dim,
            )


    class RecordingDecoder(nn.Module):
        def __init__(self, hidden_size: int, context_size: int = 3):
            super().__init__()
            self.config = types.SimpleNamespace(
                hidden_size=hidden_size,
                bos_token_id=2,
                sliding_window=context_size,
                max_position_embeddings=context_size,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=hidden_size // 2,
            )
            self.embed_tokens = nn.Embedding(3, hidden_size)
            self.layers = nn.ModuleList(
                FakeDecoderLayer(
                    hidden_size,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                )
                for _ in range(self.config.num_hidden_layers)
            )
            self.calls = []

        def forward(
            self,
            *,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            attention_mask=None,
            position_ids=None,
            cache_position=None,
            past_key_values=None,
            use_cache=False,
            **kwargs,
        ):
            cross_mask = encoder_attention_mask.get("full_attention")
            sliding_mask = (
                None
                if attention_mask is None
                else attention_mask.get("sliding_attention")
            )
            self.calls.append(
                {
                    "inputs_embeds": inputs_embeds.detach().clone(),
                    "encoder_values": encoder_hidden_states.detach().clone(),
                    "encoder_shape": tuple(encoder_hidden_states.shape),
                    "cross_mask": None
                    if cross_mask is None
                    else cross_mask.detach().clone(),
                    "self_mask": None
                    if sliding_mask is None
                    else sliding_mask.detach().clone(),
                    "past": past_key_values,
                    "self_cache_capacity": None
                    if past_key_values is None
                    else int(
                        past_key_values.self_attention_cache.key_cache[0].shape[-2]
                    ),
                    "cross_cache_id": None
                    if past_key_values is None
                    else id(past_key_values.cross_attention_cache),
                    "cache_position": None
                    if cache_position is None
                    else cache_position.detach().clone(),
                    "use_cache": use_cache,
                }
            )
            if use_cache and past_key_values is not None:
                self_cache = past_key_values.self_attention_cache
                for layer_index in range(self.config.num_hidden_layers):
                    shape = (1, self.config.num_key_value_heads, 1, self.config.head_dim)
                    key = torch.ones(shape, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                    self_cache.update(
                        key,
                        key,
                        layer_index,
                        {"cache_position": cache_position},
                    )
                    if not past_key_values.is_updated.get(layer_index):
                        cross_length = int(encoder_hidden_states.shape[1])
                        cross_shape = (
                            1,
                            self.config.num_key_value_heads,
                            cross_length,
                            self.config.head_dim,
                        )
                        cross_key = torch.ones(
                            cross_shape,
                            device=inputs_embeds.device,
                            dtype=inputs_embeds.dtype,
                        )
                        past_key_values.cross_attention_cache.update(
                            cross_key,
                            cross_key,
                            layer_index,
                        )
                        past_key_values.is_updated[layer_index] = True
                self.calls[-1]["cross_cache_length_after"] = int(
                    past_key_values.cross_attention_cache.key_cache[0].shape[-2]
                )
            # Keep gradients connected to both decoder and encoder inputs.
            hidden = inputs_embeds + encoder_hidden_states.mean(dim=1, keepdim=True)
            return types.SimpleNamespace(
                last_hidden_state=hidden,
                past_key_values=past_key_values,
            )


def _five_track_labels(classes):
    classes = torch.as_tensor(classes, dtype=torch.long)
    labels = torch.zeros((*classes.shape, 5), dtype=torch.float32)
    labels[..., INTRON_LABEL_INDEX] = classes.eq(INTRON_CLASS_ID).float()
    labels[..., EXON_LABEL_INDEX] = classes.eq(EXON_CLASS_ID).float()
    return labels


@unittest.skipIf(torch is None, "torch is not installed")
class T5GemmaSegmentationHeadTests(unittest.TestCase):
    def _head(
        self,
        encoder_dim=2,
        hidden_size=4,
        context=3,
        lookahead=3,
        multi_token_prediction=1,
        add_encoder_to_decoder_input=False,
    ):
        decoder = RecordingDecoder(hidden_size, context)
        head = T5GemmaSegmentationHead(
            encoder_dim=encoder_dim,
            num_labels=5,
            decoder_hidden_size=hidden_size,
            decoder_intermediate_size=8,
            num_decoder_layers=2,
            num_attention_heads=2,
            context_size=context,
            encoder_lookahead=lookahead,
            multi_token_prediction=multi_token_prediction,
            add_encoder_to_decoder_input=add_encoder_to_decoder_input,
            decoder=decoder,
        )
        return head, decoder

    def test_teacher_forcing_is_samplewise_unpadded_chunked_and_five_track_compatible(self):
        torch.manual_seed(0)
        head, decoder = self._head()
        head.train()
        encoder = torch.randn(2, 7, 2, requires_grad=True)
        nucleotide_mask = torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0]],
            dtype=torch.bool,
        )
        labels = _five_track_labels(
            [[0, 1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0, 0]]
        )

        loss, logits = head(
            encoder,
            nucleotide_mask=nucleotide_mask,
            labels=labels,
            labels_mask=nucleotide_mask,
        )

        self.assertIsNotNone(loss)
        self.assertEqual(tuple(logits.shape), (2, 7, 5))
        self.assertTrue(bool((logits[~nucleotide_mask] == 0).all()))
        for index in (0, 3, 4):
            self.assertTrue(
                bool(
                    (
                        logits[..., index][nucleotide_mask]
                        == UNAVAILABLE_TRACK_LOGIT
                    ).all()
                )
            )
        self.assertEqual(
            [(call["inputs_embeds"].shape[1], call["encoder_shape"][1]) for call in decoder.calls],
            [(3, 4), (1, 1), (3, 5), (2, 2)],
        )
        self.assertTrue(all(not call["use_cache"] for call in decoder.calls))
        self.assertTrue(all(call["self_mask"] is None for call in decoder.calls))
        self.assertTrue(all(call["cross_mask"] is None for call in decoder.calls))

        bos = head.decoder.embed_tokens(torch.tensor([[head.bos_token_id]]))[0, 0]
        for call in decoder.calls:
            self.assertTrue(torch.allclose(call["inputs_embeds"][0, 0], bos))

        loss.backward()
        self.assertGreater(float(encoder.grad.abs().sum()), 0.0)

    def test_teacher_forcing_shifts_categorical_target_ids(self):
        head, decoder = self._head(context=4, lookahead=4)
        encoder = torch.randn(1, 3, 2)
        classes = torch.tensor([[EXON_CLASS_ID, INTRON_CLASS_ID, EXON_CLASS_ID]])
        labels = _five_track_labels(classes)
        mask = torch.ones((1, 3), dtype=torch.bool)

        head(encoder, nucleotide_mask=mask, labels=labels, labels_mask=mask)

        seen = decoder.calls[0]["inputs_embeds"][:, 1:, :]
        expected = head.label_embedding(classes[:, :-1]).detach()
        self.assertTrue(torch.allclose(seen, expected))

    def test_teacher_forcing_can_add_aligned_encoder_states_to_decoder_inputs(self):
        head, decoder = self._head(
            encoder_dim=4,
            hidden_size=4,
            context=3,
            lookahead=2,
            add_encoder_to_decoder_input=True,
        )
        encoder = torch.arange(20, dtype=torch.float32).view(1, 5, 4)
        classes = torch.tensor([[0, 1, 0, 1, 1]])
        labels = _five_track_labels(classes)
        mask = torch.ones((1, 5), dtype=torch.bool)

        head(encoder, nucleotide_mask=mask, labels=labels, labels_mask=mask)

        projected = head.encoder_projection(encoder).detach()
        bos = head._bos_embedding(device=encoder.device).detach()
        self.assertEqual(len(decoder.calls), 2)
        for call, (start, end) in zip(decoder.calls, ((0, 3), (3, 5))):
            previous_targets = head.label_embedding(
                classes[:, start : end - 1]
            ).detach()
            expected = torch.cat(
                (bos, previous_targets + projected[:, start : end - 1, :]),
                dim=1,
            )
            self.assertTrue(torch.allclose(call["inputs_embeds"], expected))

    def test_eval_forward_with_labels_rejects_teacher_forcing(self):
        head, decoder = self._head(context=8, lookahead=3)
        head.eval()
        encoder = torch.randn(1, 4, 2)
        mask = torch.ones((1, 4), dtype=torch.bool)
        labels = _five_track_labels([[0, 1, 1, 0]])

        with self.assertRaisesRegex(RuntimeError, "cannot consume labels"):
            head(
                encoder,
                nucleotide_mask=mask,
                labels=labels,
                labels_mask=mask,
            )

        self.assertEqual(decoder.calls, [])

    def test_multi_token_loss_masks_each_offset_at_the_right_edge(self):
        head, _ = self._head(
            context=3,
            lookahead=3,
            multi_token_prediction=3,
        )
        with torch.no_grad():
            for index, classifier in enumerate(head.classifiers):
                classifier.weight.zero_()
                classifier.bias.copy_(torch.tensor([float(index), float(index) + 0.75]))

        encoder = torch.randn(1, 5, 2)
        classes = torch.tensor([[0, 1, 0, 1, 1]])
        labels = _five_track_labels(classes)
        nucleotide_mask = torch.ones((1, 5), dtype=torch.bool)
        labels_mask = torch.tensor([[True, False, True, True, True]])

        loss, logits = head(
            encoder,
            nucleotide_mask=nucleotide_mask,
            labels=labels,
            labels_mask=labels_mask,
        )

        # Head k predicts the target k positions to the right of the token
        # before its decoder state (shift indices 0, 1, 2 here).  The final k
        # query positions therefore have no target and must not enter CE.
        expected_sum = loss.new_zeros(())
        expected_count = 0
        for shift, classifier in enumerate(head.classifiers):
            valid_mask = labels_mask[:, shift:]
            repeated_logits = classifier.bias.view(1, 1, 2).expand(
                1, classes.shape[1] - shift, 2
            )
            expected_sum = expected_sum + F.cross_entropy(
                repeated_logits[valid_mask],
                classes[:, shift:][valid_mask],
                reduction="sum",
            )
            expected_count += int(valid_mask.sum().item())
        self.assertEqual(expected_count, 10)
        self.assertTrue(torch.allclose(loss, expected_sum / expected_count))

        # Public logits always come from offset 1, never an auxiliary head.
        self.assertTrue(torch.allclose(logits[..., INTRON_LABEL_INDEX], torch.zeros(1, 5)))
        self.assertTrue(
            torch.allclose(logits[..., EXON_LABEL_INDEX], torch.full((1, 5), 0.75))
        )

        loss.backward()
        self.assertTrue(all(classifier.bias.grad is not None for classifier in head.classifiers))

    def test_generate_uses_only_offset_one_cache_and_argmax_feedback(self):
        head, decoder = self._head(
            context=3,
            lookahead=3,
            multi_token_prediction=3,
        )
        with torch.no_grad():
            for classifier in head.classifiers:
                classifier.weight.zero_()
            head.classifiers[0].bias.copy_(torch.tensor([-10.0, 10.0]))
            head.classifiers[1].bias.copy_(torch.tensor([10.0, -10.0]))
            head.classifiers[2].bias.copy_(torch.tensor([10.0, -10.0]))
        auxiliary_calls = [0, 0]
        hooks = [
            classifier.register_forward_hook(
                lambda _module, _args, _output, index=index: auxiliary_calls.__setitem__(
                    index, auxiliary_calls[index] + 1
                )
            )
            for index, classifier in enumerate(head.classifiers[1:])
        ]
        cross_projection_lengths = []
        cross_hooks = [
            layer.cross_attn.k_proj.register_forward_hook(
                lambda _module, args, _output: cross_projection_lengths.append(
                    int(args[0].shape[1])
                )
            )
            for layer in decoder.layers
        ]
        head.train()
        encoder = torch.randn(1, 7, 2)
        mask = torch.ones((1, 7), dtype=torch.bool)

        logits = head.generate(encoder, nucleotide_mask=mask)
        for hook in hooks + cross_hooks:
            hook.remove()

        self.assertTrue(head.training)
        self.assertEqual(tuple(logits.shape), (1, 7, 5))
        self.assertEqual(auxiliary_calls, [0, 0])
        self.assertEqual(len(decoder.calls), 7)
        self.assertTrue(all(call["inputs_embeds"].shape[1] == 1 for call in decoder.calls))
        self.assertTrue(all(call["past"] is not None for call in decoder.calls))
        self.assertTrue(all(call["use_cache"] for call in decoder.calls))
        self.assertTrue(all(call["cross_mask"] is None for call in decoder.calls))

        # Encoder states are physically sliced; neither cross-attention keys nor
        # any mask ever scale with the complete seven-base sequence.
        projected = head.encoder_projection(encoder).detach()
        expected_bounds = [(0, 6), (0, 6), (0, 6), (1, 7), (1, 7), (1, 7), (1, 7)]
        self.assertEqual(
            [call["encoder_shape"][1] for call in decoder.calls],
            [end - start for start, end in expected_bounds],
        )
        for call, (start, end) in zip(decoder.calls, expected_bounds):
            self.assertLessEqual(call["encoder_shape"][1], head.cross_attention_span)
            self.assertTrue(
                torch.allclose(call["encoder_values"], projected[:, start:end, :])
            )

        # The self cache has fixed physical capacity C, while its one-row mask
        # exposes only filled slots before the cache reaches C.
        self.assertEqual(
            [call["self_cache_capacity"] for call in decoder.calls],
            [head.context_size] * len(decoder.calls),
        )
        self.assertEqual(
            [int((call["self_mask"] == 0).sum().item()) for call in decoder.calls],
            [1, 2, 3, 3, 3, 3, 3],
        )
        self.assertTrue(
            all(tuple(call["self_mask"].shape) == (1, 1, 1, head.context_size)
                for call in decoder.calls)
        )
        self.assertEqual(
            [int(call["cache_position"].item()) for call in decoder.calls],
            list(range(7)),
        )

        # Cross K/V remain in one bounded cache.  When [0:6] moves to [1:7],
        # each decoder layer projects only the one newly visible state.  The
        # full window then remains anchored at the right edge.
        cross_ids = [call["cross_cache_id"] for call in decoder.calls]
        self.assertTrue(all(value == cross_ids[0] for value in cross_ids))
        self.assertEqual(cross_projection_lengths, [1, 1])
        self.assertEqual(
            [call["cross_cache_length_after"] for call in decoder.calls],
            [6, 6, 6, 6, 6, 6, 6],
        )
        expected_second_input = head.label_embedding(
            torch.tensor([[EXON_CLASS_ID]])
        ).detach()
        self.assertTrue(
            torch.allclose(decoder.calls[1]["inputs_embeds"], expected_second_input)
        )
        self.assertTrue(bool((logits[..., EXON_LABEL_INDEX] > logits[..., INTRON_LABEL_INDEX]).all()))

    def test_generation_can_add_aligned_encoder_state_to_prediction_feedback(self):
        head, decoder = self._head(
            encoder_dim=4,
            hidden_size=4,
            context=4,
            lookahead=2,
            add_encoder_to_decoder_input=True,
        )
        with torch.no_grad():
            head.classifier.weight.zero_()
            head.classifier.bias.copy_(torch.tensor([-10.0, 10.0]))
        encoder = torch.arange(16, dtype=torch.float32).view(1, 4, 4)
        mask = torch.ones((1, 4), dtype=torch.bool)

        head.generate(encoder, nucleotide_mask=mask)

        projected = head.encoder_projection(encoder).detach()
        bos = head._bos_embedding(device=encoder.device).detach()
        self.assertTrue(torch.allclose(decoder.calls[0]["inputs_embeds"], bos))
        predicted_embedding = head.label_embedding(
            torch.tensor([[EXON_CLASS_ID]])
        ).detach()
        for position, call in enumerate(decoder.calls[1:], start=1):
            expected = (
                predicted_embedding
                + projected[:, position - 1 : position, :]
            )
            self.assertTrue(torch.allclose(call["inputs_embeds"], expected))

    def test_encoder_window_remains_anchored_after_reaching_right_edge(self):
        head, _ = self._head(context=3, lookahead=3)

        self.assertEqual(
            [head._encoder_window_bounds(position, 10) for position in range(10)],
            [
                (0, 6),
                (0, 6),
                (0, 6),
                (1, 7),
                (2, 8),
                (3, 9),
                (4, 10),
                (4, 10),
                (4, 10),
                (4, 10),
            ],
        )
        self.assertEqual(
            [head._encoder_window_bounds(position, 4) for position in range(4)],
            [(0, 4), (0, 4), (0, 4), (0, 4)],
        )
        head_without_lookahead, _ = self._head(context=3, lookahead=0)
        self.assertEqual(
            [
                head_without_lookahead._encoder_window_bounds(position, 10)
                for position in range(10)
            ],
            [
                (0, 3),
                (0, 3),
                (0, 3),
                (1, 4),
                (2, 5),
                (3, 6),
                (4, 7),
                (5, 8),
                (6, 9),
                (7, 10),
            ],
        )

    def test_multi_token_heads_beyond_sequence_length_receive_zero_gradients(self):
        head, _ = self._head(
            context=3,
            lookahead=2,
            multi_token_prediction=6,
        )
        encoder = torch.randn(1, 2, 2)
        mask = torch.ones((1, 2), dtype=torch.bool)
        labels = _five_track_labels([[INTRON_CLASS_ID, EXON_CLASS_ID]])

        loss, _ = head(
            encoder,
            nucleotide_mask=mask,
            labels=labels,
            labels_mask=mask,
        )
        loss.backward()

        for classifier in head.classifiers:
            self.assertIsNotNone(classifier.weight.grad)
            self.assertIsNotNone(classifier.bias.grad)
        # Offsets +3 through +6 have no target in a two-token sample, but DDP
        # sees them as used because their exact-zero graph connections remain.
        for classifier in head.classifiers[2:]:
            self.assertEqual(float(classifier.weight.grad.abs().sum()), 0.0)
            self.assertEqual(float(classifier.bias.grad.abs().sum()), 0.0)

    def test_autoregressive_forward_rejects_ground_truth(self):
        head, _ = self._head()
        encoder = torch.randn(1, 2, 2)
        mask = torch.ones((1, 2), dtype=torch.bool)
        labels = _five_track_labels([[0, 1]])
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            head(
                encoder,
                nucleotide_mask=mask,
                labels=labels,
                labels_mask=mask,
                autoregressive=True,
            )

    def test_malformed_exon_intron_targets_fail_fast(self):
        head, _ = self._head()
        encoder = torch.randn(1, 2, 2)
        mask = torch.ones((1, 2), dtype=torch.bool)
        labels = _five_track_labels([[0, 1]])
        labels[0, 0, EXON_LABEL_INDEX] = 1.0
        with self.assertRaisesRegex(RuntimeError, "exactly one active class"):
            head(encoder, nucleotide_mask=mask, labels=labels, labels_mask=mask)

    def test_projection_depth_and_multi_token_validation(self):
        projected, _ = self._head(encoder_dim=2, hidden_size=4)
        identity, _ = self._head(encoder_dim=4, hidden_size=4)
        self.assertIsInstance(projected.encoder_projection, nn.Linear)
        self.assertIsInstance(identity.encoder_projection, nn.Identity)
        self.assertEqual(len(projected.classifiers), 1)

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
        with self.assertRaisesRegex(RuntimeError, "positive integer"):
            T5GemmaSegmentationHead(
                encoder_dim=2,
                decoder_hidden_size=4,
                num_decoder_layers=2,
                num_attention_heads=2,
                context_size=3,
                encoder_lookahead=3,
                multi_token_prediction=0,
                decoder=decoder,
            )
        with self.assertRaisesRegex(RuntimeError, "must be a bool"):
            T5GemmaSegmentationHead(
                encoder_dim=2,
                decoder_hidden_size=4,
                num_decoder_layers=2,
                num_attention_heads=2,
                context_size=3,
                encoder_lookahead=3,
                add_encoder_to_decoder_input=1,
                decoder=decoder,
            )

    @unittest.skipIf(T5GemmaDecoder is None, "transformers>=4.53 with T5Gemma is not installed")
    def test_official_tiny_decoder_runs_teacher_forcing_and_generation(self):
        head = T5GemmaSegmentationHead(
            encoder_dim=8,
            num_labels=5,
            decoder_hidden_size=16,
            decoder_intermediate_size=32,
            num_decoder_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            context_size=4,
            encoder_lookahead=4,
            multi_token_prediction=3,
        )
        self.assertTrue(
            all(layer.gradient_checkpointing for layer in head.decoder.layers)
        )
        encoder = torch.randn(1, 3, 8, requires_grad=True)
        labels = _five_track_labels([[0, 1, 0]])
        mask = torch.ones((1, 3), dtype=torch.bool)
        head.train()
        loss, logits = head(
            encoder,
            nucleotide_mask=mask,
            labels=labels,
            labels_mask=mask,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(logits.shape), (1, 3, 5))
        loss.backward()
        self.assertIsNotNone(encoder.grad)
        head.eval()
        generated_logits = head.generate(encoder, nucleotide_mask=mask)
        self.assertTrue(torch.isfinite(generated_logits).all())
        self.assertEqual(tuple(generated_logits.shape), (1, 3, 5))

    @unittest.skipIf(T5GemmaDecoder is None, "transformers>=4.53 with T5Gemma is not installed")
    def test_sdpa_none_mask_teacher_forcing_remains_causal(self):
        torch.manual_seed(7)
        head = T5GemmaSegmentationHead(
            encoder_dim=8,
            num_labels=5,
            decoder_hidden_size=16,
            decoder_intermediate_size=32,
            num_decoder_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            context_size=4,
            encoder_lookahead=2,
        )
        head.train()
        encoder = torch.randn(1, 4, 8)
        mask = torch.ones((1, 4), dtype=torch.bool)
        prefix = _five_track_labels([[0, 1, 0, 0]])
        changed_future = _five_track_labels([[0, 1, 1, 1]])

        _, first_logits = head(
            encoder,
            nucleotide_mask=mask,
            labels=prefix,
            labels_mask=mask,
        )
        _, second_logits = head(
            encoder,
            nucleotide_mask=mask,
            labels=changed_future,
            labels_mask=mask,
        )

        # Changing target y2 changes the decoder input only at query 3.  Queries
        # 0..2 must be identical if the mask-free SDPA path is truly causal.
        self.assertTrue(torch.equal(first_logits[:, :3], second_logits[:, :3]))

    @unittest.skipIf(T5GemmaDecoder is None, "transformers>=4.53 with T5Gemma is not installed")
    def test_bounded_generation_matches_full_encoder_mask_reference(self):
        torch.manual_seed(11)
        head = T5GemmaSegmentationHead(
            encoder_dim=8,
            num_labels=5,
            decoder_hidden_size=16,
            decoder_intermediate_size=32,
            num_decoder_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            context_size=3,
            encoder_lookahead=2,
        )
        head.eval()
        encoder = torch.randn(1, 7, 8)
        projected = head.encoder_projection(encoder)

        # Reference implementation: retain all encoder K/V and move an
        # additive one-row mask.  It is intentionally suitable only for this
        # tiny equivalence test; production must use the bounded path.
        reference_logits = []
        previous_prediction = None
        past_key_values = None
        for position in range(projected.shape[1]):
            if previous_prediction is None:
                decoder_input = head._bos_embedding(device=projected.device)
            else:
                decoder_input = head.label_embedding(previous_prediction)
            decoder_input = decoder_input.to(projected.dtype)
            start, end = head._encoder_window_bounds(position, projected.shape[1])
            cross_mask = torch.full(
                (1, 1, 1, projected.shape[1]),
                torch.finfo(projected.dtype).min,
                dtype=projected.dtype,
                device=projected.device,
            )
            cross_mask[..., start:end] = 0
            decoder_output = head.decoder(
                inputs_embeds=decoder_input,
                encoder_hidden_states=projected,
                encoder_attention_mask={"full_attention": cross_mask},
                past_key_values=past_key_values,
                use_cache=True,
            )
            step_logits = head.classifier(
                decoder_output.last_hidden_state[:, -1:, :]
            )
            reference_logits.append(step_logits)
            previous_prediction = step_logits.argmax(dim=-1)
            past_key_values = decoder_output.past_key_values
        reference = head._five_track_compatibility_logits(
            torch.cat(reference_logits, dim=1)
        )

        bounded = head.generate(
            encoder,
            nucleotide_mask=torch.ones((1, 7), dtype=torch.bool),
        )
        self.assertTrue(torch.allclose(reference, bounded, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
