import unittest
from unittest.mock import patch


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from genatator_core.legacy_rmt import scatter_active_rows
    from genatator_core.unet import run_samplewise_chunked_unet
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch/transformers are not installed")
class SamplewiseUnetTests(unittest.TestCase):
    def test_uncovered_nucleotide_positions_are_dropped_without_info_logging(self):
        class IdentityUnet(nn.Module):
            def forward(self, x):
                return x

        with patch("genatator_core.unet.logger.info") as info_log:
            _, logits = run_samplewise_chunked_unet(
                token_hidden=torch.randn(1, 1, 2),
                token_content_mask=torch.tensor([[1]], dtype=torch.bool),
                embedding_repeater=torch.tensor([[0, -100]], dtype=torch.long),
                letter_level_tokens=torch.tensor([[1, 2]], dtype=torch.long),
                letter_level_attention_mask=torch.tensor([[1, 1]], dtype=torch.bool),
                letter_level_labels=None,
                letter_level_labels_mask=None,
                pos_weight=None,
                nucleotide_embedding=nn.Embedding(4, 2),
                unet=IdentityUnet(),
                activation_fn=nn.Identity(),
                classifier=nn.Linear(4, 1),
                cycles=1,
                chunk_size=2,
                context="silent_truncation_test",
            )

        info_log.assert_not_called()
        self.assertEqual(tuple(logits.shape), (1, 2, 1))
        self.assertTrue(bool((logits[:, 1:, :] == 0).all()))

    def test_mixed_precision_logits_are_cast_for_output_assembly(self):
        class IdentityUnet(nn.Module):
            def forward(self, x):
                return x

        class BFloat16Classifier(nn.Module):
            out_features = 1

            def forward(self, x):
                return x[..., -1:].to(torch.bfloat16)

        token_hidden = torch.randn(1, 2, 2, dtype=torch.float32, requires_grad=True)
        loss, logits = run_samplewise_chunked_unet(
            token_hidden=token_hidden,
            token_content_mask=torch.tensor([[1, 1]], dtype=torch.bool),
            embedding_repeater=torch.tensor([[0, 1]], dtype=torch.long),
            letter_level_tokens=torch.tensor([[1, 2]], dtype=torch.long),
            letter_level_attention_mask=torch.tensor([[1, 1]], dtype=torch.bool),
            letter_level_labels=None,
            letter_level_labels_mask=None,
            pos_weight=None,
            nucleotide_embedding=nn.Embedding(4, 2),
            unet=IdentityUnet(),
            activation_fn=nn.Identity(),
            classifier=BFloat16Classifier(),
            cycles=1,
            chunk_size=2,
            context="mixed_precision_test",
        )
        self.assertIsNone(loss)
        self.assertEqual(logits.dtype, torch.float32)
        logits.sum().backward()
        self.assertGreater(float(token_hidden.grad.abs().sum()), 0.0)

    def test_batched_states_use_only_single_sample_unpadded_chunks_and_global_loss(self):
        class RecordingEmbedding(nn.Embedding):
            def __init__(self):
                super().__init__(16, 2)
                self.seen = []

            def forward(self, token_ids):
                self.seen.append(token_ids.detach().clone())
                return super().forward(token_ids)

        class RecordingUnet(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, x):
                self.calls.append(tuple(x.shape))
                return x

        token_hidden = torch.randn(2, 4, 2, requires_grad=True)
        token_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
        repeater = torch.tensor(
            [[0, 0, 1, -100, -100, -100, -100], [0, 1, 1, 2, 2, -100, -100]],
            dtype=torch.long,
        )
        nucleotide_tokens = torch.tensor(
            [[1, 2, 3, 15, 15, 15, 15], [4, 3, 2, 1, 4, 15, 15]],
            dtype=torch.long,
        )
        attention = torch.tensor(
            [[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0]],
            dtype=torch.bool,
        )
        labels = torch.tensor(
            [[[0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
             [[1.0], [0.0], [1.0], [1.0], [0.0], [0.0], [0.0]]]
        )
        embedding = RecordingEmbedding()
        unet = RecordingUnet()
        classifier = nn.Linear(4, 1)

        loss, logits = run_samplewise_chunked_unet(
            token_hidden=token_hidden,
            token_content_mask=token_mask,
            embedding_repeater=repeater,
            letter_level_tokens=nucleotide_tokens,
            letter_level_attention_mask=attention,
            letter_level_labels=labels,
            letter_level_labels_mask=attention,
            pos_weight=torch.ones(2, 4, 1),
            nucleotide_embedding=embedding,
            unet=unet,
            activation_fn=nn.Identity(),
            classifier=classifier,
            cycles=1,
            chunk_size=2,
            context="test",
        )

        self.assertEqual(tuple(logits.shape), (2, 7, 1))
        self.assertEqual(unet.calls, [(1, 4, 2), (1, 4, 1), (1, 4, 2), (1, 4, 2), (1, 4, 1)])
        self.assertTrue(all(not bool((seen == 15).any()) for seen in embedding.seen))
        self.assertTrue(bool((logits[~attention] == 0).all()))
        expected = F.binary_cross_entropy_with_logits(logits[attention].float(), labels[attention].float(), reduction="sum") / 8.0
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))

        loss.backward()
        self.assertGreater(float(token_hidden.grad[0].abs().sum()), 0.0)
        self.assertGreater(float(token_hidden.grad[1].abs().sum()), 0.0)

    def test_rmt_scatter_preserves_original_sample_identity(self):
        shared_segment = scatter_active_rows(
            torch.tensor([[[1.0]], [[2.0]]]),
            [True, True],
            2,
        )
        longer_sample_only = scatter_active_rows(
            torch.tensor([[[7.0]]]),
            [False, True],
            2,
        )
        concatenated = torch.cat([shared_segment, longer_sample_only], dim=1)
        self.assertTrue(torch.equal(concatenated[0, :, 0], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.equal(concatenated[1, :, 0], torch.tensor([2.0, 7.0])))


if __name__ == "__main__":
    unittest.main()
