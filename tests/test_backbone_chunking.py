from __future__ import annotations

import types
import unittest

try:
    import torch
    import torch.nn as nn

    from genatator_core.backbones import HiddenStateBackbone
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch/transformers are not installed")
class DirectGenaLengthTests(unittest.TestCase):
    def test_long_direct_gena_input_is_rejected_without_chunking(self) -> None:
        class FakeEncoder(nn.Module):
            def forward(self, **kwargs):
                raise AssertionError("Encoder must not run for an over-limit direct GENA input")

        backbone = HiddenStateBackbone.__new__(HiddenStateBackbone)
        nn.Module.__init__(backbone)
        backbone.backbone_kind = "gena"
        backbone.hidden_size = 2
        backbone.config = types.SimpleNamespace(max_position_embeddings=3)
        backbone.encoder = FakeEncoder()
        backbone.uses_owner = False

        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        with self.assertRaisesRegex(RuntimeError, "does not support outer-input elongation"):
            backbone(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                token_type_ids=torch.zeros_like(input_ids),
            )

    def test_long_input_can_be_encoded_as_non_overlapping_unpadded_chunks(self) -> None:
        calls = []

        class FakeEncoder(nn.Module):
            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                calls.append((input_ids.detach().clone(), attention_mask.detach().clone()))
                hidden = torch.stack((input_ids.float(), input_ids.float() + 10.0), dim=-1)
                return types.SimpleNamespace(last_hidden_state=hidden, hidden_states=None, attentions=None)

        backbone = HiddenStateBackbone.__new__(HiddenStateBackbone)
        nn.Module.__init__(backbone)
        backbone.backbone_kind = "gena"
        backbone.hidden_size = 2
        backbone.config = types.SimpleNamespace(max_position_embeddings=3)
        backbone.encoder = FakeEncoder()
        backbone.uses_owner = False

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0]], dtype=torch.long)
        attention_mask = input_ids.ne(0).long()
        output = backbone.forward_chunked(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros_like(input_ids),
        )

        self.assertEqual([tuple(ids.shape) for ids, _ in calls], [(1, 3), (1, 2)])
        self.assertTrue(all(bool(mask.all()) for _, mask in calls))
        self.assertEqual(tuple(output.logits.shape), (1, 7, 2))
        self.assertTrue(torch.equal(output.logits[0, :5, 0], input_ids[0, :5].float()))
        self.assertTrue(torch.equal(output.logits[0, 5:], torch.zeros((2, 2))))


if __name__ == "__main__":
    unittest.main()
