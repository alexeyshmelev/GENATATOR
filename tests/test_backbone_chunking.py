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
class DirectGenaChunkingTests(unittest.TestCase):
    def test_long_direct_gena_inputs_are_chunked_and_reassembled(self) -> None:
        class FakeEncoder(nn.Module):
            def forward(
                self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                inputs_embeds=None,
                output_hidden_states=False,
                return_dict=True,
            ):
                if inputs_embeds is not None:
                    hidden = inputs_embeds
                else:
                    hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 2)
                return types.SimpleNamespace(
                    last_hidden_state=hidden,
                    hidden_states=None,
                    attentions=None,
                )

        backbone = HiddenStateBackbone.__new__(HiddenStateBackbone)
        nn.Module.__init__(backbone)
        backbone.backbone_kind = "gena"
        backbone.hidden_size = 2
        backbone.config = types.SimpleNamespace(max_position_embeddings=3)
        backbone.encoder = FakeEncoder()
        backbone.uses_owner = False

        input_ids = torch.tensor(
            [[1, 2, 3, 4, 5, 6], [7, 8, 0, 0, 0, 0]],
            dtype=torch.long,
        )
        attention = torch.tensor(
            [[1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0]],
            dtype=torch.long,
        )
        output = backbone(
            input_ids=input_ids,
            attention_mask=attention,
            token_type_ids=torch.zeros_like(input_ids),
        )
        self.assertEqual(tuple(output.logits.shape), (2, 6, 2))
        self.assertTrue(torch.equal(output.logits[0, :, 0], input_ids[0].float()))
        self.assertTrue(torch.equal(output.logits[1, :2, 0], input_ids[1, :2].float()))
        self.assertTrue(bool((output.logits[1, 3:] == 0).all()))


if __name__ == "__main__":
    unittest.main()
