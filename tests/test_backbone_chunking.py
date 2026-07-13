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


if __name__ == "__main__":
    unittest.main()
