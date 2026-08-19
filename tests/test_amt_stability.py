import types
import unittest
from unittest.mock import patch


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from genatator_core.amt_models import (
        AMTTokenClassifier,
        _install_stable_associative_memory,
        _run_amt_without_padding,
    )
except ImportError:
    torch = None
    nn = None
    F = None
    AMTTokenClassifier = None
    _install_stable_associative_memory = None
    _run_amt_without_padding = None

_ModuleBase = nn.Module if nn is not None else object


@unittest.skipIf(torch is None, "torch/transformers are not installed")
class AmtUseDenomConfigTests(unittest.TestCase):
    class _FakeBaseModel(_ModuleBase):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 4)

        def get_input_embeddings(self):
            return self.embedding

    class _RecordingMemoryCell(_ModuleBase):
        received_kwargs = None

        def __init__(self, **kwargs):
            super().__init__()
            type(self).received_kwargs = kwargs

    class _FakeRecurrentWrapper(_ModuleBase):
        def __init__(self, memory_cell, **kwargs):
            super().__init__()
            self.memory_cell = memory_cell

    def test_use_denom_default_and_configured_value_reach_memory_cell(self) -> None:
        remote_module = types.SimpleNamespace(
            AssociativeMemoryCell=self._RecordingMemoryCell,
            AssociativeRecurrentWrapper=self._FakeRecurrentWrapper,
        )
        for configured, expected in ((None, True), (False, False)):
            with self.subTest(configured=configured), patch(
                "genatator_core.amt_models._load_amt_base_model",
                return_value=(self._FakeBaseModel(), object(), 4, "encoder.layer"),
            ), patch(
                "genatator_core.amt_models.allow_transformers_torch_load_on_legacy_torch"
            ), patch(
                "genatator_core.amt_models.AutoModelForCausalLM.from_pretrained",
                return_value=object(),
            ), patch(
                "genatator_core.amt_models._install_stable_associative_memory"
            ), patch(
                "genatator_core.amt_models.importlib.import_module",
                return_value=remote_module,
            ):
                amt_kwargs = {} if configured is None else {"use_denom": configured}
                AMTTokenClassifier(
                    backbone_path="unused-checkpoint",
                    backbone_kind="gena",
                    num_labels=2,
                    encoder_only=True,
                    **amt_kwargs,
                )
            self.assertIs(
                self._RecordingMemoryCell.received_kwargs["use_denom"], expected
            )


@unittest.skipIf(torch is None, "torch/transformers are not installed")
class AmtPaddingCompactionTests(unittest.TestCase):
    class _FakeAMT(_ModuleBase):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.hidden_size = hidden_size
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.calls = []

        def forward(self, input_ids, attention_mask):
            self.calls.append((input_ids.detach().clone(), attention_mask.detach().clone()))
            if not bool(attention_mask.bool().all()):
                raise AssertionError("The compact AMT call still contains padding")
            logits = input_ids.float().unsqueeze(-1).repeat(
                1, 1, self.hidden_size
            ) * self.scale
            return types.SimpleNamespace(logits=logits)

    def test_padding_is_removed_samplewise_and_outputs_are_scattered_back(self) -> None:
        amt = self._FakeAMT(hidden_size=3)
        input_ids = torch.tensor(
            [
                [10, 11, 12, 0, 0, 0],
                [0, 0, 20, 21, 0, 0],
            ]
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
            ]
        )

        hidden = _run_amt_without_padding(
            amt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            hidden_size=3,
        )

        self.assertEqual([call[0].shape[1] for call in amt.calls], [3, 2])
        self.assertTrue(torch.equal(amt.calls[0][0], torch.tensor([[10, 11, 12]])))
        self.assertTrue(torch.equal(amt.calls[1][0], torch.tensor([[20, 21]])))
        self.assertEqual(tuple(hidden.shape), (2, 6, 3))
        self.assertTrue(torch.equal(hidden[0, 3:], torch.zeros_like(hidden[0, 3:])))
        self.assertTrue(torch.equal(hidden[1, :2], torch.zeros_like(hidden[1, :2])))
        self.assertTrue(torch.equal(hidden[1, 4:], torch.zeros_like(hidden[1, 4:])))

        hidden.sum().backward()
        self.assertIsNotNone(amt.scale.grad)
        self.assertTrue(torch.isfinite(amt.scale.grad))

    def test_all_padding_sample_fails_before_remote_amt_is_called(self) -> None:
        amt = self._FakeAMT(hidden_size=2)
        with self.assertRaisesRegex(RuntimeError, "no attended tokens"):
            _run_amt_without_padding(
                amt,
                input_ids=torch.zeros((1, 8), dtype=torch.long),
                attention_mask=torch.zeros((1, 8), dtype=torch.long),
                hidden_size=2,
            )
        self.assertEqual(amt.calls, [])


@unittest.skipIf(torch is None, "torch/transformers are not installed")
class AmtFp32MemoryTests(unittest.TestCase):
    class _IdentityDPFP:
        def __call__(self, values):
            # The real DPFP is non-negative before normalization.  ReLU is
            # sufficient for this compact numerical-contract test.
            return F.relu(values)

    class _FakeAssociativePrimitive(_ModuleBase):
        def __init__(self):
            super().__init__()
            self.d_model = 4
            self.d_key = 2
            self.n_heads = 1
            self.use_denom = True
            self.gating = False
            self.correction = True
            self.W_mq = nn.Linear(4, 2, bias=False)
            self.W_mk = nn.Linear(4, 2, bias=False)
            self.W_mv = nn.Linear(4, 4, bias=False)
            self.W_mb = nn.Linear(4, 1)
            self.phi = AmtFp32MemoryTests._IdentityDPFP()
            self.first_seg = True
            self.seg_num = 0
            self.W_mem = torch.empty(0)
            self.z = torch.empty(0)

        def _to_heads(self, values):
            batch, length, width = values.shape
            return values.reshape(
                batch, length, self.n_heads, width // self.n_heads
            ).permute(0, 2, 1, 3)

        def _from_heads(self, values):
            batch, heads, length, width = values.shape
            return values.permute(0, 2, 1, 3).reshape(
                batch, length, heads * width
            )

        # These three methods intentionally exist on the same defining class,
        # matching the API contract of upstream AssociativeLayerWrapper.
        def associate(self, hidden_states):  # pragma: no cover - replaced
            raise AssertionError("stable associate was not installed")

        def update_mem(self, mem_tokens):  # pragma: no cover - replaced
            raise AssertionError("stable update was not installed")

        def zero_mem(self):  # pragma: no cover - replaced
            raise AssertionError("stable reset was not installed")

    class _FakeMemoryCell(_ModuleBase):
        def __init__(self, layer):
            super().__init__()
            self.layers = nn.ModuleList([layer])

        def get_layers(self):
            return self.layers

        def zero_mem(self):
            for layer in self.layers:
                layer.zero_mem()

    def test_recurrent_state_is_fp32_while_bf16_interface_and_gradients_remain(self) -> None:
        torch.manual_seed(7)
        layer = self._FakeAssociativePrimitive()
        cell = self._FakeMemoryCell(layer)
        _install_stable_associative_memory(cell)

        self.assertEqual(layer.W_mem.dtype, torch.float32)
        self.assertEqual(layer.z.dtype, torch.float32)

        first_memory_tokens = torch.randn(1, 5, 4, dtype=torch.bfloat16)
        layer.update_mem(first_memory_tokens)
        second_memory_tokens = torch.randn(1, 5, 4, dtype=torch.bfloat16)
        layer.update_mem(second_memory_tokens)
        query = torch.randn(1, 9, 4, dtype=torch.bfloat16)
        retrieved = layer.associate(query)

        self.assertEqual(layer.W_mem.dtype, torch.float32)
        self.assertEqual(layer.z.dtype, torch.float32)
        self.assertEqual(retrieved.dtype, torch.bfloat16)
        self.assertTrue(bool(torch.isfinite(layer.W_mem).all()))
        self.assertTrue(bool(torch.isfinite(layer.z).all()))
        self.assertTrue(bool(torch.isfinite(retrieved).all()))

        retrieved.float().square().mean().backward()
        for projection in (layer.W_mq, layer.W_mk, layer.W_mv, layer.W_mb):
            for parameter in projection.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))


if __name__ == "__main__":
    unittest.main()
