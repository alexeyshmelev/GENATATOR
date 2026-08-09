from __future__ import annotations

import unittest

try:
    import torch

    from genatator_core.train_common import GenatatorTrainer
except (ImportError, ModuleNotFoundError):
    torch = None
    GenatatorTrainer = None


@unittest.skipIf(
    torch is None or GenatatorTrainer is None,
    "torch/transformers training dependencies are not installed",
)
class NonFiniteTrainingGuardTests(unittest.TestCase):
    def test_finite_gradients_are_not_modified(self):
        model = torch.nn.Linear(3, 2)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        before = [parameter.grad.clone() for parameter in model.parameters()]

        GenatatorTrainer._raise_on_nonfinite_gradients(model)

        for parameter, expected in zip(model.parameters(), before):
            self.assertTrue(torch.equal(parameter.grad, expected))

    def test_nonfinite_gradient_raises_before_optimizer(self):
        model = torch.nn.Linear(3, 2)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        next(model.parameters()).grad.view(-1)[0] = float("nan")

        with self.assertRaisesRegex(FloatingPointError, "Non-finite gradient"):
            GenatatorTrainer._raise_on_nonfinite_gradients(model)


if __name__ == "__main__":
    unittest.main()
