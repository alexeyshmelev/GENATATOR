from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

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
    @staticmethod
    def _trainer(task: str = "segmentation"):
        trainer = object.__new__(GenatatorTrainer)
        trainer.genatator_task = task
        trainer.state = SimpleNamespace(global_step=17)
        return trainer

    def test_finite_loss_is_returned_unchanged(self):
        trainer = self._trainer()
        expected = torch.tensor(1.25)

        with patch(
            "genatator_core.train_common.Trainer.compute_loss",
            return_value=expected,
        ):
            actual = trainer.compute_loss(model=None, inputs={})

        self.assertIs(actual, expected)

    def test_nonfinite_loss_raises_for_every_training_task(self):
        for task in (
            "finding_edge",
            "finding_region",
            "segmentation",
            "transcript_type",
        ):
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(task=task, value=value):
                    trainer = self._trainer(task)
                    with patch(
                        "genatator_core.train_common.Trainer.compute_loss",
                        return_value=torch.tensor(value),
                    ):
                        with self.assertRaisesRegex(
                            FloatingPointError,
                            rf"task={task} global_step=17",
                        ):
                            trainer.compute_loss(model=None, inputs={})

    def test_return_outputs_path_checks_loss_before_returning_outputs(self):
        trainer = self._trainer()
        result = (torch.tensor(float("nan")), object())

        with patch(
            "genatator_core.train_common.Trainer.compute_loss",
            return_value=result,
        ):
            with self.assertRaisesRegex(FloatingPointError, "Non-finite loss"):
                trainer.compute_loss(model=None, inputs={}, return_outputs=True)

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
