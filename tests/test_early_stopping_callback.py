from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


try:
    from genatator_core.train_common import GenatatorTrainer
except ImportError:
    GenatatorTrainer = None


@unittest.skipIf(GenatatorTrainer is None, "torch/transformers are not installed")
class EarlyStoppingCallbackTests(unittest.TestCase):
    def test_streaming_evaluation_dispatches_on_evaluate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trainer = object.__new__(GenatatorTrainer)
            trainer.args = SimpleNamespace(output_dir=str(Path(temporary)))
            trainer.state = SimpleNamespace(global_step=1000)
            initial_control = object()
            returned_control = object()
            trainer.control = initial_control
            trainer.callback_handler = Mock()
            trainer.callback_handler.on_evaluate.return_value = returned_control
            metrics = {"eval_loss": 0.25}
            trainer._streaming_evaluate_rank0 = Mock(return_value=metrics)
            trainer.log = Mock()

            with patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "1"}, clear=False):
                result = trainer.evaluate()

            self.assertEqual(result, metrics)
            trainer.log.assert_called_once_with(metrics)
            trainer.callback_handler.on_evaluate.assert_called_once_with(
                trainer.args,
                trainer.state,
                initial_control,
                metrics,
            )
            self.assertIs(trainer.control, returned_control)


if __name__ == "__main__":
    unittest.main()
