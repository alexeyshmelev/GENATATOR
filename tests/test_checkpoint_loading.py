import tempfile
import unittest
from pathlib import Path


try:
    import torch
    import torch.nn as nn

    from genatator_core.model_builders import load_finetuned_weights
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch/transformers/safetensors are not installed")
class CheckpointLoadingTests(unittest.TestCase):
    def test_missing_trainable_parameter_is_fatal(self):
        model = nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "partial.bin"
            torch.save({"weight": model.weight.detach().clone()}, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "refusing a partial load"):
                load_finetuned_weights(model, str(checkpoint))

    def test_exact_state_dict_loads(self):
        model = nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "exact.bin"
            torch.save(model.state_dict(), checkpoint)
            load_finetuned_weights(model, str(checkpoint))


if __name__ == "__main__":
    unittest.main()
