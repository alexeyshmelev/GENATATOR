import unittest


try:
    from genatator_core.amt_models import validate_gena_amt_transfer
except ImportError:
    validate_gena_amt_transfer = None


@unittest.skipIf(validate_gena_amt_transfer is None, "torch/transformers are not installed")
class GenaAmtTransferTests(unittest.TestCase):
    def test_known_task_head_differences_are_allowed(self) -> None:
        validate_gena_amt_transfer(
            ["classifier.weight", "classifier.bias"],
            ["cls.predictions.decoder.weight", "cls.predictions.bias"],
        )

    def test_missing_encoder_weight_is_fatal(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "transfer is incomplete"):
            validate_gena_amt_transfer(
                ["bert.encoder.layer.0.attention.self.query.weight"],
                [],
            )


if __name__ == "__main__":
    unittest.main()
