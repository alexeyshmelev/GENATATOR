from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _uses_unet(model: dict) -> bool:
    return model.get("family") in {"unet", "rmt"} or (
        model.get("family") == "amt" and bool(model.get("use_unet", False))
    )


class StaticConfigContractsTest(unittest.TestCase):
    def test_training_configs_use_explicit_length_units(self) -> None:
        for task_dir in ("finding", "segmentation", "transcript_type"):
            for path in sorted((ROOT / task_dir / "configs").glob("*.json")):
                cfg = _load(path)
                if not isinstance(cfg.get("training"), dict):
                    continue
                with self.subTest(path=path.relative_to(ROOT)):
                    model = cfg["model"]
                    self.assertIsInstance(cfg["training"].get("custom_prefix"), str)
                    if _uses_unet(model):
                        self.assertGreater(int(model["unet_chunk_size"]), 0)
                    for key in ("train_dataset", "eval_dataset"):
                        dataset = cfg[key]
                        if model["family"] == "caduceus":
                            self.assertGreater(int(dataset["max_nucleotides"]), 0)
                            self.assertNotIn("max_bpe_tokens", dataset)
                            self.assertNotIn("average_bpe_token_length", dataset)
                        else:
                            self.assertGreater(int(dataset["max_bpe_tokens"]), 0)
                            self.assertGreater(float(dataset["average_bpe_token_length"]), 0.0)
                            self.assertNotIn("max_nucleotides", dataset)
                        resolved_nt = (
                            int(dataset["max_nucleotides"])
                            if model["family"] == "caduceus"
                            else int(dataset["max_bpe_tokens"] * float(dataset["average_bpe_token_length"]))
                        )
                        self.assertGreaterEqual(resolved_nt, 30000)
                        self.assertLessEqual(resolved_nt, 40000)
                        self.assertNotIn("max_tokens", dataset)
                        if task_dir != "finding":
                            self.assertNotIn("overlap", dataset)
                            self.assertNotIn("random_crop", dataset)
                    self.assertNotIn("nucleotide_tokenizer_path", model)
                    if model["family"] == "rmt":
                        self.assertNotIn("input_size", model["rmt"])
                        self.assertEqual(
                            int(model["rmt"]["segment_size"]),
                            512 if model["backbone_kind"] == "gena" else 1024,
                        )
                        self.assertGreater(int(model["rmt"]["max_n_segments"]), 0)
                    if model["family"] == "amt":
                        self.assertEqual(
                            int(model["amt"]["segment_size"]),
                            512 if model["backbone_kind"] == "gena" else 1024,
                        )

    def test_training_segmentation_validation_stays_status_one(self) -> None:
        for path in sorted((ROOT / "segmentation" / "configs").glob("*.json")):
            cfg = _load(path)
            if not isinstance(cfg.get("training"), dict):
                continue
            with self.subTest(path=path.name):
                self.assertEqual(cfg["eval_dataset"].get("statuses"), [1])

    def test_separate_segmentation_template_uses_all_transcripts(self) -> None:
        path = ROOT / "segmentation" / "configs" / "infer_caduceus_ps.json"
        cfg = _load(path)
        self.assertNotIn("statuses", cfg["dataset"])
        self.assertIsInstance(cfg["inference"].get("use_cds_heuristic"), bool)


if __name__ == "__main__":
    unittest.main()
