"""Local tests: stdlib for binary format/selection; existing NumPy for signals."""

import ast
from contextlib import nullcontext, redirect_stdout
from decimal import Decimal
import io
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from bigwig import BigWigWriter
from selection import all_selections, CHANNELS, GROUPS, select_pair
import run

try:
    import numpy as np
except ImportError:
    np = None


def read_bigwig(path, start=0, end=2**32-1):
    """Independent test reader following UCSC's field layout and indexed queries.

    Deliberately does not use the writer's constants, structs, or implementation.
    This is a format test, not a substitute for testing in a genome browser.
    """
    data = path.read_bytes()
    header = struct.unpack_from("<IHHQQQHHQQIQ", data)
    magic, version, zooms, chrom_at, data_at, index_at, fields, defined, sql, summary_at, buffer_size, extra = header
    assert magic == 0x888FFC26 and version == 4 and zooms == 0
    assert (fields, defined, sql, extra) == (0, 0, 0, 0)
    assert struct.unpack_from("<I", data, len(data)-4)[0] == magic
    tree_magic, block_size, key_size, val_size, chrom_count, reserved = struct.unpack_from("<IIIIQQ", data, chrom_at)
    assert tree_magic == 0x78CA8C91 and chrom_count == 1 and val_size == 8
    assert (block_size, reserved) == (1, 0)
    assert struct.unpack_from("<BBH", data, chrom_at+32) == (1, 0, 1)
    chrom = data[chrom_at+36:chrom_at+36+key_size].decode("ascii").rstrip("\x00")
    chrom_id, chrom_length = struct.unpack_from("<II", data, chrom_at+36+key_size)
    assert chrom_id == 0
    summary = struct.unpack_from("<Qdddd", data, summary_at)
    section_count = struct.unpack_from("<Q", data, data_at)[0]
    rmagic, fanout, entries, cstart, bstart, cend, bend, file_end, per_slot, reserved = struct.unpack_from("<IIQIIIIQII", data, index_at)
    assert rmagic == 0x2468ACE0 and entries == section_count
    assert (cstart, cend, file_end, per_slot, reserved) == (0, 0, index_at, 1, 0)
    assert 0 <= bstart < bend <= chrom_length
    blocks = []
    visited = set()

    def visit(offset):
        assert offset not in visited
        visited.add(offset)
        leaf, reserved, count = struct.unpack_from("<BBH", data, offset)
        assert leaf in (0, 1) and reserved == 0 and 0 < count <= fanout
        offset += 4
        for _ in range(count):
            sc, sb, ec, eb, pointer = struct.unpack_from("<IIIIQ", data, offset)
            assert sc == ec == 0 and sb < eb <= chrom_length
            offset += 24
            size = None
            if leaf:
                size = struct.unpack_from("<Q", data, offset)[0]
                offset += 8
            if sb < end and eb > start:
                if leaf:
                    blocks.append((pointer, size, sb, eb))
                else:
                    visit(pointer)

    visit(index_at + 48)
    values = {}
    for pointer, size, sb, eb in blocks:
        assert data_at + 8 <= pointer < pointer + size <= index_at
        raw = zlib.decompress(data[pointer:pointer+size])
        assert len(raw) <= buffer_size
        cid, begin, stop, step, span, kind, reserved, count = struct.unpack_from("<IIIIIBBH", raw)
        assert (cid, begin, stop, step, span, kind, reserved) == (0, sb, eb, 1, 1, 3, 0)
        assert stop - begin == count and len(raw) == 24 + 4 * count
        for i, (value,) in enumerate(struct.iter_unpack("<f", raw[24:])):
            position = begin+i
            assert math.isfinite(value) and position not in values
            if start <= position < end:
                values[position] = value
    return chrom, chrom_length, values, summary, section_count


class BigWigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "test.bw"

    def test_dense_data_multilevel_index_and_random_access(self):
        expected = {i+7: (i % 8) / 8 for i in range(137)}
        with BigWigWriter(self.path, "NC_060944.1", 200, block_items=3, index_fanout=2) as writer:
            writer.add_values(7, list(expected.values()))
        chrom, length, values, summary, sections = read_bigwig(self.path)
        self.assertEqual((chrom, length, sections), ("NC_060944.1", 200, 46))
        self.assertEqual(values, expected)
        self.assertEqual(summary, (137, min(expected.values()), max(expected.values()),
                                   sum(expected.values()), sum(v*v for v in expected.values())))
        for begin, end in ((0, 7), (7, 8), (12, 27), (130, 144), (144, 200)):
            self.assertEqual(read_bigwig(self.path, begin, end)[2],
                             {k: v for k, v in expected.items() if begin <= k < end})

    def test_gaps_remain_missing_and_summary_uses_stored_float32(self):
        with BigWigWriter(self.path, "chr20", 20) as writer:
            writer.add_values(0, [0.1, 0.2])
            writer.add_values(10, [0.3, 0.0])
        _, _, values, summary, _ = read_bigwig(self.path)
        expected = {i: struct.unpack("<f", struct.pack("<f", value))[0]
                    for i, value in ((0, 0.1), (1, 0.2), (10, 0.3), (11, 0.0))}
        self.assertEqual(values, expected)
        self.assertEqual(summary[0], 4)
        self.assertAlmostEqual(summary[3], sum(expected.values()))
        self.assertNotIn(2, values)

    def test_nonfinite_input_aborts_without_publishing(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                with BigWigWriter(self.path, "chr20", 20) as writer:
                    writer.add_values(0, [1.0, value])
            self.assertFalse(self.path.exists())
            self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_coordinate_errors_and_empty_files(self):
        for first, second in ((0, 0), (0, -1), (19, 20)):
            with self.assertRaises(ValueError):
                with BigWigWriter(self.path, "chr20", 20) as writer:
                    writer.add_values(first, [0.5])
                    writer.add_values(second, [0.6])
            self.assertFalse(self.path.exists())
        with self.assertRaises(ValueError):
            with BigWigWriter(self.path, "chr20", 20):
                pass
        self.assertFalse(self.path.exists())

    def test_existing_file_is_not_overwritten(self):
        with BigWigWriter(self.path, "chr20", 20) as writer:
            writer.add_values(0, [0.5])
        original = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            BigWigWriter(self.path, "chr20", 20)
        self.assertEqual(original, self.path.read_bytes())

    def test_publication_race_does_not_replace_existing_file(self):
        writer = BigWigWriter(self.path, "chr20", 20)
        writer.add_values(0, [0.5])
        self.path.write_bytes(b"other writer")
        with self.assertRaises(FileExistsError):
            writer.close()
        self.assertEqual(self.path.read_bytes(), b"other writer")
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_writer_has_only_standard_library_imports(self):
        tree = ast.parse((HERE / "bigwig.py").read_text())
        modules = {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        modules.update(alias.name.split(".")[0] for node in ast.walk(tree)
                       if isinstance(node, ast.Import) for alias in node.names)
        self.assertTrue(modules <= sys.stdlib_module_names)


class SelectionTests(unittest.TestCase):
    def test_six_minima_and_candidate_counts(self):
        selections = all_selections()
        expected = {
            GROUPS[0]: (("edge_moderngena_large_rmt_unet", "0.6692375275"),
                        ("region_moderngena_large_rmt_unet", "0.54807162")),
            GROUPS[1]: (("edge_gena_large_amt_unet", "0.62119363"),
                        ("region_gena_large_amt_unet", "0.57065919")),
            GROUPS[2]: (("edge_gena_large_amt_unet", "0.615109995"),
                        ("region_moderngena_large_rmt_unet", "0.53258994")),
        }
        for group, pair in expected.items():
            for stage, (name, mean) in zip(CHANNELS, pair):
                chosen = selections[group][stage]
                self.assertEqual(chosen["model"], name)
                self.assertEqual(Decimal(chosen["mean_roc_auc"]), Decimal(mean))
                self.assertEqual(chosen["candidate_count"], 22)

    def test_configs_match_selection_training_architecture_and_group(self):
        selections = all_selections()
        for group in GROUPS:
            config = json.loads((HERE / "configs" / f"{group}.json").read_text())
            run.validate_config(config, require_weights=False)
            self.assertTrue(config["inference"]["use_reverse_complement"])
            for stage in CHANNELS:
                spec = config[stage]
                self.assertEqual(spec["model_name"], selections[group][stage]["model"])
                training = json.loads((run.REPO / spec["training_config"]).read_text())
                self.assertEqual(spec["model"], training["model"])
                self.assertEqual(spec["dataset"]["target_group"], training["train_dataset"]["target_group"])
                self.assertEqual(spec["dataset"]["max_bpe_tokens"], training["train_dataset"]["max_bpe_tokens"])
                self.assertEqual(spec["dataset"]["chromosomes"], ["NC_060944.1"])
                self.assertEqual(spec["dataset"]["genomes"], ["GCF_009914755.1_T2T-CHM13v2.0"])
                self.assertEqual(spec["dataset"]["overlap"], 0.5)

    def test_missing_checkpoint_fails_before_ml_import_or_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "outputs"
            with self.assertRaisesRegex(ValueError, "checkpoint_path"):
                run.main(["--all", "--check", "--output-root", str(destination)])
            self.assertFalse(destination.exists())

    def test_valid_checkpoint_preflight(self):
        config = json.loads((HERE / "configs" / f"{GROUPS[0]}.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "model.safetensors"
            weights.touch()
            for stage in CHANNELS:
                config[stage]["inference"]["checkpoint_path"] = temporary
            run.validate_config(config)
            # Preflight checks existence, not weight tensor contents or loading.

    def test_metrics_fields_are_rejected(self):
        config = json.loads((HERE / "configs" / f"{GROUPS[0]}.json").read_text())
        config["inference"]["metrics_json"] = "not_allowed.json"
        with self.assertRaisesRegex(ValueError, "metrics_json"):
            run.validate_config(config, require_weights=False)


@unittest.skipIf(np is None, "NumPy is part of the existing inference environment")
class SignalTests(unittest.TestCase):
    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values)
            self.shape = self.values.shape

        def cpu(self):
            return self

        def numpy(self):
            return self.values

        def to(self, device):
            return self

        def detach(self):
            return self

        def float(self):
            return self

    def batch(self):
        return {"input_ids": self.Tensor([[9, 10, 11, 9, 0]]),
                "attention_mask": self.Tensor([[1, 1, 1, 1, 0]]),
                "token_type_ids": self.Tensor([[0, 0, 0, 0, 0]]),
                "letter_level_tokens": self.Tensor([[1, 2, 3, 4, 0, 0]]),
                "dna_sequence": ["ACGT"],
                "offset_mapping": [[(0, 0), (0, 2), (2, 4), (0, 0), (0, 0)]]}

    def test_alignment_cannot_read_poisoned_truth_fields(self):
        class Poison:
            def __getattribute__(self, name):
                raise AssertionError(f"Ground truth was accessed: {name}")

        batch = self.batch()
        baseline = run.dna_alignment(batch)
        for key in ("labels", "letter_level_labels", "labels_mask", "letter_level_labels_mask",
                    "truth_labels", "pos_weight", "embedding_repeater", "letter_level_attention_mask"):
            batch[key] = Poison()
        poisoned = run.dna_alignment(batch)
        for first, second in zip(baseline, poisoned):
            np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(poisoned[0], [[False, True, True, False, False]])
        np.testing.assert_array_equal(poisoned[1], [[0, 0, 1, 1, -100, -100]])
        fake_torch = SimpleNamespace(bool=bool, long=np.int64,
                                     as_tensor=lambda data, dtype, device: np.asarray(data, dtype=dtype))
        with patch.dict(sys.modules, {"torch": fake_torch}):
            inputs = run.model_inputs(batch, "cpu", poisoned)
        self.assertEqual(set(inputs), {"input_ids", "attention_mask", "token_type_ids",
                                       "letter_level_tokens", "labels_mask", "embedding_repeater",
                                       "letter_level_attention_mask"})
        np.testing.assert_array_equal(inputs["labels_mask"], poisoned[0])

    def test_truncation_coverage_uses_only_input_offsets(self):
        batch = self.batch()
        batch["offset_mapping"][0][2] = (0, 0)
        _, repeaters, attention, coverage = run.dna_alignment(batch)
        np.testing.assert_array_equal(repeaters, [[0, 0, -100, -100, -100, -100]])
        np.testing.assert_array_equal(attention, [[1, 1, 1, 1, 0, 0]])
        np.testing.assert_array_equal(coverage, [[True, True, False, False, False, False]])

    def test_projection_preserves_gaps_and_rc_channels(self):
        probabilities = np.arange(24).reshape(6, 4) / 24
        mask = np.array([True, False, True, True, False, False])
        projected = run.project_probabilities(probabilities, mask, 4, stage="edge", is_rc=False)
        np.testing.assert_allclose(projected[[0, 2, 3]], probabilities[[0, 2, 3]])
        self.assertTrue(np.isnan(projected[1]).all())
        rc = run.project_probabilities(probabilities, mask, 4, stage="edge", is_rc=True)
        np.testing.assert_allclose(rc, projected[::-1][:, [1, 0, 3, 2]], equal_nan=True)
        region = run.project_probabilities(probabilities[:, :2], mask, 4, stage="region", is_rc=True)
        np.testing.assert_allclose(region, projected[::-1][:, [1, 0]], equal_nan=True)

    def test_invalid_prediction_is_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            run.project_probabilities(np.array([[float("nan"), 0.1]]),
                                      np.array([True]), 1, stage="region", is_rc=False)

    def test_overlapping_probabilities_and_rc_are_averaged_not_logits(self):
        sums = np.zeros((7, 2), dtype=np.float32)
        counts = np.zeros_like(sums)
        run.accumulate(sums, counts, 1, np.array([[0.0, 0.25], [0.5, 0.75], [1.0, 0.0]]))
        run.accumulate(sums, counts, 2, np.array([[1.0, 0.25], [0.0, 1.0], [0.5, 0.5]]))
        with tempfile.TemporaryDirectory() as temporary:
            paths = run.export_tracks(Path(temporary), "NC_060944.1", 7, "region", sums, counts)
            self.assertEqual([p.name for p in paths], ["intragenic_plus.bw", "intragenic_minus.bw"])
            self.assertEqual(read_bigwig(paths[0])[2], {1: 0.0, 2: 0.75, 3: 0.5, 4: 0.5})
            self.assertEqual(read_bigwig(paths[1])[2], {1: 0.25, 2: 0.5, 3: 0.5, 4: 0.5})

    def test_all_channels_export_eighteen_files_for_six_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            for group in GROUPS:
                for stage, channels in CHANNELS.items():
                    directory = Path(temporary) / group / stage
                    sums = np.full((5, len(channels)), 0.5, dtype=np.float32)
                    paths = run.export_tracks(directory, "chr20", 5, stage, sums, np.ones_like(sums))
                    self.assertEqual(len(paths), len(channels))
                    for path in paths:
                        self.assertEqual(read_bigwig(path)[2], dict.fromkeys(range(5), 0.5))
            self.assertEqual(len(list(Path(temporary).rglob("*.bw"))), 18)


@unittest.skipIf(np is None, "NumPy is part of the existing inference environment")
class RunnerTests(unittest.TestCase):
    def test_mocked_six_model_forward_rc_export_without_reference_access(self):
        calls, datasets = [], []
        batch = SignalTests().batch()
        batch.update({
            "metadata": [SimpleNamespace(genome="GCF_009914755.1_T2T-CHM13v2.0", chrom="NC_060944.1", start=2)],
            "local_start": [0],
        })

        class Poison:
            def __getattribute__(self, name):
                raise AssertionError("Reference values were accessed")

        for name in ("labels", "labels_mask", "letter_level_labels", "letter_level_labels_mask",
                     "pos_weight", "truth_labels"):
            batch[name] = Poison()

        class Dataset:
            def __init__(self, config, **kwargs):
                self.config = config
                self.released = False
                datasets.append(self)
                self.finding_store = SimpleNamespace(
                    keys=lambda: [("GCF_009914755.1_T2T-CHM13v2.0", "NC_060944.1")],
                    span=lambda key: (2, 6, 8),
                )

            def __len__(self):
                return 1

            def release_finding_cache(self):
                self.released = True

        class Model:
            def __init__(self, task):
                self.task = task

            def __call__(self, **kwargs):
                self_keys = {"labels", "letter_level_labels", "pos_weight", "letter_level_labels_mask"}
                if self_keys.intersection(kwargs):
                    raise AssertionError("Reference keys reached the model")
                np.testing.assert_array_equal(kwargs["labels_mask"], [[False, True, True, False, False]])
                calls.append(self.task)
                channels = 4 if self.task == "finding_edge" else 2
                return SimpleNamespace(logits=SignalTests.Tensor(np.zeros((1, 6, channels))))

        fake_torch = SimpleNamespace(
            bool=bool, long=np.int64, no_grad=nullcontext,
            as_tensor=lambda data, dtype, device: np.asarray(data, dtype=dtype),
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        fake_modules = {
            "torch": fake_torch,
            "torch.utils": SimpleNamespace(),
            "torch.utils.data": SimpleNamespace(DataLoader=lambda *args, **kwargs: [batch]),
            "tqdm.auto": SimpleNamespace(tqdm=lambda iterable, **kwargs: iterable),
            "genatator_core": SimpleNamespace(),
            "genatator_core.data": SimpleNamespace(GenatatorCollator=lambda: None, GenatatorDataset=Dataset),
            "genatator_core.infer_common": SimpleNamespace(
                prepare_model=lambda spec, task, device: (Model(task), None, None),
                sigmoid=lambda x: 1 / (1 + np.exp(-x)),
                suppress_repeated_rmt_inference_logs=lambda cfg: nullcontext(),
            ),
            "genatator_core.train_common": SimpleNamespace(
                dataset_family_from_model=lambda cfg: "rmt_unet" if cfg["family"] == "rmt" else "amt_unet",
            ),
        }
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, fake_modules):
            root = Path(temporary)
            checkpoint = root / "pytorch_model.bin"
            checkpoint.touch()
            for group in GROUPS:
                config = json.loads((HERE / "configs" / f"{group}.json").read_text())
                for stage in CHANNELS:
                    config[stage]["inference"]["checkpoint_path"] = str(checkpoint)
                    run.run_stage(config, stage, root / "results", "cpu")
            files = list((root / "results").rglob("*.bw"))
            self.assertEqual(len(files), 18)
            for path in files:
                self.assertEqual(read_bigwig(path)[2], dict.fromkeys(range(2, 6), 0.5))
            manifests = list((root / "results").rglob("prediction_manifest.json"))
            self.assertEqual(len(manifests), 6)
            self.assertFalse(list(root.rglob("*metrics*")))
            self.assertFalse(list(root.rglob("*.gff")))
            self.assertFalse(list(root.rglob("*.f32")))
            self.assertEqual(len(calls), 12)
            self.assertEqual([item.config["reverse_complement"] for item in datasets], [False, True] * 6)
            self.assertTrue(all(item.released for item in datasets))


if __name__ == "__main__":
    unittest.main()
