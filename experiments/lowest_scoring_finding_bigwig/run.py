#!/usr/bin/env python
"""Run the six selected finding models and export probabilities, never metrics."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import tempfile

from bigwig import BigWigWriter
from selection import CHANNELS, GROUPS


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOGGER = logging.getLogger("finding_bigwig")


def resolve_local(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else REPO / path).resolve()


def validate_config(config, stages=("edge", "region"), *, require_weights=True):
    """Validate all requested models before importing ML libraries or downloading."""
    if config.get("experiment") not in GROUPS:
        raise ValueError("Unknown experiment group")
    for forbidden in ("postprocess", "true_gff", "metrics_json", "metrics_csv", "output_gff"):
        if forbidden in config or forbidden in config.get("inference", {}):
            raise ValueError(f"BigWig-only config must not contain {forbidden}")
    targets = []
    for stage in stages:
        spec = config[stage]
        if spec["task"] != f"finding_{stage}":
            raise ValueError(f"Wrong task for {stage}")
        if not spec["model_name"].startswith(stage + "_"):
            raise ValueError(f"Wrong model name for {stage}")
        data = spec["dataset"]
        if data.get("split") != "test" or len(data.get("genomes", [])) != 1 or len(data.get("chromosomes", [])) != 1:
            raise ValueError("Require exactly one genome/chromosome from the test split")
        if data.get("prewindowed") or any(data.get(k) for k in ("max_rows", "max_windows", "statuses")):
            raise ValueError("Subset/debug dataset restrictions are not allowed for this export")
        targets.append((data["genomes"][0], data["chromosomes"][0]))
        if spec["model"].get("checkpoint_path"):
            raise ValueError("Set only inference.checkpoint_path, not model.checkpoint_path")
        if spec.get("inference", {}).get("batch_size") != 1:
            raise ValueError("Preserve the repository's batch_size=1 inference")
        checkpoint = spec["inference"].get("checkpoint_path")
        if require_weights:
            if not checkpoint or str(checkpoint).startswith("<"):
                raise ValueError(f'{config["experiment"]}.{stage}.inference.checkpoint_path must name your trained checkpoint')
            checkpoint_path = resolve_local(checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            if checkpoint_path.is_dir() and not any((checkpoint_path / filename).is_file() for filename in (
                "model.safetensors", "pytorch_model.bin",
            )):
                raise ValueError(f"The existing loader requires model.safetensors or pytorch_model.bin in {checkpoint_path}")
            if checkpoint_path.is_file() and checkpoint_path.suffix not in {".safetensors", ".bin", ".pt", ".pth"}:
                raise ValueError(f"Expected a model weight file, not {checkpoint_path}")
    if len(set(targets)) != 1:
        raise ValueError("Edge and region must use the same chromosome/assembly")


def dna_alignment(batch):
    """Build content masks/repeaters from DNA lengths and tokenizer offsets only.

    No label tensor, label mask, loss weight, or truth record is read here.
    All six selected models have BPE encoders and nucleotide-resolution UNets.
    """
    import numpy as np

    attention = batch["attention_mask"].cpu().numpy()
    batch_size, token_length = attention.shape
    letter_length = int(batch["letter_level_tokens"].shape[1])
    content = np.zeros((batch_size, token_length), dtype=bool)
    repeaters = np.full((batch_size, letter_length), -100, dtype=np.int64)
    letter_attention = np.zeros((batch_size, letter_length), dtype=np.int64)
    for sample, (dna, offsets) in enumerate(zip(batch["dna_sequence"], batch["offset_mapping"])):
        if len(offsets) != token_length:
            raise ValueError("Tokenizer offsets do not align with input tokens")
        retained_length = min(len(dna), letter_length)
        letter_attention[sample, :retained_length] = 1
        content_index = -1
        for i, ((start, end), attended) in enumerate(zip(offsets, attention[sample])):
            start, end = max(0, int(start)), min(len(dna), int(end))
            if not attended or end <= start:
                continue
            content_index += 1
            content[sample, i] = True
            repeaters[sample, min(start, retained_length):min(end, retained_length)] = content_index
    coverage = (letter_attention != 0) & (repeaters >= 0)
    if not coverage.any(axis=1).all():
        raise ValueError("An input sample has no BPE-covered nucleotide positions")
    return content, repeaters, letter_attention, coverage


def model_inputs(batch, device, alignment):
    import torch

    content, repeaters, letter_attention, _ = alignment
    values = {key: batch[key].to(device) for key in (
        "input_ids", "attention_mask", "token_type_ids", "letter_level_tokens"
    ) if key in batch}
    values.update({
        # Legacy RMT/AMT interfaces call this input-derived content mask labels_mask.
        "labels_mask": torch.as_tensor(content, dtype=torch.bool, device=device),
        "embedding_repeater": torch.as_tensor(repeaters, dtype=torch.long, device=device),
        "letter_level_attention_mask": torch.as_tensor(letter_attention, dtype=torch.long, device=device),
    })
    return values


def project_probabilities(probabilities, coverage, dna_length, *, stage, is_rc):
    import numpy as np

    if probabilities.shape != (len(coverage), len(CHANNELS[stage])):
        raise ValueError("Unexpected nucleotide logits shape")
    retained = probabilities[coverage]
    if not np.isfinite(retained).all():
        raise ValueError("Non-finite model predictions on real nucleotides")
    # UNet logits already occupy their original nucleotide positions. Preserve
    # gaps instead of shifting later predictions left if coverage is noncontiguous.
    output = np.full((dna_length, retained.shape[1]), np.nan, dtype=np.float32)
    size = min(dna_length, len(coverage))
    indices = np.flatnonzero(coverage[:size])
    output[indices] = probabilities[indices]
    if is_rc:
        order = [1, 0, 3, 2] if stage == "edge" else [1, 0]
        output = output[::-1, order]
    return output


def accumulate(sums, counts, start, values):
    import numpy as np

    end = start + len(values)
    if start < 0 or end > len(sums) or values.shape[1] != sums.shape[1]:
        raise ValueError("Prediction coordinates exceed the declared chromosome")
    finite = np.isfinite(values)
    sums[start:end] += np.where(finite, values, 0.0)
    counts[start:end] += finite.astype(np.float32)


def export_tracks(directory, chromosome, chromosome_length, stage, sums, counts):
    """Export in bounded chunks, leaving uncovered bases absent, not zero."""
    import numpy as np

    filenames = []
    with ExitStack() as stack:
        writers = []
        for channel in CHANNELS[stage]:
            name = channel.replace("+", "_plus").replace("-", "_minus") + ".bw"
            path = directory / name
            writers.append(stack.enter_context(BigWigWriter(path, chromosome, chromosome_length)))
            filenames.append(path)
        for start in range(0, chromosome_length, 262144):
            stop = min(start + 262144, chromosome_length)
            chunk_counts = counts[start:stop]
            values = np.full_like(sums[start:stop], np.nan)
            np.divide(sums[start:stop], chunk_counts, out=values, where=chunk_counts > 0)
            for channel, writer in enumerate(writers):
                finite = np.isfinite(values[:, channel])
                boundaries = np.flatnonzero(np.diff(np.r_[False, finite, False]))
                for begin, end in boundaries.reshape(-1, 2):
                    writer.add_values(start + int(begin), values[begin:end, channel])
    return filenames


def run_stage(config, stage, destination, device):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from genatator_core.data import GenatatorCollator, GenatatorDataset
    from genatator_core.infer_common import prepare_model, sigmoid, suppress_repeated_rmt_inference_logs
    from genatator_core.train_common import dataset_family_from_model

    spec = json.loads(json.dumps(config[stage]))
    spec["inference"]["checkpoint_path"] = str(resolve_local(spec["inference"]["checkpoint_path"]))
    if resolve_local(spec["dataset"]["path"]).exists():
        spec["dataset"]["path"] = str(resolve_local(spec["dataset"]["path"]))
    model, tokenizer, nucleotide_tokenizer = prepare_model(spec, spec["task"], device)
    data = dict(spec["dataset"], model_family=dataset_family_from_model(spec["model"]))
    if data["model_family"] not in {"rmt_unet", "amt_unet"}:
        raise ValueError("This experiment supports its selected RMT/AMT+UNet models only")
    expected_key = (data["genomes"][0], data["chromosomes"][0])
    directory = destination / config["experiment"] / spec["model_name"]
    directory.mkdir(parents=True, exist_ok=False)
    sizes = None
    sums = counts = None
    paths = []
    try:
        with tempfile.TemporaryDirectory(prefix="accumulator_", dir=directory) as temporary:
            try:
                passes = (False, True) if config["inference"]["use_reverse_complement"] else (False,)
                with torch.no_grad(), suppress_repeated_rmt_inference_logs(spec["model"]):
                    for is_rc in passes:
                        dataset = GenatatorDataset(dict(data, reverse_complement=is_rc), task=spec["task"],
                                                   tokenizer=tokenizer, nucleotide_tokenizer=nucleotide_tokenizer,
                                                   for_inference=True)
                        try:
                            store = dataset.finding_store
                            if store is None or store.keys() != [expected_key] or not len(dataset):
                                raise ValueError(f"Expected exactly the complete selected chromosome: {expected_key}")
                            current_sizes = store.span(expected_key)
                            if sizes is None:
                                sizes = current_sizes
                                chromosome_length = int(sizes[2])
                                shape = (chromosome_length, len(CHANNELS[stage]))
                                sums = np.memmap(Path(temporary) / "sums.f32", dtype="float32", mode="w+", shape=shape)
                                counts = np.memmap(Path(temporary) / "counts.f32", dtype="float32", mode="w+", shape=shape)
                                sums[:] = 0
                                counts[:] = 0
                            elif sizes != current_sizes:
                                raise ValueError("Chromosome coordinates changed between forward and RC passes")
                            loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=GenatatorCollator())
                            for batch in tqdm(loader, desc=f'{config["experiment"]}:{stage}:rc={is_rc}'):
                                alignment = dna_alignment(batch)
                                output = model(**model_inputs(batch, device, alignment))
                                logits = output["logits"] if isinstance(output, dict) else output.logits
                                if logits.shape[0] != 1:
                                    raise ValueError("Expected one sample per inference batch")
                                raw_logits = logits.detach().float().cpu().numpy()[0]
                                if not np.isfinite(raw_logits[alignment[3][0]]).all():
                                    raise ValueError("Non-finite model logits on real nucleotides")
                                probabilities = sigmoid(raw_logits)
                                values = project_probabilities(probabilities, alignment[3][0],
                                                               len(batch["dna_sequence"][0]), stage=stage, is_rc=is_rc)
                                meta = batch["metadata"][0]
                                if (meta.genome, meta.chrom) != expected_key:
                                    raise ValueError("Unexpected chromosome in an inference batch")
                                start = int(meta.start) + int(batch["local_start"][0])
                                accumulate(sums, counts, start, values)
                        finally:
                            dataset.release_finding_cache()
                if sizes is None:
                    raise ValueError("No chromosome predictions produced")
                paths = export_tracks(directory, expected_key[1], int(sizes[2]), stage, sums, counts)
            finally:
                # Release disk-backed accumulators before TemporaryDirectory cleanup.
                for mapped in (sums, counts):
                    if mapped is not None:
                        mapped.flush()
                        mapped._mmap.close()
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Provenance only: no reference labels, accuracy scores, metrics, or GFFs.
    record = {
        "experiment": config["experiment"], "model": spec["model_name"], "task": spec["task"],
        "checkpoint": spec["inference"]["checkpoint_path"], "genome": expected_key[0],
        "chromosome": expected_key[1], "chromosome_length": int(sizes[2]),
        "signal": "sigmoid probabilities; mean over overlapping windows and orientations",
        "coordinates": "0-based half-open", "files": [path.name for path in paths],
        "model_config": spec["model"], "dataset_config": spec["dataset"],
        "use_reverse_complement": config["inference"]["use_reverse_complement"],
    }
    with (directory / "prediction_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    LOGGER.info("Wrote %s", ", ".join(str(path) for path in paths))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--all", action="store_true", help="Run all three pairs (six checkpoints)")
    source.add_argument("--config", type=Path, help="Run one pair config")
    parser.add_argument("--stage", choices=("edge", "region", "both"), default="both")
    parser.add_argument("--device", help="Override configured device, e.g. cuda:0 or cpu")
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument("--check", action="store_true", help="Validate configs/checkpoint paths without model loading")
    args = parser.parse_args(argv)
    stages = ("edge", "region") if args.stage == "both" else (args.stage,)
    paths = [HERE / "configs" / f"{group}.json" for group in GROUPS] if args.all else [args.config]
    configs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    # Fail before any expensive work if ANY requested checkpoint is unresolved.
    for config in configs:
        validate_config(config, stages)
    if args.check:
        print(f"Validated {len(configs) * len(stages)} checkpoint paths; no models loaded.")
        return 0
    sys.path.insert(0, str(REPO))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = args.output_root.expanduser().resolve() / timestamp
    destination.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for config in configs:
        device = args.device or config["inference"]["device"]
        for stage in stages:
            run_stage(config, stage, destination, device)
    print(f"BigWig files saved under {destination}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
