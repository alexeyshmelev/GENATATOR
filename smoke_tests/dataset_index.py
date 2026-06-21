from __future__ import annotations

import gc
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

GF_REPO_ID = "AIRI-Institute/genatator-gene-finding-dataset"
SEG_REPO_ID = "AIRI-Institute/genatator-gene-segmentation-dataset"
GF_SPLIT_PREFIX = "data/test/"
SEG_REMOTE_PARQUET = "val-human/data.parquet"
INDEX_SCHEMA_VERSION = 3


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_metadata(value: Any) -> Dict[str, Any]:
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{"):
            return json.loads(text)
        if "|" in text:
            parts = text.split("|")
            start = end = 0
            if len(parts) > 6 and ":" in parts[6]:
                a, b = parts[6].split(":", 1)
                start, end = int(a), int(b)
            return {
                "transcript_id": parts[0] if len(parts) > 0 else "",
                "gene_id": parts[1] if len(parts) > 1 else "",
                "transcript_type": parts[2] if len(parts) > 2 else "",
                "strand": parts[3] if len(parts) > 3 else "+",
                "genome": parts[4] if len(parts) > 4 else "",
                "chrom": parts[5] if len(parts) > 5 else "",
                "start": start,
                "end": end,
            }
    raise RuntimeError(f"Unsupported metadata value: {type(value)}")


def _chrom_aliases(chromosome: str) -> set[str]:
    out = {str(chromosome)}
    value = str(chromosome)
    if value.lower().startswith("chr"):
        out.add(value[3:])
    elif value.isdigit():
        out.add(f"chr{value}")
    return out


def _chrom_matches(value: Any, aliases: set[str]) -> bool:
    value = str(value or "")
    candidates = _chrom_aliases(value)
    return bool(candidates & aliases)


def _hf_cache_repo_dir(repo_id: str, repo_type: str = "dataset") -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    prefix = "datasets" if repo_type == "dataset" else "models"
    owner, name = repo_id.split("/", 1)
    return Path(HF_HUB_CACHE) / f"{prefix}--{owner}--{name}"


def _latest_snapshot(repo_id: str, repo_type: str = "dataset") -> Optional[Path]:
    root = _hf_cache_repo_dir(repo_id, repo_type)
    snapshots = root / "snapshots"
    if not snapshots.exists():
        return None
    dirs = [p for p in snapshots.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def _resolve_hf_file(repo_id: str, filename: str, local_files_only: bool) -> Path:
    from huggingface_hub import hf_hub_download

    try:
        path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_files_only=True,
            )
        ).resolve()
        print(f"[dataset-location] reuse HF cache file: {path}")
        return path
    except Exception as cache_error:
        if local_files_only:
            raise RuntimeError(
                f"Required dataset file is not in the local HF cache: {repo_id}/{filename}"
            ) from cache_error
        print(f"[dataset-location] downloading one required file: {repo_id}/{filename}")
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_files_only=False,
            )
        ).resolve()


def _list_repo_files_cached(repo_id: str, manifest_path: Path, refresh: bool) -> List[str]:
    if manifest_path.exists() and not refresh:
        payload = _json_load(manifest_path)
        files = list(payload["files"])
        print(f"[dataset-index] reuse repository file manifest: {manifest_path} files={len(files)}")
        return files
    from huggingface_hub import HfApi

    print(f"[dataset-index] listing repository files once: {repo_id}")
    files = HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset")
    _json_dump(manifest_path, {"repo_id": repo_id, "files": files})
    return files


def _metadata_from_gene_finding_parquet(path: Path) -> Dict[str, Any]:
    import pyarrow.parquet as pq

    table = pq.read_table(str(path), columns=["metadata"], memory_map=True)
    if table.num_rows != 1:
        raise RuntimeError(f"Expected one row in gene-finding parquet {path}, found {table.num_rows}")
    return _parse_metadata(table.column("metadata")[0].as_py())




def _metadata_from_remote_gene_finding_parquet(
    fs: Any, repo_id: str, filename: str
) -> Dict[str, Any]:
    """Read only the metadata column from one remote parquet sample.

    HfFileSystem exposes a seekable file object, so PyArrow performs range reads
    for the parquet footer and the tiny metadata column instead of downloading
    the DNA and target columns. This is intentionally used only while building
    the persistent smoke index.
    """
    import pyarrow.parquet as pq

    remote_path = f"datasets/{repo_id}/{filename}"
    with fs.open(remote_path, "rb") as handle:
        parquet = pq.ParquetFile(handle)
        table = parquet.read(columns=["metadata"], use_threads=False)
    if table.num_rows != 1:
        raise RuntimeError(
            f"Expected one row in remote gene-finding parquet {remote_path}, "
            f"found {table.num_rows}"
        )
    return _parse_metadata(table.column("metadata")[0].as_py())


def _nested_target_scalar_to_numpy(scalar: Any) -> np.ndarray:
    """Convert one Arrow scalar containing L x C targets without Python-list expansion."""
    import pyarrow as pa

    outer = scalar.values
    if pa.types.is_fixed_size_list(outer.type):
        width = int(outer.type.list_size)
        flat = outer.values.to_numpy(zero_copy_only=False)
        return np.asarray(flat, dtype=np.float32).reshape(len(outer), width)
    if pa.types.is_list(outer.type) or pa.types.is_large_list(outer.type):
        offsets = outer.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
        widths = np.diff(offsets)
        if widths.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        if not np.all(widths == widths[0]):
            raise RuntimeError("Gene-finding target channel width is not constant")
        width = int(widths[0])
        flat_values = outer.values
        start = int(offsets[0])
        stop = int(offsets[-1])
        flat = flat_values.slice(start, stop - start).to_numpy(zero_copy_only=False)
        return np.asarray(flat, dtype=np.float32).reshape(len(outer), width)
    raise RuntimeError(f"Unsupported Arrow targets type: {outer.type}")


def _read_gene_finding_block(path: Path) -> Tuple[str, np.ndarray, Dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(str(path), columns=["dna_sequence", "targets", "metadata"], memory_map=True)
    if table.num_rows != 1:
        raise RuntimeError(f"Expected one row in gene-finding parquet {path}, found {table.num_rows}")
    dna = str(table.column("dna_sequence")[0].as_py()).upper()
    targets = _nested_target_scalar_to_numpy(table.column("targets")[0])
    metadata = _parse_metadata(table.column("metadata")[0].as_py())
    if len(dna) != targets.shape[0]:
        raise RuntimeError(f"DNA/target length mismatch in {path}: {len(dna)} vs {targets.shape}")
    return dna, targets, metadata


def _evenly_pick(values: np.ndarray, count: int) -> List[int]:
    if values.size == 0 or count <= 0:
        return []
    positions = np.linspace(0, values.size - 1, num=min(count, values.size), dtype=np.int64)
    return [int(values[i]) for i in positions]


def _window_around(position: int, length: int, context: int) -> Tuple[int, int]:
    if length <= context:
        return 0, length
    # Smoke windows are selected from the same 50%-overlap grid used by normal
    # chromosome sliding. The positive position is mapped to the nearest grid
    # window rather than creating an arbitrary crop.
    step = max(1, context // 2)
    desired = max(0, int(position) - context // 2)
    start = (desired // step) * step
    start = max(0, min(length - context, start))
    return start, start + context


def _choose_windows(targets: np.ndarray, context: int, count: int, task: str) -> List[Tuple[int, int]]:
    length = targets.shape[0]
    if task == "edge":
        positive = np.flatnonzero(np.max(targets[:, 0:4], axis=1) > 0.05)
        centers = _evenly_pick(positive, count)
    elif task == "region":
        binary = np.max(targets[:, 4:6], axis=1) >= 0.5
        transitions = np.flatnonzero(binary[1:] != binary[:-1]) + 1
        centers = _evenly_pick(transitions, count)
        if len(centers) < count:
            positive = np.flatnonzero(binary)
            for p in _evenly_pick(positive, count - len(centers)):
                if p not in centers:
                    centers.append(p)
    else:
        raise ValueError(task)
    if not centers:
        centers = [int(x) for x in np.linspace(0, max(0, length - 1), num=max(1, count), dtype=np.int64)]
    windows: List[Tuple[int, int]] = []
    for center in centers:
        window = _window_around(center, length, context)
        if window not in windows:
            windows.append(window)
    fill_positions = np.linspace(0, max(0, length - 1), num=max(1, count * 4), dtype=np.int64)
    for center in fill_positions:
        if len(windows) >= count:
            break
        window = _window_around(int(center), length, context)
        if window not in windows:
            windows.append(window)
    return windows[:count]


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    tmp.replace(path)


@dataclass
class GeneFindingSelection:
    index_path: Path
    edge_data_path: Path
    region_data_path: Path
    selected_blocks: int
    edge_samples: int
    region_samples: int
    assembled_length: int


def prepare_gene_finding_selection(
    *,
    chromosome: str,
    aliases: Sequence[str],
    index_dir: Path,
    selected_data_dir: Path,
    local_dataset_path: Optional[str],
    local_files_only: bool,
    refresh: bool,
    edge_context: int,
    region_context: int,
    windows_per_block: int,
) -> GeneFindingSelection:
    alias_set = set(aliases) | _chrom_aliases(chromosome)
    index_dir.mkdir(parents=True, exist_ok=True)
    selected_data_dir.mkdir(parents=True, exist_ok=True)
    safe_chrom = re.sub(r"[^A-Za-z0-9_.-]", "_", chromosome)
    index_path = index_dir / f"gene_finding_test_{safe_chrom}.json"
    edge_data = selected_data_dir / f"gene_finding_test_{safe_chrom}_edge.jsonl"
    region_data = selected_data_dir / f"gene_finding_test_{safe_chrom}_region.jsonl"

    if index_path.exists() and edge_data.exists() and region_data.exists() and not refresh:
        payload = _json_load(index_path)
        if (
            payload.get("schema_version") == INDEX_SCHEMA_VERSION
            and payload.get("chromosome") == chromosome
            and int(payload.get("edge_context", -1)) == int(edge_context)
            and int(payload.get("region_context", -1)) == int(region_context)
            and int(payload.get("windows_per_block", -1)) == int(windows_per_block)
        ):
            paths_ok = all(Path(x["local_path"]).exists() for x in payload["selected_blocks"])
            if paths_ok:
                print(f"[dataset-index] reuse gene-finding chromosome index: {index_path}")
                for _ in tqdm(payload["selected_blocks"], desc="reuse gene-finding selected block index"):
                    pass
                return GeneFindingSelection(
                    index_path=index_path,
                    edge_data_path=edge_data,
                    region_data_path=region_data,
                    selected_blocks=len(payload["selected_blocks"]),
                    edge_samples=int(payload["edge_samples"]),
                    region_samples=int(payload["region_samples"]),
                    assembled_length=int(payload["assembled_length"]),
                )

    if local_dataset_path:
        root = Path(local_dataset_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        print(f"[dataset-location] gene-finding local dataset: {root}")
        if root.is_file():
            all_test_paths = [root]
        else:
            candidates = list(root.glob("data/test/**/*.parquet"))
            if not candidates:
                candidates = list(root.rglob("*.parquet"))
            all_test_paths = sorted(p.resolve() for p in candidates)
        manifest_entries = [(str(p), p) for p in all_test_paths]
        print(f"[dataset-location] local gene-finding test parquet files={len(manifest_entries)}")
    else:
        snapshot = _latest_snapshot(GF_REPO_ID)
        print(f"[dataset-location] gene-finding HF cache snapshot: {snapshot or '<not cached>'}")
        manifest_file = index_dir / "gene_finding_repo_files.json"
        files = _list_repo_files_cached(GF_REPO_ID, manifest_file, refresh)
        test_files = sorted(f for f in files if f.startswith(GF_SPLIT_PREFIX) and f.endswith(".parquet"))
        manifest_entries = [(f, None) for f in test_files]
        print(f"[dataset-location] gene-finding remote test manifest files={len(manifest_entries)}")

    selected: List[Dict[str, Any]] = []
    remote_fs = None
    source_signature = (
        f"local:{Path(local_dataset_path).expanduser().resolve()}"
        if local_dataset_path
        else f"hf:{GF_REPO_ID}"
    )
    metadata_cache_path = index_dir / "gene_finding_test_sample_metadata.json"
    metadata_cache: Dict[str, Dict[str, Any]] = {}
    if metadata_cache_path.exists() and not refresh:
        cached_payload = _json_load(metadata_cache_path)
        if cached_payload.get("source_signature") == source_signature:
            metadata_cache = dict(cached_payload.get("entries", {}))
            print(
                f"[dataset-index] reuse per-sample gene-finding metadata cache: "
                f"{metadata_cache_path} entries={len(metadata_cache)}"
            )
    if not local_dataset_path:
        from huggingface_hub import HfFileSystem

        remote_fs = HfFileSystem()
        print(
            "[dataset-location] uncached gene-finding samples will be inspected "
            "through HfFileSystem parquet range reads (metadata column only)"
        )

    local_metadata_reads = 0
    remote_metadata_reads = 0
    cached_metadata_reads = 0
    progress = tqdm(
        manifest_entries,
        desc="scan every gene-finding test sample metadata",
        unit="sample",
    )
    for sample_i, (repo_name, local_path) in enumerate(progress, start=1):
        path: Optional[Path]
        if repo_name in metadata_cache:
            meta = dict(metadata_cache[repo_name])
            cached_metadata_reads += 1
            if local_path is not None:
                path = Path(local_path).resolve()
            else:
                cached_path = snapshot / repo_name if snapshot is not None else None
                path = cached_path.resolve() if cached_path is not None and cached_path.exists() else None
        elif local_path is not None:
            path = Path(local_path).resolve()
            meta = _metadata_from_gene_finding_parquet(path)
            metadata_cache[repo_name] = meta
            local_metadata_reads += 1
        else:
            cached_path = snapshot / repo_name if snapshot is not None else None
            if cached_path is not None and cached_path.exists():
                path = cached_path.resolve()
                meta = _metadata_from_gene_finding_parquet(path)
                local_metadata_reads += 1
            else:
                if local_files_only:
                    raise RuntimeError(
                        "Cannot inspect every gene-finding test sample with "
                        "--hf-local-files-only because this parquet is absent from the HF cache: "
                        f"{repo_name}"
                    )
                assert remote_fs is not None
                meta = _metadata_from_remote_gene_finding_parquet(remote_fs, GF_REPO_ID, repo_name)
                path = None
                remote_metadata_reads += 1
            metadata_cache[repo_name] = meta

        if sample_i % 25 == 0:
            _json_dump(
                metadata_cache_path,
                {
                    "schema_version": INDEX_SCHEMA_VERSION,
                    "source_signature": source_signature,
                    "entries": metadata_cache,
                },
            )
        progress.set_postfix(
            cached=cached_metadata_reads,
            local=local_metadata_reads,
            remote=remote_metadata_reads,
            selected=len(selected),
        )

        if _chrom_matches(meta.get("chrom", meta.get("chromosome", "")), alias_set):
            if path is None:
                path = _resolve_hf_file(GF_REPO_ID, repo_name, local_files_only=local_files_only)
            selected.append(
                {
                    "repo_file": repo_name,
                    "local_path": str(Path(path).resolve()),
                    "metadata": meta,
                }
            )

    _json_dump(
        metadata_cache_path,
        {
            "schema_version": INDEX_SCHEMA_VERSION,
            "source_signature": source_signature,
            "entries": metadata_cache,
        },
    )
    print(
        f"[dataset-index] gene-finding metadata scan complete: "
        f"samples={len(manifest_entries)} cached_reads={cached_metadata_reads} "
        f"local_reads={local_metadata_reads} remote_metadata_only_reads={remote_metadata_reads} "
        f"selected={len(selected)} metadata_cache={metadata_cache_path}"
    )

    if not selected:
        raise RuntimeError(f"No gene-finding test blocks found for chromosome aliases={sorted(alias_set)}")
    selected.sort(key=lambda x: int(x["metadata"].get("start", 0)))
    starts = [int(x["metadata"].get("start", 0)) for x in selected]
    ends = [int(x["metadata"].get("end", 0)) for x in selected]
    assembled_length = max(ends) - min(starts)
    print(
        f"[dataset-index] selected gene-finding blocks={len(selected)} chromosome={chromosome} "
        f"assembled_total_length={assembled_length}"
    )

    edge_rows: List[Dict[str, Any]] = []
    region_rows: List[Dict[str, Any]] = []
    block_stats: List[Dict[str, Any]] = []
    for block in tqdm(selected, desc="extract informative windows from every selected gene-finding block"):
        dna, targets, meta = _read_gene_finding_block(Path(block["local_path"]))
        block_start = int(meta.get("start", 0))
        edge_windows = _choose_windows(targets, edge_context, windows_per_block, "edge")
        region_windows = _choose_windows(targets, region_context, windows_per_block, "region")
        for task, windows, out_rows in (
            ("edge", edge_windows, edge_rows),
            ("region", region_windows, region_rows),
        ):
            for local_start, local_end in windows:
                item_meta = dict(meta)
                item_meta["start"] = block_start + local_start
                item_meta["end"] = block_start + local_end
                item_meta["sequence_length"] = local_end - local_start
                item_meta["smoke_source_block_start"] = block_start
                item_meta["smoke_task"] = task
                out_rows.append(
                    {
                        "dna_sequence": dna[local_start:local_end],
                        "targets": targets[local_start:local_end].tolist(),
                        "metadata": item_meta,
                    }
                )
        block_stats.append(
            {
                "local_path": block["local_path"],
                "start": int(meta.get("start", 0)),
                "end": int(meta.get("end", 0)),
                "edge_windows": [[int(a), int(b)] for a, b in edge_windows],
                "region_windows": [[int(a), int(b)] for a, b in region_windows],
                "edge_positive_nucleotides": int(np.count_nonzero(np.max(targets[:, 0:4], axis=1) > 0.05)),
                "region_positive_nucleotides": int(np.count_nonzero(np.max(targets[:, 4:6], axis=1) >= 0.5)),
            }
        )
        del dna, targets
        gc.collect()

    _write_jsonl(edge_data, edge_rows)
    _write_jsonl(region_data, region_rows)
    payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "repo_id": GF_REPO_ID,
        "source_split": "test",
        "chromosome": chromosome,
        "aliases": sorted(alias_set),
        "selected_blocks": selected,
        "block_stats": block_stats,
        "assembled_length": assembled_length,
        "edge_context": edge_context,
        "region_context": region_context,
        "windows_per_block": windows_per_block,
        "edge_samples": len(edge_rows),
        "region_samples": len(region_rows),
        "edge_data_path": str(edge_data),
        "region_data_path": str(region_data),
    }
    _json_dump(index_path, payload)
    print(
        f"[dataset-index] saved gene-finding index={index_path} edge_samples={len(edge_rows)} "
        f"region_samples={len(region_rows)}"
    )
    return GeneFindingSelection(index_path, edge_data, region_data, len(selected), len(edge_rows), len(region_rows), assembled_length)


@dataclass
class TranscriptSelection:
    index_path: Path
    selected_parquet_path: Path
    selected_rows: int
    total_nucleotides: int
    transcript_type_counts: Dict[str, int]


def _resolve_segmentation_parquet(local_dataset_path: Optional[str], local_files_only: bool) -> Path:
    if local_dataset_path:
        p = Path(local_dataset_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        if p.is_file():
            print(f"[dataset-location] segmentation local parquet: {p}")
            return p
        candidates = [p / SEG_REMOTE_PARQUET, p / "data.parquet"]
        candidates += sorted(p.rglob("*.parquet"))
        for candidate in candidates:
            if candidate.exists() and "val-human" in str(candidate):
                print(f"[dataset-location] segmentation local val-human parquet: {candidate}")
                return candidate.resolve()
        raise RuntimeError(f"Could not find val-human parquet under {p}")
    snapshot = _latest_snapshot(SEG_REPO_ID)
    print(f"[dataset-location] segmentation HF cache snapshot: {snapshot or '<not cached>'}")
    return _resolve_hf_file(SEG_REPO_ID, SEG_REMOTE_PARQUET, local_files_only)


def _scan_segmentation_metadata(parquet_path: Path, aliases: set[str], batch_size: int) -> Tuple[List[int], List[Dict[str, Any]], int]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(parquet_path))
    selected_indices: List[int] = []
    selected_meta: List[Dict[str, Any]] = []
    global_i = 0
    iterator = pf.iter_batches(batch_size=batch_size, columns=["metadata", "status"])
    total_batches = (pf.metadata.num_rows + batch_size - 1) // batch_size
    for batch in tqdm(iterator, total=total_batches, desc="scan every val-human transcript metadata row"):
        metas = batch.column(batch.schema.get_field_index("metadata")).to_pylist()
        statuses = batch.column(batch.schema.get_field_index("status")).to_pylist()
        for meta_value, status in zip(metas, statuses):
            meta = _parse_metadata(meta_value)
            if _chrom_matches(meta.get("chrom", ""), aliases):
                selected_indices.append(global_i)
                selected_meta.append({"row_index": global_i, "metadata": meta, "status": int(status)})
            global_i += 1
    return selected_indices, selected_meta, int(pf.metadata.num_rows)


def _extract_selected_segmentation_rows(
    parquet_path: Path,
    selected_indices: Sequence[int],
    output_path: Path,
    batch_size: int,
) -> Tuple[int, int, Dict[str, int]]:
    import math

    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(parquet_path))
    selected_sorted = sorted(int(x) for x in selected_indices)
    selected_set = set(selected_sorted)

    row_group_starts: List[int] = []
    cursor = 0
    for rg_i in range(pf.metadata.num_row_groups):
        row_group_starts.append(cursor)
        cursor += int(pf.metadata.row_group(rg_i).num_rows)

    selected_by_row_group: Dict[int, set[int]] = {}
    rg_i = 0
    for global_index in selected_sorted:
        while (
            rg_i + 1 < len(row_group_starts)
            and global_index >= row_group_starts[rg_i + 1]
        ):
            rg_i += 1
        local_index = global_index - row_group_starts[rg_i]
        selected_by_row_group.setdefault(rg_i, set()).add(local_index)

    print(
        f"[dataset-index] full transcript columns will be read only from "
        f"selected_row_groups={sorted(selected_by_row_group)} / "
        f"total_row_groups={pf.metadata.num_row_groups}; batch_size={batch_size}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    writer = None
    selected_rows = 0
    total_nt = 0
    type_counts: Dict[str, int] = {}
    total_batches = sum(
        math.ceil(int(pf.metadata.row_group(i).num_rows) / batch_size)
        for i in selected_by_row_group
    )
    progress = tqdm(total=total_batches, desc="copy only selected chromosome transcripts")
    try:
        for group_i in sorted(selected_by_row_group):
            wanted_local = selected_by_row_group[group_i]
            group_cursor = 0
            iterator = pf.iter_batches(
                batch_size=batch_size,
                row_groups=[group_i],
                columns=["dna_sequence", "labels", "metadata", "status"],
            )
            for batch in iterator:
                take_local = [
                    j
                    for j in range(batch.num_rows)
                    if group_cursor + j in wanted_local
                ]
                if take_local:
                    table = pa.Table.from_batches([batch]).take(
                        pa.array(take_local, type=pa.int64())
                    )
                    if writer is None:
                        writer = pq.ParquetWriter(str(tmp), table.schema, compression="zstd")
                    writer.write_table(table)
                    selected_rows += table.num_rows
                    dnas = table.column("dna_sequence").to_pylist()
                    metas = table.column("metadata").to_pylist()
                    total_nt += sum(len(str(x)) for x in dnas)
                    for m in metas:
                        transcript_type = str(
                            _parse_metadata(m).get("transcript_type", "")
                        )
                        type_counts[transcript_type] = type_counts.get(transcript_type, 0) + 1
                group_cursor += batch.num_rows
                progress.update(1)
                progress.set_postfix(copied=selected_rows, expected=len(selected_set))
    finally:
        progress.close()
        if writer is not None:
            writer.close()
    if selected_rows != len(selected_indices):
        raise RuntimeError(
            f"Selected transcript extraction mismatch: expected={len(selected_indices)} "
            f"wrote={selected_rows}"
        )
    tmp.replace(output_path)
    return selected_rows, total_nt, type_counts


def prepare_transcript_selection(
    *,
    chromosome: str,
    aliases: Sequence[str],
    index_dir: Path,
    selected_data_dir: Path,
    local_dataset_path: Optional[str],
    local_files_only: bool,
    refresh: bool,
    batch_size: int,
) -> TranscriptSelection:
    alias_set = set(aliases) | _chrom_aliases(chromosome)
    index_dir.mkdir(parents=True, exist_ok=True)
    selected_data_dir.mkdir(parents=True, exist_ok=True)
    safe_chrom = re.sub(r"[^A-Za-z0-9_.-]", "_", chromosome)
    index_path = index_dir / f"segmentation_val-human_validation_{safe_chrom}.json"
    selected_parquet = selected_data_dir / f"segmentation_val-human_validation_{safe_chrom}.parquet"

    if index_path.exists() and selected_parquet.exists() and not refresh:
        payload = _json_load(index_path)
        if (
            payload.get("schema_version") == INDEX_SCHEMA_VERSION
            and payload.get("chromosome") == chromosome
            and Path(payload["source_parquet"]).exists()
        ):
            print(f"[dataset-index] reuse transcript chromosome index: {index_path}")
            for _ in tqdm(payload["selected_rows"], desc="reuse selected transcript row indexes"):
                pass
            return TranscriptSelection(
                index_path=index_path,
                selected_parquet_path=selected_parquet,
                selected_rows=int(payload["selected_row_count"]),
                total_nucleotides=int(payload["selected_total_nucleotides"]),
                transcript_type_counts=dict(payload["transcript_type_counts"]),
            )

    parquet_path = _resolve_segmentation_parquet(local_dataset_path, local_files_only)
    print(f"[dataset-location] segmentation source file: {parquet_path}")
    selected_indices, selected_meta, total_rows = _scan_segmentation_metadata(parquet_path, alias_set, batch_size)
    if not selected_indices:
        raise RuntimeError(f"No val-human transcript rows found for chromosome aliases={sorted(alias_set)}")
    print(
        f"[dataset-index] val-human rows_scanned={total_rows} selected_transcripts={len(selected_indices)} "
        f"chromosome={chromosome}"
    )
    selected_count, total_nt, type_counts = _extract_selected_segmentation_rows(
        parquet_path, selected_indices, selected_parquet, batch_size
    )
    payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "repo_id": SEG_REPO_ID,
        "source_config": "val-human",
        "source_split": "validation",
        "smoke_role": "test",
        "chromosome": chromosome,
        "aliases": sorted(alias_set),
        "source_parquet": str(parquet_path),
        "total_source_rows": total_rows,
        "selected_row_count": selected_count,
        "selected_total_nucleotides": total_nt,
        "transcript_type_counts": type_counts,
        "selected_rows": selected_meta,
        "selected_parquet": str(selected_parquet),
    }
    _json_dump(index_path, payload)
    print(
        f"[dataset-index] saved transcript index={index_path} selected_parquet={selected_parquet} "
        f"transcripts={selected_count} total_nucleotides={total_nt} transcript_types={type_counts}"
    )
    return TranscriptSelection(index_path, selected_parquet, selected_count, total_nt, type_counts)
