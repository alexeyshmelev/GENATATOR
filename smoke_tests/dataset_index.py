from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm.auto import tqdm

GF_REPO_ID = "AIRI-Institute/genatator-gene-finding-dataset"
SEG_REPO_ID = "AIRI-Institute/genatator-gene-segmentation-dataset"
GF_SPLIT_PREFIX = "data/test/"
SEG_CONFIG_PREFIX = "val-human/"
INDEX_SCHEMA_VERSION = 4


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
    tmp.replace(path)


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
    value = str(chromosome)
    out = {value}
    if value.lower().startswith("chr"):
        out.add(value[3:])
    elif value.isdigit():
        out.add(f"chr{value}")
    return out


def _chrom_matches(value: Any, aliases: set[str]) -> bool:
    return bool(_chrom_aliases(str(value or "")) & aliases)


def _hf_cache_repo_dir(repo_id: str, repo_type: str = "dataset") -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    prefix = "datasets" if repo_type == "dataset" else "models"
    owner, name = repo_id.split("/", 1)
    return Path(HF_HUB_CACHE).expanduser().resolve() / f"{prefix}--{owner}--{name}"


def _latest_snapshot(repo_id: str, repo_type: str = "dataset") -> Optional[Path]:
    root = _hf_cache_repo_dir(repo_id, repo_type)
    snapshots = root / "snapshots"
    if not snapshots.exists():
        return None
    dirs = [p for p in snapshots.iterdir() if p.is_dir()]
    return max(dirs, key=lambda p: p.stat().st_mtime) if dirs else None


def _resolve_hf_file(repo_id: str, filename: str, local_files_only: bool) -> Path:
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_files_only=True,
        )
        resolved = Path(path).resolve()
        print(f"[dataset-location] reuse HF cache file: {resolved}")
        return resolved
    except Exception as cache_error:
        if local_files_only:
            raise RuntimeError(
                f"Required dataset file is absent from the local HF cache: {repo_id}/{filename}"
            ) from cache_error
        print(f"[dataset-location] downloading required selected file: {repo_id}/{filename}")
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

    print(f"[dataset-index] list repository files once: {repo_id}")
    files = list(HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset"))
    _json_dump(manifest_path, {"repo_id": repo_id, "files": files})
    return files


def _metadata_from_gene_finding_parquet(path: Path) -> Dict[str, Any]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(str(path))
    table = parquet.read(columns=["metadata"], use_threads=False)
    if table.num_rows != 1:
        raise RuntimeError(f"Expected one row in gene-finding parquet {path}, found {table.num_rows}")
    return _parse_metadata(table.column("metadata")[0].as_py())


def _metadata_from_remote_gene_finding_parquet(fs: Any, filename: str) -> Dict[str, Any]:
    """Read only the Parquet footer and metadata column through HTTP range requests."""
    import pyarrow.parquet as pq

    remote_path = f"datasets/{GF_REPO_ID}/{filename}"
    with fs.open(remote_path, "rb") as handle:
        parquet = pq.ParquetFile(handle)
        table = parquet.read(columns=["metadata"], use_threads=False)
    if table.num_rows != 1:
        raise RuntimeError(
            f"Expected one row in remote gene-finding parquet {remote_path}, found {table.num_rows}"
        )
    return _parse_metadata(table.column("metadata")[0].as_py())


def _local_gene_finding_parquets(local_dataset_path: str) -> List[Tuple[str, Path]]:
    root = Path(local_dataset_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    print(f"[dataset-location] gene-finding local source: {root}")
    if root.is_file():
        return [(root.name, root)]

    split_root = root / "data" / "test"
    if split_root.exists():
        files = sorted(split_root.rglob("*.parquet"))
    elif root.name == "test" or "test" in root.parts:
        files = sorted(root.rglob("*.parquet"))
    else:
        files = sorted(
            p for p in root.rglob("*.parquet")
            if "/data/test/" in p.as_posix()
        )
    if not files:
        raise RuntimeError(
            f"No gene-finding test-split parquet files found under {root}. "
            "Pass the dataset snapshot root, data/test directory, or one parquet file."
        )
    return [(str(p.relative_to(root)), p.resolve()) for p in files]


def _remote_gene_finding_parquets(index_dir: Path, refresh: bool) -> Tuple[List[Tuple[str, Optional[Path]]], Optional[Path]]:
    cache_root = _hf_cache_repo_dir(GF_REPO_ID)
    snapshot = _latest_snapshot(GF_REPO_ID)
    print(f"[dataset-location] HF_HUB_CACHE={cache_root.parent}")
    print(f"[dataset-location] gene-finding cache repository={cache_root}")
    print(f"[dataset-location] gene-finding cache snapshot={snapshot or '<not cached>'}")

    manifest = _list_repo_files_cached(
        GF_REPO_ID,
        index_dir / "gene_finding_repo_files.json",
        refresh,
    )
    names = sorted(
        f for f in manifest
        if f.startswith(GF_SPLIT_PREFIX) and f.endswith(".parquet")
    )
    if not names:
        raise RuntimeError(f"No parquet files found under {GF_REPO_ID}/{GF_SPLIT_PREFIX}")
    entries: List[Tuple[str, Optional[Path]]] = []
    for name in names:
        local = snapshot / name if snapshot is not None else None
        entries.append((name, local.resolve() if local is not None and local.exists() else None))
    print(
        f"[dataset-location] gene-finding required source=test parquet_files={len(entries)} "
        f"already_cached={sum(path is not None for _, path in entries)}"
    )
    return entries, snapshot


@dataclass
class GeneFindingSelection:
    index_path: Path
    selected_index_path: Path
    source_samples_scanned: int
    selected_blocks: int
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
) -> GeneFindingSelection:
    """Scan every gene-finding test sample and retain all blocks of one chromosome.

    Only metadata is read for rejected samples. Full DNA/target arrays are never copied
    into the smoke index. The selected block files are loaded lazily, one block at a
    time, by the training/validation/inference datasets.
    """
    alias_set = set(aliases) | _chrom_aliases(chromosome)
    index_dir.mkdir(parents=True, exist_ok=True)
    selected_data_dir.mkdir(parents=True, exist_ok=True)
    safe_chrom = re.sub(r"[^A-Za-z0-9_.-]", "_", chromosome)
    index_path = index_dir / f"gene_finding_test_{safe_chrom}.json"
    selected_index_path = index_dir / f"gene_finding_test_{safe_chrom}_selected_blocks.jsonl"

    if index_path.exists() and selected_index_path.exists() and not refresh:
        payload = _json_load(index_path)
        if (
            payload.get("schema_version") == INDEX_SCHEMA_VERSION
            and payload.get("chromosome") == chromosome
            and all(Path(row["local_path"]).exists() for row in payload.get("selected_blocks", []))
        ):
            print(f"[dataset-index] reuse gene-finding chromosome index: {index_path}")
            for _ in tqdm(payload["selected_blocks"], desc="reuse all selected gene-finding block indexes", unit="block"):
                pass
            return GeneFindingSelection(
                index_path=index_path,
                selected_index_path=selected_index_path,
                source_samples_scanned=int(payload["source_samples_scanned"]),
                selected_blocks=len(payload["selected_blocks"]),
                assembled_length=int(payload["assembled_length"]),
            )

    if local_dataset_path:
        manifest_entries = [(name, path) for name, path in _local_gene_finding_parquets(local_dataset_path)]
        snapshot = None
        source_signature = f"local:{Path(local_dataset_path).expanduser().resolve()}"
        print(f"[dataset-location] gene-finding local test parquet_files={len(manifest_entries)}")
    else:
        manifest_entries, snapshot = _remote_gene_finding_parquets(index_dir, refresh)
        source_signature = f"hf:{GF_REPO_ID}:test"

    metadata_cache_path = index_dir / "gene_finding_test_sample_metadata.json"
    metadata_cache: Dict[str, Dict[str, Any]] = {}
    if metadata_cache_path.exists() and not refresh:
        cached_payload = _json_load(metadata_cache_path)
        if cached_payload.get("source_signature") == source_signature:
            metadata_cache = dict(cached_payload.get("entries", {}))
            print(
                f"[dataset-index] reuse per-sample metadata cache: {metadata_cache_path} "
                f"entries={len(metadata_cache)}"
            )

    remote_fs = None
    if not local_dataset_path:
        from huggingface_hub import HfFileSystem

        remote_fs = HfFileSystem()
        print(
            "[dataset-location] uncached rejected test samples are inspected through "
            "Parquet metadata range reads; only selected chromosome files are downloaded"
        )

    selected: List[Dict[str, Any]] = []
    cached_reads = local_reads = remote_reads = 0
    progress = tqdm(manifest_entries, desc="scan every gene-finding test sample metadata", unit="sample")
    for sample_i, (repo_name, local_path) in enumerate(progress, start=1):
        path: Optional[Path] = Path(local_path).resolve() if local_path is not None else None
        if repo_name in metadata_cache:
            meta = dict(metadata_cache[repo_name])
            cached_reads += 1
        elif path is not None:
            meta = _metadata_from_gene_finding_parquet(path)
            metadata_cache[repo_name] = meta
            local_reads += 1
        else:
            if local_files_only:
                raise RuntimeError(
                    "Cannot scan the complete gene-finding test split with --hf-local-files-only; "
                    f"this parquet is missing from the cache: {repo_name}"
                )
            assert remote_fs is not None
            meta = _metadata_from_remote_gene_finding_parquet(remote_fs, repo_name)
            metadata_cache[repo_name] = meta
            remote_reads += 1

        if _chrom_matches(meta.get("chrom", meta.get("chromosome", "")), alias_set):
            if path is None:
                path = _resolve_hf_file(GF_REPO_ID, repo_name, local_files_only=local_files_only)
            selected.append(
                {
                    "repo_file": repo_name,
                    "local_path": str(path.resolve()),
                    "metadata": meta,
                }
            )

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
            cached=cached_reads,
            local=local_reads,
            remote=remote_reads,
            selected=len(selected),
        )
    progress.close()

    _json_dump(
        metadata_cache_path,
        {
            "schema_version": INDEX_SCHEMA_VERSION,
            "source_signature": source_signature,
            "entries": metadata_cache,
        },
    )
    if not selected:
        raise RuntimeError(f"No gene-finding test samples found for chromosome aliases={sorted(alias_set)}")

    selected.sort(key=lambda row: int(row["metadata"].get("start", 0)))
    starts = [int(row["metadata"].get("start", 0)) for row in selected]
    ends = [int(row["metadata"].get("end", 0)) for row in selected]
    assembled_length = max(ends) - min(starts)

    # The JSONL contains only selected block references and metadata. It is the
    # complete chromosome input for every edge/region train, validation and test job.
    _write_jsonl(
        selected_index_path,
        [
            {
                "parquet_path": row["local_path"],
                "repo_file": row["repo_file"],
                "metadata": row["metadata"],
            }
            for row in selected
        ],
    )
    payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "repo_id": GF_REPO_ID,
        "source_split": "test",
        "chromosome": chromosome,
        "aliases": sorted(alias_set),
        "source_samples_scanned": len(manifest_entries),
        "selected_blocks": selected,
        "assembled_length": assembled_length,
        "selected_index_path": str(selected_index_path),
        "loading_policy": "selected parquet blocks are loaded lazily one at a time; every 50%-overlap model window is used",
    }
    _json_dump(index_path, payload)
    print(
        f"[dataset-index] gene-finding scan complete: source_test_samples={len(manifest_entries)} "
        f"selected_blocks={len(selected)} chromosome={chromosome} "
        f"assembled_total_length={assembled_length}"
    )
    print(f"[dataset-index] saved selected block index: {selected_index_path}")
    return GeneFindingSelection(
        index_path=index_path,
        selected_index_path=selected_index_path,
        source_samples_scanned=len(manifest_entries),
        selected_blocks=len(selected),
        assembled_length=assembled_length,
    )


def _local_segmentation_parquets(local_dataset_path: str) -> List[Tuple[str, Path]]:
    root = Path(local_dataset_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    print(f"[dataset-location] segmentation local source: {root}")
    if root.is_file():
        return [(root.name, root)]

    config_root = root / "val-human"
    if config_root.exists():
        files = sorted(config_root.rglob("*.parquet"))
    elif root.name == "val-human" or "val-human" in root.parts:
        files = sorted(root.rglob("*.parquet"))
    else:
        files = sorted(p for p in root.rglob("*.parquet") if "/val-human/" in p.as_posix())
    if not files:
        raise RuntimeError(
            f"No val-human parquet files found under {root}. "
            "Smoke tests intentionally exclude train-human and train-multi-specie."
        )
    return [(str(p.relative_to(root)), p.resolve()) for p in files]


def _remote_segmentation_parquets(
    index_dir: Path,
    refresh: bool,
    local_files_only: bool,
) -> List[Tuple[str, Path]]:
    cache_root = _hf_cache_repo_dir(SEG_REPO_ID)
    snapshot = _latest_snapshot(SEG_REPO_ID)
    print(f"[dataset-location] HF_HUB_CACHE={cache_root.parent}")
    print(f"[dataset-location] segmentation cache repository={cache_root}")
    print(f"[dataset-location] segmentation cache snapshot={snapshot or '<not cached>'}")
    manifest = _list_repo_files_cached(
        SEG_REPO_ID,
        index_dir / "segmentation_repo_files.json",
        refresh,
    )
    names = sorted(
        f for f in manifest
        if f.startswith(SEG_CONFIG_PREFIX) and f.endswith(".parquet")
    )
    if not names:
        raise RuntimeError(f"No parquet files found under {SEG_REPO_ID}/{SEG_CONFIG_PREFIX}")
    print(
        f"[dataset-location] segmentation required source=val-human parquet_files={len(names)}; "
        "train-human and train-multi-specie are excluded"
    )
    paths = []
    for name in tqdm(names, desc="download/reuse val-human parquet files only", unit="file"):
        paths.append((name, _resolve_hf_file(SEG_REPO_ID, name, local_files_only=local_files_only)))
    return paths


@dataclass
class TranscriptSelection:
    index_path: Path
    selected_parquet_path: Path
    source_rows_scanned: int
    selected_rows: int
    total_nucleotides: int
    transcript_type_counts: Dict[str, int]


def _scan_transcript_metadata_file(
    parquet_path: Path,
    aliases: set[str],
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], int]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(str(parquet_path))
    columns = ["metadata"]
    has_status = "status" in parquet.schema_arrow.names
    if has_status:
        columns.append("status")
    total_batches = (parquet.metadata.num_rows + batch_size - 1) // batch_size
    selected: List[Dict[str, Any]] = []
    global_i = 0
    iterator = parquet.iter_batches(batch_size=batch_size, columns=columns, use_threads=False)
    for batch in tqdm(
        iterator,
        total=total_batches,
        desc=f"scan transcript metadata: {parquet_path.name}",
        unit="batch",
        leave=False,
    ):
        metas = batch.column(batch.schema.get_field_index("metadata")).to_pylist()
        statuses = (
            batch.column(batch.schema.get_field_index("status")).to_pylist()
            if has_status
            else [None] * len(metas)
        )
        for meta_value, status in zip(metas, statuses):
            meta = _parse_metadata(meta_value)
            if _chrom_matches(meta.get("chrom", meta.get("chromosome", "")), aliases):
                selected.append(
                    {
                        "row_index": global_i,
                        "metadata": meta,
                        "status": None if status is None else int(status),
                    }
                )
            global_i += 1
    return selected, int(parquet.metadata.num_rows)


def _selected_rows_by_row_group(parquet: Any, global_indices: Sequence[int]) -> Dict[int, set[int]]:
    starts: List[int] = []
    cursor = 0
    for group_i in range(parquet.metadata.num_row_groups):
        starts.append(cursor)
        cursor += int(parquet.metadata.row_group(group_i).num_rows)
    out: Dict[int, set[int]] = {}
    group_i = 0
    for global_index in sorted(int(x) for x in global_indices):
        while group_i + 1 < len(starts) and global_index >= starts[group_i + 1]:
            group_i += 1
        out.setdefault(group_i, set()).add(global_index - starts[group_i])
    return out


def _copy_selected_transcript_rows(
    selected_entries: Sequence[Dict[str, Any]],
    output_path: Path,
    batch_size: int,
) -> Tuple[int, int, Dict[str, int]]:
    import math

    import pyarrow as pa
    import pyarrow.parquet as pq

    by_file: Dict[str, List[int]] = {}
    for entry in selected_entries:
        by_file.setdefault(str(entry["source_parquet"]), []).append(int(entry["row_index"]))

    plans: Dict[str, Tuple[Any, Dict[int, set[int]]]] = {}
    total_batches = 0
    for filename, indices in by_file.items():
        parquet = pq.ParquetFile(filename)
        plan = _selected_rows_by_row_group(parquet, indices)
        plans[filename] = (parquet, plan)
        total_batches += sum(
            math.ceil(int(parquet.metadata.row_group(group_i).num_rows) / batch_size)
            for group_i in plan
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    writer = None
    copied = 0
    total_nt = 0
    type_counts: Dict[str, int] = {}
    progress = tqdm(total=total_batches, desc="copy every selected chromosome transcript", unit="batch")
    try:
        for filename in sorted(plans):
            parquet, plan = plans[filename]
            for group_i in sorted(plan):
                wanted_local = plan[group_i]
                group_cursor = 0
                iterator = parquet.iter_batches(
                    batch_size=batch_size,
                    row_groups=[group_i],
                    columns=["dna_sequence", "labels", "metadata", "status"],
                    use_threads=False,
                )
                for batch in iterator:
                    take = [j for j in range(batch.num_rows) if group_cursor + j in wanted_local]
                    if take:
                        table = pa.Table.from_batches([batch]).take(pa.array(take, type=pa.int64()))
                        if writer is None:
                            writer = pq.ParquetWriter(str(tmp), table.schema, compression="zstd")
                        writer.write_table(table)
                        copied += table.num_rows
                        dnas = table.column("dna_sequence").to_pylist()
                        metas = table.column("metadata").to_pylist()
                        total_nt += sum(len(str(dna)) for dna in dnas)
                        for meta_value in metas:
                            tx_type = str(_parse_metadata(meta_value).get("transcript_type", ""))
                            type_counts[tx_type] = type_counts.get(tx_type, 0) + 1
                    group_cursor += batch.num_rows
                    progress.update(1)
                    progress.set_postfix(copied=copied, expected=len(selected_entries))
    finally:
        progress.close()
        if writer is not None:
            writer.close()
    if copied != len(selected_entries):
        raise RuntimeError(
            f"Selected transcript extraction mismatch: expected={len(selected_entries)} copied={copied}"
        )
    tmp.replace(output_path)
    return copied, total_nt, type_counts


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
    """Scan every val-human transcript row and persist all requested-chromosome rows."""
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
            and all(Path(path).exists() for path in payload.get("source_parquets", []))
        ):
            print(f"[dataset-index] reuse transcript chromosome index: {index_path}")
            for _ in tqdm(payload["selected_rows"], desc="reuse all selected transcript row indexes", unit="transcript"):
                pass
            return TranscriptSelection(
                index_path=index_path,
                selected_parquet_path=selected_parquet,
                source_rows_scanned=int(payload["source_rows_scanned"]),
                selected_rows=int(payload["selected_row_count"]),
                total_nucleotides=int(payload["selected_total_nucleotides"]),
                transcript_type_counts=dict(payload["transcript_type_counts"]),
            )

    if local_dataset_path:
        source_files = _local_segmentation_parquets(local_dataset_path)
        print(f"[dataset-location] local val-human parquet_files={len(source_files)}")
    else:
        source_files = _remote_segmentation_parquets(index_dir, refresh, local_files_only)

    selected_entries: List[Dict[str, Any]] = []
    source_rows_scanned = 0
    file_progress = tqdm(source_files, desc="scan every val-human parquet file", unit="file")
    for repo_name, parquet_path in file_progress:
        selected, rows = _scan_transcript_metadata_file(parquet_path, alias_set, batch_size)
        source_rows_scanned += rows
        for row in selected:
            row["repo_file"] = repo_name
            row["source_parquet"] = str(parquet_path.resolve())
            selected_entries.append(row)
        file_progress.set_postfix(rows=source_rows_scanned, selected=len(selected_entries))
    file_progress.close()

    if not selected_entries:
        raise RuntimeError(f"No val-human transcripts found for chromosome aliases={sorted(alias_set)}")
    selected_entries.sort(
        key=lambda row: (
            str(row["source_parquet"]),
            int(row["row_index"]),
        )
    )
    selected_count, total_nt, type_counts = _copy_selected_transcript_rows(
        selected_entries,
        selected_parquet,
        batch_size,
    )
    payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "repo_id": SEG_REPO_ID,
        "source_config": "val-human",
        "source_split": "validation",
        "smoke_role": "test",
        "chromosome": chromosome,
        "aliases": sorted(alias_set),
        "source_parquets": [str(path.resolve()) for _, path in source_files],
        "source_rows_scanned": source_rows_scanned,
        "selected_row_count": selected_count,
        "selected_total_nucleotides": total_nt,
        "transcript_type_counts": type_counts,
        "selected_rows": selected_entries,
        "selected_parquet": str(selected_parquet),
    }
    _json_dump(index_path, payload)
    print(
        f"[dataset-index] transcript scan complete: source_val-human_rows={source_rows_scanned} "
        f"selected_transcripts={selected_count} chromosome={chromosome} "
        f"selected_total_nucleotides={total_nt} transcript_types={type_counts}"
    )
    print(f"[dataset-index] saved transcript index={index_path}")
    print(f"[dataset-index] saved selected transcript parquet={selected_parquet}")
    return TranscriptSelection(
        index_path=index_path,
        selected_parquet_path=selected_parquet,
        source_rows_scanned=source_rows_scanned,
        selected_rows=selected_count,
        total_nucleotides=total_nt,
        transcript_type_counts=type_counts,
    )
