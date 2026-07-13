from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from .intervals import binary_intervals


def _norm_tx_type(value: str) -> str:
    v = (value or "").lower().replace("-", "_")
    if v in {"lnc", "lncrna", "lnc_rna", "long_noncoding", "long_non_coding"}:
        return "lnc_RNA"
    return "mRNA"


def _attr(parts: dict) -> str:
    return ";".join(f"{k}={v}" for k, v in parts.items() if v not in {None, ""})


def write_finding_gff(records: List[Dict], path: str | Path) -> None:
    """Write genome-oriented GFF3 for transcript-boundary/gene-finding metrics.

    The annotation leaderboard expects predicted transcript, exon, and optionally
    CDS features on chromosome coordinates. Gene finding predicts transcript
    intervals, so each predicted interval is represented as one gene, one
    transcript, and one exon spanning the full predicted transcript interval.
    """
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("##gff-version 3\n")
        for i, r in enumerate(records, 1):
            chrom = str(r["chrom"])
            start = int(r["start"])
            end = int(r["end"])
            strand = str(r.get("strand", "+"))
            if end <= start:
                continue
            gene_id = str(r.get("gene_id") or f"GENATATOR_gene_{i}")
            tx_id = str(r.get("transcript_id") or f"GENATATOR_tx_{i}")
            typ = _norm_tx_type(str(r.get("transcript_type", "mRNA")))
            gff_start = start + 1
            gff_end = end
            f.write(f"{chrom}\tGENATATOR\tgene\t{gff_start}\t{gff_end}\t.\t{strand}\t.\t{_attr({'ID': gene_id})}\n")
            f.write(f"{chrom}\tGENATATOR\t{typ}\t{gff_start}\t{gff_end}\t.\t{strand}\t.\t{_attr({'ID': tx_id, 'Parent': gene_id})}\n")
            f.write(f"{chrom}\tGENATATOR\texon\t{gff_start}\t{gff_end}\t.\t{strand}\t.\t{_attr({'ID': f'{tx_id}.exon1', 'Parent': tx_id})}\n")


def write_segmentation_gff(records: List[Dict], path: str | Path, coordinate_mode: str = "transcript") -> None:
    """Write GFF3 for the official segmentation metric.

    The segmentation metric expects prediction files with ``seqid`` equal to the
    reference transcript ID. Therefore, by default this writer uses
    transcript-relative coordinates rather than chromosome coordinates. This is
    intentionally different from the gene-finding GFF writer.
    """
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("##gff-version 3\n")
        for i, r in enumerate(records, 1):
            typ = _norm_tx_type(str(r.get("transcript_type", "mRNA")))
            gene_id = str(r.get("gene_id") or f"GENATATOR_gene_{i}")
            tx_id = str(r.get("transcript_id") or f"GENATATOR_tx_{i}")
            strand = str(r.get("strand", "+"))
            exons = [(int(s), int(e)) for s, e in r.get("exons", []) if int(e) > int(s)]
            cds = [(int(s), int(e)) for s, e in r.get("cds", []) if int(e) > int(s)]

            if coordinate_mode == "transcript":
                seqid = tx_id
                tx_len = int(r.get("transcript_length") or r.get("length") or r.get("end", 0) or 0)
                max_end = max([e for _, e in exons + cds] + [tx_len, 1])
                tx_start = 1
                tx_end = max_end
                source_start = 0
            elif coordinate_mode == "genome":
                seqid = str(r["chrom"])
                source_start = int(r.get("start", 0))
                tx_start = source_start + 1
                tx_end = int(r.get("end", source_start + max([e for _, e in exons + cds] + [1])))
            else:
                raise RuntimeError(f"Unsupported segmentation GFF coordinate_mode={coordinate_mode!r}")

            f.write(f"{seqid}\tGENATATOR\tgene\t{tx_start}\t{tx_end}\t.\t{strand}\t.\t{_attr({'ID': gene_id})}\n")
            f.write(f"{seqid}\tGENATATOR\t{typ}\t{tx_start}\t{tx_end}\t.\t{strand}\t.\t{_attr({'ID': tx_id, 'Parent': gene_id})}\n")
            for j, (s, e) in enumerate(exons, 1):
                a = source_start + s + 1 if coordinate_mode == "genome" else s + 1
                b = source_start + e if coordinate_mode == "genome" else e
                f.write(f"{seqid}\tGENATATOR\texon\t{a}\t{b}\t.\t{strand}\t.\t{_attr({'ID': f'{tx_id}.exon{j}', 'Parent': tx_id})}\n")
            if typ == "mRNA":
                for j, (s, e) in enumerate(cds, 1):
                    a = source_start + s + 1 if coordinate_mode == "genome" else s + 1
                    b = source_start + e if coordinate_mode == "genome" else e
                    f.write(f"{seqid}\tGENATATOR\tCDS\t{a}\t{b}\t.\t{strand}\t0\t{_attr({'ID': f'{tx_id}.cds{j}', 'Parent': tx_id})}\n")


def _best_interval_from_scores(scores: np.ndarray, min_len: int = 16) -> List[tuple[int, int]]:
    if scores.size == 0:
        return []
    n = len(scores)
    min_len = max(1, min(int(min_len), n))
    center = int(np.nanargmax(scores))
    start = max(0, min(n - min_len, center - min_len // 2))
    return [(start, start + min_len)]


def labels_to_segmentation_record(meta, probs, threshold=0.5, force_nonempty: bool = False) -> Dict:
    """Decode segmentation tracks with the same class-group argmax as validation.

    ``threshold`` is retained only for backward-compatible call signatures and is
    intentionally ignored: interval decoding does not use independent thresholds.
    """
    scores = np.asarray(probs)
    if scores.ndim != 2 or scores.shape[1] < 5:
        raise RuntimeError(f"Segmentation decoding expects [length, 5] scores, got {scores.shape}")
    exon_track = np.argmax(scores[:, [1, 0, 3]], axis=1) == 0
    cds_track = np.argmax(scores[:, [4, 2]], axis=1) == 0
    exons = binary_intervals(exon_track)
    cds = binary_intervals(cds_track)
    if force_nonempty and not exons and probs.shape[1] > 1:
        exons = _best_interval_from_scores(probs[:, 1], min_len=min(64, len(probs)))
    transcript_type = _norm_tx_type(meta.transcript_type)
    if force_nonempty and transcript_type == "mRNA" and not cds and probs.shape[1] > 4:
        cds = _best_interval_from_scores(probs[:, 4], min_len=min(32, len(probs)))
    tx_id = meta.transcript_id or f"GENATATOR_tx_{meta.chrom}_{meta.start}_{meta.end}"
    return {
        "chrom": meta.chrom,
        "start": meta.start,
        "end": meta.end,
        "strand": meta.strand,
        "transcript_type": transcript_type,
        "gene_id": meta.gene_id or f"{tx_id}.gene",
        "transcript_id": tx_id,
        "transcript_length": int(len(probs)),
        "exons": exons,
        "cds": cds,
    }
