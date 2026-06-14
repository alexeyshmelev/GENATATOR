from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .intervals import binary_intervals


def write_finding_gff(records: List[Dict], path: str | Path) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("##gff-version 3\n")
        for i, r in enumerate(records, 1):
            chrom, start, end, strand = r["chrom"], int(r["start"]), int(r["end"]), r["strand"]
            gene_id = f"GENATATOR_gene_{i}"
            tx_id = f"GENATATOR_tx_{i}"
            typ = r.get("transcript_type", "mRNA")
            f.write(f"{chrom}\tGENATATOR\tgene\t{start+1}\t{end}\t.\t{strand}\t.\tID={gene_id}\n")
            f.write(f"{chrom}\tGENATATOR\t{typ}\t{start+1}\t{end}\t.\t{strand}\t.\tID={tx_id};Parent={gene_id}\n")
            f.write(f"{chrom}\tGENATATOR\texon\t{start+1}\t{end}\t.\t{strand}\t.\tID={tx_id}.exon1;Parent={tx_id}\n")


def write_segmentation_gff(records: List[Dict], path: str | Path) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("##gff-version 3\n")
        for i, r in enumerate(records, 1):
            chrom, start, end, strand = r["chrom"], int(r["start"]), int(r["end"]), r.get("strand", "+")
            typ = r.get("transcript_type", "mRNA")
            gene_id = r.get("gene_id") or f"GENATATOR_gene_{i}"
            tx_id = r.get("transcript_id") or f"GENATATOR_tx_{i}"
            f.write(f"{chrom}\tGENATATOR\tgene\t{start+1}\t{end}\t.\t{strand}\t.\tID={gene_id}\n")
            f.write(f"{chrom}\tGENATATOR\t{typ}\t{start+1}\t{end}\t.\t{strand}\t.\tID={tx_id};Parent={gene_id}\n")
            for j, (s, e) in enumerate(r.get("exons", []), 1):
                f.write(f"{chrom}\tGENATATOR\texon\t{start+s+1}\t{start+e}\t.\t{strand}\t.\tID={tx_id}.exon{j};Parent={tx_id}\n")
            if typ == "mRNA":
                for j, (s, e) in enumerate(r.get("cds", []), 1):
                    f.write(f"{chrom}\tGENATATOR\tCDS\t{start+s+1}\t{start+e}\t.\t{strand}\t0\tID={tx_id}.cds{j};Parent={tx_id}\n")


def labels_to_segmentation_record(meta, probs, threshold=0.5) -> Dict:
    pred = probs >= threshold
    exons = binary_intervals(pred[:, 1])
    cds = binary_intervals(pred[:, 4]) if pred.shape[1] > 4 else []
    return {
        "chrom": meta.chrom,
        "start": meta.start,
        "end": meta.end,
        "strand": meta.strand,
        "transcript_type": "lnc_RNA" if meta.transcript_type.lower() in {"lnc_rna", "lncrna"} else "mRNA",
        "gene_id": meta.gene_id,
        "transcript_id": meta.transcript_id,
        "exons": exons,
        "cds": cds,
    }
