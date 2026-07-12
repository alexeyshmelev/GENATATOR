"""Benchmark-compatible CDS inference from predicted exon intervals.

This is a direct port of the heuristic used by GENATATOR-PIPELINE.  Internal
coordinates are zero-based, half-open intervals.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from Bio import BiopythonWarning
from Bio.Seq import Seq


def _find_segments_ones(array: np.ndarray) -> list[tuple[int, int]]:
    ones_idx = np.where(array == 1)[0]
    if ones_idx.size == 0:
        return []
    split_idx = np.where(np.diff(ones_idx) > 1)[0] + 1
    split_ones_idx = np.split(ones_idx, split_idx)
    return [(int(segment[0]), int(segment[-1]) + 1) for segment in split_ones_idx]


def _exon_mask_to_cds_mask_benchmark(
    exon_preds: np.ndarray,
    seq: str,
    strand: str = "+",
) -> np.ndarray:
    """Return the longest complete ORF as a mask on the original sequence.

    Exonic bases are spliced together before all three reading frames are
    translated.  The winning ORF must start with methionine and end with a stop
    codon.  The strict longest-ORF comparison and tie behavior intentionally
    match the benchmark implementation.
    """

    exon_preds = np.asarray(exon_preds, dtype=np.uint8)

    if len(seq) < 3 or int(np.sum(exon_preds)) < 3:
        return np.zeros_like(exon_preds)

    if strand == "-":
        seq = str(Seq(seq).reverse_complement())
        exon_preds = exon_preds[::-1]

    exon_positions = np.where(exon_preds == 1)[0]
    exon_seq = "".join(np.array(list(seq.upper()), dtype=object)[exon_positions])

    best_len_aa = 0
    best_nt_start = None
    best_nt_end = None

    for frame in range(3):
        sub_seq = exon_seq[frame:]
        if len(sub_seq) < 3:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BiopythonWarning)
            aa_seq = str(Seq(sub_seq).translate(to_stop=False))
        protein_split = aa_seq.split("*")
        aa_seqs = [
            protein + "*" if i < len(protein_split) - 1 else protein
            for i, protein in enumerate(protein_split)
        ]

        aa_start = 0
        for protein in aa_seqs:
            prot_len = len(protein)
            if prot_len == 0:
                continue

            nt_start = frame + aa_start * 3
            nt_end = nt_start + prot_len * 3
            aa_start += prot_len

            if "M" in protein and "*" in protein:
                m_pos = protein.find("M")
                orf = protein[m_pos:]
                orf_len_aa = len(orf)

                if orf[0] == "M" and orf[-1] == "*" and orf_len_aa > best_len_aa:
                    best_len_aa = orf_len_aa
                    best_nt_start = nt_start + m_pos * 3
                    best_nt_end = nt_end

    if best_nt_start is None:
        return np.zeros_like(exon_preds)

    cds_mask = np.zeros_like(exon_preds)
    cds_positions = exon_positions[best_nt_start:best_nt_end]
    cds_mask[cds_positions] = 1

    if strand == "-":
        cds_mask = cds_mask[::-1]

    return cds_mask


def infer_cds_with_benchmark_heuristic(
    sequence: str,
    interval_start: int,
    exons: Sequence[tuple[int, int]],
    strand: str,
) -> list[tuple[int, int]]:
    """Infer CDS intervals using the exact GENATATOR benchmark heuristic."""

    if not exons:
        return []

    exon_mask = np.zeros(len(sequence), dtype=np.uint8)
    for start, end in sorted(exons):
        rel_start = max(0, int(start) - int(interval_start))
        rel_end = min(len(sequence), int(end) - int(interval_start))
        if rel_end > rel_start:
            exon_mask[rel_start:rel_end] = 1

    cds_mask = _exon_mask_to_cds_mask_benchmark(exon_mask, sequence, strand=strand)
    return [
        (start + int(interval_start), end + int(interval_start))
        for start, end in _find_segments_ones(cds_mask)
    ]
