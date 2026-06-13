#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def main():
    root = Path(__file__).resolve().parent / 'data'
    seq = ('ACGT' * 256)[:1024]
    # finding: 12 channels, combined + mRNA-only. One + transcript.
    targets = [[0.0] * 12 for _ in seq]
    for i in range(95, 106):
        targets[i][0] = 1.0
        targets[i][6] = 1.0
    for i in range(795, 806):
        targets[i][2] = 1.0
        targets[i][8] = 1.0
    for i in range(100, 801):
        targets[i][4] = 1.0
        targets[i][10] = 1.0
    finding_meta = {
        'genome': 'smoke_genome', 'split': 'train', 'chrom': 'chrSmoke',
        'start': 0, 'end': len(seq), 'chrom_length': len(seq),
        'sequence_length': len(seq), 'target_shape': [len(seq), 12], 'block_size_bp': len(seq)
    }
    finding_row = {'dna_sequence': seq, 'targets': targets, 'metadata': json.dumps(finding_meta)}
    write_jsonl(root / 'finding' / 'train.jsonl', [finding_row, finding_row])
    write_jsonl(root / 'finding' / 'validation.jsonl', [finding_row])

    # segmentation: labels order [5UTR, exon, intron, 3UTR, CDS].
    labels = []
    for i in range(len(seq)):
        y = [0.0, 0.0, 1.0, 0.0, 0.0]
        if 100 <= i < 200 or 300 <= i < 420 or 600 <= i < 800:
            y[1] = 1.0; y[2] = 0.0
        if 320 <= i < 400 or 610 <= i < 760:
            y[4] = 1.0
        if 100 <= i < 200:
            y[0] = 1.0
        if 760 <= i < 800:
            y[3] = 1.0
        labels.append(y)
    seg_meta = 'tx1|gene1|mRNA|+|smoke_genome|chrSmoke|0:1024'
    seg_row = {'dna_sequence': seq, 'labels': labels, 'metadata': seg_meta, 'status': 'representative'}
    write_jsonl(root / 'segmentation' / 'train.jsonl', [seg_row, seg_row])
    write_jsonl(root / 'segmentation' / 'validation.jsonl', [seg_row])

    gff = root / 'reference.gff'
    with open(gff, 'w', encoding='utf-8') as f:
        f.write('##gff-version 3\n')
        f.write('chrSmoke\tsmoke\tgene\t1\t1024\t.\t+\t.\tID=gene1\n')
        f.write('chrSmoke\tsmoke\tmRNA\t1\t1024\t.\t+\t.\tID=tx1;Parent=gene1\n')
        for j, (s, e) in enumerate([(100, 200), (300, 420), (600, 800)], 1):
            f.write(f'chrSmoke\tsmoke\texon\t{s+1}\t{e}\t.\t+\t.\tID=tx1.exon{j};Parent=tx1\n')
        for j, (s, e) in enumerate([(320, 400), (610, 760)], 1):
            f.write(f'chrSmoke\tsmoke\tCDS\t{s+1}\t{e}\t.\t+\t0\tID=tx1.cds{j};Parent=tx1\n')
    print(root)

if __name__ == '__main__':
    main()
