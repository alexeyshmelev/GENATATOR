"""Single-chromosome, lossless float32 BigWig v4 writer; standard library only.

Writes zlib-compressed fixedStep sections, a chromosome B+ tree, an R-tree,
and the total signal summary. No external converter, library, or executable.
Zoom levels are omitted: all queries use the original one-base signal.
Format references (not runtime dependencies):
https://github.com/ucscGenomeBrowser/kent/blob/master/src/lib/bwgCreate.c
https://github.com/ucscGenomeBrowser/kent/blob/master/src/lib/cirTree.c
https://github.com/ucscGenomeBrowser/kent/blob/master/src/lib/bPlusTree.c
"""

from array import array
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import zlib


BIGWIG_MAGIC = 0x888FFC26
BPT_MAGIC = 0x78CA8C91
RTREE_MAGIC = 0x2468ACE0
HEADER = struct.Struct("<IHHQQQHHQQIQ")
SECTION = struct.Struct("<IIIIIBBH")


class BigWigWriter:
    """Append ordered, nonoverlapping, finite one-base values on one chromosome.

    Publication is exclusive and atomic: an existing output is never replaced.
    Missing positions must be skipped by the caller, never converted to zeros.
    """

    def __init__(self, path, chromosome, chromosome_length, *, block_items=8192,
                 index_fanout=64):
        self.path = Path(path)
        key = chromosome.encode("ascii")
        if not key or b"\x00" in key or any(chr(c).isspace() for c in key):
            raise ValueError("Chromosome must be a nonempty ASCII name without whitespace")
        if not 0 < chromosome_length < 2**32:
            raise ValueError("Chromosome length must fit an unsigned 32-bit integer")
        if not 1 <= block_items <= 65535 or not 2 <= index_fanout <= 65535:
            raise ValueError("Invalid section size or R-tree fanout")
        if self.path.exists():
            raise FileExistsError(self.path)
        self.length = int(chromosome_length)
        self.block_items = block_items
        self.fanout = index_fanout
        self.blocks = []
        self.end = 0
        self.count = 0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.total = 0.0
        self.squares = 0.0
        self.max_uncompressed = 0
        self.closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".partial",
                                         dir=self.path.parent)
        self.temporary = Path(temporary)
        self.file = os.fdopen(fd, "w+b")
        self.file.write(bytes(HEADER.size + 40))  # header and total-summary reservation
        self.chrom_offset = self.file.tell()
        self.file.write(struct.pack("<IIIIQQ", BPT_MAGIC, 1, len(key), 8, 1, 0))
        self.file.write(struct.pack("<BBH", 1, 0, 1))
        self.file.write(key + struct.pack("<II", 0, self.length))
        self.data_offset = self.file.tell()
        self.file.write(bytes(8))  # section count, not nucleotide count

    def add_values(self, start, values):
        if self.closed:
            raise ValueError("Writer is closed")
        if not isinstance(start, int) or start < self.end:
            raise ValueError("Values must be ordered and nonoverlapping")
        if start + len(values) > self.length:
            raise ValueError("Values exceed the declared chromosome length")
        for offset in range(0, len(values), self.block_items):
            # array('f') rounds to the on-disk values before computing the summary.
            block = array("f", values[offset:offset + self.block_items])
            if not all(math.isfinite(value) for value in block):
                raise ValueError("BigWig values must be finite; omit missing intervals")
            begin = start + offset
            stop = begin + len(block)
            self.count += len(block)
            self.minimum = min(self.minimum, min(block))
            self.maximum = max(self.maximum, max(block))
            self.total += math.fsum(block)
            self.squares += math.fsum(value * value for value in block)
            if sys.byteorder != "little":
                block.byteswap()
            raw = SECTION.pack(0, begin, stop, 1, 1, 3, 0, len(block)) + block.tobytes()
            payload = zlib.compress(raw)
            position = self.file.tell()
            self.file.write(payload)
            self.blocks.append((begin, stop, position, len(payload)))
            self.max_uncompressed = max(self.max_uncompressed, len(raw))
            self.end = stop

    def _write_index(self, index_offset):
        # Compact variable-sized nodes, addressed by absolute file offsets.
        nodes = [dict(leaf=True, entries=self.blocks[i:i + self.fanout])
                 for i in range(0, len(self.blocks), self.fanout)]
        for node in nodes:
            node.update(start=node["entries"][0][0], end=node["entries"][-1][1])
        while len(nodes) > 1:
            nodes = [dict(leaf=False, entries=nodes[i:i + self.fanout],
                          start=nodes[i]["start"],
                          end=nodes[min(i + self.fanout, len(nodes)) - 1]["end"])
                     for i in range(0, len(nodes), self.fanout)]
        root = nodes[0]
        self.file.write(struct.pack("<IIQIIIIQII", RTREE_MAGIC, self.fanout,
                                    len(self.blocks), 0, root["start"], 0, root["end"],
                                    index_offset, 1, 0))

        def write_node(node):
            position = self.file.tell()
            entries = node["entries"]
            self.file.write(struct.pack("<BBH", int(node["leaf"]), 0, len(entries)))
            if node["leaf"]:
                for begin, end, data_offset, size in entries:
                    self.file.write(struct.pack("<IIIIQQ", 0, begin, 0, end, data_offset, size))
            else:
                table_offset = self.file.tell()
                self.file.write(bytes(24 * len(entries)))
                children = [(child, write_node(child)) for child in entries]
                after = self.file.tell()
                self.file.seek(table_offset)
                for child, child_offset in children:
                    self.file.write(struct.pack("<IIIIQ", 0, child["start"], 0,
                                                child["end"], child_offset))
                self.file.seek(after)
            return position

        write_node(root)

    def close(self):
        if self.closed:
            return
        try:
            if not self.blocks:
                raise ValueError("Refusing to publish a BigWig without predictions")
            index_offset = self.file.tell()
            self._write_index(index_offset)
            self.file.write(struct.pack("<I", BIGWIG_MAGIC))
            self.file.seek(0)
            self.file.write(HEADER.pack(BIGWIG_MAGIC, 4, 0, self.chrom_offset,
                                        self.data_offset, index_offset, 0, 0, 0, 64,
                                        self.max_uncompressed, 0))
            self.file.write(struct.pack("<Qdddd", self.count, self.minimum, self.maximum,
                                        self.total, self.squares))
            self.file.seek(self.data_offset)
            self.file.write(struct.pack("<Q", len(self.blocks)))
            self.file.flush()
            os.fsync(self.file.fileno())
            self.file.close()
            # Atomic no-replace publication on the same filesystem.
            os.link(self.temporary, self.path)
        finally:
            self.abort()

    def abort(self):
        if not self.closed:
            self.file.close()
            self.temporary.unlink(missing_ok=True)
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        else:
            self.abort()
