#!/usr/bin/env python3
"""Stream standard ANN fvecs into cuVS-bench's contiguous fbin format."""

from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path

import numpy as np


def geometry(path: Path) -> tuple[int, int, int]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(4)
    if len(prefix) != 4:
        raise ValueError(f"truncated fvecs input: {path}")
    dimensions = struct.unpack("<I", prefix)[0]
    if dimensions <= 0:
        raise ValueError(f"invalid fvecs dimension {dimensions}: {path}")
    row_bytes = 4 * (dimensions + 1)
    if size == 0 or size % row_bytes:
        raise ValueError(
            f"fvecs size is not divisible by its row width: {size} % {row_bytes}"
        )
    rows = size // row_bytes
    if rows > 0xFFFFFFFF:
        raise ValueError("fbin row count does not fit uint32")
    return rows, dimensions, row_bytes


def valid_fbin(path: Path, rows: int, dimensions: int) -> bool:
    if not path.is_file() or path.stat().st_size != 8 + rows * dimensions * 4:
        return False
    with path.open("rb") as stream:
        header = stream.read(8)
    return header == struct.pack("<II", rows, dimensions)


def convert(source: Path, output: Path, chunk_rows: int, reuse_valid: bool) -> None:
    rows, dimensions, row_bytes = geometry(source)
    if output.exists():
        if reuse_valid and valid_fbin(output, rows, dimensions):
            print(f"reusing valid {output} ({rows} x {dimensions})")
            return
        raise FileExistsError(f"refusing to replace existing output: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.partial.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    record_dtype = np.dtype([("dimension", "<u4"), ("vector", "<f4", (dimensions,))])
    completed = 0
    try:
        with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
            output_stream.write(struct.pack("<II", rows, dimensions))
            while completed < rows:
                current_rows = min(chunk_rows, rows - completed)
                payload = input_stream.read(current_rows * row_bytes)
                if len(payload) != current_rows * row_bytes:
                    raise ValueError(f"truncated fvecs payload after row {completed}")
                records = np.frombuffer(payload, dtype=record_dtype, count=current_rows)
                if not np.all(records["dimension"] == dimensions):
                    bad = int(np.flatnonzero(records["dimension"] != dimensions)[0])
                    raise ValueError(
                        f"inconsistent fvecs dimension at row {completed + bad}"
                    )
                records["vector"].tofile(output_stream)
                completed += current_rows
                if completed == rows or completed % 100_000 == 0:
                    print(f"converted {completed}/{rows} rows", flush=True)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if not valid_fbin(temporary, rows, dimensions):
            raise ValueError(f"converted fbin failed final geometry check: {temporary}")
        temporary.rename(output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    print(f"created {output} ({rows} x {dimensions})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--reuse-valid", action="store_true")
    args = parser.parse_args()
    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")
    convert(args.source.resolve(), args.output.resolve(), args.chunk_rows, args.reuse_valid)


if __name__ == "__main__":
    main()
