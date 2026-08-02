#!/usr/bin/env python3
"""Crop a headered row-major fbin without loading it into memory."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("rows", type=int)
    args = parser.parse_args()
    with args.source.open("rb") as source:
        source_rows, dimensions = struct.unpack("<II", source.read(8))
        if not 0 < args.rows <= source_rows:
            raise ValueError("crop row count is outside the source extent")
        expected = 8 + source_rows * dimensions * 4
        if args.source.stat().st_size != expected:
            raise ValueError("source is not a float32 fbin of the declared extent")
        args.destination.parent.mkdir(parents=True, exist_ok=True)
        with args.destination.open("xb") as destination:
            destination.write(struct.pack("<II", args.rows, dimensions))
            remaining = args.rows * dimensions * 4
            while remaining:
                block = source.read(min(16 * 1024 * 1024, remaining))
                if not block:
                    raise EOFError("source ended during crop")
                destination.write(block)
                remaining -= len(block)
    print(f"cropped {args.source} to {args.rows} x {dimensions}: {args.destination}")


if __name__ == "__main__":
    main()
