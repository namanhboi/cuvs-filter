#!/usr/bin/env python3
"""Convert copies of graph-only CAGRA v5 and IVF-Flat v4 indexes for pinned cuVS 26.12."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

CHUNK = 8 * 1024 * 1024


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class Reader:
    def __init__(self, path: Path):
        self.stream = path.open("rb")
        self.size = os.fstat(self.stream.fileno()).st_size
        self.digest = hashlib.sha256()
        self.components = {}

    def read(self, count: int) -> bytes:
        data = self.stream.read(count)
        self.digest.update(data)
        return data

    def tell(self) -> int:
        return self.stream.tell()

    def header(self) -> tuple[tuple, np.dtype, int]:
        version = np.lib.format.read_magic(self)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(self)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(self)
        else:
            raise ValueError(f"unsupported NumPy wire header {version}")
        require(
            not fortran and not dtype.hasobject and dtype.isnative,
            "unsupported NumPy layout",
        )
        size = math.prod(shape) * dtype.itemsize
        require(self.tell() + size <= self.size, "truncated array payload")
        return shape, dtype, size

    def scalar(self, dtype: str, output=None, replacement=None):
        shape, actual_dtype, size = self.header()
        require(
            shape == () and actual_dtype == np.dtype(dtype),
            "incorrect scalar wire type",
        )
        data = self.read(size)
        value = np.frombuffer(data, dtype=actual_dtype)[0].item()
        if output is not None:
            np.save(
                output,
                np.asarray(
                    value if replacement is None else replacement, dtype=dtype
                ),
            )
        return value

    def array(
        self,
        name: str,
        dtype: str,
        shape: tuple,
        output=None,
        trim_rows=None,
        materialize=False,
    ):
        actual_shape, actual_dtype, size = self.header()
        require(
            actual_shape == shape and actual_dtype == np.dtype(dtype),
            f"incorrect {name} layout",
        )
        target_shape = (trim_rows,) if trim_rows is not None else shape
        copied_bytes = math.prod(target_shape) * actual_dtype.itemsize
        require(copied_bytes <= size, f"invalid {name} truncation")
        if output is not None:
            np.lib.format.write_array_header_1_0(
                output,
                {
                    "descr": actual_dtype.str,
                    "fortran_order": False,
                    "shape": target_shape,
                },
            )
        digest = hashlib.sha256()
        pieces = []
        remaining = copied_bytes
        while remaining:
            data = self.read(min(CHUNK, remaining))
            require(bool(data), f"truncated {name}")
            digest.update(data)
            if output is not None:
                output.write(data)
            if materialize:
                pieces.append(data)
            remaining -= len(data)
        # Legacy IVF files include padding IDs. Consume them for the source digest;
        # only the original logical IDs are written to the new format.
        remaining = size - copied_bytes
        while remaining:
            data = self.read(min(CHUNK, remaining))
            require(bool(data), f"truncated {name} padding")
            remaining -= len(data)
        self.components[name] = {
            "dtype": actual_dtype.str,
            "shape": list(target_shape),
            "sha256": digest.hexdigest(),
        }
        if materialize:
            return np.frombuffer(b"".join(pieces), dtype=actual_dtype).reshape(
                target_shape
            )
        return None

    def finish(self) -> None:
        require(self.tell() == self.size, "unexpected trailing index bytes")

    def close(self) -> None:
        self.stream.close()


def dtype_prefix(reader: Reader, output) -> np.dtype:
    prefix = reader.read(4)
    require(len(prefix) == 4, "truncated dtype prefix")
    dtype = np.dtype(prefix.rstrip(b"\0").decode("ascii"))
    require(
        dtype
        in (np.dtype("f4"), np.dtype("f2"), np.dtype("u1"), np.dtype("i1")),
        "unsupported data type",
    )
    if output is not None:
        output.write(prefix)
    return dtype


def walk_cagra(reader: Reader, version: int, output=None) -> dict:
    dtype = dtype_prefix(reader, output)
    actual = reader.scalar("i4", output, replacement=6)
    require(actual == version, f"expected CAGRA v{version}, got v{actual}")
    if version == 6:
        require(
            reader.scalar("u4", output) == 0,
            "only graph-only CAGRA indexes are supported",
        )
    elif output is not None:
        np.save(
            output, np.asarray(0, dtype="u4")
        )  # serialized_dataset_kind::none
    rows = reader.scalar("u4", output)
    dim = reader.scalar("u4", output)
    degree = reader.scalar("u4", output)
    metric = reader.scalar("i4", output)
    require(rows > 0 and degree > 0, "invalid CAGRA graph dimensions")
    reader.array("graph", "u4", (rows, degree), output)
    content_map = reader.scalar("u4", output)
    require(
        content_map in (0, 2),
        "embedded CAGRA datasets need a separate typed conversion",
    )
    if content_map & 2:
        reader.array("source_indices", "u4", (rows,), output)
    reader.finish()
    return {
        "dtype": dtype.str,
        "rows": rows,
        "stored_dim": dim,
        "degree": degree,
        "metric": metric,
    }


def walk_flat(reader: Reader, version: int, output=None) -> dict:
    dtype = dtype_prefix(reader, output)
    actual = reader.scalar("i4", output, replacement=5)
    require(actual == version, f"expected IVF-Flat v{version}, got v{actual}")
    rows = reader.scalar("i8", output)
    dim = reader.scalar("u4", output)
    lists = reader.scalar("u4", output)
    metric = reader.scalar("i4", output)
    adaptive = reader.scalar("u1", output)
    conservative = reader.scalar("u1", output)
    require(
        adaptive in (0, 1) and conservative in (0, 1),
        "invalid IVF boolean flag",
    )
    require(rows > 0 and dim > 0 and lists > 0, "invalid IVF-Flat dimensions")
    reader.array("centroids", "f4", (lists, dim), output)
    has_norms = reader.scalar("u1", output)
    require(has_norms in (0, 1), "invalid IVF norms flag")
    if has_norms:
        reader.array("centroid_norms", "f4", (lists,), output)
    sizes = reader.array(
        "list_sizes", "u4", (lists,), output, materialize=True
    )
    require(
        sum(map(int, sizes)) == rows,
        "IVF logical list sizes do not sum to the dataset size",
    )
    for label, value in enumerate(sizes):
        logical = int(value)
        padded = (logical + 31) // 32 * 32
        stored = reader.scalar("u4", output, replacement=logical)
        expected = padded if version == 4 else logical
        require(
            stored == expected,
            f"unexpected serialized size for IVF list {label}",
        )
        if stored:
            reader.array(
                f"list_{label}/vectors", dtype.str, (padded, dim), output
            )
            reader.array(
                f"list_{label}/ids", "i8", (stored,), output, trim_rows=logical
            )
    reader.finish()
    return {
        "dtype": dtype.str,
        "rows": rows,
        "dim": dim,
        "lists": lists,
        "metric": metric,
        "adaptive_centers": adaptive,
        "conservative_memory_allocation": conservative,
    }


def convert(source: Path, output: Path, kind: str) -> dict:
    require(kind in ("cagra", "ivf_flat"), "unsupported index kind")
    source, output = Path(source).resolve(), Path(output).resolve()
    metadata_path = output.with_name(output.name + ".json")
    if source == output or output.exists() or metadata_path.exists():
        raise FileExistsError("conversion outputs must be new paths")
    walk, source_version, target_version = (
        (walk_cagra, 5, 6) if kind == "cagra" else (walk_flat, 4, 5)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".index-conversion-", dir=output.parent)
    temporary = Path(name)
    try:
        original = Reader(source)
    except Exception:
        os.close(fd)
        temporary.unlink(missing_ok=True)
        raise
    try:
        before = source.stat()
        with os.fdopen(fd, "wb") as stream:
            summary = walk(original, source_version, stream)
            stream.flush()
            os.fsync(stream.fileno())
        converted = Reader(temporary)
        try:
            observed_summary = walk(converted, target_version)
            require(
                summary == observed_summary, "converted index metadata differs"
            )
            require(
                original.components == converted.components,
                "converted index components differ",
            )
            output_hash = converted.digest.hexdigest()
        finally:
            converted.close()
        after = source.stat()
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            "source index changed during conversion",
        )
        metadata = {
            "kind": kind,
            "source": str(source),
            "output": str(output),
            "source_version": source_version,
            "target_version": target_version,
            "source_sha256": original.digest.hexdigest(),
            "output_sha256": output_hash,
            "summary": summary,
            "components": original.components,
            "verified_components": True,
        }
        os.link(temporary, output)
        try:
            with metadata_path.open("x") as stream:
                json.dump(metadata, stream, indent=2)
                stream.write("\n")
        except Exception:
            output.unlink()
            raise
        return metadata
    finally:
        original.close()
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("cagra", "ivf_flat"))
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = convert(args.source, args.output, args.kind)
    print(
        json.dumps(
            {
                key: metadata[key]
                for key in ("output", "kind", "verified_components")
            }
        )
    )


if __name__ == "__main__":
    main()
