#!/usr/bin/env python3
"""Estimate and enforce GPU-memory requirements for the exact bitmap control.

The estimate intentionally models every long-lived allocation visible in the current
``brute_force_search_filtered`` implementation and adds a sizeable reserve for CUDA/RMM and
library workspaces.  It is a preflight guard, not a claim about byte-exact allocator peaks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

GIB = 1024**3
MIB = 1024**2
SPARSE_SELECTIVITY_NUMERATOR = 1
SPARSE_SELECTIVITY_DENOMINATOR = 10
FIXED_SAFETY_BYTES = 2 * GIB
FRACTIONAL_SAFETY = 0.20


def estimate_gpu_memory(
    *,
    base_rows: int,
    dim: int,
    query_rows: int,
    k: int,
    bitmap_storage_bytes: int,
    passing_count: int,
) -> dict[str, object]:
    """Return a conservative allocation estimate for one complete benchmark shard."""
    values = {
        "base_rows": base_rows,
        "dim": dim,
        "query_rows": query_rows,
        "k": k,
        "bitmap_storage_bytes": bitmap_storage_bytes,
        "passing_count": passing_count,
    }
    if any(
        value <= 0 for key, value in values.items() if key != "passing_count"
    ):
        raise ValueError(
            f"memory-estimate dimensions must be positive: {values}"
        )
    total_predicate_slots = base_rows * query_rows
    if not 0 <= passing_count <= total_predicate_slots:
        raise ValueError(
            f"passing_count={passing_count} is outside [0,{total_predicate_slots}]"
        )

    # cuVS selects the sparse CSR path when sparsity >= 0.9, i.e. selectivity <= 0.1.
    sparse = (
        passing_count * SPARSE_SELECTIVITY_DENOMINATOR
        <= total_predicate_slots * SPARSE_SELECTIVITY_NUMERATOR
    )
    components: dict[str, int] = {
        "resident_float32_base": base_rows * dim * 4,
        "resident_base_l2_norms": base_rows * 4,
        "resident_float32_queries": query_rows * dim * 4,
        "resident_bitmap": bitmap_storage_bytes,
        "benchmark_output_ids_and_distances": query_rows * k * (8 + 4),
        "timed_query_l2_norms": query_rows * 4,
    }
    if sparse:
        # device_csr_matrix<float,int64_t> + the int64 COO row vector retained through select-k.
        components.update(
            {
                "timed_csr_values": passing_count * 4,
                "timed_csr_column_indices": passing_count * 8,
                "timed_csr_row_offsets": (query_rows + 1) * 8,
                "timed_coo_row_indices": passing_count * 8,
            }
        )
        execution_path = "sparse_csr_masked_matmul"
    else:
        # chooseTileSize uses at most a 1-GiB double-streaming target on >8-GiB GPUs.
        # Model its larger half plus all per-column-tile intermediate top-k output.
        tile_rows = min(1024 if dim <= 32 else 512, query_rows)
        initial_target = 512 * MIB
        largest_target = 1 * GIB
        if 2 * tile_rows * base_rows * 4 <= initial_target:
            tile_cols = base_rows
        else:
            tile_cols = min(
                base_rows, max(k, largest_target // (2 * 4 * tile_rows))
            )
        column_tiles = math.ceil(base_rows / tile_cols)
        temporary_output_columns = k * column_tiles
        components.update(
            {
                "timed_dense_distance_tile": tile_rows * tile_cols * 4,
                "timed_dense_tile_topk_distances": (
                    tile_rows * temporary_output_columns * 4
                ),
                "timed_dense_tile_topk_indices": (
                    tile_rows * temporary_output_columns * 8
                ),
            }
        )
        execution_path = "dense_tiled_masked_distance"

    modeled_peak = sum(components.values())
    safety = max(
        FIXED_SAFETY_BYTES, math.ceil(modeled_peak * FRACTIONAL_SAFETY)
    )
    required = modeled_peak + safety
    return {
        "schema_version": 1,
        "execution_path": execution_path,
        "base_rows": base_rows,
        "dim": dim,
        "query_rows": query_rows,
        "k": k,
        "passing_count": passing_count,
        "selectivity": passing_count / total_predicate_slots,
        "components_bytes": components,
        "modeled_peak_bytes": modeled_peak,
        "safety_margin_bytes": safety,
        "required_free_device_bytes": required,
        "safety_policy": (
            "max(2 GiB, 20% of modeled allocations), covering CUDA/RMM context, "
            "conversion/select-k workspaces, and allocator fragmentation"
        ),
    }


def _nvidia_smi_devices() -> list[dict[str, object]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {completed.stderr.strip()}")
    devices: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise ValueError(f"unexpected nvidia-smi row: {line!r}")
        devices.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "free_bytes": int(fields[3]) * MIB,
                "total_bytes": int(fields[4]) * MIB,
            }
        )
    if not devices:
        raise RuntimeError("nvidia-smi reported no GPUs")
    return devices


def selected_device() -> dict[str, object]:
    devices = _nvidia_smi_devices()
    visible = (
        os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", 1)[0].strip()
    )
    if not visible:
        return devices[0]
    if visible.isdigit():
        physical_index = int(visible)
        for device in devices:
            if device["index"] == physical_index:
                return device
    else:
        normalized = visible.removeprefix("GPU-")
        for device in devices:
            uuid = str(device["uuid"])
            if uuid == visible or uuid.removeprefix("GPU-").startswith(
                normalized
            ):
                return device
    raise ValueError(
        f"CUDA_VISIBLE_DEVICES selects {visible!r}, which was not found in nvidia-smi"
    )


def validate_estimate(estimate: dict[str, object]) -> int:
    if int(estimate.get("schema_version", -1)) != 1:
        raise ValueError("unsupported/missing GPU-memory estimate schema")
    required = int(estimate.get("required_free_device_bytes", -1))
    modeled = int(estimate.get("modeled_peak_bytes", -1))
    safety = int(estimate.get("safety_margin_bytes", -1))
    if (
        required <= 0
        or modeled <= 0
        or safety <= 0
        or required != modeled + safety
    ):
        raise ValueError("malformed GPU-memory estimate")
    return required


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-manifest", type=Path, required=True)
    parser.add_argument("--shard-number", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--available-bytes",
        type=int,
        help="test/debug override; otherwise query the first CUDA-visible GPU with nvidia-smi",
    )
    args = parser.parse_args()

    manifest = json.loads(args.exact_manifest.read_text())
    shards = {
        int(shard["shard_number"]): shard
        for shard in manifest.get("shards", [])
    }
    if args.shard_number not in shards:
        raise ValueError(
            f"unknown shard {args.shard_number} in {args.exact_manifest}"
        )
    estimate = shards[args.shard_number].get("gpu_memory_estimate")
    if not isinstance(estimate, dict):
        raise TypeError("exact manifest lacks a per-shard GPU-memory estimate")
    required = validate_estimate(estimate)
    if args.available_bytes is None:
        device = selected_device()
        available = int(device["free_bytes"])
    else:
        if args.available_bytes < 0:
            parser.error("--available-bytes must be nonnegative")
        available = args.available_bytes
        device = {
            "index": "override",
            "uuid": "override",
            "name": "test/debug override",
            "free_bytes": available,
            "total_bytes": None,
        }
    passed = available >= required
    result = {
        "schema_version": 1,
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "exact_manifest": str(args.exact_manifest.resolve()),
        "shard_number": args.shard_number,
        "device": device,
        "available_bytes": available,
        "required_free_device_bytes": required,
        "headroom_bytes": available - required,
        "status": "PASS" if passed else "FAIL",
        "estimate": estimate,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(
            f".{args.output.name}.tmp.{os.getpid()}"
        )
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        os.replace(temporary, args.output)
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(
            "insufficient free GPU memory for exact bitmap shard: "
            f"available={available}, required={required}"
        )


if __name__ == "__main__":
    main()
