#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Build and search a tiny packed-bitmap dataset with each IVF ANN executable."""

import argparse
import copy
import csv
import json
import random
import struct
import subprocess
import tempfile
from pathlib import Path

ROWS, QUERIES, DIM, GT_K = 769, 9, 32, 300
METHODS = ("flat", "pq", "sq")


def passes(query, candidate):
    if query == 0:
        return True
    if query == 1:
        return False
    if query == 2:
        return candidate in (17, 513, 768)
    return (candidate * 7 + query * 11) % 19 < query


def write_matrix(path, rows, cols, values, dtype):
    path.write_bytes(
        struct.pack("<II", rows, cols)
        + struct.pack(f"<{len(values)}{dtype}", *values)
    )


def generate(directory):
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(2026)
    # Round before computing GT so the oracle sees the same float32 vectors as the GPU.
    vectors = [
        struct.unpack("<f", struct.pack("<f", rng.uniform(-1, 1)))[0]
        for _ in range((ROWS + QUERIES) * DIM)
    ]
    base, queries = vectors[: ROWS * DIM], vectors[ROWS * DIM :]
    write_matrix(directory / "base.fbin", ROWS, DIM, base, "f")
    write_matrix(directory / "query.fbin", QUERIES, DIM, queries, "f")
    words = [0] * ((ROWS * QUERIES + 31) // 32)
    truth = []
    for query in range(QUERIES):
        eligible = []
        for candidate in range(ROWS):
            if not passes(query, candidate):
                continue
            bit = query * ROWS + candidate
            words[bit // 32] |= 1 << (bit % 32)
            distance = sum(
                (queries[query * DIM + d] - base[candidate * DIM + d]) ** 2
                for d in range(DIM)
            )
            eligible.append((distance, candidate))
        ids = [candidate for _, candidate in sorted(eligible)[:GT_K]]
        # The benchmark GT map requires unique IDs, including out-of-range padding entries.
        truth.extend(
            ids + [0xFFFFFFFF - rank for rank in range(GT_K - len(ids))]
        )
    header = struct.pack(
        "<8sIIQQQ", b"CUVSBMAP", 1, 32, QUERIES, ROWS, len(words)
    )
    (directory / "filter.bmap").write_bytes(
        header + struct.pack(f"<{len(words)}I", *words)
    )
    write_matrix(directory / "groundtruth.ibin", QUERIES, GT_K, truth, "I")
    configs = {}
    for method in METHODS:
        search = [
            dict(nprobe=4, n_queries=batch, k=k)
            for batch, k in ((9, 8), (3, 8), (3, 300))
        ]
        build = dict(nlist=4, niter=10)
        if method == "pq":
            build.update(pq_bits=4, pq_dim=8)
            for params in search:
                params.update(
                    internalDistanceDtype="float", smemLutDtype="float"
                )
            search.append(dict(search[1], refine_ratio=2))
            search.append(
                dict(
                    search[1],
                    refine_ratio=2,
                    dataset_memory_type="device",
                    max_internal_batch_size=1,
                )
            )
        config = {
            "dataset": dict(
                name="ivf-bitmap-smoke",
                base_file="base.fbin",
                query_file="query.fbin",
                groundtruth_neighbors_file="groundtruth.ibin",
                distance="euclidean",
                dtype="float",
                filter=dict(kind="bitmap", file="filter.bmap"),
            ),
            "search_basic_param": dict(k=8, batch_size=3),
            "index": [
                dict(
                    name=f"ivf-{method}-bitmap",
                    algo=f"cuvs_ivf_{method}",
                    file=f"{method}.index",
                    build_param=build,
                    search_params=search,
                )
            ],
        }
        config_path = directory / f"{method}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        latency = copy.deepcopy(config)
        latency["index"][0]["search_params"] = [
            dict(
                search[1],
                n_queries=1,
                max_queries=1,
                benchmark_latency_trace_file=str(
                    directory / f"{method}-latency.csv"
                ),
                benchmark_latency_workload="ivf-bitmap-smoke",
                benchmark_latency_method=method,
                benchmark_latency_repetition=0,
                benchmark_latency_shard_index=0,
                benchmark_latency_warmup_queries=2,
                benchmark_latency_timer_floor_samples=4,
            )
        ]
        latency_path = directory / f"{method}-latency.json"
        latency_path.write_text(json.dumps(latency, indent=2) + "\n")
        configs[method] = (config_path, latency_path)
    return configs


def verify_report(path):
    rows = json.loads(path.read_text())["benchmarks"]
    assert rows, f"No benchmark results in {path}"
    for row in rows:
        assert not row.get("error_occurred", False), row
        for counter in (
            "FilterViolations",
            "InvalidSentinelErrors",
            "SentinelOrderErrors",
            "InvalidSentinelDistanceErrors",
            "DuplicateOutputQueries",
        ):
            assert counter in row and row[counter] == 0, (path, counter, row)


def run(build_dir, directory, configs):
    for method, (config, latency_config) in configs.items():
        binary = (
            build_dir
            / "bench"
            / "ann"
            / f"CUVS_IVF_{method.upper()}_ANN_BENCH"
        )
        common = [
            str(binary),
            f"--data_prefix={directory}",
            f"--index_prefix={directory}",
            "--benchmark_min_time=0.001s",
            "--benchmark_min_warmup_time=0.001",
            "--benchmark_repetitions=1",
            "--benchmark_out_format=json",
        ]
        subprocess.run(
            common + ["--build", "--force", str(config)], check=True
        )
        for threads in (1, 2):
            report = directory / f"{method}-throughput-{threads}.json"
            subprocess.run(
                common
                + [
                    "--search",
                    "--mode=throughput",
                    f"--threads={threads}",
                    f"--benchmark_out={report}",
                    str(config),
                ],
                check=True,
            )
            verify_report(report)
        report = directory / f"{method}-latency-results.json"
        subprocess.run(
            common
            + [
                "--search",
                "--mode=latency",
                "--threads=1",
                f"--benchmark_out={report}",
                str(latency_config),
            ],
            check=True,
        )
        verify_report(report)
        with (directory / f"{method}-latency.csv").open(newline="") as source:
            samples = [
                row
                for row in csv.DictReader(source)
                if row["record_kind"] == "query"
            ]
        assert sorted(int(row["query_local_id"]) for row in samples) == list(
            range(QUERIES)
        )
        print(
            f"IVF-{method.upper()}: bitmap throughput, thread copies, and latency passed"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "build",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Prepare inputs without running CUDA",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="cuvs-ivf-bitmap-") as temporary:
        directory = (args.output_dir or Path(temporary)).resolve()
        configs = generate(directory)
        if args.generate_only:
            if not args.output_dir:
                parser.error(
                    "--generate-only requires --output-dir to preserve inputs"
                )
            print(f"Generated bitmap smoke inputs in {directory}")
        else:
            run(args.build_dir.resolve(), directory, configs)


if __name__ == "__main__":
    main()
