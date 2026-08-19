#!/usr/bin/env python3
"""Static and synthetic checks for the CPU context orchestration and aggregation."""

from __future__ import annotations

import csv
import importlib.util
import json
import pathlib
import py_compile
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent


def load_analyzer():
    spec = importlib.util.spec_from_file_location("cpu_context_analyze", HERE / "analyze.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: pathlib.Path, method: str, queries: int, shard: int, rep: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method", "filtered_seeds", "ef_search", "queries", "chunk", "qps", "recall", "search_seconds",
        "build_seconds", "filter_violations", "underfilled_queries",
        "intrinsic_underfilled_queries", "invalid_count_mismatch_queries",
        "sentinel_error_queries", "duplicate_output_queries", "index_bytes",
        "search_threads", "build_threads",
    ]
    seconds = (queries / 1000.0) * (1.0 + 0.01 * rep + 0.001 * shard)
    row = {
        "method": method,
        "filtered_seeds": 10 if method.endswith("_navix_seeded") else 0,
        "ef_search": 10, "queries": queries,
        "chunk": 512 if queries != 10_000 else 10_000,
        "qps": queries / seconds, "recall": 0.5, "search_seconds": seconds,
        "build_seconds": 0, "filter_violations": 0, "underfilled_queries": 0,
        "intrinsic_underfilled_queries": 0, "invalid_count_mismatch_queries": 0,
        "sentinel_error_queries": 0, "duplicate_output_queries": 0,
        "index_bytes": 1234, "search_threads": 16 if method == "faiss_navix" and queries == 10_000 else 32,
        "build_threads": 24,
    }
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def synthetic_run(root: pathlib.Path) -> None:
    methods = [
        "faiss_navix", "acorn_1", "acorn_gamma",
        "acorn_1_navix_seeded", "acorn_gamma_navix_seeded",
    ]
    workloads = ["yfcc", "em", "emis", "r"]
    created = {
        "run_id": "synthetic",
        "graph_configurations": {
            f"{method}/{workload}": {"bytes": 1234}
            for method in methods for workload in workloads
        },
        "run_config": {
            "repetitions": 3, "methods": methods, "workloads": workloads,
            "ef_search": [10], "yfcc_shard_counts": [2048, 2048, 2048, 2048, 1808],
            "arxiv_shard_counts": [10_000],
            "seeded_methods": ["acorn_1_navix_seeded", "acorn_gamma_navix_seeded"],
        },
    }
    (root / "run.created.json").write_text(json.dumps(created))
    (root / "run.completed.json").write_text(json.dumps({"run_id": "synthetic"}))
    for rep in range(1, 4):
        for method in methods:
            for workload in workloads:
                counts = [2048, 2048, 2048, 2048, 1808] if workload == "yfcc" else [10_000]
                cursor = 0
                for shard, count in enumerate(counts):
                    path = root / f"rep_{rep:02d}/raw/{method}/{workload}/shard_{cursor:05d}_{cursor + count:05d}.csv"
                    write_csv(path, method, count, shard, rep)
                    cursor += count


def main() -> None:
    py_compile.compile(str(HERE / "run.py"), doraise=True)
    py_compile.compile(str(HERE / "analyze.py"), doraise=True)
    analyze = load_analyzer()
    with tempfile.TemporaryDirectory(prefix="cpu_context_test_") as directory:
        root = pathlib.Path(directory)
        synthetic_run(root)
        output = analyze.analyze_run(root)
        rows = list(csv.DictReader((output / "per_rep.csv").open()))
        yfcc = next(
            row for row in rows
            if row["repetition"] == "1" and row["method"] == "faiss_navix" and row["workload"] == "yfcc"
        )
        expected_seconds = sum(
            (count / 1000.0) * (1.0 + 0.01 + 0.001 * shard)
            for shard, count in enumerate([2048, 2048, 2048, 2048, 1808])
        )
        assert abs(float(yfcc["qps"]) - 10_000 / expected_seconds) < 1e-9
        assert (output / "cpu_context_pareto.png").is_file()
        assert len(list(csv.DictReader((output / "summary.csv").open()))) == 20

    with tempfile.TemporaryDirectory(prefix="cpu_context_bad_test_") as directory:
        root = pathlib.Path(directory)
        synthetic_run(root)
        path = root / "rep_01/raw/faiss_navix/em/shard_00000_10000.csv"
        rows = list(csv.DictReader(path.open()))
        fields = list(rows[0])
        rows[0]["filter_violations"] = "1"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        try:
            analyze.analyze_run(root)
        except ValueError as error:
            assert "correctness failure" in str(error)
        else:
            raise AssertionError("filter violation was not rejected")
    print("cpu_context pipeline tests: PASS")


if __name__ == "__main__":
    main()
