#!/usr/bin/env python3
"""CPU-only contract tests for the serialized per-query latency pipeline."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_SPEC = importlib.util.spec_from_file_location(
    "latency_pipeline", SCRIPT_DIR / "latency.py"
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
LATENCY = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(LATENCY)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def shards(root: Path, total: int, shard_size: int) -> list[dict[str, object]]:
    result = []
    first = 0
    number = 0
    while first < total:
        count = min(shard_size, total - first)
        directory = (
            root / f"shard_{number:02d}_{first:05d}_{first + count:05d}"
        )
        result.append(
            {
                "shard_number": number,
                "first_query": first,
                "query_count": count,
                "directory": str(directory),
                "query_file": str(directory / "query.bin"),
                "groundtruth_file": str(directory / "groundtruth.ibin"),
                "bitmap_file": str(directory / "filter.bitmap"),
            }
        )
        first += count
        number += 1
    return result


def write_ibin(path: Path, rows: int, cols: int, offset: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(struct.pack("<II", rows, cols))
        for row in range(rows):
            output.write(
                struct.pack(
                    f"<{cols}I",
                    *(offset + row * cols + rank for rank in range(cols)),
                )
            )


class LatencyPipelineTest(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path, Path, Path]:
        data = root / "data"
        exact = root / "exact"
        profile_path = root / "profile.json"
        selected_path = root / "selected.csv"
        datasets = {}
        for workload in LATENCY.WORKLOADS:
            bitmap_directory = f"navix_bitmap/{'yfcc' if workload == 'yfcc' else f'arxiv-large/{workload}'}"
            datasets[workload] = {
                "bitmap_directory": bitmap_directory,
                "base_file": f"{workload}/base.bin",
                "index_file": f"{workload}/graph.index",
                "dtype": "uint8" if workload == "yfcc" else "float",
                "dataset_size": 10_000_000
                if workload == "yfcc"
                else 2_735_264,
                "dimension": 192 if workload == "yfcc" else 4096,
                "graph_degree": 64,
                "intermediate_graph_degree": 128,
            }
            for phase, total, size in (
                ("correctness_1000", 1_000, 1_000),
                ("throughput_10000", 10_000, 2_048),
            ):
                source_shards = shards(
                    data / bitmap_directory / phase, total, size
                )
                write_json(
                    data / bitmap_directory / phase / "manifest.json",
                    {"schema_version": 1, "shards": source_shards},
                )
                exact_dir = (
                    exact / "yfcc" / phase
                    if workload == "yfcc"
                    else exact / "arxiv-large" / workload / phase
                )
                exact_shards = shards(exact_dir, total, size)
                write_json(
                    exact_dir / "manifest.json",
                    {
                        "schema_version": 1,
                        "method": "cuvs_brute_force_bitmap",
                        "base_file": str(exact / workload / "base.fbin"),
                        "shards": exact_shards,
                    },
                )
        write_json(
            profile_path,
            {
                "schema_version": 1,
                "name": "synthetic-a100-latency",
                "max_queries": 2048,
                "matched_widths": [1, 2],
                "datasets": datasets,
            },
        )
        fields = (
            "workload",
            "method",
            "itopk",
            "search_width",
            "max_iterations",
            "recall_median",
            "qps_median",
            "target_recall",
            "selected",
        )
        with selected_path.open("w", newline="") as output:
            writer = csv.DictWriter(
                output, fieldnames=fields, lineterminator="\n"
            )
            writer.writeheader()
            for workload in LATENCY.WORKLOADS:
                for method in LATENCY.GRAPH_METHODS:
                    writer.writerow(
                        {
                            "workload": workload,
                            "method": method,
                            "itopk": 128,
                            "search_width": 1,
                            "max_iterations": 0,
                            "recall_median": 0.8
                            if workload == "yfcc"
                            else 0.95,
                            "qps_median": 10_000,
                            "target_recall": 0.8
                            if workload == "yfcc"
                            else 0.95,
                            "selected": True,
                        }
                    )
        return data, exact, profile_path, selected_path

    def test_generator_freezes_one_query_calls_and_cyclic_method_order(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, exact, profile, selected = self.fixture(root)
            result = root / "result"
            environment = dict(os.environ)
            environment["RETRIEVE_DATASET_PROFILE"] = str(profile)
            subprocess.run(
                (
                    sys.executable,
                    str(SCRIPT_DIR / "latency.py"),
                    "generate",
                    "--root",
                    str(result),
                    "--data-root",
                    str(data),
                    "--exact-data-root",
                    str(exact),
                    "--selected-points",
                    str(selected),
                    "--profile",
                    str(profile),
                ),
                env=environment,
                check=True,
            )
            manifest = json.loads((result / "manifest.json").read_text())
            self.assertEqual(
                manifest["contracts"]["serialized_max_queries"], 1
            )
            self.assertEqual(
                manifest["contracts"]["queries_per_search_call"], 1
            )
            self.assertEqual(len(manifest["records"]), 76)
            graph_trace = next(
                row
                for row in manifest["records"]
                if row["stage"] == "trace_graph" and row["workload"] == "yfcc"
            )
            config = json.loads(Path(graph_trace["config"]).read_text())
            self.assertEqual(config["search_basic_param"]["batch_size"], 1)
            searches = config["index"][0]["search_params"]
            self.assertEqual(len(searches), 9)
            self.assertEqual(
                {int(row["max_queries"]) for row in searches}, {1}
            )
            self.assertEqual(
                [row["bitmap_method"] for row in searches],
                [
                    "default_cagra",
                    "default_cagra_accumulator",
                    "navix_reference",
                    "default_cagra_accumulator",
                    "navix_reference",
                    "default_cagra",
                    "navix_reference",
                    "default_cagra",
                    "default_cagra_accumulator",
                ],
            )
            exact_trace = next(
                row
                for row in manifest["records"]
                if row["stage"] == "trace_exact"
            )
            exact_config = json.loads(Path(exact_trace["config"]).read_text())
            self.assertEqual(
                exact_config["search_basic_param"]["batch_size"], 1
            )
            self.assertEqual(len(exact_config["index"][0]["search_params"]), 3)
            self.assertTrue(
                all(
                    row["exact_control"] == "bitmap_count_csr_search"
                    for row in exact_config["index"][0]["search_params"]
                )
            )

    def test_trace_validation_requires_complete_rotated_pass_and_floor(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / "trace.csv"
            fields = (
                "schema_version",
                "record_kind",
                "workload",
                "method",
                "repetition",
                "shard_index",
                "query_local_id",
                "query_global_id",
                "call_order",
                "host_latency_ns",
                "itopk",
                "search_width",
                "max_iterations",
                "max_queries",
                "k",
            )
            rows = []
            for order, local in enumerate((2, 0, 1)):
                rows.append(
                    {
                        "schema_version": 1,
                        "record_kind": "query",
                        "workload": "yfcc",
                        "method": "navix_reference",
                        "repetition": 1,
                        "shard_index": 0,
                        "query_local_id": local,
                        "query_global_id": 10 + local,
                        "call_order": order,
                        "host_latency_ns": 100 + order,
                        "itopk": 128,
                        "search_width": 1,
                        "max_iterations": 0,
                        "max_queries": 1,
                        "k": 10,
                    }
                )
            for sample in range(LATENCY.TIMER_FLOOR_SAMPLES):
                row = dict(rows[0])
                row.update(
                    {
                        "record_kind": "timer_floor",
                        "query_local_id": -1,
                        "query_global_id": -1,
                        "call_order": sample,
                        "host_latency_ns": 10,
                    }
                )
                rows.append(row)
            with trace.open("w", newline="") as output:
                writer = csv.DictWriter(
                    output, fieldnames=fields, lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)
            case = {
                "trace": str(trace),
                "method": "navix_reference",
                "repetition": 1,
            }
            record = {
                "workload": "yfcc",
                "shard_index": 0,
                "first_query": 10,
                "query_count": 3,
            }
            result = LATENCY.validate_trace(case, record)
            self.assertEqual(result["queries"], 3)
            rows[1]["query_global_id"] = 99
            with trace.open("w", newline="") as output:
                writer = csv.DictWriter(
                    output, fieldnames=fields, lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "global query IDs"):
                LATENCY.validate_trace(case, record)

    def test_nearest_rank_and_neighbor_set_reader(self) -> None:
        self.assertEqual(
            LATENCY.nearest_rank([5.0, 1.0, 4.0, 2.0, 3.0], 0.50), 3.0
        )
        self.assertEqual(
            LATENCY.nearest_rank([5.0, 1.0, 4.0, 2.0, 3.0], 0.99), 5.0
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "neighbors.ibin"
            write_ibin(path, 3, 10, offset=7)
            rows, cols, values = LATENCY.load_ibin(path)
            self.assertEqual((rows, cols), (3, 10))
            self.assertEqual(values[2][-1], 36)

    def test_correctness_rejects_duplicates_outside_native_base(self) -> None:
        row = {
            "FilterViolations": 0,
            "InvalidSentinelErrors": 0,
            "SentinelOrderErrors": 0,
            "InvalidSentinelDistanceErrors": 0,
            "DuplicateOutputQueries": 0.1,
        }
        LATENCY.validate_correctness("default_cagra", row, Path("native.json"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            LATENCY.validate_correctness(
                "default_cagra_accumulator", row, Path("retain.json")
            )


if __name__ == "__main__":
    unittest.main()
