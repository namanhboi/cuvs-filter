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
    def fixture(
        self, root: Path, k: int = 10
    ) -> tuple[Path, Path, Path, Path, Path]:
        data = root / "data"
        exact = root / "exact"
        profile_path = root / "profile.json"
        matched = root / "matched_recall"
        selected_path = matched / "analysis" / "selected_points.csv"
        selected_provenance = matched / "analysis" / "provenance.json"
        run_provenance = matched / "provenance" / "run.json"
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
                for shard in source_shards:
                    write_ibin(
                        Path(str(shard["directory"])) / "groundtruth.ibin",
                        int(shard["query_count"]),
                        k,
                    )
                write_json(
                    data / bitmap_directory / phase / "manifest.json",
                    {"schema_version": 1, "k": k, "shards": source_shards},
                )
                exact_dir = (
                    exact / "yfcc" / phase
                    if workload == "yfcc"
                    else exact / "arxiv-large" / workload / phase
                )
                exact_shards = shards(exact_dir, total, size)
                for shard in exact_shards:
                    write_ibin(
                        Path(str(shard["groundtruth_file"])),
                        int(shard["query_count"]),
                        k,
                    )
                write_json(
                    exact_dir / "manifest.json",
                    {
                        "schema_version": 1,
                        "method": "cuvs_brute_force_bitmap",
                        "k": k,
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
            "group",
            "phase",
            "workload",
            "graph_degree",
            "intermediate_graph_degree",
            "method",
            "itopk",
            "search_width",
            "max_iterations",
            "repetitions",
            "shards_per_repetition",
            "queries_per_repetition",
            "recall_median",
            "qps_median",
            "target_recall",
            "target_reached",
            "within_target_window",
            "selected",
            "paper_included",
        )
        selected_rows: list[dict[str, object]] = []
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        with selected_path.open("w", newline="") as output:
            writer = csv.DictWriter(
                output, fieldnames=fields, lineterminator="\n"
            )
            writer.writeheader()
            for workload in LATENCY.WORKLOADS:
                for method in LATENCY.GRAPH_METHODS:
                    target = 0.8 if workload == "yfcc" else 0.95
                    unmatched = k == 100 and (
                        (
                            workload in ("yfcc", "emis")
                            and method != "navix_reference"
                        )
                        or (workload == "r" and method == "default_cagra")
                    )
                    row: dict[str, object] = {
                        "group": "matched_recall_final",
                        "phase": "throughput",
                        "workload": workload,
                        "graph_degree": 64,
                        "intermediate_graph_degree": 128,
                        "method": method,
                        "itopk": 128,
                        "search_width": 1,
                        "max_iterations": 0,
                        "repetitions": 3,
                        "shards_per_repetition": 5,
                        "queries_per_repetition": 10_000,
                        "recall_median": target - 0.1 if unmatched else target,
                        "qps_median": 10_000,
                        "target_recall": target,
                        "target_reached": not unmatched,
                        "within_target_window": not unmatched,
                        "selected": True,
                        "paper_included": True,
                    }
                    writer.writerow(row)
                    selected_rows.append(row)
        write_json(
            run_provenance,
            {
                "schema_version": 2,
                "experiment": "retrieve_workshop_gpu_graph",
                "fixed_contract": {
                    "gpu_algo": "SINGLE_CTA",
                    "k": k,
                    "max_queries": 2_048,
                    "reported_throughput_repetitions": 3,
                    "correctness_repetitions": 1,
                    "throughput_queries": 10_000,
                    "correctness_queries": 1_000,
                    "output_set_semantics": "distinct_valid_output_ids_v1",
                },
            },
        )
        # The hash-bound run provenance remains authoritative; the top-level field gives new
        # k=100 analyses an additional early mismatch check.
        write_json(
            selected_provenance,
            {
                "schema_version": 1,
                "experiment": "retrieve_workshop_matched_recall",
                "max_queries": 2_048,
                "k": k,
                "targets": LATENCY.TARGETS,
                "target_window": 0.002,
                "run_provenance": {
                    "path": str(run_provenance.resolve()),
                    "sha256": LATENCY.sha256(run_provenance),
                },
                "analysis_inputs": [
                    {
                        "path": str(selected_path.resolve()),
                        "sha256": LATENCY.sha256(selected_path),
                    }
                ],
                "selected_rows": selected_rows,
            },
        )
        return data, exact, profile_path, selected_path, selected_provenance

    def run_generate(
        self,
        *,
        root: Path,
        data: Path,
        exact: Path,
        profile: Path,
        selected: Path,
        selected_provenance: Path,
        k: int = 10,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        environment["RETRIEVE_DATASET_PROFILE"] = str(profile)
        return subprocess.run(
            (
                sys.executable,
                str(SCRIPT_DIR / "latency.py"),
                "generate",
                "--root",
                str(root),
                "--data-root",
                str(data),
                "--exact-data-root",
                str(exact),
                "--selected-points",
                str(selected),
                "--selected-provenance",
                str(selected_provenance),
                "--profile",
                str(profile),
                "--k",
                str(k),
            ),
            env=environment,
            check=check,
            capture_output=True,
            text=True,
        )

    def test_generator_freezes_one_query_calls_and_cyclic_method_order(
        self,
    ) -> None:
        for k in LATENCY.SUPPORTED_K:
            with self.subTest(k=k), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                data, exact, profile, selected, selected_provenance = (
                    self.fixture(root, k)
                )
                result = root / "result"
                self.run_generate(
                    root=result,
                    data=data,
                    exact=exact,
                    profile=profile,
                    selected=selected,
                    selected_provenance=selected_provenance,
                    k=k,
                )
                manifest = json.loads((result / "manifest.json").read_text())
                self.assertEqual(
                    manifest["contracts"]["serialized_max_queries"], 1
                )
                self.assertEqual(
                    manifest["contracts"]["queries_per_search_call"], 1
                )
                self.assertEqual(manifest["contracts"]["k"], k)
                self.assertEqual(
                    manifest["selected_provenance"]["contract"]["k"], k
                )
                wrong_k = 100 if k == 10 else 10
                with self.assertRaisesRegex(
                    ValueError, "but k=.* was requested"
                ):
                    LATENCY.validate_frozen_contract(
                        result,
                        selected,
                        selected_provenance,
                        expected_k=wrong_k,
                    )
                self.assertEqual(len(manifest["records"]), 76)
                graph_trace = next(
                    row
                    for row in manifest["records"]
                    if row["stage"] == "trace_graph"
                    and row["workload"] == "yfcc"
                )
                self.assertEqual(graph_trace["k"], k)
                config = json.loads(Path(graph_trace["config"]).read_text())
                self.assertEqual(config["search_basic_param"]["batch_size"], 1)
                self.assertEqual(config["search_basic_param"]["k"], k)
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
                if k == 100:
                    self.assertEqual(
                        graph_trace["cases"][0]["operating_point_status"],
                        "maximum_reachable_recall",
                    )
                    self.assertEqual(
                        graph_trace["cases"][2]["operating_point_status"],
                        "target_matched",
                    )
                exact_trace = next(
                    row
                    for row in manifest["records"]
                    if row["stage"] == "trace_exact"
                )
                exact_config = json.loads(
                    Path(exact_trace["config"]).read_text()
                )
                self.assertEqual(
                    exact_config["search_basic_param"]["batch_size"], 1
                )
                self.assertEqual(exact_config["search_basic_param"]["k"], k)
                self.assertEqual(
                    len(exact_config["index"][0]["search_params"]), 3
                )
                self.assertTrue(
                    all(
                        row["exact_control"] == "bitmap_count_csr_search"
                        for row in exact_config["index"][0]["search_params"]
                    )
                )

    def test_generator_rejects_k100_selected_run_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, exact, profile, selected, analysis = self.fixture(root)
            analysis_payload = json.loads(analysis.read_text())
            run = Path(analysis_payload["run_provenance"]["path"])
            run_payload = json.loads(run.read_text())
            run_payload["fixed_contract"]["k"] = 100
            write_json(run, run_payload)
            analysis_payload["run_provenance"]["sha256"] = LATENCY.sha256(run)
            write_json(analysis, analysis_payload)
            completed = self.run_generate(
                root=root / "result",
                data=data,
                exact=exact,
                profile=profile,
                selected=selected,
                selected_provenance=analysis,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("not the k=10 A100 contract", completed.stderr)
            self.assertFalse((root / "result/manifest.json").exists())

    def test_generator_rejects_selected_csv_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, exact, profile, selected, analysis = self.fixture(root)
            selected.write_text(selected.read_text() + "\n")
            completed = self.run_generate(
                root=root / "result",
                data=data,
                exact=exact,
                profile=profile,
                selected=selected,
                selected_provenance=analysis,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("SHA-256", completed.stderr)

    def test_generator_rejects_k100_graph_and_exact_sources(self) -> None:
        for source_kind in ("graph", "exact"):
            with (
                self.subTest(source_kind=source_kind),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                data, exact, profile, selected, analysis = self.fixture(root)
                if source_kind == "graph":
                    manifest = (
                        data
                        / "navix_bitmap/yfcc/throughput_10000/manifest.json"
                    )
                else:
                    manifest = exact / "yfcc/throughput_10000/manifest.json"
                payload = json.loads(manifest.read_text())
                payload["k"] = 100
                write_json(manifest, payload)
                completed = self.run_generate(
                    root=root / "result",
                    data=data,
                    exact=exact,
                    profile=profile,
                    selected=selected,
                    selected_provenance=analysis,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("source manifest is not k=10", completed.stderr)

    def test_resume_contract_rejects_changed_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, exact, profile, selected, analysis = self.fixture(root)
            result = root / "result"
            self.run_generate(
                root=result,
                data=data,
                exact=exact,
                profile=profile,
                selected=selected,
                selected_provenance=analysis,
            )
            manifest = json.loads((result / "manifest.json").read_text())
            ground_truth = Path(
                manifest["source_inputs"][0]["ground_truth"][0]["path"]
            )
            with ground_truth.open("r+b") as output:
                output.seek(8)
                value = struct.unpack("<I", output.read(4))[0]
                output.seek(8)
                output.write(struct.pack("<I", value + 1))
            with self.assertRaisesRegex(ValueError, "ground truth changed"):
                LATENCY.validate_frozen_contract(result, selected, analysis)

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
            for row in rows:
                row["k"] = 100
            record["k"] = 100
            with trace.open("w", newline="") as output:
                writer = csv.DictWriter(
                    output, fieldnames=fields, lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)
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
        row["DuplicateOutputQueries"] = 0
        row["InvalidSentinelDistanceErrors"] = 0.1
        for method in (
            "default_cagra",
            "default_cagra_accumulator",
            "navix_reference",
        ):
            with self.assertRaisesRegex(ValueError, "noncanonical"):
                LATENCY.validate_correctness(method, row, Path(f"{method}.json"))


if __name__ == "__main__":
    unittest.main()
