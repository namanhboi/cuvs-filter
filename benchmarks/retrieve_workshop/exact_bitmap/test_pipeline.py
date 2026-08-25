#!/usr/bin/env python3
"""Synthetic completeness/correctness tests for the exact bitmap pipeline."""

from __future__ import annotations

import csv
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
MATRIX_HEADER = struct.Struct("<II")


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args], text=True, capture_output=True, check=check
    )


def create_fixture(root: Path, mode: str) -> Path:
    destination = root / mode
    run(
        str(SCRIPT_DIR / "make_smoke_fixture.py"),
        "--output",
        str(destination),
        "--mode",
        mode,
    )
    return destination / "manifest.json"


def generate(root: Path, manifest: Path, workload: str, phase: str) -> Path:
    destination = root / "configs" / phase / workload
    run(
        str(SCRIPT_DIR / "generate_configs.py"),
        "--exact-manifest",
        str(manifest),
        "--workload",
        workload,
        "--phase",
        phase,
        "--output",
        str(destination),
        "--index-marker",
        str(root / "marker.index"),
    )
    return destination / "manifest.json"


def read_matrix(path: Path, dtype: str) -> np.ndarray:
    with path.open("rb") as stream:
        rows, cols = MATRIX_HEADER.unpack(stream.read(MATRIX_HEADER.size))
        return np.fromfile(stream, dtype=dtype, count=rows * cols).reshape(
            rows, cols
        )


def write_matrix(path: Path, values: np.ndarray) -> None:
    values = np.ascontiguousarray(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(MATRIX_HEADER.pack(*values.shape))
        values.tofile(stream)


def create_preparation_source(
    root: Path, *, source_dtype: str = "float32", gt_width: int = 10
) -> tuple[Path, Path]:
    exact_manifest = create_fixture(root / "fixture", "sparse")
    exact = json.loads(exact_manifest.read_text())
    exact_shard = exact["shards"][0]
    source = root / "source"
    shard = source / "shard_00000_00004"
    shard.mkdir(parents=True)

    base_values = read_matrix(Path(exact["base_file"]), "<f4")
    query_values = read_matrix(Path(exact_shard["query_file"]), "<f4")
    if source_dtype == "uint8":
        base_values = base_values.astype("u1")
        query_values = query_values.astype("u1")
        base = source / "base.u8bin"
    else:
        base = source / "base.fbin"
    write_matrix(base, base_values)
    write_matrix(shard / "query.bin", query_values)

    gt = read_matrix(Path(exact_shard["groundtruth_file"]), "<u4")
    if gt_width != gt.shape[1]:
        widened = np.full(
            (gt.shape[0], gt_width), np.iinfo(np.uint32).max, dtype="<u4"
        )
        widened[:, : min(gt.shape[1], gt_width)] = gt[
            :, : min(gt.shape[1], gt_width)
        ]
        gt = widened
    write_matrix(shard / "groundtruth.ibin", gt)
    bitmap = shard / "filter.bitmap"
    shutil.copyfile(exact_shard["bitmap_file"], bitmap)
    manifest = {
        "schema_version": 1,
        "bitmap_schema": "CUVSBMAP/v1/u32/row-major",
        "base_rows": int(exact["base_rows"]),
        "query_rows": int(exact["query_rows"]),
        "shards": [
            {
                "first_query": 0,
                "query_count": int(exact["query_rows"]),
                "directory": str(shard.resolve()),
                "bitmap": str(bitmap.resolve()),
                "min_passing": int(exact_shard["min_passing"]),
                "max_passing": int(exact_shard["max_passing"]),
                "mean_selectivity": float(exact_shard["mean_selectivity"]),
                "empty_queries": int(exact_shard["empty_queries"]),
            }
        ],
    }
    path = source / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path, base


def raw_record(
    repetition: int, underfilled: float, missing: float, queries: int = 4
) -> dict:
    return {
        "name": "synthetic",
        "run_type": "iteration",
        "repetition_index": repetition,
        "n_queries": queries,
        "k": 10,
        "items_per_second": 1000.0 + repetition,
        "Recall": 1.0,
        "ValidGTRecall": 1.0,
        "ValidGTFraction": 1.0 - missing,
        "native_l2_cutoff_validation": 1.0,
        "resident_bitmap": 1.0,
        "label": 'exact_control="bitmap_count_csr_search"',
        "FilterViolations": 0.0,
        "InvalidSentinelErrors": 0.0,
        "SentinelOrderErrors": 0.0,
        "InvalidSentinelDistanceErrors": 0.0,
        "DuplicateOutputQueries": 0.0,
        "NativeL2CutoffRecall": 1.0,
        "NativeL2CutoffErrors": 0.0,
        "NativeL2StrictPrefixErrors": 0.0,
        "NativeL2CutoffValidated": 1.0,
        "OutputSetSemanticsVersion": 1,
        "UnderfilledQueries": underfilled,
        "MissingResultSlots": missing,
    }


class ExactPipelineTest(unittest.TestCase):
    def test_k100_preparation_and_config_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_manifest, base = create_preparation_source(
                root, gt_width=100
            )
            prepared = root / "prepared"
            run(
                str(SCRIPT_DIR / "prepare_exact_workload.py"),
                "--bitmap-manifest",
                str(source_manifest),
                "--base-file",
                str(base),
                "--source-dtype",
                "float32",
                "--output",
                str(prepared),
                "--k",
                "100",
            )
            manifest = json.loads((prepared / "manifest.json").read_text())
            self.assertEqual(manifest["k"], 100)
            config_root = root / "configs"
            run(
                str(SCRIPT_DIR / "generate_configs.py"),
                "--exact-manifest",
                str(prepared / "manifest.json"),
                "--workload",
                "k100",
                "--phase",
                "smoke",
                "--output",
                str(config_root),
                "--index-marker",
                str(root / "marker.index"),
                "--k",
                "100",
            )
            generated = json.loads((config_root / "manifest.json").read_text())
            config = json.loads(Path(generated["configs"][0]["config"]).read_text())
            self.assertEqual(generated["k"], 100)
            self.assertEqual(config["search_basic_param"]["k"], 100)

    def test_preparation_cache_is_content_bound_and_force_replaces_it(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_manifest, base = create_preparation_source(
                root, source_dtype="uint8"
            )
            output = root / "prepared"
            converted = root / "converted" / "base.fbin"
            command = (
                str(SCRIPT_DIR / "prepare_exact_workload.py"),
                "--bitmap-manifest",
                str(source_manifest),
                "--base-file",
                str(base),
                "--source-dtype",
                "uint8",
                "--converted-base",
                str(converted),
                "--output",
                str(output),
            )
            run(*command)
            first = json.loads((output / "manifest.json").read_text())
            self.assertEqual(
                first["preparation"]["source_base"]["path"],
                str(base.resolve()),
            )
            self.assertTrue(
                Path(str(converted) + ".conversion.json").is_file()
            )
            self.assertGreater(
                first["shards"][0]["gpu_memory_estimate"][
                    "required_free_device_bytes"
                ],
                0,
            )
            self.assertIs(
                first["timed_invalid_sentinel_normalization"], True
            )
            self.assertIn(
                "invalid-sentinel normalization", first["timing_contract"]
            )
            run(*command)  # Provenance-identical cache reuse is accepted.

            stale_timing = dict(first)
            stale_timing["timed_invalid_sentinel_normalization"] = False
            (output / "manifest.json").write_text(
                json.dumps(stale_timing, indent=2) + "\n"
            )
            stale = run(*command, check=False)
            self.assertNotEqual(stale.returncode, 0)
            run(*command, "--force")

            with base.open("r+b") as stream:
                stream.seek(MATRIX_HEADER.size)
                original = stream.read(1)
                stream.seek(MATRIX_HEADER.size)
                stream.write(bytes([original[0] ^ 1]))
            stale = run(*command, check=False)
            self.assertNotEqual(stale.returncode, 0)
            run(*command, "--force")
            second = json.loads((output / "manifest.json").read_text())
            self.assertNotEqual(
                first["preparation"]["source_base"]["sha256"],
                second["preparation"]["source_base"]["sha256"],
            )

            source = json.loads(source_manifest.read_text())
            source["cache_test_revision"] = 1
            source_manifest.write_text(json.dumps(source, indent=2) + "\n")
            stale = run(*command, check=False)
            self.assertNotEqual(stale.returncode, 0)
            run(*command, "--force")

    def test_preparation_rejects_non_k10_gt_and_memory_gate_is_enforced(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad_manifest, bad_base = create_preparation_source(
                root / "bad", gt_width=11
            )
            rejected = run(
                str(SCRIPT_DIR / "prepare_exact_workload.py"),
                "--bitmap-manifest",
                str(bad_manifest),
                "--base-file",
                str(bad_base),
                "--source-dtype",
                "float32",
                "--output",
                str(root / "bad-prepared"),
                check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)

            source_manifest, base = create_preparation_source(root / "good")
            output = root / "prepared"
            run(
                str(SCRIPT_DIR / "prepare_exact_workload.py"),
                "--bitmap-manifest",
                str(source_manifest),
                "--base-file",
                str(base),
                "--source-dtype",
                "float32",
                "--output",
                str(output),
            )
            prepared = output / "manifest.json"
            estimate = json.loads(prepared.read_text())["shards"][0][
                "gpu_memory_estimate"
            ]
            required = int(estimate["required_free_device_bytes"])
            generated = generate(root, prepared, "memory", "smoke")
            generated_row = json.loads(generated.read_text())["configs"][0]
            self.assertEqual(
                int(generated_row["required_free_device_bytes"]), required
            )
            passed_record = root / "memory-pass.json"
            run(
                str(SCRIPT_DIR / "gpu_memory_preflight.py"),
                "--exact-manifest",
                str(prepared),
                "--shard-number",
                "0",
                "--available-bytes",
                str(required),
                "--output",
                str(passed_record),
            )
            self.assertEqual(
                json.loads(passed_record.read_text())["status"], "PASS"
            )
            failed_record = root / "memory-fail.json"
            failed = run(
                str(SCRIPT_DIR / "gpu_memory_preflight.py"),
                "--exact-manifest",
                str(prepared),
                "--shard-number",
                "0",
                "--available-bytes",
                str(required - 1),
                "--output",
                str(failed_record),
                check=False,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(
                json.loads(failed_record.read_text())["status"], "FAIL"
            )

    def test_sparse_and_dense_paths_pass_strict_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for mode, underfilled, missing in (
                ("sparse", 0.75, 0.30),
                ("dense", 0.50, 0.45),
            ):
                manifest = create_fixture(root / "fixtures", mode)
                workload = "yfcc" if mode == "sparse" else mode
                generate(root, manifest, workload, "smoke")
                raw = (
                    root
                    / "raw"
                    / "smoke"
                    / workload
                    / "shard_00.json"
                )
                raw.parent.mkdir(parents=True, exist_ok=True)
                record = raw_record(0, underfilled, missing)
                # Fixed-width GT names only one valid tied-ID choice.  Exactness is determined by
                # the native-distance cutoff for every workload, while canonical overlap remains
                # a diagnostic.  Exercise that rule for both YFCC and a non-YFCC fixture.
                record["ValidGTRecall"] = 0.9
                raw.write_text(
                    json.dumps({"benchmarks": [record]})
                    + "\n"
                )
            run(
                str(SCRIPT_DIR / "analyze_exact.py"),
                "--result-root",
                str(root),
                "--phase",
                "smoke",
            )
            with (root / "analysis" / "exact_summary.csv").open() as stream:
                summary = list(csv.DictReader(stream))
            self.assertEqual(len(summary), 2)
            for row in summary:
                self.assertEqual(row["sentinel_order_errors"], "0.0")
                self.assertEqual(
                    row["invalid_sentinel_distance_errors"], "0.0"
                )
                self.assertEqual(
                    row["timed_invalid_sentinel_normalization"], "True"
                )

    def test_rejects_native_l2_cutoff_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = create_fixture(root / "fixtures", "sparse")
            generate(root, manifest, "yfcc", "smoke")
            raw = root / "raw" / "smoke" / "yfcc" / "shard_00.json"
            raw.parent.mkdir(parents=True, exist_ok=True)
            record = raw_record(0, 0.75, 0.30)
            record["ValidGTRecall"] = 0.9
            record["NativeL2CutoffRecall"] = 0.9
            record["NativeL2CutoffErrors"] = 0.1
            record["NativeL2StrictPrefixErrors"] = 0.1
            raw.write_text(json.dumps({"benchmarks": [record]}) + "\n")
            result = run(
                str(SCRIPT_DIR / "analyze_exact.py"),
                "--result-root",
                str(root),
                "--phase",
                "smoke",
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exact-control correctness failed", result.stderr)

    def test_accepts_tightly_bounded_fast_f32_rank_k_exchange(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = create_fixture(root / "fixtures", "sparse")
            generate(root, manifest, "yfcc", "smoke")
            raw = root / "raw" / "smoke" / "yfcc" / "shard_00.json"
            raw.parent.mkdir(parents=True, exist_ok=True)
            record = raw_record(0, 0.75, 0.30)
            record["NativeL2CutoffRecall"] = 0.99995
            record["NativeL2CutoffErrors"] = 0.00005
            raw.write_text(json.dumps({"benchmarks": [record]}) + "\n")
            result = run(
                str(SCRIPT_DIR / "analyze_exact.py"),
                "--result-root",
                str(root),
                "--phase",
                "smoke",
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(
                (root / "analysis" / "exact_results.json").read_text()
            )
            contract = payload["correctness"]["numerical_contract"]
            self.assertEqual(
                contract["minimum_native_l2_cutoff_recall_per_shard"],
                0.9999,
            )
            self.assertEqual(
                contract["maximum_native_l2_cutoff_error_rate_per_shard"],
                0.0001,
            )

    def test_rejects_missing_and_duplicate_repetitions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = create_fixture(root / "fixtures", "sparse")
            generate(root, manifest, "yfcc", "throughput")
            raw = root / "raw" / "throughput" / "yfcc" / "shard_00.json"
            raw.parent.mkdir(parents=True, exist_ok=True)
            base = raw_record(0, 0.75, 0.30)
            raw.write_text(json.dumps({"benchmarks": [base]}) + "\n")
            result = run(
                str(SCRIPT_DIR / "analyze_exact.py"),
                "--result-root",
                str(root),
                "--phase",
                "throughput",
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            duplicate = [raw_record(0, 0.75, 0.30) for _ in range(3)]
            raw.write_text(json.dumps({"benchmarks": duplicate}) + "\n")
            result = run(
                str(SCRIPT_DIR / "analyze_exact.py"),
                "--result-root",
                str(root),
                "--phase",
                "throughput",
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)

    def test_rejects_legacy_slot_counted_output_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = create_fixture(root / "fixtures", "sparse")
            generate(root, manifest, "sparse", "smoke")
            raw = root / "raw" / "smoke" / "sparse" / "shard_00.json"
            raw.parent.mkdir(parents=True, exist_ok=True)
            record = raw_record(0, 0.75, 0.30)
            record.pop("OutputSetSemanticsVersion")
            raw.write_text(json.dumps({"benchmarks": [record]}) + "\n")
            result = run(
                str(SCRIPT_DIR / "analyze_exact.py"),
                "--result-root",
                str(root),
                "--phase",
                "smoke",
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("output semantics", result.stderr)

    def test_rejects_stale_workload_timing_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = create_fixture(root / "fixtures", "sparse")
            generate(root, manifest, "sparse", "smoke")
            raw = root / "raw" / "smoke" / "sparse" / "shard_00.json"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text(
                json.dumps(
                    {"benchmarks": [raw_record(0, 0.75, 0.30)]}
                )
                + "\n"
            )

            stale = json.loads(manifest.read_text())
            stale["timed_invalid_sentinel_normalization"] = False
            manifest.write_text(json.dumps(stale, indent=2) + "\n")
            rejected_config = run(
                str(SCRIPT_DIR / "generate_configs.py"),
                "--exact-manifest",
                str(manifest),
                "--workload",
                "stale",
                "--phase",
                "smoke",
                "--output",
                str(root / "stale-config"),
                "--index-marker",
                str(root / "marker.index"),
                check=False,
            )
            self.assertNotEqual(rejected_config.returncode, 0)
            rejected_analysis = run(
                str(SCRIPT_DIR / "analyze_exact.py"),
                "--result-root",
                str(root),
                "--phase",
                "smoke",
                check=False,
            )
            self.assertNotEqual(rejected_analysis.returncode, 0)
            self.assertIn("timed sentinel normalization", rejected_analysis.stderr)

    def test_rejects_incomplete_shard_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = create_fixture(root / "fixtures", "sparse")
            payload = json.loads(source.read_text())
            second = dict(payload["shards"][0])
            second["shard_number"] = 1
            second["first_query"] = 4
            payload["shards"].append(second)
            payload["query_rows"] = 8
            source.write_text(json.dumps(payload) + "\n")
            generate(root, source, "two_shards", "smoke")
            raw = root / "raw" / "smoke" / "two_shards" / "shard_00.json"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text(
                json.dumps({"benchmarks": [raw_record(0, 0.75, 0.30)]}) + "\n"
            )
            result = run(
                str(SCRIPT_DIR / "analyze_exact.py"),
                "--result-root",
                str(root),
                "--phase",
                "smoke",
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
