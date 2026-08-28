#!/usr/bin/env python3
"""Synthetic tests for the k=100 Retain occupancy verifier."""

from __future__ import annotations

import csv
import importlib.util
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_PATH = SCRIPT_DIR / "k100_retain_occupancy.py"
SPEC = importlib.util.spec_from_file_location("k100_retain_occupancy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

WORKLOADS = ("yfcc", "em", "emis", "r")
POINTS = {
    "yfcc": (100, 1, 7569, 7569),
    "em": (200, 2, 0, 105),
    "emis": (128, 1, 7569, 7569),
    "r": (314, 2, 0, 162),
}
INTERNAL_L = {"yfcc": 128, "em": 256, "emis": 128, "r": 512}


class K100RetainOccupancyTest(unittest.TestCase):
    def make_bundle(self, base: Path) -> Path:
        tree = base / "bundle/paper_gpu_bundle_k100_matched/matched_recall"
        tree.mkdir(parents=True)
        selected = tree / "selected_points.csv"
        fields = [
            "phase",
            "workload",
            "graph_degree",
            "method",
            "itopk",
            "search_width",
            "max_iterations",
            "resolved_iterations",
            "target_recall",
            "target_reached",
            "selected",
            "paper_included",
            "selection_rule",
        ]
        with selected.open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for workload in WORKLOADS:
                itopk, width, maximum, resolved = POINTS[workload]
                writer.writerow(
                    {
                        "phase": "throughput",
                        "workload": workload,
                        "graph_degree": 64,
                        "method": "default_cagra_accumulator",
                        "itopk": itopk,
                        "search_width": width,
                        "max_iterations": maximum,
                        "resolved_iterations": resolved,
                        "target_recall": 0.8 if workload == "yfcc" else 0.95,
                        "target_reached": workload in {"em", "r"},
                        "selected": True,
                        "paper_included": True,
                        "selection_rule": "fixture",
                    }
                )
        archive = base / "reference.tar.gz"
        with tarfile.open(archive, "w:gz") as output:
            output.add(tree.parent, arcname=tree.parent.name)
        return archive

    def make_source_configs(self, base: Path) -> Path:
        root = base / "source"
        for workload in WORKLOADS:
            destination = root / workload / "shard_00.json"
            destination.parent.mkdir(parents=True)
            payload = {
                "dataset": {
                    "name": f"fixture-{workload}",
                    "base_file": "base.bin",
                    "query_file": "query.bin",
                    "groundtruth_neighbors_file": "groundtruth.ibin",
                    "distance": "euclidean",
                    "dtype": "float",
                    "filter": {"kind": "bitmap", "file": "filter.bitmap"},
                },
                "search_basic_param": {"batch_size": 2048, "k": 100},
                "index": [
                    {
                        "name": "cagra-g64-ig128",
                        "algo": "cuvs_cagra",
                        "file": "graph.index",
                        "build_param": {
                            "graph_build_algo": "NN_DESCENT",
                            "graph_degree": 64,
                            "intermediate_graph_degree": 128,
                        },
                        "search_params": [
                            {
                                "algo": "single_cta",
                                "filter_mode": "default",
                                "max_queries": 2048,
                                "itopk": 128,
                                "search_width": 1,
                                "max_iterations": 0,
                                "favor_udf_passing_accumulator": False,
                                "require_identity_source_indices": True,
                                "bitmap_method": "default_cagra",
                                "k": 100,
                            },
                            {
                                "algo": "single_cta",
                                "filter_mode": "default",
                                "max_queries": 2048,
                                "itopk": 128,
                                "search_width": 1,
                                "max_iterations": 0,
                                "favor_udf_passing_accumulator": True,
                                "require_identity_source_indices": True,
                                "bitmap_method": "default_cagra_accumulator",
                                "k": 100,
                            },
                        ],
                    }
                ],
            }
            destination.write_text(json.dumps(payload, indent=2) + "\n")
        return root

    def generate(self, base: Path) -> Path:
        result = base / "result"
        MODULE.generate_configs(
            self.make_source_configs(base), self.make_bundle(base), result / "configs"
        )
        return result

    @staticmethod
    def resource_record(
        workload: str, method: str, **overrides: int | bool | str
    ) -> dict:
        _, width, _, _ = POINTS[workload]
        internal = INTERNAL_L[workload]
        threads = 64 if workload == "yfcc" else 256
        active = 16 if workload == "yfcc" else 4
        base_smem = 4000 + internal * 8 + width * 64 * 8
        record: dict[str, int | bool | str] = {
            "method": method,
            "diagnostics": False,
            "graph_degree": 64,
            "itopk": internal,
            "search_width": width,
            "threads_per_cta": threads,
            "dynamic_smem_bytes": base_smem
            + (MODULE.EXPECTED_RETAIN_SMEM_DELTA if method == "retain" else 0),
            "static_smem_bytes": 0,
            "registers_per_thread": 64,
            "active_ctas_per_sm": active,
        }
        record.update(overrides)
        return record

    def populate_capture(self, result: Path) -> None:
        manifest = json.loads((result / "configs/manifest.json").read_text())
        (result / "raw").mkdir()
        (result / "resources").mkdir()
        provenance = result / "provenance/run.json"
        provenance.parent.mkdir()
        provenance.write_text(json.dumps({"fixture": True}) + "\n")
        for case in manifest["cases"]:
            workload = case["workload"]
            rows = []
            for method in ("default_cagra", "default_cagra_accumulator"):
                rows.append(
                    {
                        "name": "fixture",
                        "run_type": "iteration",
                        "label": f'bitmap_method="{method}"',
                        "k": 100,
                        "max_queries": 2048,
                        "itopk": case["requested_itopk"],
                        "search_width": case["search_width"],
                        "max_iterations": case["max_iterations"],
                        "n_queries": 2048,
                    }
                )
            (result / f"raw/{workload}.json").write_text(
                json.dumps(
                    {
                        "context": {
                            "gpu_name": "NVIDIA A100 80GB PCIe",
                            "max_k": "100",
                            "max_n_queries": "2048",
                        },
                        "benchmarks": rows,
                    }
                )
                + "\n"
            )
            records = [
                self.resource_record(workload, "base"),
                self.resource_record(workload, "retain"),
            ]
            lines = []
            for record in records:
                line = MODULE.RESOURCE_PREFIX + json.dumps(record, separators=(",", ":"))
                lines.extend([line, line])
            (result / f"resources/{workload}.log").write_text("\n".join(lines) + "\n")

    def prepared_result(self, base: Path) -> Path:
        result = self.generate(base)
        self.populate_capture(result)
        return result

    def rewrite_resource(self, result: Path, workload: str, records: list[dict]) -> None:
        (result / f"resources/{workload}.log").write_text(
            "\n".join(
                MODULE.RESOURCE_PREFIX + json.dumps(record, separators=(",", ":"))
                for record in records
            )
            + "\n"
        )

    def test_generator_uses_exact_selected_retain_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self.generate(Path(temporary))
            manifest = json.loads((result / "configs/manifest.json").read_text())
            self.assertEqual(manifest["expected_retain_dynamic_smem_delta_bytes"], 804)
            self.assertEqual(
                {
                    case["workload"]: (
                        case["requested_itopk"],
                        case["search_width"],
                        case["max_iterations"],
                    )
                    for case in manifest["cases"]
                },
                {
                    workload: POINTS[workload][:3]
                    for workload in WORKLOADS
                },
            )
            for case in manifest["cases"]:
                payload = json.loads(Path(case["config"]).read_text())
                searches = payload["index"][0]["search_params"]
                self.assertEqual(
                    [row["bitmap_method"] for row in searches],
                    ["default_cagra", "default_cagra_accumulator"],
                )
                self.assertEqual(
                    {row["itopk"] for row in searches}, {case["requested_itopk"]}
                )
                self.assertEqual(
                    {row["search_width"] for row in searches}, {case["search_width"]}
                )

    def test_analysis_accepts_equal_active_cta_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self.prepared_result(Path(temporary))
            report = MODULE.analyze(result)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(len(report["rows"]), 4)
            self.assertTrue(all(row["passed"] for row in report["rows"]))
            self.assertEqual(
                {row["retain_dynamic_smem_delta_bytes"] for row in report["rows"]},
                {804},
            )

    def test_analysis_rejects_resource_regressions(self) -> None:
        mutations = {
            "dynamic shared-memory delta": {"dynamic_smem_bytes": 9999},
            "registers per thread": {"registers_per_thread": 65},
            "active CTAs per SM": {"active_ctas_per_sm": 3},
            "threads per CTA": {"threads_per_cta": 128},
        }
        for expected, override in mutations.items():
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temporary:
                result = self.prepared_result(Path(temporary))
                self.rewrite_resource(
                    result,
                    "em",
                    [
                        self.resource_record("em", "base"),
                        self.resource_record("em", "retain", **override),
                    ],
                )
                report = MODULE.analyze(result)
                self.assertEqual(report["status"], "FAIL")
                errors = " ".join(
                    error
                    for row in report["rows"]
                    if row["workload"] == "em"
                    for error in row["errors"]
                )
                self.assertIn(expected, errors)

    def test_analysis_rejects_inconsistent_duplicate_resource_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self.prepared_result(Path(temporary))
            self.rewrite_resource(
                result,
                "r",
                [
                    self.resource_record("r", "base"),
                    self.resource_record("r", "base", registers_per_thread=63),
                    self.resource_record("r", "retain"),
                ],
            )
            report = MODULE.analyze(result)
            self.assertEqual(report["status"], "FAIL")
            row = next(row for row in report["rows"] if row["workload"] == "r")
            self.assertIn("one consistent base resource tuple", row["errors"][0])

    def test_analysis_rejects_non_a100_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self.prepared_result(Path(temporary))
            path = result / "raw/yfcc.json"
            payload = json.loads(path.read_text())
            payload["context"]["gpu_name"] = "NVIDIA L4"
            path.write_text(json.dumps(payload) + "\n")
            report = MODULE.analyze(result)
            self.assertEqual(report["status"], "FAIL")
            row = next(row for row in report["rows"] if row["workload"] == "yfcc")
            self.assertIn("did not run on an A100", row["errors"][0])


if __name__ == "__main__":
    unittest.main()
