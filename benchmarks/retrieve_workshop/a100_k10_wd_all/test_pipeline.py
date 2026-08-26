#!/usr/bin/env python3
"""Synthetic contract tests for the all-workload W*D NaviX workflow."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RETRIEVE_DIR = SCRIPT_DIR.parent
WORKFLOW = SCRIPT_DIR / "workflow.py"
MATCHED = RETRIEVE_DIR / "matched_recall" / "matched_recall.py"
WORKLOADS = ("yfcc", "em", "emis", "r")
TARGETS = {"yfcc": 0.800, "em": 0.950, "emis": 0.950, "r": 0.950}


class PipelineTest(unittest.TestCase):
    def fixture(self, base: Path) -> tuple[Path, Path, Path, Path]:
        data = base / "data"
        profile = base / "profile.json"
        datasets: dict[str, dict[str, object]] = {}
        for workload in WORKLOADS:
            bitmap = f"navix_bitmap/{workload}"
            datasets[workload] = {
                "bitmap_directory": bitmap,
                "base_file": f"datasets/{workload}/base.bin",
                "index_file": f"datasets/{workload}/graph.index",
                "dtype": "uint8" if workload == "yfcc" else "float",
                "dataset_size": 10_000_000 if workload == "yfcc" else 2_735_264,
                "dimension": 192 if workload == "yfcc" else 4_096,
                "graph_degree": 64,
                "intermediate_graph_degree": 128,
            }
            for relative in (datasets[workload]["base_file"], datasets[workload]["index_file"]):
                path = data / str(relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            for phase, counts in (
                ("correctness_1000", [1_000]),
                ("throughput_10000", [2_048, 2_048, 2_048, 2_048, 1_808]),
            ):
                root = data / bitmap / phase
                shards = []
                first = 0
                for index, count in enumerate(counts):
                    directory = root / f"shard_{index:02d}_{first:05d}_{first+count:05d}"
                    directory.mkdir(parents=True, exist_ok=True)
                    for name in ("query.bin", "groundtruth.ibin", "filter.bitmap"):
                        (directory / name).write_bytes(b"fixture")
                    shards.append(
                        {
                            "shard_index": index,
                            "first_query": first,
                            "query_count": count,
                            "directory": str(directory.resolve()),
                        }
                    )
                    first += count
                root.mkdir(parents=True, exist_ok=True)
                (root / "manifest.json").write_text(
                    json.dumps({"query_rows": first, "shards": shards}, indent=2) + "\n"
                )
        profile.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "name": "synthetic-a100-wd-all",
                    "max_queries": 2_048,
                    "matched_widths": [1, 2],
                    "datasets": datasets,
                },
                indent=2,
            )
            + "\n"
        )
        selected = base / "selected.csv"
        fields = [
            "workload",
            "method",
            "selected",
            "itopk",
            "search_width",
            "max_iterations",
            "recall_median",
            "recall_min",
            "recall_max",
            "qps_median",
        ]
        coordinates = {
            "yfcc": (129, 2),
            "em": (30, 2),
            "emis": (56, 2),
            "r": (31, 1),
        }
        with selected.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for workload in WORKLOADS:
                itopk, width = coordinates[workload]
                writer.writerow(
                    {
                        "workload": workload,
                        "method": "navix_reference",
                        "selected": "True",
                        "itopk": itopk,
                        "search_width": width,
                        "max_iterations": 0,
                        "recall_median": TARGETS[workload] + 0.001,
                        "recall_min": TARGETS[workload],
                        "recall_max": TARGETS[workload] + 0.001,
                        "qps_median": 10_000,
                    }
                )
        provenance = base / "selected_provenance.json"
        run_provenance = base / "reference_run.json"
        run_provenance.write_text(
            json.dumps({"fixed_contract": {"k": 10, "max_queries": 2_048}})
            + "\n"
        )
        provenance.write_text(
            json.dumps(
                {
                    "max_queries": 2_048,
                    "navix_seed_policy": "k",
                    "run_provenance": {"path": str(run_provenance.resolve())},
                }
            )
            + "\n"
        )
        return data, profile, selected, provenance

    def environment(self, profile: Path) -> dict[str, str]:
        return {
            **os.environ,
            "RETRIEVE_DATASET_PROFILE": str(profile),
            "RETRIEVE_MATCHED_METHODS": "navix_reference",
            "RETRIEVE_MATCHED_NAVIX_SEED_POLICY": "wd",
            "RETRIEVE_MATCHED_ALLOW_SHALLOW_NAVIX": "1",
        }

    def test_generates_complete_frontier_and_wd_caps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            data, profile, selected, provenance = self.fixture(base)
            root = base / "run"
            subprocess.run(
                [
                    sys.executable,
                    str(WORKFLOW),
                    "initialize",
                    "--root",
                    str(root),
                    "--data-root",
                    str(data),
                    "--reference-selected",
                    str(selected),
                    "--reference-provenance",
                    str(provenance),
                ],
                check=True,
                env=self.environment(profile),
            )
            for workload in WORKLOADS:
                b0 = json.loads(
                    (root / "frontier/configs/b0" / workload / "manifest.json").read_text()
                )
                self.assertEqual(b0["navix_seed_policy"], "wd")
                self.assertEqual(len(b0["search_points"]), 6)
                self.assertEqual(
                    {int(row["navix_seed_cap"]) for row in b0["search_points"]},
                    {64, 128},
                )
                self.assertEqual(b0["repetitions"], 3)
                self.assertEqual(b0["expected_queries"], 10_000)
                correctness = json.loads(
                    (root / "frontier/configs/correctness" / workload / "manifest.json").read_text()
                )
                self.assertEqual(
                    [int(row["navix_seed_cap"]) for row in correctness["search_points"]],
                    [64, 128],
                )

    def test_matched_group_uses_only_navix_and_wd_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            data, profile, _, _ = self.fixture(base)
            root = base / "matched"
            points = base / "points.json"
            points.write_text(
                json.dumps(
                    {
                        "points": [
                            {
                                "workload": workload,
                                "method": "navix_reference",
                                "itopk": 10,
                                "search_width": 2,
                                "max_iterations": 0,
                            }
                            for workload in WORKLOADS
                        ]
                    }
                )
                + "\n"
            )
            subprocess.run(
                [
                    sys.executable,
                    str(MATCHED),
                    "generate-group",
                    "--result-root",
                    str(root),
                    "--data-root",
                    str(data),
                    "--group",
                    "calibration_r00",
                    "--stage",
                    "calibration",
                    "--repetitions",
                    "1",
                    "--points",
                    str(points),
                ],
                check=True,
                env=self.environment(profile),
            )
            for workload in WORKLOADS:
                manifest = json.loads(
                    (root / "configs/calibration_r00" / workload / "manifest.json").read_text()
                )
                self.assertEqual(manifest["navix_seed_policy"], "wd")
                self.assertEqual(manifest["search_points"][0]["navix_seed_cap"], 128)
                config = json.loads(
                    (root / "configs/calibration_r00" / workload / "shard_00.json").read_text()
                )
                searches = config["index"][0]["search_params"]
                self.assertEqual(len(searches), 1)
                self.assertEqual(searches[0]["bitmap_method"], "navix_reference")
                self.assertEqual(searches[0]["navix_seed_cap"], 128)

    def test_controls_pair_both_policies_at_both_configurations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            data, profile, selected, provenance = self.fixture(base)
            root = base / "run"
            env = self.environment(profile)
            subprocess.run(
                [
                    sys.executable,
                    str(WORKFLOW),
                    "initialize",
                    "--root",
                    str(root),
                    "--data-root",
                    str(data),
                    "--reference-selected",
                    str(selected),
                    "--reference-provenance",
                    str(provenance),
                ],
                check=True,
                env=env,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(WORKFLOW),
                    "create-controls",
                    "--root",
                    str(root),
                    "--data-root",
                    str(data),
                    "--selected",
                    str(selected),
                ],
                check=True,
                env=env,
            )
            for group in ("paired_incumbent", "paired_winner"):
                for workload in WORKLOADS:
                    manifest = json.loads(
                        (root / "controls/configs" / group / workload / "manifest.json").read_text()
                    )
                    self.assertEqual(
                        {row["seed_policy"] for row in manifest["search_points"]},
                        {"k_seed", "wd_seed"},
                    )
                    widths = {int(row["search_width"]) for row in manifest["search_points"]}
                    self.assertEqual(
                        {int(row["navix_seed_cap"]) for row in manifest["search_points"]},
                        {10, *(width * 64 for width in widths)},
                    )

    def test_hash_manifest_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            root.mkdir()
            (root / "data.txt").write_text("original")
            sys.path.insert(0, str(SCRIPT_DIR))
            import workflow

            workflow.make_manifest(root, {"schema_version": 1})
            workflow.validate_hash_manifest(root)
            (root / "data.txt").write_text("changed")
            with self.assertRaisesRegex(ValueError, "bundle hash mismatch"):
                workflow.validate_hash_manifest(root)


if __name__ == "__main__":
    unittest.main()
