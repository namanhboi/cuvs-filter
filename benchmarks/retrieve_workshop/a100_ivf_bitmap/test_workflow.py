"""Tests for comparison semantics, independent of CUDA and primary input files."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

module_spec = importlib.util.spec_from_file_location(
    "ivf_workflow", Path(__file__).with_name("workflow.py")
)
workflow = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(workflow)
status_spec = importlib.util.spec_from_file_location(
    "ivf_status", Path(__file__).with_name("status.py")
)
status_module = importlib.util.module_from_spec(status_spec)
status_spec.loader.exec_module(status_module)


def point(probe, recall, qps, ratio=1, index="sample"):
    value = dict(
        index_id=index,
        nprobe=probe,
        refine_ratio=ratio,
        recall_median=recall,
        recall_min=recall,
        recall_max=recall,
        qps_median=qps,
        seconds_median=10000 / qps,
    )
    value["point_id"] = workflow.point_id(value)
    return value


class ComparisonTests(unittest.TestCase):
    def test_eta_uses_complete_cases_and_reports_unknown_families(self):
        specs = [
            dict(id="one", corpus="yfcc", kind="flat"),
            dict(id="two", corpus="yfcc", kind="flat"),
        ]
        manifest = dict(
            indexes=specs,
            runtime_revisions=[dict(captured_utc="2026-10-14T03:00:00")],
        )
        completed = {("one", "yfcc", 10), ("one", "yfcc", 100)}
        timings = [
            dict(
                stage="calibration",
                detail=f"one yfcc k={k} coarse shard=0 points=11",
                elapsed_seconds=seconds,
                completed_utc=date,
            )
            for k, seconds, date in [
                (10, 999, "2026-10-14T02:00:00"),
                (10, 30, "2026-10-14T04:00:00"),
                (100, 50, "2026-10-14T04:00:00"),
            ]
        ]
        result = status_module.calibration_eta(manifest, completed, timings)
        self.assertEqual(result["estimated_remaining_calibration_seconds"], 80)
        self.assertEqual(result["remaining_calibrations"], 2)
        specs.append(dict(id="unseen", corpus="arxiv", kind="pq"))
        result = status_module.calibration_eta(manifest, completed, timings)
        self.assertIsNone(result["estimated_remaining_calibration_seconds"])
        self.assertEqual(result["estimated_seconds_for_sampled_families"], 80)
        self.assertEqual(
            result["families_without_timing_samples"], ["arxiv/pq"]
        )

    def test_resume_keeps_successful_shards_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                run_root=root, build_dir=root, data_root=root
            )
            spec = dict(
                id="sample",
                kind="flat",
                dtype="float",
                file="shared.index",
                build=dict(nlist=1024),
            )
            shards = [
                dict(
                    first_query=i * 2048,
                    query_count=count,
                    query_file="query",
                    float_query_file="query",
                    groundtruth_file="gt",
                    bitmap_file="bitmap",
                )
                for i, count in enumerate((2048, 2048, 2048, 2048, 1808))
            ]
            inp = dict(
                workload="yfcc",
                k=10,
                base_file="base",
                float_base_file="base",
                shards=shards,
            )
            runner = workflow.Runner(
                args,
                dict(
                    indexes=[spec],
                    inputs=dict(yfcc_k10=inp),
                    minimum_time="0.10s",
                ),
            )
            candidate = dict(index_id="sample", nprobe=1, refine_ratio=1)

            def generate(command, *_):
                output = Path(
                    next(
                        arg.split("=", 1)[1]
                        for arg in command
                        if arg.startswith("--benchmark_out=")
                    )
                )
                config = workflow.read_json(command[-1])
                count = config["search_basic_param"]["batch_size"]
                row = dict(
                    name="ivf-" + workflow.point_id(candidate) + "/search",
                    run_type="iteration",
                    repetition_index=0,
                    k=10,
                    n_queries=count,
                    threads=1,
                    iterations=1,
                    total_queries=count,
                    ValidGTRecall=0.8,
                    ValidGTFraction=1,
                    items_per_second=100,
                    Latency=count / 100,
                )
                row.update({name: 0 for name in workflow.ERROR_COUNTERS})
                workflow.write_json(output, dict(benchmarks=[row]))

            with patch.object(
                runner, "invoke", side_effect=generate
            ) as invoke:
                first = runner.measure(
                    spec, "yfcc", 10, [candidate], "calibration", "coarse", 1
                )
                self.assertEqual(invoke.call_count, 5)
            completion = (
                root / "measurements/calibration/sample/yfcc_k10/coarse.json"
            )
            completion.unlink()  # Interrupted between the last shard and its aggregate marker.
            with patch.object(
                runner,
                "invoke",
                side_effect=AssertionError("Completed shards must not rerun"),
            ):
                resumed = runner.measure(
                    spec, "yfcc", 10, [candidate], "calibration", "coarse", 1
                )
                self.assertEqual(first, resumed)
            raw = root / "raw/calibration/sample/yfcc_k10/coarse/shard_04.json"
            report = workflow.read_json(raw)
            report["benchmarks"][0]["DuplicateOutputQueries"] = 0.1
            workflow.write_json(raw, report)
            self.assertRaises(
                ValueError,
                runner.measure,
                spec,
                "yfcc",
                10,
                [candidate],
                "calibration",
                "coarse",
                1,
            )

    def test_index_reuse_and_parameter_grid(self):
        specs = workflow.index_specs()
        self.assertEqual(len(specs), 24)
        self.assertEqual(len({spec["id"] for spec in specs}), 24)
        for spec in specs:
            points = workflow.coarse_points(spec)
            self.assertEqual(
                max(p["nprobe"] for p in points), spec["build"]["nlist"]
            )
            self.assertEqual(min(p["nprobe"] for p in points), 1)
            self.assertEqual(spec["build"]["niter"], 25)

    def test_final_partial_shard_and_gpu_reranking_config(self):
        spec = dict(
            id="sample",
            kind="pq",
            dtype="float",
            file="shared.index",
            build=dict(nlist=1024),
        )
        inp = dict(
            workload="yfcc",
            k=100,
            base_file="integer.base",
            float_base_file="float.base",
        )
        shard = dict(
            first_query=8192,
            query_count=1808,
            query_file="integer.query",
            float_query_file="float.query",
            groundtruth_file="gt",
            bitmap_file="bitmap",
        )
        config = workflow.make_config(
            spec,
            inp,
            shard,
            [dict(index_id="sample", nprobe=4, refine_ratio=8)],
        )
        params = config["index"][0]["search_params"][0]
        self.assertEqual(params["dataset_memory_type"], "device")
        self.assertEqual(params["n_queries"], 1808)
        self.assertEqual(config["search_basic_param"]["batch_size"], 1808)
        self.assertEqual(config["dataset"]["base_file"], "float.base")
        self.assertEqual(config["dataset"]["filter"]["file"], "bitmap")

    def test_tight_window_all_repetitions_must_reach(self):
        self.assertTrue(workflow.within_window(point(1, 0.95, 10), 0.95))
        self.assertTrue(workflow.within_window(point(1, 0.952, 10), 0.95))
        self.assertFalse(workflow.within_window(point(1, 0.95201, 10), 0.95))
        value = point(1, 0.951, 10)
        value["recall_min"] = 0.94999
        self.assertFalse(workflow.within_window(value, 0.95))

    def test_interpolate_time_not_qps(self):
        a, b = point(1, 0.94, 100), point(2, 0.96, 50)
        result = workflow.interpolate(a, b, 0.95)
        self.assertAlmostEqual(result["qps"], 10000 / 150)
        self.assertEqual(result["status"], "interpolated")
        self.assertEqual(result["lower_point_id"], a["point_id"])
        self.assertRaises(ValueError, workflow.interpolate, a, b, 0.98)
        self.assertRaises(
            ValueError, workflow.interpolate, a, point(3, 0.96, 50), 0.95
        )
        self.assertRaises(
            ValueError,
            workflow.interpolate,
            a,
            point(2, 0.96, 50, ratio=2),
            0.95,
        )

    def test_nonmonotonic_crossings_are_refined(self):
        rows = [
            point(1, 0.94, 100),
            point(8, 0.97, 90),
            point(16, 0.93, 80),
            point(32, 0.98, 70),
        ]
        proposals = workflow.refinement_proposals(rows, 0.95)
        self.assertTrue({4, 12, 24}.issubset({p["nprobe"] for p in proposals}))
        self.assertTrue(all(1 < p["nprobe"] < 32 for p in proposals))
        self.assertEqual(
            workflow.refinement_proposals(
                [point(1, 0.94, 100), point(2, 0.96, 90)], 0.95
            ),
            [],
        )

    def test_finalists_preserve_maximum_and_separate_families(self):
        rows = [point(1, 0.94, 100), point(2, 0.96, 80), point(3, 0.99, 50)]
        finals = workflow.finalists(rows, 0.95)
        self.assertEqual({p["nprobe"] for p in finals}, {1, 2, 3})
        unreachable = workflow.finalists(
            [point(1, 0.8, 100), point(2, 0.9, 80)], 0.95
        )
        self.assertEqual(len(unreachable), 1)
        self.assertEqual(unreachable[0]["nprobe"], 2)
        mismatched = [point(1, 0.94, 100), point(2, 0.96, 80, index="other")]
        self.assertEqual(workflow.adjacent_brackets(mismatched, 0.95), [])

    def test_valid_gt_weighting_and_serial_shard_time(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = dict(index_id="sample", nprobe=1, refine_ratio=1)
            paths = []
            for i, count in enumerate((2048, 2048, 2048, 2048, 1808)):
                rows = []
                for rep in range(3):
                    row = dict(
                        name="ivf-"
                        + workflow.point_id(candidate)
                        + "/process_time",
                        run_type="iteration",
                        repetition_index=rep,
                        k=100,
                        n_queries=count,
                        threads=1,
                        iterations=2,
                        total_queries=count * 2,
                        ValidGTRecall=0.5 if i == 4 else 1.0,
                        ValidGTFraction=0.5 if i == 4 else 1.0,
                        items_per_second=100 if i == 4 else 200,
                        Latency=count / 200,
                    )
                    row.update({name: 0 for name in workflow.ERROR_COUNTERS})
                    rows.append(row)
                path = Path(directory) / f"shard{i}.json"
                path.write_text(json.dumps(dict(benchmarks=rows)))
                paths.append((count, path))
            aggregated = workflow.aggregate(paths, [candidate], 100, 3)[0]
            self.assertAlmostEqual(
                aggregated["recall_median"], (8192 + 904 * 0.5) / (8192 + 904)
            )
            self.assertAlmostEqual(
                aggregated["qps_median"], 10000 / (8192 / 200 + 1808 / 100)
            )
            self.assertEqual(len(aggregated["repetition_rows"]), 3)
            content = workflow.read_json(paths[-1][1])
            content["benchmarks"][0]["FilterViolations"] = 1
            paths[-1][1].write_text(json.dumps(content))
            self.assertRaises(
                ValueError, workflow.aggregate, paths, [candidate], 100, 3
            )

    def test_incomplete_and_missing_counter_reports_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.json"
            path.write_text(json.dumps(dict(benchmarks=[])))
            candidate = dict(index_id="sample", nprobe=1, refine_ratio=1)
            self.assertRaises(
                ValueError,
                workflow.validate_raw,
                path,
                [candidate],
                2048,
                10,
                1,
            )


if __name__ == "__main__":
    unittest.main()
