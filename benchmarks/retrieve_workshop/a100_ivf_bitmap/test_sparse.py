"""Sparse-run correctness and compatibility without primary datasets or CUDA."""

import json
import tempfile
import unittest
from pathlib import Path

import sparse

core = sparse.core


def point(probe, recall, qps, index="flat"):
    row = dict(
        index_id=index,
        nprobe=probe,
        refine_ratio=1,
        recall_min=recall,
        recall_median=recall,
        recall_max=recall,
        qps_median=qps,
        seconds_median=2048 / qps,
        repetitions=3,
    )
    row["point_id"] = core.point_id(row)
    return row


class SparseTests(unittest.TestCase):
    def test_single_shard_aggregation_preserves_qps_and_query_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = dict(index_id="flat", nprobe=10, refine_ratio=1)
            row = dict(
                name="ivf-" + core.point_id(p) + "/real_time",
                threads=1,
                k=10,
                n_queries=2048,
                iterations=2,
                total_queries=4096,
                ValidGTRecall=0.951,
                ValidGTFraction=1,
                items_per_second=5120,
                Latency=0.4,
                **{name: 0 for name in core.ERROR_COUNTERS},
            )
            path = Path(tmp) / "raw.json"
            path.write_text(json.dumps(dict(benchmarks=[row])))
            shards = [(2048, path)]
            result = core.aggregate(shards, [p], 10, 1, 2048, 1)[0]
            self.assertEqual(result["qps_median"], 5120)
            self.assertEqual(result["seconds_median"], 0.4)
            self.assertEqual(result["repetition_rows"][0]["queries"], 2048)
            with self.assertRaises(ValueError):
                core.aggregate(shards, [p], 10, 1)
            row["FilterViolations"] = 1
            path.write_text(json.dumps(dict(benchmarks=[row])))
            with self.assertRaises(ValueError):
                core.aggregate(shards, [p], 10, 1, 2048, 1)

    def test_interpolation_uses_sparse_seconds_not_10000_queries(self):
        low, high = point(10, 0.949, 200), point(11, 0.953, 100)
        result = core.interpolate(low, high, 0.951, 2048)
        self.assertAlmostEqual(result["qps"], 400 / 3)
        self.assertEqual(result["status"], "interpolated")
        self.assertAlmostEqual(sparse.best_result([low, high])["qps"], 400 / 3)
        with self.assertRaises(ValueError):
            core.interpolate(low, point(12, 0.953, 100), 0.951, 2048)
        with self.assertRaises(ValueError):
            core.interpolate(low, point(11, 0.953, 100, "other"), 0.951, 2048)
        with self.assertRaises(ValueError):
            core.interpolate(low, high, 0.96, 2048)

    def test_all_repetitions_must_match_and_measured_is_preferred(self):
        bad = point(9, 0.951, 1000)
        bad["recall_max"] = 0.953
        good = point(10, 0.951, 100)
        result = sparse.best_result([bad, good])
        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["qps"], 100)
        self.assertFalse(sparse.in_window(bad))
        good["recall_min"] = 0.949
        self.assertFalse(sparse.in_window(good))

    def test_finalists_include_fastest_three_and_maximum(self):
        rows = [point(i + 1, 0.951, 100 - i) for i in range(5)]
        rows.append(point(99, 1.0, 1))
        selected = sparse.select_finalists(rows)
        self.assertEqual({p["nprobe"] for p in selected}, {1, 2, 3, 99})
        unmatched = sparse.best_result([point(1, 0.3, 50), point(2, 0.8, 20)])
        self.assertEqual(unmatched["status"], "unmatched")
        self.assertEqual(unmatched["max_recall"], 0.8)
        self.assertEqual(unmatched["qps_at_max_recall"], 20)

    def test_resume_rejects_changed_input_and_runtime_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input"
            path.write_text("original")
            source = {**core.identity(path), "sha256": core.digest(path)}
            manifest = dict(files=[source])
            sparse.validate_sources(manifest)
            source["sha256"] = "wrong hash"
            with self.assertRaises(ValueError):
                sparse.validate_sources(manifest)
            path.write_text("changed")
            with self.assertRaises(ValueError):
                sparse.validate_sources(manifest)

    def test_final_stage_rejects_incomplete_calibration(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(run_root=Path(tmp))
            runner = sparse.SparseRunner(args, dict(indexes=[dict(id="flat")]))
            with self.assertRaises(ValueError):
                runner.finalize()

    def test_complete_report_contains_five_figures_and_all_methods(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs = root / "references"
            refs.mkdir()
            graph, selected, calibration, exact, final = [], [], [], [], []
            inputs = {}
            for cohort, rate in zip(
                sparse.COHORTS, (0.0001, 0.0003, 0.001, 0.003, 0.01)
            ):
                inputs[cohort + "_k10"] = dict(selectivity=rate)
                row = point(10, 0.951, 25000)
                row.update(workload=cohort, phase="final")
                final.append(row)
                for method in (
                    "default_cagra",
                    "default_cagra_accumulator",
                    "navix_reference",
                ):
                    graph.append(
                        dict(
                            workload=cohort,
                            method=method,
                            queries_per_repetition=2048,
                            repetitions=3,
                            recall_median=0.8,
                            qps_median=100,
                        )
                    )
                    selected.append(
                        dict(cohort=cohort, method=method, status="unmatched")
                    )
                    calibration.append(
                        dict(
                            cohort=cohort,
                            method=method,
                            recall=0.8,
                            qps=100,
                            source="reference",
                        )
                    )
                exact.extend(
                    dict(
                        cohort=cohort,
                        repetition=rep,
                        qps=30000,
                        native_l2_cutoff_recall=1,
                    )
                    for rep in range(3)
                )
            for name, rows in (
                ("graph_summary", graph),
                ("matched_selected", selected),
                ("calibration_points", calibration),
                ("exact_control", exact),
            ):
                core.write_csv(refs / (name + ".csv"), rows)
            (root / "state").mkdir()
            (root / "state/manifest.json").write_text("{}")
            runner = sparse.SparseRunner(
                SimpleNamespace(run_root=root),
                dict(indexes=[dict(id="flat")], inputs=inputs),
            )
            with patch.object(
                runner,
                "load_rows",
                side_effect=lambda phase: final if phase == "final" else [],
            ):
                runner.report(complete=True)
            self.assertEqual(len(list((root / "analysis").glob("*.pdf"))), 5)
            self.assertEqual(len(list((root / "analysis").glob("*.png"))), 5)
            self.assertTrue((root / "analysis/provenance.json").exists())
            self.assertIn(
                "25,000.0", (root / "analysis/comparison.md").read_text()
            )


if __name__ == "__main__":
    unittest.main()
