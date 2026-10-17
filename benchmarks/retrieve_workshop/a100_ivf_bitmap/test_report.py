"""Check report completeness, measured envelopes, and unmatched table labels."""

import importlib.util
import csv
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ivf_report", Path(__file__).with_name("report.py")
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def sparse_rows():
    rows = []
    for rate_index, rate in enumerate(report.SPARSE_RATES):
        for method in report.SPARSE_METHODS:
            unmatched = (
                method == "default_cagra"
                or (method == "default_cagra_accumulator" and rate_index == 0)
                or (method == "navix_reference" and rate_index < 3)
            )
            exact = method == "exact"
            rows.append(
                dict(
                    cohort=f"synthetic_low_{rate}_random",
                    method=method,
                    status="unmatched"
                    if unmatched
                    else "exact"
                    if exact
                    else "measured",
                    recall="" if unmatched else 1.0 if exact else 0.951,
                    qps="" if unmatched else 100000 if exact else 25000,
                    max_recall=0.8 if unmatched else 1.0,
                    qps_at_max_recall=500,
                    recall_min="" if unmatched or exact else 0.951,
                    recall_max="" if unmatched or exact else 0.951,
                    repetitions=3,
                )
            )
    return rows


class ReportTests(unittest.TestCase):
    def test_sparse_bars_use_maximum_recall_qps_for_unmatched_methods(self):
        values = report.sparse_bar_values(sparse_rows())
        self.assertEqual(len(values), 25)
        self.assertEqual(len({v["selectivity"] for v in values}), 5)
        self.assertEqual(sum(v["unmatched"] for v in values), 9)
        base = next(v for v in values if v["method"] == "default_cagra")
        self.assertEqual(base["qps"], 500)
        self.assertEqual(base["recall"], 0.8)
        flat = next(v for v in values if v["method"] == "ivf_flat")
        self.assertEqual(flat["qps"], 25000)
        exact = next(v for v in values if v["method"] == "exact")
        self.assertEqual(exact["recall"], 1)

    def test_sparse_loader_requires_complete_run_and_unchanged_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "analysis").mkdir()
            progress = root / "state/progress.json"
            progress.write_text(json.dumps(dict(state="running")))
            with self.assertRaisesRegex(ValueError, "must finish"):
                report.load_sparse_results(root)
            progress.write_text(json.dumps(dict(state="complete")))
            manifest = root / "state/manifest.json"
            manifest.write_text(
                json.dumps(
                    dict(
                        experiment="arxiv_sparse_ivf_flat",
                        query_count=2048,
                        k=10,
                        repetitions=3,
                    )
                )
            )
            comparison = root / "analysis/comparison.csv"
            rows = sparse_rows()
            with comparison.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            curves = root / "analysis/curve_points.csv"
            curves.write_text("test\n")
            (root / "analysis/provenance.json").write_text(
                json.dumps(
                    dict(
                        manifest_sha256=report.sha(manifest),
                        comparison_sha256=report.sha(comparison),
                        curve_points_sha256=report.sha(curves),
                    )
                )
            )
            self.assertEqual(len(report.load_sparse_results(root)), 25)
            comparison.write_text(comparison.read_text() + "changed\n")
            with self.assertRaisesRegex(ValueError, "Changed sparse"):
                report.load_sparse_results(root)

    def test_sparse_section_distinguishes_maximum_and_matched_recall(self):
        text = report.sparse_section(sparse_rows())
        self.assertIn("0.01\\%", text)
        self.assertIn("2,048", text)
        self.assertIn("sparse_selectivity_qps.pdf", text)
        self.assertIn("not matched-recall comparisons", text)
        self.assertIn("logarithmic", text)

    def test_pareto_preserves_only_undominated_measured_points(self):
        rows = [
            dict(recall=r, qps=q)
            for r, q in [
                (0.7, 100),
                (0.8, 90),
                (0.8, 95),
                (0.75, 80),
                (0.9, 70),
            ]
        ]
        self.assertEqual(
            [(r["recall"], r["qps"]) for r in report.pareto(rows)],
            [(0.7, 100), (0.8, 95), (0.9, 70)],
        )

    def test_table_uses_maximum_only_for_unmatched_cells(self):
        row = dict(
            status="interpolated",
            recall=0.95,
            qps=2000.0,
            max_observed_recall=0.99,
            qps_at_max_recall=500.0,
        )
        cell = report.table_cell(row)
        self.assertIn("2,000", cell)
        self.assertIn("0.95000", cell)
        self.assertIn("^{*}", cell)
        self.assertNotIn("max", cell)
        row["status"] = "unmatched"
        cell = report.table_cell(row)
        self.assertIn("500", cell)
        self.assertIn("0.99000", cell)
        self.assertIn("max", cell)
        self.assertNotIn("^{*}", cell)

    def test_incomplete_final_data_cannot_create_a_final_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "analysis").mkdir()
            (root / "analysis/comparison.json").write_text(
                json.dumps(dict(complete=False, rows=[]))
            )
            with self.assertRaisesRegex(ValueError, "32 completed"):
                report.load_results(root)

    def test_flat_scope_requires_finals_unless_calibration_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "analysis").mkdir()
            (root / "analysis/comparison.json").write_text(
                json.dumps(dict(complete=False, rows=[]))
            )
            with self.assertRaisesRegex(ValueError, "8 completed"):
                report.load_results(root, ("ivf_flat",))

    def test_calibration_selection_checks_every_index_and_marks_one_repetition(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "state").mkdir()
            indexes = [
                dict(id=name, corpus="arxiv", kind="flat")
                for name in ("one", "two", "three")
            ]
            (root / "state/manifest.json").write_text(
                json.dumps(dict(indexes=indexes))
            )
            for index, recall, qps in [
                ("one", 0.96, 50000.0),
                ("two", 0.9503, 2000.0),
                ("three", 0.951, 1000.0),
            ]:
                marker = root / "state/calibrated" / index / "em_k10.json"
                marker.parent.mkdir(parents=True)
                marker.write_text(
                    json.dumps(
                        [
                            dict(
                                method="ivf_flat",
                                index_id=index,
                                point_id=index,
                                nprobe=10,
                                refine_ratio=1,
                                recall_median=recall,
                                recall_min=recall,
                                qps_median=qps,
                                seconds_median=10000 / qps,
                            )
                        ]
                    )
                )
            row = report.calibration_cell(root, "em", 10, "ivf_flat")
            self.assertEqual(row["index_id"], "two")
            self.assertEqual(row["qps"], 2000.0)
            self.assertEqual(row["repetitions"], 1)
            self.assertEqual(row["source"], "ivf_calibration_1rep")
            self.assertIn(r"\dagger", report.table_cell(row))
            (root / "state/calibrated/three/em_k10.json").unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete calibration"):
                report.calibration_cell(root, "em", 10, "ivf_flat")

    def test_document_contains_all_eight_workload_figures_and_both_tables(
        self,
    ):
        cells = [
            dict(
                workload=w,
                k=k,
                method=m,
                status="measured",
                recall=report.TARGETS[w],
                qps=1234.0,
                max_observed_recall=0.99,
                qps_at_max_recall=500.0,
            )
            for w in report.WORKLOADS
            for k in (10, 100)
            for m in report.METHODS
        ]
        source = report.make_tex(cells)
        self.assertIn(r"\documentclass[lettersize,journal]{IEEEtran}", source)
        self.assertEqual(source.count(r"\begin{figure*}"), 8)
        self.assertEqual(source.count(r"\begin{table*}"), 2)
        for w in report.WORKLOADS:
            for k in (10, 100):
                self.assertIn(f"figures/{w}_qps_recall_k{k}.pdf", source)


if __name__ == "__main__":
    unittest.main()
