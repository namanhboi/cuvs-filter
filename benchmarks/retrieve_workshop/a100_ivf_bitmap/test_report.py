"""Check report completeness, measured envelopes, and unmatched table labels."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ivf_report", Path(__file__).with_name("report.py")
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class ReportTests(unittest.TestCase):
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
