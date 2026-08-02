import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import generate_automatic_retention_report as report


class AutomaticRetentionReportTest(unittest.TestCase):
    @staticmethod
    def evaluation(gpu_name: str, target_recall: float) -> report.Evaluation:
        max_recall_rows = []
        target_rows = []
        parameter_rows = []
        for dataset in report.DATASETS:
            for selectivity in report.SELECTIVITIES:
                encoded = selectivity / 100
                parameter_rows.append(
                    {
                        "dataset": dataset.title,
                        "selectivity": f"{encoded:g}",
                        "itopk": "32",
                        "search_width": "1",
                        "max_iterations": "0",
                        "thread_block_size": "0",
                    }
                )
                for workload in ("throughput", "latency"):
                    for method in report.METHOD_ORDER:
                        max_recall_rows.append(
                            {
                                "dataset": dataset.title,
                                "selectivity": f"{encoded:g}",
                                "workload": workload,
                                "batch_size": "1",
                                "series": method,
                                "method": report.METHOD_LABELS[method],
                                "max_recall": "0.999",
                            }
                        )
                        target_rows.append(
                            {
                                "dataset": dataset.title,
                                "selectivity": f"{encoded:g}",
                                "workload": workload,
                                "series": method,
                                "target_recall": f"{target_recall:g}",
                                "value": "1000",
                            }
                        )
        return report.Evaluation(
            max_recall_rows, target_rows, parameter_rows, gpu_name
        )

    def test_multi_plot_uses_one_native_result_root(self) -> None:
        dataset = report.Dataset("sift", "sift", "SIFT-1M")
        command = report.build_plot_command(
            Path("/bench"),
            Path("/fresh-l4"),
            Path("/processed"),
            dataset,
            automatic_overlay_root=None,
            cta_mode="MULTI_CTA",
            batch_size=1,
            target_recall=0.99,
            latency_derived_qps=True,
        )

        result_index = command.index("--result-dir")
        self.assertEqual(command[result_index + 1], "/fresh-l4/sift")
        self.assertNotIn("--overlay-series", command)
        self.assertIn("automatic_retention", command)
        self.assertIn("--latency-derived-qps", command)

    def test_single_plot_keeps_explicit_automatic_overlay(self) -> None:
        dataset = report.Dataset("sift", "sift", "SIFT-1M")
        command = report.build_plot_command(
            Path("/bench"),
            Path("/single-baseline"),
            Path("/processed"),
            dataset,
            automatic_overlay_root=Path("/single-automatic"),
            cta_mode="SINGLE_CTA",
            batch_size=10,
            target_recall=0.90,
            latency_derived_qps=False,
        )

        overlay_index = command.index("--overlay-series")
        self.assertEqual(
            command[overlay_index + 1 : overlay_index + 5],
            [
                "automatic_retention",
                "automatic_retention",
                "Automatic retention FAVOR",
                "/single-automatic/sift",
            ],
        )

    def test_config_cells_require_matched_native_methods(self) -> None:
        rows = []
        for itopk in (32, 64):
            rows.extend(
                [
                    {
                        "filter_mode": "default",
                        "itopk": itopk,
                        "search_width": 1,
                    },
                    {
                        "filter_mode": "favor",
                        "favor_penalty_mode": "cagra_retention_safe",
                        "favor_retention_fraction": 0.5,
                        "itopk": itopk,
                        "search_width": 1,
                    },
                    {
                        "filter_mode": "favor",
                        "favor_penalty_mode": "cagra_retention_safe",
                        "favor_retention_fraction": 0.0,
                        "itopk": itopk,
                        "search_width": 1,
                    },
                ]
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"index": [{"search_params": rows}]}))
            cells = report.load_config_cells([path])

        self.assertEqual(cells, {(32, 1, 0, 0), (64, 1, 0, 0)})

    def test_config_cells_reject_mismatched_automatic_frontier(self) -> None:
        rows = [
            {"filter_mode": "default", "itopk": 32, "search_width": 1},
            {
                "filter_mode": "favor",
                "favor_penalty_mode": "cagra_retention_safe",
                "favor_retention_fraction": 0.5,
                "itopk": 32,
                "search_width": 1,
            },
            {
                "filter_mode": "favor",
                "favor_penalty_mode": "cagra_retention_safe",
                "favor_retention_fraction": 0.0,
                "itopk": 64,
                "search_width": 1,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"index": [{"search_params": rows}]}))
            with self.assertRaisesRegex(ValueError, "identical search cells"):
                report.load_config_cells([path])

    def test_benchmark_gpu_requires_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            valid = Path(directory) / "valid.json"
            valid.write_text(json.dumps({"context": {"gpu_name": "NVIDIA L4"}}))
            self.assertEqual(report.benchmark_gpu(valid), "NVIDIA L4")

            invalid = Path(directory) / "invalid.json"
            invalid.write_text(json.dumps({"context": {}}))
            with self.assertRaisesRegex(ValueError, "GPU provenance"):
                report.benchmark_gpu(invalid)

    def test_report_separates_hardware_and_describes_single_only_caveat(self) -> None:
        single = self.evaluation("NVIDIA A30", 0.90)
        multi = self.evaluation("NVIDIA L4", 0.99)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.org"
            report.render_report(path, single, multi)
            text = path.read_text()

        self.assertIn("SINGLE_CTA and MULTI_CTA Performance Evaluation", text)
        self.assertIn("SINGLE_CTA= curves were measured on NVIDIA A30", text)
        self.assertIn("curves were all measured on NVIDIA L4", text)
        self.assertIn("MULTI_CTA= uses a linear sorted CTA-local", text)
        self.assertIn("All three fresh =MULTI_CTA=", text)
        self.assertIn("** Control-region interpretation", text)
        self.assertIn("=itopk>=1024= at 1%", text)
        self.assertIn("sampling noise, not an automatic-policy", text)
        self.assertIn("must not be used to claim a win or loss", text)
        self.assertNotIn("results_retention_safe_multi_confirmed", text)

    @mock.patch.object(report, "DATASETS", (report.Dataset("sift", "sift", "SIFT-1M"),))
    @mock.patch.object(report, "SELECTIVITIES", (1,))
    @mock.patch.object(report, "MULTI_ITOPK_VALUES", {"sift": (32,)})
    def test_source_preflight_rejects_the_wrong_gpu(self) -> None:
        rows = [
            {"filter_mode": "default", "itopk": 32, "search_width": 1},
            {
                "filter_mode": "favor",
                "favor_penalty_mode": "cagra_retention_safe",
                "favor_retention_fraction": 0.5,
                "itopk": 32,
                "search_width": 1,
            },
            {
                "filter_mode": "favor",
                "favor_penalty_mode": "cagra_retention_safe",
                "favor_retention_fraction": 0.0,
                "itopk": 32,
                "search_width": 1,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir = root / "sift" / "raw"
            config_dir = root / "sift" / "configs"
            raw_dir.mkdir(parents=True)
            config_dir.mkdir(parents=True)
            (raw_dir / "sift_s01_nq1.json").write_text(
                json.dumps(
                    {
                        "context": {"gpu_name": "NVIDIA A30"},
                        "benchmarks": [
                            {"run_type": "iteration", "run_name": f"run-{index}"}
                            for index in range(3)
                        ],
                    }
                )
            )
            (config_dir / "sift_s01_nq1.json").write_text(
                json.dumps({"index": [{"search_params": rows}]})
            )

            with self.assertRaisesRegex(ValueError, "expected 'NVIDIA L4'"):
                report.validate_evaluation_sources(
                    root,
                    automatic_overlay_root=None,
                    cta_mode="MULTI_CTA",
                    batch_size=1,
                    config_batch_size=1,
                    expected_gpu="NVIDIA L4",
                )


if __name__ == "__main__":
    unittest.main()
