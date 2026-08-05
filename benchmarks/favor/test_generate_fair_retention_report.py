import json
import tempfile
import unittest
from pathlib import Path

import generate_fair_retention_report as report


class FairRetentionReportTest(unittest.TestCase):
    def test_raw_method_distinguishes_all_current_series(self) -> None:
        self.assertEqual(
            report.raw_method(
                {"label": 'algo="single_cta"#filter_mode="default"'}
            ),
            "default",
        )
        common = {
            "label": (
                'algo="single_cta"#favor_penalty_mode="cagra_retention_safe"'
                '#filter_mode="favor"'
            ),
            "favor_penalty_lambda": 1.0,
        }
        self.assertEqual(
            report.raw_method({**common, "favor_retention_fraction": 0.5}),
            "favor_retention_safe",
        )
        self.assertEqual(
            report.raw_method({**common, "favor_retention_fraction": 0.0}),
            "automatic_retention",
        )

    def test_report_contains_only_the_24_fair_plots(self) -> None:
        rendered = report.build_report(
            Path("results_fair_retention_single_comparison"),
            Path("results_fair_retention_multi_comparison"),
            gpu_name="NVIDIA L4",
            result_date="2026-08-05",
        )

        self.assertEqual(rendered.count("[[file:"), 24)
        self.assertEqual(rendered.count("results_fair_retention_single"), 12)
        self.assertEqual(rendered.count("results_fair_retention_multi"), 12)
        self.assertNotIn("archive", rendered.lower())
        self.assertNotIn("historical", rendered.lower())
        self.assertNotIn("results_automatic_retention_", rendered)
        self.assertNotIn("results_retention_safe_", rendered)
        self.assertNotIn("results_corrected_retention_", rendered)

    def test_fair_report_rejects_a_missing_filtering_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            config_dir = raw_root / "gist" / "configs"
            config_dir.mkdir(parents=True)
            for selectivity in report.SELECTIVITIES:
                rate = round(1.0 - selectivity / 100.0, 2)
                row = {"filtering_rate": rate}
                if selectivity == 10:
                    row = {}
                payload = {"index": [{"search_params": [row]}]}
                path = config_dir / f"gist_s{selectivity:02d}_nq1.json"
                path.write_text(json.dumps(payload))
            spec = report.ModeSpec(
                "MULTI_CTA", raw_root, raw_root / "plots", 1, 1
            )

            with self.assertRaisesRegex(ValueError, "missing filtering_rate"):
                report.validate_filtering_rates(spec, "gist", "gist")


if __name__ == "__main__":
    unittest.main()
