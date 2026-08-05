import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import generate_configs


class GenerateConfigsTest(unittest.TestCase):
    def test_rates_are_explicit_and_methods_are_interleaved_by_cell(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            argv = [
                "generate_configs.py",
                "--output-dir",
                str(output),
                "--selectivities",
                "1",
                "90",
                "--batch-sizes",
                "1",
                "--itopk-values",
                "32",
                "64",
                "--search-widths",
                "1",
                "--algo",
                "multi_cta",
                "--modes",
                "default",
                "favor_retention_safe",
                "--retention-fractions",
                "0.5",
                "0",
            ]
            with mock.patch.object(sys, "argv", argv):
                generate_configs.main()

            for selectivity, expected_rate in ((1, 0.99), (90, 0.1)):
                path = output / f"sift_s{selectivity:02d}_nq1.json"
                rows = json.loads(path.read_text())["index"][0][
                    "search_params"
                ]
                self.assertEqual(len(rows), 6)
                self.assertTrue(
                    all(row["filtering_rate"] == expected_rate for row in rows)
                )
                observed = [
                    (
                        row["itopk"],
                        row["filter_mode"],
                        row.get("favor_retention_fraction"),
                    )
                    for row in rows
                ]
                self.assertEqual(
                    observed,
                    [
                        (32, "default", None),
                        (32, "favor", 0.5),
                        (32, "favor", 0.0),
                        (64, "default", None),
                        (64, "favor", 0.5),
                        (64, "favor", 0.0),
                    ],
                )


if __name__ == "__main__":
    unittest.main()
