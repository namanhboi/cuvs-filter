from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_multi_cta_retention_regression as regression


class MultiCtaRetentionRegressionTest(unittest.TestCase):
    def test_manifest_refuses_changed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = {
                name: root / name
                for name in (
                    "baseline-binary",
                    "baseline-library",
                    "candidate-binary",
                    "candidate-library",
                )
            }
            for name, path in artifacts.items():
                path.write_bytes(name.encode())
            args = SimpleNamespace(
                result_root=root / "results",
                baseline="baseline",
                baseline_commit="a" * 40,
                baseline_binary=artifacts["baseline-binary"],
                baseline_library=artifacts["baseline-library"],
                candidate="candidate",
                candidate_commit="b" * 40,
                candidate_binary=artifacts["candidate-binary"],
                candidate_library=artifacts["candidate-library"],
                iterations=100,
                repetitions=3,
            )
            regression.write_manifest(args)
            regression.write_manifest(args)
            artifacts["candidate-binary"].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "experiment identity changed"):
                regression.write_manifest(args)

    def test_generated_configs_are_old_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory) / "configs"
            regression.generate_configs(config_dir, Path("/data"))
            self.assertEqual(
                {path.stem for path in config_dir.glob("*.json")},
                set(regression.DATASETS),
            )
            for path in config_dir.glob("*.json"):
                config = json.loads(path.read_text())
                params = config["index"][0]["search_params"]
                self.assertNotIn("favor_retention_fraction", path.read_text())
                self.assertEqual(
                    {regression._method(param) for param in params}, set(regression.METHODS)
                )
                self.assertTrue(all(param["search_width"] == 1 for param in params))

    def test_analysis_detects_normalized_fixed_regression(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            favor_dir = Path(directory) / "favor"
            result_root = favor_dir / "results"
            config_dir = result_root / "configs"
            regression.generate_configs(config_dir, Path("/data"))
            manifest = {
                "builds": {"baseline": "a" * 40, "candidate": "b" * 40}
            }
            result_root.mkdir(parents=True, exist_ok=True)
            (result_root / "manifest.json").write_text(json.dumps(manifest))
            for build in ("baseline", "candidate"):
                raw_dir = result_root / "raw" / build
                raw_dir.mkdir(parents=True)
                for key in regression.DATASETS:
                    config = json.loads((config_dir / f"{key}.json").read_text())
                    rows = []
                    for family, param in enumerate(config["index"][0]["search_params"]):
                        method = regression._method(param)
                        gpu = 0.0011 if build == "candidate" and method == "fixed" else 0.001
                        for repetition in range(3):
                            rows.append(
                                {
                                    "name": f"family-{family}-rep-{repetition}",
                                    "run_name": f"family-{family}-rep-{repetition}",
                                    "run_type": "iteration",
                                    "family_index": family,
                                    "repetition_index": repetition,
                                    "iterations": 100,
                                    "real_time": 1.0,
                                    "time_unit": "ms",
                                    "GPU": gpu,
                                    "Latency": gpu,
                                    "Recall": 0.99,
                                    "UnderfilledQueries": 0.0,
                                    "MissingResultSlots": 0.0,
                                    "total_queries": 100,
                                }
                            )
                    raw = {"context": {"gpu_name": regression.EXPECTED_GPU}, "benchmarks": rows}
                    (raw_dir / f"{key}.json").write_text(json.dumps(raw))
            regression.analyze(
                SimpleNamespace(
                    result_root=result_root,
                    baseline="baseline",
                    candidate="candidate",
                    repetitions=3,
                    iterations=100,
                )
            )
            summary = json.loads((result_root / "analysis" / "summary.json").read_text())
            self.assertEqual(summary["verdict"], "PERFORMANCE_REGRESSION")
            self.assertAlmostEqual(summary["normalized_fixed_ratio_geomean"], 1.1)
            self.assertTrue((favor_dir / "MULTI_CTA_RETENTION_REGRESSION_REPORT.org").exists())


if __name__ == "__main__":
    unittest.main()
