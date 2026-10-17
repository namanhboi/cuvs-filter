import copy
import json
import struct
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import validate


class ValidationContractTest(unittest.TestCase):
    def test_cache_grouping_preserves_every_frozen_job_and_config(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            jobs, before = [], {}
            for i, (variant, index) in enumerate(
                (
                    ("old", "old.index"),
                    ("integrated", "new.index"),
                    ("old", "old.index"),
                    ("integrated", "new.index"),
                )
            ):
                config = root / f"{i}.json"
                config.write_text(
                    json.dumps(
                        {
                            "index": [
                                {"algo": "cuvs_ivf_flat", "file": index}
                            ],
                            "dataset": {"base_file": "frozen.fbin"},
                        }
                    )
                )
                before[config] = config.read_bytes()
                jobs.append(
                    {"id": str(i), "variant": variant, "config": str(config)}
                )
            snapshot = copy.deepcopy(jobs)
            result = validate.cache_order(jobs)
            self.assertEqual(jobs, snapshot)
            self.assertEqual(sorted(result, key=lambda x: x["id"]), jobs)
            self.assertEqual(
                [
                    json.loads(Path(x["config"]).read_text())["index"][0][
                        "file"
                    ]
                    for x in result
                ],
                ["new.index", "new.index", "old.index", "old.index"],
            )
            self.assertEqual({p: p.read_bytes() for p in before}, before)

    def test_native_missing_slot_encoding_does_not_hide_neighbor_changes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            summary = {"n_queries": 1, "k": 2, "n_rows": 100}

            def output(name, node, empty):
                prefix = root / name
                prefix.with_suffix(".ids").write_bytes(
                    struct.pack("<qq", node, empty)
                )
                prefix.with_suffix(".distances").write_bytes(
                    struct.pack("<ff", 1.0, float.fromhex("0x1.fffffep127"))
                )
                return validate.native_output_identity(prefix, summary)

            upstream = output("upstream", 7, (1 << 31) - 1)
            canonical = output("canonical", 7, (1 << 63) - 1)
            different = output("different", 8, (1 << 63) - 1)
            self.assertNotEqual(
                upstream["ids_sha256"], canonical["ids_sha256"]
            )
            self.assertEqual(
                upstream["comparable_ids_sha256"],
                canonical["comparable_ids_sha256"],
            )
            self.assertNotEqual(
                upstream["comparable_ids_sha256"],
                different["comparable_ids_sha256"],
            )
            self.assertEqual(upstream["legacy_sentinel_slots"], 1)

    def test_native_unknown_ids_and_invalid_sentinel_distances_are_rejected(
        self,
    ):
        with tempfile.TemporaryDirectory() as d:
            prefix = Path(d) / "point"
            summary = {"n_queries": 1, "k": 1, "n_rows": 100}
            prefix.with_suffix(".ids").write_bytes(struct.pack("<q", 101))
            prefix.with_suffix(".distances").write_bytes(
                struct.pack("<f", 1.0)
            )
            with self.assertRaisesRegex(
                ValueError, "unexpected native invalid ID"
            ):
                validate.native_output_identity(prefix, summary)
            prefix.with_suffix(".ids").write_bytes(
                struct.pack("<q", (1 << 31) - 1)
            )
            with self.assertRaisesRegex(
                ValueError, "sentinel has a valid-looking distance"
            ):
                validate.native_output_identity(prefix, summary)

    def test_correctness_resume_requires_unchanged_library_and_results(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            binary, library, output = (
                root / "binary",
                root / "library",
                root / "tests.json",
            )
            binary.write_bytes(b"code")
            library.write_bytes(b"library")

            def run(*args):
                output.write_text('{"tests": 2, "failures": 0}')

            with patch.object(validate, "execute", side_effect=run) as execute:

                def check():
                    validate.correctness_suite(
                        [str(binary)], binary, library, output, root, 0
                    )

                check()
                check()
                self.assertEqual(execute.call_count, 1)
                library.write_bytes(b"new library")
                check()
                self.assertEqual(execute.call_count, 2)
                output.write_text('{"tests": 2, "failures": 1}')
                check()
                self.assertEqual(execute.call_count, 3)

    def test_failed_correctness_suite_is_never_marked_complete(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            binary, output = root / "binary", root / "tests.json"
            binary.write_bytes(b"code")
            with patch.object(validate, "execute"):
                for result in ({"tests": 0}, {"tests": 2, "failures": 1}):
                    output.write_text(json.dumps(result))
                    with self.assertRaisesRegex(
                        ValueError, "empty or failing"
                    ):
                        validate.correctness_suite(
                            [str(binary)], binary, binary, output, root, 0
                        )
                    self.assertFalse(
                        output.with_suffix(".complete.json").exists()
                    )

    def config(self):
        return {
            "dataset": {
                "name": "frozen",
                "dtype": "float",
                "base_file": "base.fbin",
                "query_file": "query.bin",
                "groundtruth_neighbors_file": "gt.ibin",
                "filter": {"kind": "bitmap", "file": "filter.bitmap"},
            },
            "search_basic_param": {"k": 10, "batch_size": 2048},
            "index": [
                {
                    "algo": "cuvs_cagra",
                    "file": "saved.index",
                    "build_param": {},
                    "search_params": [
                        {
                            "bitmap_method": "navix_reference",
                            "itopk": 35,
                            "search_width": 1,
                            "max_iterations": 0,
                            "navix_seed_cap": 64,
                            "rand_xor_mask": 123456,
                            "favor_udf_passing_accumulator": False,
                        }
                    ],
                }
            ],
        }

    def test_conversion_preserves_search_and_seed_settings_without_mutating_source(
        self,
    ):
        original = self.config()
        snapshot = copy.deepcopy(original)
        result = validate.normalize_config(
            original,
            Path("/inputs"),
            True,
            Path("/copies"),
            "roaring",
            "case",
            Path("/absent"),
        )
        self.assertEqual(original, snapshot)
        point = result["index"][0]["search_params"][0]
        self.assertEqual(
            (point["itopk"], point["search_width"], point["max_iterations"]),
            (35, 1, 0),
        )
        self.assertEqual(
            (point["rand_xor_mask"], point["navix_seed_cap"]), (123456, 64)
        )
        self.assertFalse(point["passing_result_accumulator"])
        self.assertNotIn("favor_udf_passing_accumulator", point)
        self.assertEqual(
            result["index"][0]["file"], "/copies/arxiv_cagra.index"
        )

    def records(self):
        return [
            {
                "run_type": "iteration",
                "repetition_index": i,
                "n_queries": 2048,
                "label": '#validation_case_id="case"',
                "ValidGTRecall": 0.95,
                "items_per_second": qps,
                "FilterViolations": 0,
            }
            for i, qps in enumerate([1000, 1100, 900])
        ]

    def parse(self, rows):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "raw.json"
            p.write_text(json.dumps({"benchmarks": rows}))
            return validate.read_measurements(p, ["case"])

    def test_three_repetitions_and_aggregate_rows(self):
        rows = self.records() + [
            {"run_type": "aggregate", "aggregate_name": "median"}
        ]
        result = self.parse(rows)[0]
        self.assertEqual(result["qps"], 1000)
        self.assertEqual(result["recall"], 0.95)

    def test_incomplete_or_duplicate_repetition_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete repetitions"):
            self.parse(self.records()[:2])
        rows = self.records()
        rows[2]["repetition_index"] = 0
        with self.assertRaisesRegex(ValueError, "incomplete repetitions"):
            self.parse(rows)

    def test_wrong_subset_or_invalid_filtered_output_is_rejected(self):
        rows = self.records()
        rows[0]["n_queries"] = 10000
        with self.assertRaisesRegex(ValueError, "query count"):
            self.parse(rows)
        rows = self.records()
        rows[0]["FilterViolations"] = 0.0001
        with self.assertRaisesRegex(ValueError, "invalid output"):
            self.parse(rows)

    def test_missing_recall_and_benchmark_errors_cannot_be_reported_as_success(
        self,
    ):
        rows = self.records()
        rows[0].pop("ValidGTRecall")
        with self.assertRaises(KeyError):
            self.parse(rows)
        rows = self.records()
        rows[0]["error_occurred"] = True
        with self.assertRaisesRegex(ValueError, "benchmark failure"):
            self.parse(rows)

    def test_regression_threshold_boundaries(self):
        old = {"qps": 1000, "recall": 0.95}
        same = validate.comparison(old, {"qps": 900, "recall": 0.952})
        self.assertFalse(same["qps_flag"])
        self.assertFalse(same["recall_flag"])
        worse = validate.comparison(old, {"qps": 899, "recall": 0.947})
        self.assertTrue(worse["qps_flag"])
        self.assertTrue(worse["recall_flag"])

    def test_exact_control_requires_independent_native_distance_validation(
        self,
    ):
        rows = self.records()
        for row in rows:
            row["label"] = '#validation_case_id="case__exact"'
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "raw.json"
            p.write_text(json.dumps({"benchmarks": rows}))
            with self.assertRaisesRegex(
                ValueError, "native-distance validation"
            ):
                validate.read_measurements(p, ["case__exact"])
            for row in rows:
                row.update(NativeL2CutoffValidated=1, NativeL2CutoffRecall=1)
            p.write_text(json.dumps({"benchmarks": rows}))
            self.assertEqual(
                len(validate.read_measurements(p, ["case__exact"])), 1
            )

    def test_archived_scalar_reconstruction_requires_a_matching_method_and_width(
        self,
    ):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cohort/shard_00.json"
            p.parent.mkdir()
            config = self.config()
            p.write_text(json.dumps(config))
            csv = Path(d) / "frozen.csv"
            csv.write_text("recorded search parameters\n")
            row = {
                "method": "navix_reference",
                "itopk": 100,
                "search_width": 1,
                "max_iterations": 8184,
            }
            result = validate.source_point(
                [(p, config)], row, 10, "cohort", False, csv
            )
            point = result["config"]["index"][0]["search_params"][0]
            self.assertEqual(point["max_iterations"], 8184)
            self.assertEqual(point["navix_seed_cap"], 64)
            self.assertEqual(
                result["parameter_source_sha256"], validate.sha(csv)
            )
            row["search_width"] = 2
            with self.assertRaisesRegex(
                ValueError, "missing frozen configuration"
            ):
                validate.source_point(
                    [(p, config)], row, 10, "cohort", False, csv
                )


if __name__ == "__main__":
    unittest.main()
