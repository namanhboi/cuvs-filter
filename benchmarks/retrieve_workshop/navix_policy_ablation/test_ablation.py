import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import ablation as a


class AblationTest(unittest.TestCase):
    def test_policy_only_fixed_change(self):
        template = dict(
            navix_mode="adaptive_kuzu",
            itopk=54,
            search_width=2,
            max_iterations=0,
            navix_seed_cap=128,
            rand_xor_mask=1213332,
        )
        left = a.point(template, "adaptive_kuzu", 54, 2, 0)
        right = a.point(template, "adaptive_blind", 54, 2, 0)
        self.assertEqual(
            {k for k in left if left[k] != right[k]},
            {"navix_mode", "validation_case_id"},
        )
        self.assertEqual(template["navix_mode"], "adaptive_kuzu")

    def test_tuning_seed_rule(self):
        self.assertEqual(
            a.point(
                dict(navix_seed_cap=128),
                "adaptive_blind",
                100,
                1,
                8,
                tune=True,
            )["navix_seed_cap"],
            64,
        )

    def test_boundaries_and_hash_limit(self):
        self.assertTrue(a.within(0.952, 0.95))
        self.assertFalse(a.within(0.95201, 0.95))
        self.assertTrue(a.legal(100, 1, 100, 100))
        self.assertFalse(a.legal(99, 1, 100, 100))
        self.assertFalse(a.legal(512, 2, 7569, 100))

    def test_execution_identity(self):
        def p(frontier, m):
            return dict(
                navix_mode="adaptive_blind",
                itopk=frontier,
                search_width=2,
                max_iterations=m,
                navix_seed_cap=128,
            )

        self.assertEqual(a.fingerprint(p(100, 0)), a.fingerprint(p(101, 0)))
        self.assertNotEqual(a.fingerprint(p(100, 0)), a.fingerprint(p(102, 0)))

    def fixture(self):
        sp = a.point(dict(navix_seed_cap=64), "adaptive_blind", 10, 1, 0)
        row = dict(
            label=f'validation_case_id="{sp["validation_case_id"]}"',
            n_queries=2048,
            FilterViolations=0,
            InvalidSentinelErrors=0,
            SentinelOrderErrors=0,
            DuplicateOutputQueries=0,
            ValidGTRecall=0.951,
            items_per_second=100,
            repetition_index=0,
        )
        return sp, row

    def test_validation(self):
        sp, row = self.fixture()
        self.assertEqual(
            a.validate(dict(benchmarks=[row]), [sp], 1)[0]["qps"], 100
        )
        for field in (
            "FilterViolations",
            "DuplicateOutputQueries",
            "n_queries",
            "repetition_index",
        ):
            bad = copy.deepcopy(row)
            bad[field] = 7
            with self.assertRaises(ValueError):
                a.validate(dict(benchmarks=[bad]), [sp], 1)
        with self.assertRaises(ValueError):
            a.validate(dict(benchmarks=[]), [sp], 1)

    def test_completed_job_rejects_changed_raw_and_configuration(self):
        sp, row = self.fixture()
        case = dict(id="test", config=dict(search_basic_param={}, index=[{}]))

        def fake_run(command, **kwargs):
            raw = next(
                c.split("=", 1)[1]
                for c in command
                if c.startswith("--benchmark_out=")
            )
            a.save(raw, dict(benchmarks=[row]))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(
                a.subprocess, "run", side_effect=fake_run
            ) as run:
                a.job(root, case, "fixed", [copy.deepcopy(sp)])
                a.job(root, case, "fixed", [copy.deepcopy(sp)])
                self.assertEqual(run.call_count, 1)
                changed = copy.deepcopy(sp)
                changed["itopk"] = 11
                with self.assertRaises(ValueError):
                    a.job(root, case, "fixed", [changed])
                (root / "jobs/test/fixed/raw.json").write_text("{}")
                with self.assertRaises(ValueError):
                    a.job(root, case, "fixed", [copy.deepcopy(sp)])

    def test_tuner_completes_both_modes_with_strict_matches(self):
        def fake_job(root, case, stage, points, reps=1, diagnostic=False):
            result = []
            for sp in points:
                budget = sp["max_iterations"] or a.b0(
                    sp["itopk"], sp["search_width"]
                )
                recall = min(0.999, 0.5 + budget * 0.01)
                result.append(
                    dict(
                        search=sp,
                        recall=recall,
                        recalls=[recall] * reps,
                        qps=1000 / budget,
                        qps_min=1000 / budget,
                        qps_max=1000 / budget,
                    )
                )
            return result

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(a, "job", side_effect=fake_job):
                result = a.tune(
                    Path(directory),
                    dict(id="test", k=10, target=0.8),
                    dict(navix_seed_cap=64),
                )
            self.assertEqual({r["mode"] for r in result}, set(a.MODES))
            self.assertTrue(all(r["matched"] for r in result))


if __name__ == "__main__":
    unittest.main()
