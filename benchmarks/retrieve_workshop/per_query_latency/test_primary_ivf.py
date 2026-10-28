import copy
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import primary_ivf as p


class FrozenBatchOneTest(unittest.TestCase):
    def test_frozen_content_mismatch_is_rejected_even_after_timestamp_change(
        self,
    ):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "input"
            file.write_bytes(b"original")
            previous = p.identity(file)
            file.write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "frozen experiment"):
                p.identity(file, {str(file): previous})

    def test_single_cta_and_original_scalar_selection_are_required(self):
        cfg = {
            "index": [
                {
                    "search_params": [
                        {
                            "algo": "single_cta",
                            "itopk": 33,
                            "search_width": 2,
                            "max_iterations": 0,
                        }
                    ]
                }
            ]
        }
        row = dict(
            method="navix_reference",
            itopk=33,
            search_width=2,
            max_iterations=0,
        )
        p.selected_settings(cfg, row)
        with self.assertRaises(ValueError):
            p.selected_settings(cfg, row | {"itopk": 64})
        cfg["index"][0]["search_params"][0]["algo"] = "auto"
        with self.assertRaises(ValueError):
            p.selected_settings(cfg, row)

    def test_partial_last_shard_is_required(self):
        shards = [
            dict(first_query=i * 2048, query_count=2048 if i < 4 else 1808)
            for i in range(5)
        ]
        p.validate_shards(shards)
        with self.assertRaises(ValueError):
            p.validate_shards(shards[:4])

    def test_trace_changes_batch_only_and_preserves_seed_and_search_budget(
        self,
    ):
        with tempfile.TemporaryDirectory() as d:
            search = dict(
                algo="single_cta",
                itopk=100,
                search_width=2,
                max_iterations=0,
                max_queries=2048,
                navix_seed_cap=128,
                rand_xor_mask=123,
                hashmap_max_fill_rate=0.5,
            )
            template = dict(
                dataset={},
                search_basic_param=dict(batch_size=2048, k=100),
                index=[dict(search_params=[search])],
            )
            point = dict(
                id="em__k100__navix_reference",
                search=search,
                workload="em",
                method="navix_reference",
                source_recall=0.951,
                target=0.95,
            )
            before = copy.deepcopy((template, point))
            cfg, _ = p.job_config(
                template,
                [point],
                {},
                Path(d),
                "trace",
                dict(query_count=1808, first_query=8192, index=4),
                2,
            )
            sp = cfg["index"][0]["search_params"][0]
            self.assertEqual((template, point), before)
            for field in (
                "itopk",
                "search_width",
                "max_iterations",
                "navix_seed_cap",
                "rand_xor_mask",
                "hashmap_max_fill_rate",
            ):
                self.assertEqual(sp[field], search[field])
            self.assertEqual(
                (
                    cfg["search_basic_param"]["batch_size"],
                    sp["n_queries"],
                    sp["max_queries"],
                ),
                (1, 1, 1),
            )
            self.assertEqual(sp["benchmark_latency_global_query_offset"], 8192)

    def fixture(self, root):
        job = dict(
            stage="trace",
            count=3,
            first=8192,
            k=100,
            workload="em",
            repetition=2,
            shard=4,
        )
        case = dict(
            id="em__k100__ivf_flat",
            trace=str(root / "trace.csv"),
            neighbors=None,
        )
        rows = []
        for order, local in enumerate((2, 0, 1)):
            rows.append(
                dict(
                    schema_version=2,
                    record_kind="query",
                    workload="em",
                    method="ivf_flat",
                    repetition=2,
                    shard_index=4,
                    query_local_id=local,
                    query_global_id=8192 + local,
                    call_order=order,
                    host_latency_ns=100,
                    gpu_latency_ns=50,
                    max_queries=1,
                    k=100,
                )
            )
        rows.append(
            rows[0]
            | dict(
                record_kind="timer_floor",
                query_local_id=-1,
                query_global_id=-1,
                gpu_latency_ns=0,
            )
        )
        return job, case, rows

    def test_both_timings_and_rotated_complete_query_coverage(self):
        with tempfile.TemporaryDirectory() as d, patch.object(p, "FLOOR", 1):
            job, case, rows = self.fixture(Path(d))
            p.write_csv(case["trace"], rows)
            self.assertEqual(len(p.trace_rows(job, case)[0]), 3)
            rows[1]["query_global_id"] = rows[0]["query_global_id"]
            p.write_csv(case["trace"], rows)
            with self.assertRaisesRegex(ValueError, "every query"):
                p.trace_rows(job, case)

    def test_missing_gpu_time_or_incomplete_trace_cannot_pass(self):
        with tempfile.TemporaryDirectory() as d, patch.object(p, "FLOOR", 1):
            job, case, rows = self.fixture(Path(d))
            rows[0]["gpu_latency_ns"] = 0
            p.write_csv(case["trace"], rows)
            with self.assertRaisesRegex(ValueError, "timing"):
                p.trace_rows(job, case)
            p.write_csv(case["trace"], rows[:-1])
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                p.trace_rows(job, case)

    def test_recall_weights_valid_gt_slots_not_shard_sizes(self):
        self.assertEqual(
            p.aggregate(
                [dict(matching=9, valid_gt=10), dict(matching=1, valid_gt=2)]
            ),
            10 / 12,
        )

    def test_population_stats_and_nearest_rank_percentiles(self):
        stats = p.stats([1000, 2000, 3000, 100000])
        self.assertEqual(stats["p50_us"], 2)
        self.assertEqual(stats["p95_us"], 100)
        self.assertEqual(stats["mean_us"], 26.5)

    def test_resume_rejects_modified_trace(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            raw = root / "raw.json"
            cfg = root / "config.json"
            trace = root / "trace.csv"
            raw.write_text("{}")
            cfg.write_text("{}")
            trace.write_text("original")
            job = dict(
                raw=str(raw),
                config=str(cfg),
                cases=[dict(trace=str(trace), neighbors=None)],
                id="case",
            )
            p.save(
                str(raw) + ".complete.json",
                dict(
                    signature="frozen",
                    config_sha256=p.sha(cfg),
                    outputs=p.output_hashes(job),
                ),
            )
            trace.write_text("changed")
            with self.assertRaisesRegex(ValueError, "refusing reuse"):
                p.completed(job, "frozen")

    def test_gate_copy_preserves_k100_gt_and_nonzero_bitmap_rows(self):
        with tempfile.TemporaryDirectory() as d, patch.object(p, "GATE", 2):
            root = Path(d)
            q = root / "q.bin"
            gt = root / "gt.ibin"
            bm = root / "filter.bitmap"
            q.write_bytes(struct.pack("<II", 3, 2) + bytes(range(6)))
            gt.write_bytes(struct.pack("<II", 3, 100) + bytes(3 * 100 * 4))
            bm.write_bytes(
                struct.pack("<8sIIQQQ", b"CUVSBMAP", 1, 32, 3, 32, 3)
                + struct.pack("<III", 1, 2, 4)
            )
            ds = dict(
                name="em-k100",
                dtype="uint8",
                query_file=str(q),
                groundtruth_neighbors_file=str(gt),
                filter=dict(file=str(bm)),
            )
            out = p.gate_inputs(root, ds)
            self.assertEqual(
                Path(out["groundtruth_neighbors_file"]).read_bytes()[:8],
                struct.pack("<II", 2, 100),
            )
            self.assertEqual(
                Path(out["filter"]["file"]).read_bytes()[40:],
                struct.pack("<II", 1, 2),
            )
            self.assertEqual(
                bm.read_bytes()[40:], struct.pack("<III", 1, 2, 4)
            )


if __name__ == "__main__":
    unittest.main()
