# Paused batch-1 latency results — 2026-10-28

The user stopped the full 10,000-query experiment. The measurement worker and
automatic publication worker are stopped; no benchmark remains running.

## Complete subset measurements

The table below uses one serialized pass over the same first 1,000 queries for
every method/workload/k combination. CAGRA is SINGLE_CTA, all methods use dense
resident bitmaps, and the exact frozen QPS–recall configurations are preserved.
Mean host-call-to-GPU-completion / CUDA-event elapsed latency is shown in ms.
These are preliminary subset latencies, not the planned three-pass, 10,000-query
statistics. Subset achieved recalls differ from full-workload recalls.

| Workload | k | Base | Retain | NaviX | IVF-Flat |
|---|---:|---:|---:|---:|---:|
| YFCC | 10 | 95.592 / 95.582 † | 4.293 / 4.286 | 4.593 / 4.585 | 0.136 / 0.128 |
| ArXiv EM | 10 | 2.566 / 2.559 | 1.984 / 1.976 | 1.873 / 1.865 | 2.157 / 2.149 |
| ArXiv EMIS | 10 | 176.784 / 176.773 † | 62.704 / 62.695 | 3.530 / 3.522 | 3.305 / 3.297 |
| ArXiv R | 10 | 4.341 / 4.333 | 2.380 / 2.372 | 1.712 / 1.705 | 1.443 / 1.436 |
| YFCC | 100 | 95.591 / 95.582 † | 70.027 / 70.018 † | 7.099 / 7.091 | 0.222 / 0.215 |
| ArXiv EM | 100 | 17.249 / 17.241 | 11.068 / 11.060 | 4.089 / 4.082 | 2.376 / 2.369 |
| ArXiv EMIS | 100 | 176.809 / 176.798 † | 137.210 / 137.200 † | 6.224 / 6.216 | 4.463 / 4.455 |
| ArXiv R | 100 | 70.752 / 70.742 † | 13.866 / 13.858 | 3.839 / 3.831 | 1.610 / 1.603 |

† The source sweep never reached its target: this is its saved maximum-recall
configuration, not a target-matched point. Target windows on the full workloads
are [0.800,0.802] for YFCC and [0.950,0.952] for ArXiv. The CSV includes the actual
1,000-query achieved recall, the full 10,000-query batched-reference recall, and
both timing metrics’ mean/std/p50/p95/p99/max and timer floors. No floor is subtracted.

## Correctness and full-run progress

All 32 paired 1,000-query batched/serialized checks passed. Maximum absolute
recall difference was 0.000070046, below the agreed 0.0001 limit. All 32 full
10,000-query batched references also completed; their maximum recall difference
from the saved source measurements was 0.000040.

Only the first full-run YFCC k=10 shard completed: 2,048 queries each for Base,
Retain, and NaviX, one pass each (6,144 verified measured calls of the planned
960,000). Its values are in completed_main_shards.csv and are not mixed into
the common-subset table. The interrupted next shard remains preserved as an
unmarked partial attempt for an explicit future resume.

## Files and provenance

- subset_latency_1000.csv/.json: all 32 subset latency distributions and recalls.
- completed_main_shards.csv: the three completed first-shard measurements.
- configurations.csv: the exact selected settings and source target status.
- gate.json and baseline_vs_source.csv: paired and full-reference checks.
- manifest.json, build_provenance.json, and configuration_audit.json: immutable
  configurations, input/runtime identities, and validated build details.
- Full raw traces, runtime copies, logs, and partial attempts remain at `/data/retrieve_workshop_runs/a100_batch1_primary_ivf_20261028T073000Z`.

Do not restart full measurement or publication automatically. To resume when the
user asks, restart the original supervisor.json with run_job.py so the existing
hash-verified outputs are retained and only incomplete jobs are rerun. Restart
the finalizer separately with the then-current expected main revision if desired.
