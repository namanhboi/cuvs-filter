# Directed-expansion ablation

`adaptive_blind` retains adaptive_kuzu's per-parent one-hop decision (P/D >= 0.4),
then uses capped blind two-hop expansion below that boundary. `adaptive_kuzu`
remains the default. Both modes use the same runtime kernel and workspace.

The runner compares YFCC and ArXiv EM/EMIS/R, k=10 and k=100, on the same first
2,048 queries, dense bitmaps, degree-64 graphs, SINGLE_CTA and max_queries=2048.
Frozen settings come from the paused primary latency experiment's manifest;
only its settings and first batched shard are read. That experiment stays paused.

Fixed settings: change only policy, retaining frontier, iteration cap and seeds.
Subset recall may differ from the original 10,000-query paper recall. Matched
settings: independently calibrate both modes at YFCC [0.800,0.802] and ArXiv
[0.950,0.952], with W=1/2, k<=L<=512 and the W*D seed rule. Coarse L anchors
are refined at integer target crossings. Iteration refinement explores shallow
caps and doubles to the legal hash limit, then resolves observed crossings.
Calibration uses one repetition; up to four in-window finalists receive three
fresh repetitions. A match requires every repetition to remain in the window.
Unmatched results are explicitly labeled; calibration CSVs preserve all points.
No interpolation is used.

Timing includes the bitmap-seed prepass and search. Uploads are excluded.
Three-repetition comparisons use median QPS with ranges. Diagnostic runs are
separate: their work counters must not be used as throughput measurements.
Diagnostic GT tracing covers top-10 even for k=100; recall uses full workload GT.

Build the CAGRA benchmark and bitmap/UDF correctness targets in the isolated
26.12 build first, then run using a new absolute result directory:

```bash
python3 ablation.py prepare --root /data/retrieve_workshop_runs/navix_policy_TIMESTAMP
python3 ablation.py run --root /data/retrieve_workshop_runs/navix_policy_TIMESTAMP
python3 ablation.py status --root /data/retrieve_workshop_runs/navix_policy_TIMESTAMP
```

The run holds the shared GPU lock and its own exclusive run lock. Completed jobs
are checked against config signatures, raw hashes, repetitions and correctness
counters before reuse. Partial raw outputs are preserved before retry. Runtime
and input hashes are recorded; changed runtime/runner or changed input metadata
refuses resume. Analysis is updated after each workload and retains raw JSON,
configs, logs, fixed and matched tables, calibration CSVs and recall–QPS plots.

CPU checks: `python3 -m unittest discover -s . -v` in this directory.
