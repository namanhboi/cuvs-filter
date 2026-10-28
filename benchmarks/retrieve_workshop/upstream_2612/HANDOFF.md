# Handoff — 2026-10-28

The user is switching to a new task. Save this state and do not resume tuning or
launch benchmarks unless the user asks. No tuning or benchmark jobs were running
at the last process check on 2026-10-28.

## Completed work

The agreed upstream integration plan is complete, including review, merge, and
push to the user's repository. Main's integration merge is
`0f99eaab1ab66a8822167d0b45fef1c92e481d11`; the subsequent handoff commit changes
documentation only. The integration branch is `integrate-upstream-26.12`, at
`6aa8eb673f352dd8fc7b1f3d40af67d0928ae9ad`.

- Merged pinned NVIDIA cuVS `3721d31fb98ba927417f42fbe3f93ab9ff15ed8e`, 26.12
  development, into the user's fork.
- Deprecated FAVOR execution, APIs, setup, sidecars, and experiment entrypoints;
  preserved shared Retain/NaviX functionality and historical results.
- Preserved CAGRA Base, Retain, and NaviX and their dense per-query bitmaps.
- Added native per-query Roaring support for those CAGRA methods, including
  passing seeds, remapped IDs, query offsets, and container ownership.
- Preserved IVF-Flat/SQ/PQ per-query dense bitmap filtering and refinement checks.
- Converted copies of two CAGRA and four IVF-Flat saved indexes, preserving graph,
  centroid, vector, list-assignment, and ID contents. All six loaded in native cuVS.
- Converted five sparse query bitmaps to independently verified sorted-ID
  interchange artifacts; native Roaring owners are constructed from these IDs.
- Completed correctness and representative performance validation, committed the
  evidence and review, and pushed the merge to `namanhboi/cuvs-filter` main.

Push only to `git@github.com:namanhboi/cuvs-filter.git`. Never push to NVIDIA.

## Validation and qualifications

- GPU suites: 518 passed, 17 intentional skips, zero failures. CPU workflows:
  51 tests passed. A separate early Roaring seed check passed 20 GPU cases.
- Frozen first-2,048-query replay, three repetitions: 75/75 benchmark processes,
  covering 57 old-fork/integrated source cases and Roaring for all 39 CAGRA cases.
- Public-API comparison: 26/26 executions, covering 13 upstream/integration pairs.
- Old versus integrated dense QPS ratios: 0.983–1.013. Maximum absolute recall
  change: 0.000009775. All agreed compatibility thresholds passed.
- All 13 integrated native recall values equal the corresponding previous-fork
  recall and exceed pinned upstream's recall. Source inspection identifies
  inherited result-compaction fixes absent upstream; this is not a complete
  end-to-end ablation of every changed neighbor.
- YFCC NaviX k=100 has a four-ground-truth-hit dense/Roaring median difference
  (0.000019551). Repetition ranges overlap across old dense, integrated dense, and
  Roaring. Traversal source differs only by a diagnostic namespace rename.
  Concurrent visited-set insertion is plausible, not a proven isolated cause.
- Raw native API outputs occasionally contain duplicates in both checkouts;
  benchmark output is deduplicated and recall counts unique hits.
- Roaring is usually slower, particularly for NaviX; dense remains the default.
  Across the 39 CAGRA cases, Roaring/dense QPS ratios are 0.894–1.007 for Base,
  0.800–1.004 for Retain, and 0.413–0.964 for NaviX.
- Raw automated flags and `ready_for_review: false` remain unchanged. The explicit
  merge review documents why these strict-equality findings were accepted.

These were compatibility runs with frozen settings, not newly tuned matched-recall
comparisons. Some recalls differ from the original 10,000-query paper workload.
Do not present their QPS as a new matched-recall ranking.

## What remains, currently paused or deferred

1. **Matched-recall dense/Roaring tuning:** the user chose tuning on the first
   2,048 queries at k=10 and k=100, then explicitly stopped it before any new
   tuning sweep was launched. No new matched-recall Roaring sweep or results exist.
   Read each workload's existing recall target/tolerance from the frozen paper
   records rather than assuming one target applies to every workload/result width.
2. **Full parameter sweeps:** deferred beyond representative integration validation.
3. **IVF-PQ performance tuning:** stopped earlier at the user's request. Existing
   partial results remain saved; do not restart automatically.
4. **IVF Roaring support:** not implemented. Our IVF filtering currently uses dense
   per-query bitmaps only.
5. **IVF-RaBitQ filtering and benchmarks:** not implemented. Pinned upstream has
   IVF-RaBitQ, but its search API has no predicate-filter parameter. Upstream
   Flat/SQ/PQ accept a shared dense bitset; they do not dispatch per-query bitmap,
   Roaring, Bloom, or UDF filters in this pinned version. Our fork extends
   Flat/SQ/PQ with per-query dense bitmap filtering.
6. Further profiling of Roaring overhead, native duplicate handling, and complete
   ablation of upstream compaction differences are possible follow-ups, not active
   jobs or requirements left unfinished in the completed integration plan.

If tuning is resumed, create a separate run directory, preserve the frozen evidence,
hold query subsets/seeds/indexes constant, and tune each method/representation to
the established target. Report actual recall with QPS, and max recall plus its QPS
when the target is unreachable. Label any interpolation explicitly. Do not start
with a full sweep or overwrite paper results just to resume this handoff.

## Locations and build warning

- Main source checkout: `/home/ubuntu/cuvs-filter`.
- Integration source and validated build: `/data/cuvs-upstream-integration` and
  `/data/cuvs-upstream-integration/cpp/build`.
- Isolated 26.12 environment: `/data/cuvs-upstream-envs/cuvs-2612`.
- Unmodified upstream control: `/data/cuvs-upstream-control`, pinned at `3721d31f`.
- Validation workspace, converted index copies, allowlists, logs, binaries, and
  native output arrays: `/data/cuvs-upstream-validation`.
- Representative measurements: `/data/cuvs-upstream-validation/benchmarks`.
- Original datasets/indexes: `/data/retrieve_data`; original runs:
  `/data/retrieve_workshop_runs`.
- Existing IVF runs: `a100_ivf_bitmap_20261014T024338Z` and
  `a100_ivf_flat_sparse_20261015T025433Z` under that original-runs directory.
- Paper matched-recall records:
  `/home/ubuntu/paper_selectivity_revision/data/current_gpu_matched_recall.csv`
  and `current_gpu_matched_recall_k100.csv` in the same directory.

**The original checkout's `/home/ubuntu/cuvs-filter/cpp/build` still contains the
pre-merge binaries. Updating main did not rebuild them.** Use the validated build
above or configure a new build with the 26.12 environment. Previous source is
available at `0b3a56692631a186b875a1542930d7e773a75e69`; do not infer an old binary's
source revision from the main checkout's current HEAD.

Do not remove old environments, binaries, datasets, indexes, or paper artifacts.
Any future runs should use the shared GPU lock
`/data/cuvs-upstream-validation/gpu.lock` and survive client disconnection.

## Start reading here

- [Integration README and replay commands](README.md)
- [Completed validation](VALIDATION.md)
- [Merge review and accepted limitations](MERGE_REVIEW.md)
- [Archived assessment](results/a100_20261017/ASSESSMENT.md)
- [All comparison rows](results/a100_20261017/comparison.csv)
- [Raw comparisons and flags](results/a100_20261017/comparison.json)
- [Frozen configurations, raw repetitions, provenance, and checksums](results/a100_20261017/)

The archived evidence is immutable. No code changes or benchmark runs are needed
merely to switch to the user's next task.
