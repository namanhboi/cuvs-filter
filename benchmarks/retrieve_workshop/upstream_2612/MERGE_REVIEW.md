# Integration merge review — 2026-10-17

Accept the integration for the user's `main` branch. The user authorized completing
the review and merge after publication of the validation evidence. Push only to
`git@github.com:namanhboi/cuvs-filter.git`.

Reviewed integration: `f9dde9f9d8f6b384b197b0ef609594b5eda62861`.
Previous main: `0b3a56692631a186b875a1542930d7e773a75e69`.
Pinned upstream: `3721d31fb98ba927417f42fbe3f93ab9ff15ed8e`.

## Evidence and disposition

- Both isolated builds passed. GPU suites completed with 518 passes, 17 intentional
  skips, and no failures; 51 CPU workflow tests passed. No production source changed
  after the recorded validation, so another GPU replay is unnecessary for this
  documentation and Git merge checkpoint.
- Rechecked all 321 archived artifact checksums and 12 executable/library hashes.
  All 57 previous-fork versus integrated comparisons pass the agreed thresholds:
  QPS ratios 0.983–1.013 and maximum absolute recall change 0.000009775.
- All 13 native integration recalls equal the corresponding previous-fork recall.
  Their differences from pinned upstream are consistent with inherited result
  compaction fixes. The helper is unchanged from previous main. The source
  investigation is not a complete ablation of every changed neighbor; raw native
  ID/recall flags remain recorded. Native QPS ratios are 0.944–1.127.
- The YFCC NaviX k=100 dense/Roaring median difference is four ground-truth hits
  (0.000019551). Repetition ranges overlap in the old fork and both integrated
  representations, and output invariant checks pass. Retain this strict equality
  flag as measured variation, without retuning or asserting a proven root cause.
- Raw native outputs occasionally contain duplicates in both checkouts. The
  benchmark adapter deduplicates, and recall counts unique ground-truth hits.
  This remains a documented native-API limitation rather than a newly established
  merge regression.
- Roaring can be slower, especially for NaviX. It is optional; dense remains the
  default. Container ownership, query offsets, graph-ID seed order, remapped IDs,
  and odd payload alignment are covered by the implementation and GPU checks.
- Six index copies and five sparse bitmap artifacts were verified losslessly;
  native cuVS loaded all six copied indexes. Originals and paper data are preserved.

The automatic `ready_for_review` field intentionally remains false because its
strict equality checks still flag the documented differences. This review accepts
the integration based on the old-fork compatibility results and investigations;
it does not change the measurements, suppress flags, or claim upstream identity.
The [archived assessment](results/a100_20261017/ASSESSMENT.md) and raw evidence
remain immutable records of the pre-merge checkpoint.

## Operational notes

The validated 26.12 build is in `/data/cuvs-upstream-integration/cpp/build`, using
the isolated environment `/data/cuvs-upstream-envs/cuvs-2612`. The existing
`/home/ubuntu/cuvs-filter/cpp/build` contains the previous fork's binaries; advancing
its source checkout does not rebuild them. Use the validated build, or configure a
new build with the 26.12 environment. The previous source remains available at the
recorded main commit; archived baseline binary paths identify the historical build.

Matched-recall tuning stays paused. Full parameter sweeps, IVF Roaring support,
PQ performance tuning, and IVF-RaBitQ filtering remain deferred. No original paper
measurements are replaced by this representative 2,048-query replay.
