# Integration validation, 2026-10-17

The agreed representative validation and [merge review](MERGE_REVIEW.md) are
complete. Original paper results are unchanged; NVIDIA's repository is read-only.
See [the assessment](results/a100_20261017/ASSESSMENT.md) for measurements,
investigations, and remaining limitations.

Completed checks:

- FAVOR API/search requests reject execution; legacy experiment runners and the
  retired comparison example report deprecation. FAVOR-only kernels, GPU test
  targets, and the example target are excluded from active builds.
- Public estimator, penalty, and sidecar declarations are deprecated. Estimator
  and sidecar compatibility implementations only throw; no statistics or files
  are produced. Private search planning rejects retired modes before setup.
- Shared Retain/NaviX data preparation and diagnostics no longer require active
  FAVOR experiment modules. New generated controls have generic names; frozen
  result readers accept both spellings.
- CPU workflow tests: paper pipeline 7, GPU graph pipeline 8, selectivity follow-up
  7, k=100 Retain occupancy pipeline 5, bitmap/index conversion 11, replay contract
  13: all 51 passed.
- Changed Python sources pass syntax checks and Ruff apart from the existing
  E402/E741 conventions in legacy workflow files.
- Focused CUDA compilation passed for the CAGRA wrapper, CAGRA bitmap/Roaring GPU
  test source, and IVF bitmap GPU test source. Earlier focused builds also passed
  the Roaring search bridge and single-CTA launcher instantiations.
- All five real sparse query bitmaps were converted losslessly to separate
  sorted-ID artifacts; 2,048 membership rows were independently checked per bitmap.
- Six saved-index copies passed independent component verification and native
  cuVS loading: two CAGRA graphs and four IVF-Flat indexes.
- All frozen benchmark inputs have recorded SHA-256 values. The 57 selected cases
  are grouped into 75 benchmark processes, plus 13 public-API control comparisons.
- Flat, SQ, PQ, and exact benchmark adapters and their frontends compiled. The exact
  frontend check caught and resolved an old raw-CUDA-stream conversion.
- Both public-API probes compiled against their own checkout's headers. The control
  library remains unmodified. Its benchmark frontend is not used: it lacks the custom
  query-bitmap adapter, and the pinned target also emits a malformed main macro.
- A standalone GPU seed check passed 20 cases against dense and CPU references,
  covering identity/remapped IDs, array/bitmap/mixed containers, odd container
  payload alignment, clipped graph bounds, multiple seed widths, and empty rows.
  Identity-ID selection now enumerates sorted Roaring containers inside search timing;
  remapped IDs retain the exact membership-scan fallback.

Both isolated builds now pass. GPU suites completed with 518 passed tests,
17 intentional skips, and no failures:

| Checkout | Suite | Passed | Skipped |
|---|---|---:|---:|
| Upstream control | Roaring core | 6 | 0 |
| Upstream control | UDF | 24 | 0 |
| Upstream control | BBQ | 135 | 0 |
| Upstream control | Multi-partition | 51 | 0 |
| Integration | Roaring core | 6 | 0 |
| Integration | Bitmap/Roaring CAGRA | 60 | 2 |
| Integration | UDF CAGRA | 37 | 14 |
| Integration | Bitmap IVF Flat/SQ/PQ | 13 | 1 |
| Integration | BBQ | 135 | 0 |
| Integration | Multi-partition | 51 | 0 |

Skips cover single-CTA-only custom controls in other search modes and IVF-SQ's
absent dynamic batching wrapper. Dense/Roaring Base, Retain, and NaviX comparisons,
retirement rejection, empty/underfilled rows, remapped IDs, query offsets,
mixed-container passing seeds, k=100 outputs, and native multi-partition search
passed. The separate 20-case GPU seed check above is not included in this table.

Representative benchmark findings:

- All 75 benchmark processes and 26 native public-API executions completed,
  each with three timed repetitions. All 57 previous-fork versus integration
  comparisons are within the agreed thresholds: QPS ratios 0.983–1.013 and maximum
  absolute recall change 0.000009775. Benchmark output invariant checks passed.
- All 13 native integration recall values exactly equal the previous fork's
  corresponding benchmark recall. They exceed pinned upstream by 0.000391–0.082437.
  The inherited final-result compaction fixes explain the direction of this
  difference; their helper is unchanged from the previous fork. This is a source
  investigation, not an end-to-end ablation. Native QPS ratios are 0.944–1.127.
- Raw native results contain a small number of duplicates in both checkouts;
  recall counts each ground-truth ID once. The benchmark adapter deduplicates.
  Raw arrays and native difference flags remain available for review.
- Dense/Roaring recall matches except YFCC NaviX k=100, whose median difference
  is four ground-truth hits (0.000019551). Repetition ranges overlap across the
  previous fork, integrated dense, and Roaring. The strict equality flag remains.
- Roaring is an optional representation and is slower on most measured cases;
  NaviX Roaring/dense QPS ratios range from 0.413 to 0.964. Dense remains the default.

The automated `ready_for_review` field remains false because it requires exact
native equality and no representation flags. The assessment documents these
findings without changing the raw flags. This is a review checkpoint, not automatic
approval to update `main`; the separate merge review records the disposition.

Settings were frozen and replayed on the first 2,048 queries. These are compatibility
measurements, not a newly tuned matched-recall method ranking. Full sweeps, tuning,
and IVF Roaring support remain deferred.

Detached build states and logs live under `/data/cuvs-upstream-validation` on the
validation host. Large datasets, copied indexes, allowlist artifacts, and build
products are external artifacts and are not Git source files.
