# Integration checkpoint, 2026-10-17

This is a source checkpoint, not a completed validation report. Neither `main`
nor the original paper results have been changed. NVIDIA's repository is read-only.

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
  8: all 46 passed.
- Changed Python sources pass syntax checks and Ruff apart from the existing
  E402/E741 conventions in legacy workflow files.
- Focused CUDA compilation passed for the CAGRA wrapper, CAGRA bitmap/Roaring GPU
  test source, and IVF bitmap GPU test source. Earlier focused builds also passed
  the Roaring search bridge and single-CTA launcher instantiations.
- All five real sparse query bitmaps were converted losslessly to separate
  sorted-ID artifacts; 2,048 membership rows were independently checked per bitmap.
- Six saved-index copies passed independent component verification: two CAGRA
  graphs and four IVF-Flat indexes. Native loading is still a separate pending gate.
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

Still required:

- Complete final control/integration builds after regenerating changed sources.
- Execute GPU tests, including retirement rejection, dense/Roaring equivalence,
  empty/underfilled allowlists, remapped IDs, query offsets, k=100 outputs, and
  native multi-partition search.
- Load independently verified converted index copies with native cuVS.
- Run and assess the agreed representative benchmarks against reruns of the old
  binaries and the pinned upstream control. No new performance conclusion is claimed.

Detached build states and logs live under `/data/cuvs-upstream-validation` on the
validation host. Large datasets, copied indexes, allowlist artifacts, and build
products are external artifacts and are not Git source files.
