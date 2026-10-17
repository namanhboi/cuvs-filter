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
  7, k=100 Retain occupancy pipeline 5, bitmap/index conversion 11: all passed.
- Changed Python sources pass syntax checks and Ruff apart from the existing
  E402/E741 conventions in legacy workflow files.
- Focused CUDA compilation passed for the CAGRA wrapper, CAGRA bitmap/Roaring GPU
  test source, and IVF bitmap GPU test source. Earlier focused builds also passed
  the Roaring search bridge and single-CTA launcher instantiations.
- All five real sparse query bitmaps were converted losslessly to separate
  sorted-ID artifacts; 2,048 membership rows were independently checked per bitmap.

Still required:

- Complete final control/integration builds after regenerating changed sources.
- Execute GPU tests, including retirement rejection, dense/Roaring equivalence,
  empty/underfilled allowlists, remapped IDs, query offsets, and k=100 outputs.
- Load independently verified converted index copies with native cuVS.
- Run and assess the agreed representative benchmarks against reruns of the old
  binaries and the pinned upstream control. No new performance conclusion is claimed.

Detached build states and logs live under `/data/cuvs-upstream-validation` on the
validation host. Large datasets, copied indexes, allowlist artifacts, and build
products are external artifacts and are not Git source files.
