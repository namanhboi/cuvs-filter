# Active NaviX policy ablation

Approved scope: first 2,048 queries of YFCC and ArXiv EM/EMIS/R, k=10/100,
dense bitmaps, SINGLE_CTA batched throughput. Compare adaptive_kuzu against
adaptive_blind at identical frozen settings and independently matched recall.

Result root: `/data/retrieve_workshop_runs/navix_policy_20261028T211816Z`.
Detached coordinator: `/data/cuvs-upstream-validation/navix-policy-supervisor.py`.
Coordinator state/log: `supervisor.state.json` and `supervisor.log` in the root.
Build log: `/data/cuvs-upstream-validation/navix-policy-build.log`.

The coordinator waits for the already running isolated build, checks freshly
linked artifacts, runs bitmap/UDF GPU correctness suites under the GPU lock,
prepares the immutable runtime and inputs, then runs all comparisons and
separate diagnostics. `progress.json` identifies the current benchmark stage;
`failure.json` records runner failures. Resume with the README's run command
only after inspecting any failure. Do not start duplicate workers.

Seven CPU tests and Ruff checks pass. Device-header static assertions check all
passing counts for both graph degrees and scheduler encodings. GPU validation
is pending the build; do not describe results as validated until its reports
are present. The default policy remains unchanged. The old 10,000-query batch-1
experiment and its publication worker remain paused.

When complete, the coordinator exports compact analysis, provenance, and GPU
correctness reports into this directory's results subtree. It commits/pushes
only if main remains at the expected clean HEAD; otherwise it leaves an explicit
export-only state. All raw calibration/diagnostic evidence remains in the root.
Only the user's repository is a push destination.

## GPU availability recovery

The 439-step build completed. The first correctness invocation aborted at RMM
initialization because MIG was enabled with no configured GPU instances; CUDA
reported zero devices. With no GPU workloads or instances present, MIG was
disabled and the temporarily stopped DCGM/persistence services were restarted.
CUDA now detects one full A100 and persistence is enabled. The detached
coordinator was restarted; archived failed logs remain in the result root.
This failure occurred before any correctness test or benchmark measurement.
