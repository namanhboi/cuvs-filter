# Batch-1 primary CAGRA/IVF-Flat progress — 2026-10-28

This is the active task following the completed upstream integration. Implement
and run the plan in `PRIMARY_IVF_README.md`; do not restart matched-recall tuning.

## Frozen scope

- Four methods: Base, Retain, NaviX, IVF-Flat; dense resident bitmaps only.
- YFCC and ArXiv EM/EMIS/R, k=10 and k=100, all 10,000 queries each.
- CAGRA SINGLE_CTA, batch 1, max_queries 1. Preserve every other source setting.
- Merged 26.12 runtime and verified converted index copies.
- Three full passes: 960,000 measured calls, both host completion and CUDA-event
  timing. Twenty-five source selections match their target; seven are CAGRA
  source-maximum fallbacks and remain labeled unmatched.

## Implemented and checked

- Optional benchmark trace schema 2 adds per-query GPU event timing while retaining
  the schema-1 format and ordinary throughput control flow.
- New `primary_ivf.py` and `run_primary_ivf.sh` freeze source configurations,
  snapshot runtime/libraries, run paired gates and full workloads, validate/resume
  records, and generate CSV/Markdown/CDF reports. See the README for details.
- Both frontends rebuilt successfully in the isolated integration build.
- CPU pipeline tests: 18 passed (new and existing); Ruff and shell syntax checks pass.
- Preparation completed: 32 selections, 352 benchmark processes, 185 frozen input
  identities. Original-source hashes were cross-checked against the prepared
  records, including those inherited from the source experiments.
- Original integration frontends were preserved before rebuilding under
  `/data/cuvs-upstream-validation/pre_batch1_runtime`; libcuvs is unchanged.

## Active run

Result root:
`/data/retrieve_workshop_runs/a100_batch1_primary_ivf_20261028T073000Z`

The detached supervisor is described by `supervisor.json` and
`supervisor.state.json` there; `run.log` has per-job completion messages,
`progress.json` identifies the active job, and `failure.json` records errors.
The supervisor PID initially was 126829 and the worker PID initially was 128600;
check current processes rather than assuming those IDs remain valid.

The initial four YFCC batched-gate processes completed. The first ArXiv load
reads approximately 44 GB of source vectors, so cold loading is substantial.
Continue monitoring the saved state for current progress rather than treating
this paragraph as a live count. All input setup and the detached worker run on
the server and do not depend on the user's laptop staying connected.

The worker owns the shared GPU lock. Do not run another GPU experiment alongside
it. A gated estimate is saved in `analysis/eta.json` after all paired checks pass.
Fresh and resumed raw outputs are validated before acceptance. The exact source
SHA and runner hash are in `manifest.json`; changing the runner after preparation
requires an explicit recorded revision or a fresh run, not a silent restart.

To inspect progress:

```bash
python_bin=/home/ubuntu/micromamba/envs/cuvs/bin/python
runner=/home/ubuntu/cuvs-filter/benchmarks/retrieve_workshop/per_query_latency/primary_ivf.py
result_root=/data/retrieve_workshop_runs/a100_batch1_primary_ivf_20261028T073000Z
"$python_bin" "$runner" status --root "$result_root"
```

If the worker has stopped, inspect the failure and per-job log before resuming
with `run --root`. Never run two workers for the same directory. If outputs are
partial and unmarked, the runner archives them under `attempts/` before retrying;
changed completed records fail closed. Base's historical duplicate-slot counts
remain diagnostics under distinct-ID recall; other methods require zero duplicates.

## Work still required

1. Finish the paired 1,000-query gates for all 32 cases; investigate recall drift
   above 0.0001 without changing source settings.
2. Finish merged-runtime batched references over all 10,000 queries and all
   960,000 main serialized calls, including the 1,808-query final shards.
3. Review achieved recall/status, filter/sentinel/duplicate diagnostics, timing
   floors, full coverage, and source/serialized recall differences. Do not report
   an out-of-window or source-maximum cell as target matched.
4. Check all CSV/JSON summaries and PDF/PNG CDFs. Commit compact results and update
   this handoff with the completed outcome; keep large traces and runtimes external.
5. Push code/docs/results only to the user's `namanhboi/cuvs-filter` repository.
   Do not modify the original paper or throughput evidence.

Runtime executables and libcuvs for this experiment are immutable snapshots in
the run's `runtime/`, resolved with a run-specific library search path. The older
main-checkout build still holds pre-merge binaries. GPU jobs create a `cufile.log`
in their working directory; preserve that generated diagnostic with the run rather
than committing it as source.
