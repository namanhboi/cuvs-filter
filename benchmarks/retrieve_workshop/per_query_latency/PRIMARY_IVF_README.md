# Primary-workload batch-1 latency

This is a separate frozen-config comparison of Base, Retain, NaviX, and IVF-Flat,
using dense resident bitmaps on the merged cuVS 26.12 implementation. CAGRA uses
SINGLE_CTA explicitly. It covers YFCC and ArXiv EM/EMIS/R at k=10 and k=100,
all 10,000 queries in each workload, three complete passes: 960,000 measured calls.
The older graph/exact `run.sh` workflow and its schema-1 traces are unchanged.

## Sources and measurement contract

The runner binds the 32 source configurations in the integration validation manifest
to the paper CAGRA selection CSVs and the IVF-Flat report's eight selections. The
validated converted index copies preserve the original index contents. Original
queries, GT, masks, frontier/search-width/iteration policies, random seeds, NaviX
seed caps, and IVF probe counts are unchanged. Only batch size and CAGRA's query
capacity change to one. No tuning occurs. Seven CAGRA selections are source-sweep
maximum-recall fallbacks, not target matches.

The host timer covers the search call through CUDA-event completion. The device
timer measures the CUDA-stream event interval around that same call; it is not
summed kernel time. Events are allocated once per configuration. Loading, uploads,
JIT, 32 warm-up calls, and 128 empty-event floor samples are excluded from the
query distribution. Normal wrapper work and passing-seed selection are included.
Both timing fields are recorded in opt-in trace schema 2. No floor is subtracted.

Each query is visited once per pass, with cyclic starting positions varied by pass.
Statistics summarize each query's median over the three passes: mean, population
standard deviation, nearest-rank p50/p95/p99, and maximum, separately for host and
device timings. These are measured batch-1 latencies, not inverses of batched QPS.

## Build and run

Rebuild `CUVS_CAGRA_ANN_BENCH` and `CUVS_IVF_FLAT_ANN_BENCH` in the isolated
26.12 environment after installing the timing change. The defaults use
`/data/cuvs-upstream-integration/cpp/build`, not the old binaries still present in
the main checkout's build directory.

```bash
cd /home/ubuntu/cuvs-filter
python_bin=/home/ubuntu/micromamba/envs/cuvs/bin/python
runner=benchmarks/retrieve_workshop/per_query_latency/primary_ivf.py
result_root=/data/retrieve_workshop_runs/a100_batch1_primary_ivf_RUN_TAG

"$python_bin" "$runner" prepare --root "$result_root"
nohup "$python_bin" "$runner" run --root "$result_root" \
  > "$result_root/supervisor.log" 2>&1 < /dev/null &
"$python_bin" "$runner" status --root "$result_root"
```

`--help` lists overrides for the frozen sources, dataset/index paths, build, and
environment. `run` resumes, `analyze` regenerates summaries, and `--stage` permits
individual stages. All jobs share `/data/cuvs-upstream-validation/gpu.lock` and a
per-run execution lock. The worker runs on one allowed CPU and one GPU; no other
GPU experiment should overlap it. Snapshot binaries and libcuvs are loaded from
the result directory and their resolution is checked before preparation succeeds.

Inputs and settings are immutable within a run. Reuse requires unchanged input
identities, runtime hashes, manifest/configuration signatures, and saved output
hashes. Unmarked interrupted raw outputs are archived under `attempts/` before a
retry. Modified completed outputs fail closed. Runner changes require a recorded
revision or a fresh run rather than silently reusing old signatures.

## Correctness and outputs

The paired gate compares the first 1,000 queries in batched and serialized modes,
including nonzero bitmap offsets. It records output-set agreement and requires
aggregate valid-GT recall agreement within 0.0001. Complete batched baselines cover
all five shards, including the final 1,808 queries. Full serialized recall is also
checked against this merged-runtime baseline. Large drift is flagged, without
retuning or relabeling settings.

Every trace must contain every expected query exactly once, all three passes,
both timing fields, and the expected identity. Filter violations, invalid slots,
and malformed traces fail validation. Retain/NaviX/IVF duplicate outputs fail;
Base's pre-existing duplicate-slot diagnostic is retained and recall counts unique
GT hits, matching its source experiment. This is reported, not hidden or promoted
to distinct neighbors.

`configurations.csv` and `manifest.json` freeze all selected points, sources,
indexes, settings, and provenance. `raw/`, `traces/`, `neighbors/`, and `logs/`
retain each measurement and gate. `progress.json` identifies the active job.
`analysis/` contains gate diagnostics, `latency_summary.csv`/`.json`, raw-repeat
and median values in `per_query_samples.csv`, `REPORT.md`, and host/device CDFs
in PDF/PNG at both k values. Completed selections appear in partial reports while
later workloads run.

Target windows are [0.800,0.802] for YFCC and [0.950,0.952] for ArXiv. Source
maximum-recall rows stay unmatched. A source-matched point whose serialized recall
falls outside the window is labeled `serialized_out_of_window`. Neither category
is included in matched-recall conclusions. The achieved recall and its range,
source recall, merged batched recall, drift, and duplicate rate accompany latency.

Large raw traces and runtime snapshots stay outside Git. Preserve the original
paper, archived integration validation, and historical binaries. Publish compact
results and code only to the user's repository.

`collect_primary_ivf.py --root RUN --output SNAPSHOT` verifies completion and every
saved output signature before exporting compact tables, figures, configurations,
scalar benchmark JSON, and external-output checksums. It refuses incomplete runs
or recall-drift failures and does not duplicate raw traces or runtime binaries.

The optional `finalize_primary_ivf.py` waits for the detached supervisor, runs the
collector, and can publish the snapshot after successful validation. Publication
requires the exact expected repository HEAD, a clean tracked working tree, and
the user's origin URL. Concurrent repository changes leave a complete bundle in
the run directory with an explicit pending-commit status; no changes are forced.
Its separate `finalize.state.json` records progress and any publication failure.
