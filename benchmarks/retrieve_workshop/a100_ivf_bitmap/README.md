# A100 bitmap-filtered IVF comparison

Runs IVF-Flat, IVF-SQ, plain IVF-PQ, and IVF-PQ with GPU reranking against the
saved Retain/NaviX measurements used by the paper. Covers YFCC, ArXiv EM,
EMIS, and R at k=10 and k=100: 10,000 queries, five shards, batches of at most
2,048, and three final repetitions. Input construction/loading is outside the
timed search; PQ GPU reranking is inside it.
GPU reranking processes query chunks with at most approximately 1 GiB of
temporary candidate-vector storage; the initial PQ search still receives the
complete query batch.
Plain and reranked PQ use separate processes, so plain PQ's memory budget
contains only its compressed index and search workspace.

```bash
export RETRIEVE_IVF_RUN_ROOT=/data/retrieve_workshop_runs/a100_ivf_bitmap_$(date -u +%Y%m%dT%H%M%SZ)
bash benchmarks/retrieve_workshop/a100_ivf_bitmap/run.sh all
```

The same command resumes validated completed work. Stages `prepare`, `build`,
`calibrate`, `final`, and `analyze` are also available. Inputs, runtime binaries,
configs, and saved results must remain unchanged within a run. Failed partial
reports are preserved; successful reports are promoted only after validation.

The index sweep uses nlist=1024/2048/4096, 25 training iterations, Flat/PQ
training ratio 10, and SQ's 256 training points per cluster. PQ uses 8-bit
subspace codebooks, dimensions 32/64 for YFCC and 64/128 for ArXiv, float
distance accumulation, half lookup tables, and reranking ratios 2–64.
Probe counts start at powers of two and are refined to adjacent integers
around every observed target crossing. Each refinement measures a midpoint
and the predicted target probe count with its neighbors, reducing repeated
index loads. Calibration uses one repetition;
the fastest three matched configurations (or three bracketing pairs) and
each method's maximum observed recall point receive three final repetitions.
The workflow finalizes YFCC before starting ArXiv, so its comparison becomes
available while the remaining workloads are being tuned.

Targets are [0.800,0.802] for YFCC and [0.950,0.952] for ArXiv. Prefer measured
matches. When adjacent probes skip the window, interpolate seconds per query
between measured endpoints with the same index/reranking settings and label
the resulting QPS estimate. Never extrapolate a target that was not reached.
Maximum recall always means the maximum observed within this sweep.

`state/progress.json` reports the active stage. Per-command logs and durations
are under `logs/`; raw JSON/configs, index files, measurement aggregates,
input/runtime hashes, and baseline snapshots remain in the run directory.
`analysis/comparison.md` and `.csv` compare final recall/QPS and speedups;
`analysis/calibration_points.csv` retains the full sweep, with final points
and QPS–recall PDF/PNG plots beside it. Unreached cells show maximum observed
recall and its QPS; speedups require both methods to match the target.

Use `bash benchmarks/retrieve_workshop/a100_ivf_bitmap/run.sh status` with the
same `RETRIEVE_IVF_RUN_ROOT` to read progress while the job owns its execution
lock. Calibration ETA uses completed cases from the same corpus and IVF
method; it remains unknown for families that have no timing samples yet.
Final measurements and index building are excluded from that estimate.

Runtime fixes require recording a new runtime revision and archiving all
affected measurements before restarting. This experiment preserves its first
failed calibration under `revisions/`: full-data validation exposed dummy
result slots with passing IDs, and GPU refinement exposed underfilled lists.
The PQ sweep also exposed requests for more reranking candidates than the
largest probed pool. The materialized PQ distance rows now include enough
dummy-score padding to satisfy dense top-k selection, and its workspace
estimate accounts for that padding. Float/uint8 regression tests cover
6,400-candidate pools, empty and underfilled filters, bitmap row offsets,
and partial refinement chunks. All PQ measurements from before this fix
are archived and rerun; trained indexes and unaffected Flat/SQ measurements
are retained with their original runtime provenance.

The workflow does not edit the paper or rerun its CAGRA baselines.

After `all` finishes, it generates `report/main.tex` and compiles `report/main.pdf`
with the submission's `IEEEtran` journal/letter-size format. The report includes
eight workload/k figures with full-range and target-recall views, all four IVF
variants and saved Base/Retain/NaviX curves, and matched-recall tables for both
k values. Interpolated cells are starred; unmatched cells show maximum observed
recall and its QPS. Calibration points use one repetition; final IVF and saved
CAGRA points use three. Lines are measured Pareto envelopes across settings.
Combined curve/table CSVs, PDF/PNG plots, and hashed provenance accompany it.

The saved paper curve CSVs and formatting source are snapshotted before `all`
starts. `bash benchmarks/retrieve_workshop/a100_ivf_bitmap/run.sh report`
regenerates the document for the selected run root and rejects incomplete IVF
results. The report requires Matplotlib, IEEEtran, and `latexmk`.

For the requested comparison of IVF-Flat with Base/Retain/NaviX using currently
available measurements:

```bash
bash benchmarks/retrieve_workshop/a100_ivf_bitmap/run.sh report \
  --ivf-methods ivf_flat --allow-calibration
```

This writes `report_ivf_flat/main.tex` and `main.pdf`. It uses final measurements
where available, otherwise requires every IVF-Flat index calibration to be
complete for that workload/k and labels the one-repetition result with a dagger.
Its plots and CSVs contain only IVF-Flat and the three CAGRA methods. The full
experiment's completion flag and the preserved PQ/SQ data are unchanged.

## Sparse synthetic IVF-Flat experiments

```bash
export RETRIEVE_IVF_RUN_ROOT=/data/retrieve_workshop_runs/a100_ivf_flat_sparse_$(date -u +%Y%m%dT%H%M%SZ)
bash benchmarks/retrieve_workshop/a100_ivf_bitmap/run.sh sparse all
bash benchmarks/retrieve_workshop/a100_ivf_bitmap/run.sh sparse status
```

This separate run reuses the five original ArXiv random bitmaps at 0.01%,
0.03%, 0.1%, 0.3%, and 1%, their 2,048 queries and k=10 ground truth, and the
three successful ArXiv IVF-Flat builds (nlist=1024/2048/4096). It preserves
both source experiments. Override `--reference-root`, `--index-source-root`,
and `--paper-data` when moving the artifacts to another machine.
Large base/index files must retain their original identities and successful
build provenance; the original base SHA-256 is inherited. Queries, bitmaps,
ground truth, reference snapshots, runtime binaries, and runner code are hashed.

Stages are `prepare`, `calibrate`, `final`, `analyze`, and `status`; use the
same run root to resume. Calibration tests powers of two through all lists,
then refines crossings of 0.950, 0.951, and 0.952 to adjacent integer probes.
Every final repetition must be in [0.950,0.952]. Measured matches take priority;
otherwise adjacent confirmed probes may interpolate latency at 0.951, marked
with an asterisk. Unmatched methods show maximum observed recall and its QPS.
The fastest three calibration matches (or bracketing pairs) and maximum-recall
point receive three final repetitions. All-list scans must reach the target,
and every accepted IVF result must pass the correctness counters.

`analysis/comparison.csv` and `.md` compare IVF-Flat with the paper's frozen
Base/Retain/NaviX and exact-KNN results. The five PDF/PNG plots include full
recall and target-window views; curve CSVs retain all measurements. Exact KNN
is a full-recall throughput reference rather than a matched-recall ANN point.
The runner reports calibration progress and an ETA based on completed cases;
the estimate excludes final measurements and plotting.

To add the completed sparse analysis and a grouped selectivity/QPS bar chart
to the IVF-Flat report, keep `RETRIEVE_IVF_RUN_ROOT` set to the primary IVF run:

```bash
bash benchmarks/retrieve_workshop/a100_ivf_bitmap/run.sh report \
  --ivf-methods ivf_flat --allow-calibration \
  --sparse-run-root /data/retrieve_workshop_runs/a100_ivf_flat_sparse_20261015T025433Z
```

The five selectivity groups each contain Base, Retain, NaviX, IVF-Flat, and
exact-KNN bars. A logarithmic QPS axis keeps all methods visible. Unmatched
methods use hatched bars at their maximum-recall QPS, with achieved recall
annotated; their throughput is excluded from matched-recall conclusions.
The report includes separate sparse comparison, curve, and bar-value CSVs
and hashes both source experiments in its provenance.
