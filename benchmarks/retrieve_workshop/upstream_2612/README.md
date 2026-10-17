# Upstream 26.12 integration

This branch integrates NVIDIA cuVS commit
`3721d31fb98ba927417f42fbe3f93ab9ff15ed8e` with the filtering implementation at
`0b3a56692631a186b875a1542930d7e773a75e69`. The upstream commit is pinned;
following upstream `main` during validation would change the experiment.

## Scope

- Preserve Base, Retain, NaviX, shared diagnostics, and existing per-query dense
  bitmap filtering for CAGRA and IVF-Flat/SQ/PQ.
- Add lossless dense-bitmap-to-Roaring conversion and per-query Roaring support
  for Base, Retain, and NaviX, including passing-seed selection.
- Preserve native upstream Roaring and BBQ behavior. Reject unsupported custom
  method/dataset combinations explicitly.
- Convert copies of old CAGRA v5 and IVF-Flat v4 indexes to the new formats while
  preserving graphs, vectors, centroids, list assignments, and IDs. Updating a
  version number alone is not a conversion.
- Retire FAVOR execution, kernels, parameters, sidecar computation, and experiment
  entrypoints. Preserve historical results and shared Retain/NaviX functionality.
- Defer full parameter sweeps and Roaring filtering for IVF.

## Isolation and review

The original checkout, environment, binaries, indexes, and paper results remain
available for comparison. Integration and unmodified upstream use separate
worktrees and build directories. Generated indexes, build products, and benchmark
results are external artifacts, not Git source files.

Push only to `git@github.com:namanhboi/cuvs-filter.git`. NVIDIA's repository is a
read-only source. Publish the integration branch and validation evidence for
review before advancing the user's `main` branch.

## Validation gate

1. Run relevant GPU correctness tests for bitmap/UDF/Roaring filtering,
   Retain/NaviX, rejection of retired FAVOR requests, k=100 result compaction, and IVF
   refinement with empty or underfilled candidate pools.
2. Check converted index component identity and dense/Roaring membership
   equivalence, including empty and full allowlists and shared query mappings.
3. Benchmark the first 2,048 queries of YFCC and ArXiv EM/EMIS/R at k=10 and
   k=100, plus all five sparse selectivity rates at k=10 and exact controls.
   Use three repetitions and identical configurations/seeds. Rerun the old
   build on those subsets; frozen 10,000-query throughput is not a comparable
   baseline.
4. Compare integrated default CAGRA to an unmodified pinned-upstream control.
   Retain legacy baseline contracts when comparing old results. Investigate
   recall shifts above 0.002 or QPS regressions above 10%; do not hide changes
   through retuning.
5. Run relevant Python workflow/report tests and formatting checks. Capture
   source revisions, commands, environment details, and artifact provenance.

## Commit checkpoints

Record the integration contract first, then resolve the upstream merge. Commit
subsequent converter, Roaring adapter, correctness, and validation workflow
changes as focused, reviewable units. A committed checkpoint is not a claim that
the complete validation gate has passed.

## Bitmap conversion

Convert an immutable query bitmap to a separate allowlist artifact:

```bash
python benchmarks/retrieve_workshop/upstream_2612/convert_bitmap.py \
  /path/to/filter.bitmap --output /path/to/filter.allowids
python -m unittest discover -s benchmarks/retrieve_workshop/upstream_2612
```

Conversion reconstructs and compares every membership row, records source/output
SHA-256 values in `filter.allowids.json`, and refuses to replace an existing file.
Only one bitmap row and its passing IDs are materialized at a time.

`CUVSRIDS/v1` stores sorted **input IDs**, not portable Roaring bytes or a GPU
memory-size measurement. Its little-endian header is `<8sIIQQQ>`: magic
`CUVSRIDS`, version 1, ID width 32, query rows, dataset rows, and total ID count.
The header is followed by `query_rows + 1` uint64 offsets and the uint32 ID payload.
Each row is strictly increasing; empty rows have equal adjacent offsets. cuVS
constructs resident Roaring owners with `from_ids(..., pre_sorted=true)` outside
search timing. The original dense bitmap remains the membership reference.

## CAGRA filter representations

CAGRA benchmark search parameters accept `filter_representation: "bitmap"`
(the default) or `filter_representation: "roaring"`. The dataset still names the
original query bitmap. Roaring mode constructs one resident cuVS allowlist per
query and retains its owners for the entire search; it does not upload a dense
GPU bitmap. Base, Retain, and NaviX share the same query offsets and source-ID
mapping. Passing-seed selection remains part of search timing.

An optional `roaring_allowlist_file` names the `CUVSRIDS` artifact above. The
adapter compares every saved ID row with the original bitmap before constructing
the native allowlists. Owner construction and this verification happen outside
search timing. Identity-ID Roaring seeds enumerate the sorted portable containers; remapped IDs
use membership scanning to preserve internal graph-ID order. Seed inspection
counters count representation-specific tiles for Roaring and bitmap words for
the dense identity path; they are not equivalent work measures.

New configurations use `passing_result_accumulator` and `search_diagnostics_*`.
Frozen `favor_udf_passing_accumulator` and `favor_diagnostics_*` names remain
compatibility aliases for these shared controls. They never enable FAVOR.
FAVOR modes, penalty/sampling/sidecar controls, and termination shadows are retired.

Custom bitmap/Retain/NaviX benchmark paths require uncompressed CAGRA datasets.
IVF currently accepts dense per-query bitmaps only; requesting Roaring is an error.
Native upstream compressed-index paths are validated separately.

## Saved index conversion

Convert immutable saved indexes to separate paths:

```bash
python benchmarks/retrieve_workshop/upstream_2612/convert_indexes.py cagra \
  /old/cagra.index --output /new/cagra.index
python benchmarks/retrieve_workshop/upstream_2612/convert_indexes.py ivf_flat \
  /old/flat.index --output /new/flat.index
```

The converter handles graph-only CAGRA v5 to v6 and IVF-Flat v4 to v5 with int64
IDs. It refuses embedded CAGRA datasets and unsupported versions. CAGRA graph
and source mappings are preserved. IVF centroid, norm, logical list-size, and
interleaved vector bytes are preserved; only padded, nonlogical list IDs are
removed to match the new serialization convention. No clustering or graph rebuild
is performed.

Each output is reread independently and its components compared with the source.
A `.json` manifest records component and full-file SHA-256 values. Existing outputs
are never overwritten. Wire-format verification is followed by native cuVS loading
and GPU correctness tests before a converted index is used for benchmark comparisons.

## Frozen representative replay

On the validation host, after configuring both isolated builds:

```bash
python benchmarks/retrieve_workshop/upstream_2612/validate.py prepare
python benchmarks/retrieve_workshop/upstream_2612/validate.py compile-native
python benchmarks/retrieve_workshop/upstream_2612/validate.py hash-inputs
python benchmarks/retrieve_workshop/upstream_2612/validate.py run --wait
python benchmarks/retrieve_workshop/upstream_2612/validate.py analyze
```

Paths can be supplied as command-line options. Outputs default to
`/data/cuvs-upstream-validation/benchmarks`. `prepare` refuses to replace an
existing manifest. It selects 32 primary cases and 25 sparse cases from frozen
measurements, then groups them into 75 cuVS-bench processes. Every measured case
uses the original first 2,048-query shard and three repetitions. The sparse CAGRA
cases use recorded matched settings or the maximum observed recall; omitted
calibration JSON is reconstructed only from archived scalar settings and a frozen
method/width template, with both sources recorded. This performs no tuning.

The runner waits for the recorded build, GPU-test, index-conversion, and native
probe compilation prerequisites. It runs the native upstream Roaring/UDF/BBQ
smoke suites, reruns old binaries, measures integrated dense and CAGRA Roaring
paths, and compares public CAGRA search with an independently compiled upstream
control on 13 datasets/result widths. Both native probes use their own checkout's
headers and library ABI. Native probe timing is compared only with native probe
timing, separately from the cuVS-bench comparisons.

A shared GPU lock prevents overlapping workloads. Input checksums, executable
and library hashes, environments, exact configurations, raw repetitions, and
comparison tables are recorded. Completed benchmark files are reused only when
their configuration, binary, library, and manifest signatures still match.
Threshold flags remain explicit and require investigation; this runner never
pushes, modifies saved paper results, or updates `main`.
