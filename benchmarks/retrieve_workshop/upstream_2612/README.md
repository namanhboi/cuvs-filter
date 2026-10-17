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
