# Representative integration assessment — 2026-10-17

The integration preserves the previous fork's measured behavior. All 57
old-main/integrated dense comparisons pass the agreed recall/QPS thresholds:
QPS ratios are 0.983–1.013; the maximum absolute recall change is 0.000009775.
All measured benchmark filter, uniqueness, missing-slot, and exact-distance
checks pass. Both isolated builds pass. GPU suites report 518 passes, 17
intentional skips, and zero failures; 51 CPU workflow tests passed.

This is a review checkpoint for the integration branch. Main and original
paper results remain unchanged. No parameters were retuned. The automatic
`ready_for_review` field remains false: the strict comparison retains one
representation-equality flag and 13 native-control difference flags. The
investigations below explain these findings without deleting or waiving them.

## Coverage and provenance

- Pinned upstream: `3721d31fb98ba927417f42fbe3f93ab9ff15ed8e`.
- Previous fork: `0b3a56692631a186b875a1542930d7e773a75e69`.
- A100 80 GB PCIe; first 2,048 original queries; three timed repetitions.
- 75 cuVS-bench processes: 57 source cases with old/integrated results, plus
  Roaring for all 39 CAGRA cases (459 iteration records).
- 26 native public-API executions: 13 upstream/integration pairs, 78 timed
  repetitions. Both probes use their own headers/ABI and the same public UDF
  reading identical dense bitmap bits. Native timings are compared separately.
- Six converted index copies passed independent component verification and
  native loading. All five sparse bitmap artifacts passed every membership row.
- 41 frozen input SHA-256 records, configurations, build options, environments,
  linked libraries, binary hashes, and raw repetitions are archived here.

These replay frozen paper selections (or maximum-observed-recall settings)
on a smaller query subset. Some target recalls are missed. They are not a
new matched-recall ranking or a replacement for the original paper results.

## Why native recall exceeds upstream

All 13 integrated native recall values exactly equal the corresponding
previous-fork benchmark recall; every upstream comparison has lower recall.
Pinned upstream's final compaction masks the visited bit on one side of its
invalid-ID comparison only. Thus an empty uint32 slot is counted as valid.
Its finalization also operates on physical positions in a swizzled buffer.
The merge preserves our earlier logical-order compactor, which masks both
sides and synchronizes before other warps consume the result. This helper is
byte-identical to old main and predates the merge (commit `1bba08d8`).

This explains the direction of the quality change from upstream; it is a
source-based investigation, not an end-to-end ablation of every changed ID.
See [the source investigation](investigations/upstream-native-compaction.md)
and [the bit-mask proof](investigations/upstream-compaction-proof.json).

| Native workload | k | Upstream recall | Integrated recall | Integrated/upstream QPS | Duplicate slots upstream/integrated |
|---|---:|---:|---:|---:|---:|
| YFCC | 10 | 0.630957 | 0.655371 | 0.947 | 0/3 |
| ArXiv EM | 10 | 0.940723 | 0.956738 | 1.013 | 3/2 |
| ArXiv EMIS | 10 | 0.690918 | 0.700781 | 1.011 | 0/0 |
| ArXiv R | 10 | 0.930322 | 0.942480 | 1.005 | 11/3 |
| YFCC | 100 | 0.311565 | 0.337299 | 0.944 | 9/8 |
| ArXiv EM | 100 | 0.873799 | 0.956235 | 1.006 | 23/16 |
| ArXiv EMIS | 100 | 0.448228 | 0.465356 | 1.013 | 13/7 |
| ArXiv R | 100 | 0.867324 | 0.931187 | 1.127 | 0/1 |
| Sparse 0.01% | 10 | 0.083203 | 0.085303 | 0.994 | 0/0 |
| Sparse 0.03% | 10 | 0.187549 | 0.187939 | 0.983 | 1/1 |
| Sparse 0.1% | 10 | 0.279980 | 0.280859 | 0.984 | 1/1 |
| Sparse 0.3% | 10 | 0.381006 | 0.424902 | 0.995 | 0/0 |
| Sparse 1% | 10 | 0.767578 | 0.779004 | 0.992 | 0/0 |

Raw native API outputs occasionally contain duplicate slots in both
checkouts; recall counts each ground-truth ID once, and cuVS-bench additionally
deduplicates output. These counts remain visible above. The native probe does
not rewrite raw arrays. Only known missing-slot IDs paired with FLT_MAX or
positive infinity are normalized for ID comparison. Even after that
normalization, all 13 native ID sequences differ: this is not merely a sentinel
encoding difference. Filter violations are zero. No native QPS regression
crosses the 10% threshold (ratios 0.944–1.127).

## Dense versus Roaring

Roaring uses the same membership rows and passing seeds, with construction
outside search timing and seed selection inside it. Recall matches dense
except YFCC NaviX k=100: a four-hit median difference (0.000019551). Repeated
recall ranges overlap in old dense, integrated dense, and Roaring, and the
NaviX traversal source is unchanged except a diagnostic namespace rename.
Concurrent visited-set insertion is a plausible cause, not independently
isolated. See [the repetition evidence](investigations/yfcc-k100-navix-variation.json).

Roaring is usually slower here; dense remains the default. Across all 39
CAGRA cases, Roaring/dense QPS ratios are 0.894–1.007 for Base, 0.800–1.004
for Retain, and 0.413–0.964 for NaviX. Container access and passing-seed work
are plausible overheads; no profiler attribution is claimed. Representation
performance is reported separately from old-main compatibility.

| Sparse selectivity | Base Roaring/dense QPS | Retain Roaring/dense QPS | NaviX Roaring/dense QPS |
|---|---:|---:|---:|
| 0.01% | 0.992 | 1.000 | 0.608 |
| 0.03% | 0.993 | 0.988 | 0.651 |
| 0.1% | 0.987 | 0.989 | 0.597 |
| 0.3% | 0.990 | 0.985 | 0.546 |
| 1% | 0.989 | 0.979 | 0.413 |

## Deferred work

Matched-recall tuning stays stopped at the user's request. Full sweeps,
IVF Roaring, PQ performance tuning, and RabitQ bitmap work are outside this
checkpoint. IVF Flat/SQ/PQ dense bitmap correctness is covered; representative
IVF performance replay here uses Flat. Native duplicate handling and full
ablation of the upstream compaction differences remain explicit limitations.

Large datasets, index copies, logs, binaries, and native arrays remain under
`/data/cuvs-upstream-validation`; this snapshot includes their provenance and
native output checksums rather than duplicating them in Git.
