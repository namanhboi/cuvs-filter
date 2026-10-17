# Native upstream differences

The native probes call the same public UDF on the same first 2,048 queries,
converted graphs, source vectors, bitmap bits, and frozen search settings. Both
are compiled against their own checkout's public headers and library.

Pinned upstream's `search_single_cta_jit.cuh:352` tests validity as:

```cpp
(index & ~index_msb_1_mask) == invalid_index ? 0 : 1
```

For an invalid uint32 ID, the left operand is `0x7fffffff` and the right operand
is `0xffffffff`. The condition is false, so that missing slot is counted as valid.
This means the final compaction can leave missing entries in the returned prefix.
The same finalization also reads physical positions in a swizzled bitonic buffer.

The integration preserves our old main's logical-order compactor, which masks
both sides of the invalid-ID test, keeps sorted logical order, and synchronizes
before other warps read the result. The helper is byte-identical to old main's
implementation; it predates this merge (commit
`1bba08d8eac06ce760009df76f6f0957be4d5e97`). A separate high-dimensional GPU
regression test for the compaction synchronization passed.

The higher native recall is consistent with these inherited compaction fixes.
This is a source-based explanation, not a full end-to-end ablation attributing
every changed neighbor to one statement. The frozen old-main comparison is the
compatibility reference; it shows unchanged quality. Native-control quality and
ID-equality flags remain in comparison.json and are explicitly reported for review.
No parameters were retuned and no upstream source was edited.
