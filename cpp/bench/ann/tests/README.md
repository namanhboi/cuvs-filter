# IVF per-query bitmap checks

IVF-Flat, IVF-PQ, and IVF-SQ accept the existing dataset configuration:

```json
"filter": {"kind": "bitmap", "file": "filter.bmap"}
```

The benchmark loads the bitmap once and uses the original query row for each batch and latency call.
A bitmap bit of one permits the source dataset ID in that column. Query and dataset order must match
the bitmap. Bitmap filtering requires ordinary search; dynamic batching rejects this filter.

To compile and run the library regression tests:

```bash
cmake -S cpp -B cpp/build -DBUILD_TESTS=ON
cmake --build cpp/build --target NEIGHBORS_ANN_IVF_BITMAP_TEST \
  CUVS_IVF_FLAT_ANN_BENCH CUVS_IVF_PQ_ANN_BENCH CUVS_IVF_SQ_ANN_BENCH -j 4
cpp/build/gtests/NEIGHBORS_ANN_IVF_BITMAP_TEST
python3 cpp/bench/ann/tests/ivf_bitmap_smoke.py
```

The smoke check generates a small dataset, a packed bitmap, and CPU exact ground truth in a temporary
directory. It checks full and partial query batches, empty and underfilled result rows, IVF-PQ
refinement, two benchmark threads, and serialized latency. Use `--output-dir /tmp/ivf-bitmap-smoke`
to retain its configurations, inputs, and reports. Adding `--generate-only` prepares inputs without CUDA.

For direct C++ calls, pass `bitmap_filter<uint32_t, int64_t>` to the existing `search` overload. Set
`search_params.filter_query_offset` when searching a batch whose first query belongs to a later
bitmap row; the default is zero. Keep the device bitmap alive until the search stream completes.
Missing results have the maximum output-ID value and `std::numeric_limits<float>::max()` distance.
