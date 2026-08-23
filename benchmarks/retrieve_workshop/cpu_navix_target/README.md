# Target-matched CPU NaviX benchmark

This workflow calibrates native FAISS-NaviX to the paper's closed recall windows, measures
thread scaling on the 16-core/32-thread CPU host, and produces a hardware-qualified
CAGRA-NaviX/FAISS-NaviX QPS ratio. It reuses the immutable graphs and dataset manifests from the
CPU-context experiment; it neither constructs nor modifies an index.

The comparison is method-matched but not hardware- or topology-matched. CAGRA and HNSW are
different graph families, and the GPU and CPU adapters have different execution stacks. Therefore
the output must be described as a measured QPS ratio on the named systems, not as a universal GPU
speedup.

Run the static preflight and tests:

```bash
cd /home/ubuntu/cuvs-filter-retrieve
/home/ubuntu/micromamba/envs/cuvs/bin/python \
  benchmarks/retrieve_workshop/cpu_navix_target/test_pipeline.py
/home/ubuntu/micromamba/envs/cuvs/bin/python \
  benchmarks/retrieve_workshop/cpu_navix_target/run.py --preflight
```

Run the complete experiment into a new immutable result directory:

```bash
/home/ubuntu/micromamba/envs/cuvs/bin/python \
  benchmarks/retrieve_workshop/cpu_navix_target/run.py \
  --results-root /home/ubuntu/retrieve_workshop_runs/cpu \
  --run-id cpu_navix_target_<UTC_TAG>
```

The runner performs one-run integer `efSearch` calibration, one-run thread screening at
`threads={1,2,4,8,16,32}` for ArXiv, and a fresh three-run final measurement at the fastest CPU
thread count. YFCC fixes the maximum hardware thread count (32): its selected `efSearch=8192`
already takes roughly 146 seconds per 10,000-query repetition, and every ArXiv thread screen peaks
at 32 threads. The YFCC CPU point (0.79973 recall) is accepted as tolerance-matched to the GPU's
0.80001 point; both achieved recalls must be printed with the ratio. Every run contains 10,000
queries. YFCC QPS is computed as 10,000 divided by the sum of its five shard times. Native
packed-bitmap-to-byte-mask conversion remains outside the timed search call, matching the
CPU-context protocol.
