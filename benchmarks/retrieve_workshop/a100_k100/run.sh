#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")"; pwd)
repo_dir=$(cd "${script_dir}/../../.."; pwd)
profile=${RETRIEVE_DATASET_PROFILE:-"${script_dir}/profiles/a100_yfcc10m_arxiv_large_k100.json"}
python_bin=${PYTHON:-/home/ubuntu/micromamba/envs/cuvs/bin/python}
data_root=${RETRIEVE_DATA_ROOT:-/data/retrieve_data}
raw_arxiv=${RETRIEVE_ARXIV_LARGE_RAW:-"${data_root}/arxiv-for-fanns-large"}
run_tag=${RETRIEVE_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
run_root=${RETRIEVE_A100_K100_RUN_ROOT:-/data/retrieve_workshop_runs/a100_k100_${run_tag}}
bench_bin=${RETRIEVE_BENCH_BIN:-"${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"}
exact_bin=${RETRIEVE_EXACT_BENCH_BIN:-"${repo_dir}/cpp/build/bench/ann/CUVS_BRUTE_FORCE_ANN_BENCH"}
libcuvs=${RETRIEVE_LIBCUVS:-"${repo_dir}/cpp/build/libcuvs.so"}
yfcc_gt_root=${RETRIEVE_YFCC_GT100_ROOT:-"${data_root}/retrieve_workshop/k100_groundtruth/yfcc"}
yfcc_float_base=${RETRIEVE_K100_YFCC_FLOAT_BASE:-"${data_root}/retrieve_workshop/exact_bitmap_a100/yfcc/base.10M.fbin"}
exact_data_root=${RETRIEVE_K100_EXACT_DATA_ROOT:-"${data_root}/retrieve_workshop/exact_bitmap_a100_k100"}
matched_root=${RETRIEVE_K100_MATCHED_ROOT:-"${run_root}/matched_recall"}
stage=${1:-all}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHON="${python_bin}"
export RETRIEVE_DATASET_PROFILE="${profile}"
export RETRIEVE_DATA_ROOT="${data_root}"
export RETRIEVE_BENCH_BIN="${bench_bin}"
export RETRIEVE_LIBCUVS="${libcuvs}"
export RETRIEVE_GPU_ARCH=80-real
export RETRIEVE_PROVENANCE_MAX_QUERIES=2048
export RETRIEVE_PROVENANCE_K=100
export RETRIEVE_RESUME_COMPLETE=1
export RETRIEVE_HASH_CACHE="${run_root}/provenance/input_hash_cache.json"

mkdir -p "$(dirname "${run_root}")"
exec 9>"${run_root}.lock"
flock -n 9 || { echo "another k=100 stage owns ${run_root}.lock" >&2; exit 2; }
mkdir -p "${run_root}/logs" "${run_root}/state"
exec > >(tee -a "${run_root}/logs/orchestrator.log") 2>&1

profile_hash=$(sha256sum "${profile}" | awk '{print $1}')
if test -f "${run_root}/state/profile.sha256"; then
  test "$(<"${run_root}/state/profile.sha256")" = "${profile_hash}" || {
    echo "dataset profile changed inside immutable run root" >&2
    exit 2
  }
else
  printf '%s\n' "${profile_hash}" >"${run_root}/state/profile.sha256"
  cp "${profile}" "${run_root}/state/dataset_profile.json"
fi

done_marker() { printf '%s/.done/%s\n' "${run_root}" "$1"; }
is_done() { test -f "$(done_marker "$1")"; }
mark_done() {
  mkdir -p "${run_root}/.done"
  date -u +%Y-%m-%dT%H:%M:%SZ >"$(done_marker "$1")"
}

require_inputs() {
  local required=(
    "${data_root}/yfcc-10M/base.10M.u8bin"
    "${data_root}/yfcc-10M/GT.public.ibin"
    "${data_root}/yfcc-10M/cagra_g64_ig128.index"
    "${data_root}/arxiv-for-fanns-large/base.fbin"
    "${data_root}/arxiv-for-fanns-large/cagra_g64_ig128.index"
    "${data_root}/navix_bitmap/yfcc/correctness_1000/manifest.json"
    "${data_root}/navix_bitmap/yfcc/throughput_10000/manifest.json"
  )
  for workload in em emis r; do
    required+=(
      "${raw_arxiv}/ground_truth_${workload}.ivecs"
      "${data_root}/navix_bitmap/arxiv-large/${workload}/correctness_1000/manifest.json"
      "${data_root}/navix_bitmap/arxiv-large/${workload}/throughput_10000/manifest.json"
    )
  done
  for path in "${required[@]}"; do
    test -f "${path}" || { echo "missing required k=100 input: ${path}" >&2; exit 2; }
  done
}

preflight() {
  is_done preflight && return
  require_inputs
  local extra=()
  test "${RETRIEVE_ALLOW_NON_A100:-0}" = 1 && extra+=(--allow-non-a100)
  "${python_bin}" "${script_dir}/../a100_paper/preflight.py" \
    --repo "${repo_dir}" --data-root "${data_root}" --run-root "${run_root}" \
    --profile "${profile}" --minimum-free-gib "${RETRIEVE_K100_MINIMUM_FREE_GIB:-20}" \
    "${extra[@]}"
  mark_done preflight
}

build_binaries() {
  is_done build && return
  (cd "${repo_dir}" && env PARALLEL_LEVEL=${PARALLEL_LEVEL:-12} \
    ./build.sh libcuvs tests bench-ann -n \
      --limit-tests=NEIGHBORS_ANN_CAGRA_FILTER_BITMAP_TEST \
      --limit-bench-ann="CUVS_CAGRA_ANN_BENCH;CUVS_BRUTE_FORCE_ANN_BENCH" \
      --gpu-arch=80-real)
  test -x "${bench_bin}" && test -x "${exact_bin}" && test -f "${libcuvs}"
  mark_done build
}

test_gate() {
  is_done test && return
  env LD_PRELOAD="${libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
    "${repo_dir}/cpp/build/gtests/NEIGHBORS_ANN_CAGRA_FILTER_BITMAP_TEST" --gtest_color=no
  env -u RETRIEVE_DATASET_PROFILE MPLBACKEND=Agg \
    "${python_bin}" "${repo_dir}/benchmarks/retrieve_workshop/gpu_graph/test_pipeline.py"
  env -u RETRIEVE_DATASET_PROFILE MPLBACKEND=Agg \
    "${python_bin}" "${repo_dir}/benchmarks/retrieve_workshop/exact_bitmap/test_pipeline.py"
  env MPLBACKEND=Agg "${python_bin}" "${script_dir}/test_pipeline.py"
  env MPLBACKEND=Agg "${python_bin}" "${script_dir}/test_matched_recall.py"
  env -u RETRIEVE_DATASET_PROFILE MPLBACKEND=Agg \
    "${python_bin}" "${repo_dir}/benchmarks/retrieve_workshop/per_query_latency/test_pipeline.py"
  mark_done test
}

prepare_yfcc_gt() {
  is_done yfcc_gt100 && return
  local started_epoch
  started_epoch=$(date +%s)
  local generated_shards=0
  local reused_shards=0
  "${python_bin}" "${script_dir}/prepare_yfcc_gt100.py" prepare \
    --bitmap-manifest "${data_root}/navix_bitmap/yfcc/throughput_10000/manifest.json" \
    --base "${data_root}/yfcc-10M/base.10M.u8bin" \
    --official-gt "${data_root}/yfcc-10M/GT.public.ibin" \
    --converted-base "${yfcc_float_base}" --output "${yfcc_gt_root}"
  mkdir -p "${run_root}/yfcc_gt100/raw"
  while IFS=$'\t' read -r shard config output; do
    if test -s "${output}" && "${python_bin}" - "${output}" <<'PY'
import pathlib
import struct
import sys

path = pathlib.Path(sys.argv[1])
with path.open("rb") as source:
    rows, cols = struct.unpack("<II", source.read(8))
valid = cols == 100 and path.stat().st_size == 8 + rows * cols * 4
raise SystemExit(0 if valid else 1)
PY
    then
      echo "reuse generated YFCC GT shard ${output}"
      reused_shards=$((reused_shards + 1))
      continue
    fi
    test ! -e "${output}" || {
      echo "refusing malformed generated GT shard; remove it and retry: ${output}" >&2
      exit 2
    }
    env LD_PRELOAD="${libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
      "${exact_bin}" --search --mode=throughput --threads=1 \
      --data_prefix=/ --index_prefix=/ --benchmark_repetitions=1 \
      --benchmark_min_time=0.001s --benchmark_min_warmup_time=0.001 \
      --benchmark_report_aggregates_only=false --benchmark_out_format=json \
      --benchmark_out="${run_root}/yfcc_gt100/raw/shard_${shard}.json" "${config}"
    generated_shards=$((generated_shards + 1))
  done < <("${python_bin}" - "${yfcc_gt_root}/manifest.json" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
for row in manifest["shards"]:
    print(f"{row['shard_number']}\t{row['config']}\t{row['groundtruth_file']}")
PY
  )
  "${python_bin}" "${script_dir}/prepare_yfcc_gt100.py" finalize --output "${yfcc_gt_root}"
  mkdir -p "${run_root}/provenance"
  cp "${yfcc_gt_root}/manifest.json" "${run_root}/provenance/yfcc_gt100_manifest.json"
  sha256sum "${yfcc_gt_root}/manifest.json" \
    >"${run_root}/provenance/yfcc_gt100_manifest.sha256"
  "${python_bin}" - "${yfcc_gt_root}/manifest.json" <<'PY' | \
    xargs sha256sum >"${run_root}/provenance/yfcc_gt100_shards.sha256"
import json
import pathlib
import sys

for row in json.loads(pathlib.Path(sys.argv[1]).read_text())["shards"]:
    print(row["groundtruth_file"])
PY
  local elapsed_seconds
  elapsed_seconds=$(( $(date +%s) - started_epoch ))
  "${python_bin}" - "${run_root}/provenance/yfcc_gt100_stage_timing.json" \
    "${started_epoch}" "${elapsed_seconds}" \
    "${generated_shards}" "${reused_shards}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

output = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "started_epoch_seconds": int(sys.argv[2]),
    "wall_seconds": int(sys.argv[3]),
    "generated_shards": int(sys.argv[4]),
    "reused_shards": int(sys.argv[5]),
    "scope": "conversion/provenance checks, five masked exact searches when needed, and strict GT validation",
    "timing_use": "preprocessing diagnostic only; excluded from reported online QPS",
}
output.write_text(json.dumps(payload, indent=2) + "\n")
PY
  mark_done yfcc_gt100
}

prepare_views() {
  is_done views_unique_padding_v1 && return
  "${python_bin}" "${script_dir}/prepare_k100_views.py" \
    --data-root "${data_root}" --arxiv-raw "${raw_arxiv}" \
    --yfcc-gt-manifest "${yfcc_gt_root}/manifest.json"
  mark_done views_unique_padding_v1
}

run_graph() {
  is_done graph && return
  env RETRIEVE_RESULT_ROOT="${run_root}/gpu_graph" \
    "${repo_dir}/benchmarks/retrieve_workshop/gpu_graph/run_gpu_graph.sh" all \
      --k 100 --primary-methods-only --cartesian-b0
  mark_done graph
}

run_exact() {
  is_done exact && return
  env RETRIEVE_BITMAP_ROOT="${data_root}/navix_bitmap_k100" \
    RETRIEVE_ARXIV_BITMAP_NAMESPACE=arxiv-large \
    RETRIEVE_ARXIV_DATASET_DIR=arxiv-for-fanns-large \
    RETRIEVE_EXACT_DATA_ROOT="${exact_data_root}" \
    RETRIEVE_EXACT_RESULT_ROOT="${run_root}/exact_bitmap" \
    RETRIEVE_EXACT_K=100 RETRIEVE_EXACT_YFCC_FLOAT_BASE="${yfcc_float_base}" \
    "${repo_dir}/benchmarks/retrieve_workshop/exact_bitmap/run_exact.sh" all
  mark_done exact
}

run_matched() {
  is_done matched_recall && return
  local baseline_summary="${run_root}/gpu_graph/analysis/summary_points.csv"
  local baseline_provenance="${run_root}/gpu_graph/analysis/provenance.json"
  test -f "${baseline_summary}" && test -f "${baseline_provenance}" || {
    echo "run the k=100 graph stage before matched recall" >&2
    exit 2
  }
  env RETRIEVE_RESULT_ROOT="${matched_root}" \
    RETRIEVE_MATCHED_K=100 \
    RETRIEVE_MATCHED_ALLOW_SHALLOW_NAVIX=1 \
    RETRIEVE_MATCHED_BASELINE_SUMMARY="${baseline_summary}" \
    RETRIEVE_MATCHED_BASELINE_PROVENANCE="${baseline_provenance}" \
    RETRIEVE_PROVENANCE_K=100 \
    RETRIEVE_PROVENANCE_MAX_QUERIES=2048 \
    "${repo_dir}/benchmarks/retrieve_workshop/matched_recall/run_matched_recall.sh" all
  env RETRIEVE_MATCHED_K=100 RETRIEVE_MATCHED_ALLOW_SHALLOW_NAVIX=1 MPLBACKEND=Agg \
    "${python_bin}" "${script_dir}/matched_table.py" --result-root "${matched_root}"
  mark_done matched_recall
}

run_per_query_latency() {
  local latency_root="${run_root}/per_query_latency"
  local selected="${matched_root}/analysis/selected_points.csv"
  local selected_provenance="${matched_root}/analysis/provenance.json"
  if is_done per_query_latency; then
    env RETRIEVE_LATENCY_K=100 RETRIEVE_LATENCY_RESULT_ROOT="${latency_root}" \
      RETRIEVE_LATENCY_SELECTED_POINTS="${selected}" \
      RETRIEVE_LATENCY_SELECTED_PROVENANCE="${selected_provenance}" \
      RETRIEVE_LATENCY_EXACT_DATA_ROOT="${exact_data_root}" \
      RETRIEVE_EXACT_BENCH_BIN="${exact_bin}" \
      "${repo_dir}/benchmarks/retrieve_workshop/per_query_latency/run.sh" validate
    return
  fi
  test -f "${selected}" && test -f "${selected_provenance}" || {
    echo "run matched-recall before serialized k=100 latency" >&2
    exit 2
  }
  test -f "${exact_data_root}/arxiv-large/em/throughput_10000/manifest.json" || {
    echo "run exact before serialized k=100 latency" >&2
    exit 2
  }
  env RETRIEVE_LATENCY_K=100 RETRIEVE_LATENCY_RESULT_ROOT="${latency_root}" \
    RETRIEVE_LATENCY_SELECTED_POINTS="${selected}" \
    RETRIEVE_LATENCY_SELECTED_PROVENANCE="${selected_provenance}" \
    RETRIEVE_LATENCY_EXACT_DATA_ROOT="${exact_data_root}" \
    RETRIEVE_EXACT_BENCH_BIN="${exact_bin}" \
    "${repo_dir}/benchmarks/retrieve_workshop/per_query_latency/run.sh" all
  mark_done per_query_latency
}

run_seed_ablation() {
  env RETRIEVE_A100_K100_RUN_ROOT="${run_root}" \
    RETRIEVE_A100_SEED_RUN_ROOT="${RETRIEVE_A100_SEED_RUN_ROOT:-${run_root}/seed_ablation}" \
    "${repo_dir}/benchmarks/retrieve_workshop/a100_seed_ablation/run.sh" all
}

analyze() {
  env RETRIEVE_RESULT_ROOT="${run_root}/gpu_graph" \
    "${repo_dir}/benchmarks/retrieve_workshop/gpu_graph/run_gpu_graph.sh" analyze
  env RETRIEVE_BITMAP_ROOT="${data_root}/navix_bitmap_k100" \
    RETRIEVE_ARXIV_BITMAP_NAMESPACE=arxiv-large \
    RETRIEVE_ARXIV_DATASET_DIR=arxiv-for-fanns-large \
    RETRIEVE_EXACT_DATA_ROOT="${exact_data_root}" \
    RETRIEVE_EXACT_RESULT_ROOT="${run_root}/exact_bitmap" \
    RETRIEVE_EXACT_K=100 RETRIEVE_EXACT_YFCC_FLOAT_BASE="${yfcc_float_base}" \
    "${repo_dir}/benchmarks/retrieve_workshop/exact_bitmap/run_exact.sh" analyze
  env MPLBACKEND=Agg "${python_bin}" "${script_dir}/analyze.py" --run-root "${run_root}"
  if test -d "${matched_root}/configs"; then
    env RETRIEVE_RESULT_ROOT="${matched_root}" RETRIEVE_MATCHED_K=100 \
      RETRIEVE_MATCHED_ALLOW_SHALLOW_NAVIX=1 RETRIEVE_PROVENANCE_K=100 \
      RETRIEVE_PROVENANCE_MAX_QUERIES=2048 \
      "${repo_dir}/benchmarks/retrieve_workshop/matched_recall/run_matched_recall.sh" analyze
    env RETRIEVE_MATCHED_K=100 RETRIEVE_MATCHED_ALLOW_SHALLOW_NAVIX=1 MPLBACKEND=Agg \
      "${python_bin}" "${script_dir}/matched_table.py" --result-root "${matched_root}"
  fi
  if test -d "${run_root}/per_query_latency/configs"; then
    env RETRIEVE_LATENCY_K=100 \
      RETRIEVE_LATENCY_RESULT_ROOT="${run_root}/per_query_latency" \
      RETRIEVE_LATENCY_SELECTED_POINTS="${matched_root}/analysis/selected_points.csv" \
      RETRIEVE_LATENCY_SELECTED_PROVENANCE="${matched_root}/analysis/provenance.json" \
      RETRIEVE_LATENCY_EXACT_DATA_ROOT="${exact_data_root}" \
      RETRIEVE_EXACT_BENCH_BIN="${exact_bin}" \
      "${repo_dir}/benchmarks/retrieve_workshop/per_query_latency/run.sh" analyze
  fi
}

bundle() {
  "${python_bin}" "${script_dir}/bundle.py" --run-root "${run_root}"
}

case "${stage}" in
  preflight) preflight ;;
  build) build_binaries ;;
  test) test_gate ;;
  prepare-yfcc-gt) prepare_yfcc_gt ;;
  prepare-views) prepare_views ;;
  graph) run_graph ;;
  exact) run_exact ;;
  matched-recall) run_matched ;;
  seed-ablation) run_seed_ablation ;;
  latency) run_per_query_latency ;;
  analyze) analyze ;;
  bundle) bundle ;;
  all)
    preflight
    build_binaries
    test_gate
    prepare_yfcc_gt
    prepare_views
    run_graph
    run_exact
    run_matched
    analyze
    bundle
    ;;
  *)
    echo "usage: $0 {preflight|build|test|prepare-yfcc-gt|prepare-views|graph|exact|matched-recall|seed-ablation|latency|analyze|bundle|all}" >&2
    exit 2
    ;;
esac

printf '%s\n' "${run_root}"
