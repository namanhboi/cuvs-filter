#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/../../.."; pwd)
data_root=${NAVIX_BITMAP_DATA_ROOT:-"${repo_dir}/datasets"}
bitmap_root=${NAVIX_BITMAP_ROOT:-"${data_root}/navix_bitmap"}
result_root=${NAVIX_BITMAP_RESULT_ROOT:-"${repo_dir}/benchmarks/favor/navix_bitmap/results_20260809"}
bench_bin="${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"
build_libcuvs="${repo_dir}/cpp/build/libcuvs.so"
python_bin=${PYTHON:-/home/ubuntu/micromamba/envs/cuvs/bin/python}
stage=${1:-all}

prepare_sparse() {
  local base_metadata=$1
  local query_metadata=$2
  local query_vectors=$3
  local groundtruth=$4
  local dtype=$5
  local output=$6
  local limit=$7
  local shard_size=$8
  if [[ -f "${output}/manifest.json" && "${NAVIX_BITMAP_FORCE_PREPARE:-0}" != 1 ]]; then
    return
  fi
  "${python_bin}" "${repo_dir}/benchmarks/favor/navix_bitmap/prepare_bitmaps.py" sparse \
    --base-metadata "${base_metadata}" --query-metadata "${query_metadata}" \
    --query-vectors "${query_vectors}" --groundtruth "${groundtruth}" \
    --vector-dtype "${dtype}" --output "${output}" --limit "${limit}" \
    --shard-size "${shard_size}"
}

prepare_range() {
  local base_metadata=$1
  local query_metadata=$2
  local query_vectors=$3
  local groundtruth=$4
  local output=$5
  local limit=$6
  if [[ -f "${output}/manifest.json" && "${NAVIX_BITMAP_FORCE_PREPARE:-0}" != 1 ]]; then
    return
  fi
  "${python_bin}" "${repo_dir}/benchmarks/favor/navix_bitmap/prepare_bitmaps.py" range \
    --base-metadata "${base_metadata}" --query-metadata "${query_metadata}" \
    --query-vectors "${query_vectors}" --groundtruth "${groundtruth}" \
    --vector-dtype float32 --output "${output}" --limit "${limit}"
}

prepare_data() {
  prepare_sparse \
    "${data_root}/yfcc-10M/base.metadata.10M.spmat" \
    "${data_root}/yfcc-10M/workloads/correctness_1000/query.metadata.spmat" \
    "${data_root}/yfcc-10M/workloads/correctness_1000/query.u8bin" \
    "${data_root}/yfcc-10M/workloads/correctness_1000/groundtruth.ibin" uint8 \
    "${bitmap_root}/yfcc/correctness_1000" 1000 1000
  prepare_sparse \
    "${data_root}/yfcc-10M/base.metadata.10M.spmat" \
    "${data_root}/yfcc-10M/workloads/throughput_10000/query.metadata.spmat" \
    "${data_root}/yfcc-10M/workloads/throughput_10000/query.u8bin" \
    "${data_root}/yfcc-10M/workloads/throughput_10000/groundtruth.ibin" uint8 \
    "${bitmap_root}/yfcc/throughput_10000" 10000 2048

  for predicate in em emis; do
    for phase in correctness throughput; do
      local count=10000
      local source_phase=throughput_10000
      local output_phase=throughput_10000
      if [[ "${phase}" == correctness ]]; then
        count=1000
        source_phase=correctness_10000
        output_phase=correctness_1000
      fi
      prepare_sparse \
        "${data_root}/arxiv-for-fanns-medium/${predicate}/base_metadata.spmat" \
        "${data_root}/arxiv-for-fanns-medium/${predicate}/${source_phase}/query_metadata.spmat" \
        "${data_root}/arxiv-for-fanns-medium/${predicate}/${source_phase}/query.fbin" \
        "${data_root}/arxiv-for-fanns-medium/${predicate}/${source_phase}/groundtruth.ibin" \
        float32 "${bitmap_root}/arxiv/${predicate}/${output_phase}" "${count}" 0
    done
  done
  prepare_range \
    "${data_root}/arxiv-for-fanns-medium/r/base_metadata.rmeta" \
    "${data_root}/arxiv-for-fanns-medium/r/correctness_10000/query_metadata.rmeta" \
    "${data_root}/arxiv-for-fanns-medium/r/correctness_10000/query.fbin" \
    "${data_root}/arxiv-for-fanns-medium/r/correctness_10000/groundtruth.ibin" \
    "${bitmap_root}/arxiv/r/correctness_1000" 1000
  prepare_range \
    "${data_root}/arxiv-for-fanns-medium/r/base_metadata.rmeta" \
    "${data_root}/arxiv-for-fanns-medium/r/throughput_10000/query_metadata.rmeta" \
    "${data_root}/arxiv-for-fanns-medium/r/throughput_10000/query.fbin" \
    "${data_root}/arxiv-for-fanns-medium/r/throughput_10000/groundtruth.ibin" \
    "${bitmap_root}/arxiv/r/throughput_10000" 10000
}

generate_one() {
  local workload=$1
  local phase=$2
  local manifest=$3
  local base_file=$4
  local index_file=$5
  local dtype=$6
  "${python_bin}" "${repo_dir}/benchmarks/favor/navix_bitmap/generate_configs.py" \
    --bitmap-manifest "${manifest}" --base-file "${base_file}" \
    --index-file "${index_file}" --dataset-name "${workload}-${phase}" \
    --dtype "${dtype}" --output "${result_root}/configs/${phase}/${workload}"
}

generate_configs() {
  generate_one yfcc correctness "${bitmap_root}/yfcc/correctness_1000/manifest.json" \
    yfcc-10M/base.10M.u8bin yfcc-10M/cagra_g32_ig64.index uint8
  generate_one yfcc throughput "${bitmap_root}/yfcc/throughput_10000/manifest.json" \
    yfcc-10M/base.10M.u8bin yfcc-10M/cagra_g32_ig64.index uint8
  for predicate in em emis r; do
    generate_one "${predicate}" correctness \
      "${bitmap_root}/arxiv/${predicate}/correctness_1000/manifest.json" \
      arxiv-for-fanns-medium/base.fbin arxiv-for-fanns-medium/cagra_g32_ig64.index float
    generate_one "${predicate}" throughput \
      "${bitmap_root}/arxiv/${predicate}/throughput_10000/manifest.json" \
      arxiv-for-fanns-medium/base.fbin arxiv-for-fanns-medium/cagra_g32_ig64.index float
  done
}

run_group() {
  local phase=$1
  local workload=$2
  local min_time=$3
  local config_dir="${result_root}/configs/${phase}/${workload}"
  local raw_dir="${result_root}/raw/${phase}/${workload}"
  local log_dir="${result_root}/logs/${phase}/${workload}"
  mkdir -p "${raw_dir}" "${log_dir}"
  mapfile -t configs < <("${python_bin}" -c \
    'import json,sys; print(*[x["config"] for x in json.load(open(sys.argv[1]))["configs"]], sep="\n")' \
    "${config_dir}/manifest.json")
  local shard=0
  for config in "${configs[@]}"; do
    local output="${raw_dir}/shard_$(printf '%02d' "${shard}").json"
    env LD_PRELOAD="${build_libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
      "${bench_bin}" --search --mode=throughput --threads=1 \
      --data_prefix="${data_root}" --index_prefix="${data_root}" \
      --benchmark_repetitions=1 --benchmark_min_time="${min_time}" \
      --benchmark_min_warmup_time=0.01 --benchmark_report_aggregates_only=false \
      --benchmark_out_format=json --benchmark_out="${output}" "${config}" \
      2>&1 | tee "${log_dir}/shard_$(printf '%02d' "${shard}").log"
    shard=$((shard + 1))
  done
}

run_phase() {
  local phase=$1
  local min_time=$2
  for workload in yfcc em emis r; do
    run_group "${phase}" "${workload}" "${min_time}"
  done
}

analyze() {
  "${python_bin}" "${repo_dir}/benchmarks/favor/navix_bitmap/analyze.py" \
    --result-root "${result_root}" --bitmap-root "${bitmap_root}"
}

case "${stage}" in
  prepare) prepare_data ;;
  configs) generate_configs ;;
  correctness) generate_configs; run_phase correctness 0.01s ;;
  throughput) generate_configs; run_phase throughput 0.05s ;;
  analyze) analyze ;;
  all)
    prepare_data
    generate_configs
    run_phase correctness 0.01s
    run_phase throughput 0.05s
    analyze
    ;;
  *)
    echo "usage: $0 {prepare|configs|correctness|throughput|analyze|all}" >&2
    exit 2
    ;;
esac
