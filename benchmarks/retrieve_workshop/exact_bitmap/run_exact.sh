#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/../../.."; pwd)
python_bin=${PYTHON:-/home/ubuntu/micromamba/envs/cuvs/bin/python}
data_root=${RETRIEVE_DATA_ROOT:-"${repo_dir}/datasets"}
if [[ ! -d "${data_root}" && -d /home/ubuntu/cuvs-filter/datasets ]]; then
  data_root=/home/ubuntu/cuvs-filter/datasets
fi
bitmap_root=${RETRIEVE_BITMAP_ROOT:-"${data_root}/navix_bitmap"}
arxiv_namespace=${RETRIEVE_ARXIV_BITMAP_NAMESPACE:-arxiv}
arxiv_dataset_dir=${RETRIEVE_ARXIV_DATASET_DIR:-arxiv-for-fanns-medium}
exact_data_root=${RETRIEVE_EXACT_DATA_ROOT:-"${data_root}/retrieve_workshop/exact_bitmap"}
result_root=${RETRIEVE_EXACT_RESULT_ROOT:-"${repo_dir}/benchmarks/retrieve_workshop/exact_bitmap/results"}
bench_bin=${RETRIEVE_EXACT_BENCH_BIN:-"${repo_dir}/cpp/build/bench/ann/CUVS_BRUTE_FORCE_ANN_BENCH"}
build_libcuvs=${RETRIEVE_EXACT_LIBCUVS:-"${repo_dir}/cpp/build/libcuvs.so"}
stage=${1:-all}

prepare_one() {
  local source_manifest=$1
  local base_file=$2
  local source_dtype=$3
  local output=$4
  shift 4
  local force_args=()
  if [[ "${RETRIEVE_EXACT_FORCE_PREPARE:-0}" = 1 ]]; then
    force_args+=(--force)
  fi
  "${python_bin}" "${repo_dir}/benchmarks/retrieve_workshop/exact_bitmap/prepare_exact_workload.py" \
    --bitmap-manifest "${source_manifest}" --base-file "${base_file}" \
    --source-dtype "${source_dtype}" --output "${output}" "${force_args[@]}" "$@"
}

prepare_data() {
  local yfcc_float_base="${exact_data_root}/yfcc/base.10M.fbin"
  prepare_one "${bitmap_root}/yfcc/correctness_1000/manifest.json" \
    "${data_root}/yfcc-10M/base.10M.u8bin" uint8 \
    "${exact_data_root}/yfcc/correctness_1000" --converted-base "${yfcc_float_base}"
  prepare_one "${bitmap_root}/yfcc/throughput_10000/manifest.json" \
    "${data_root}/yfcc-10M/base.10M.u8bin" uint8 \
    "${exact_data_root}/yfcc/throughput_10000" --converted-base "${yfcc_float_base}"
  for predicate in em emis r; do
    for phase in correctness_1000 throughput_10000; do
      prepare_one "${bitmap_root}/${arxiv_namespace}/${predicate}/${phase}/manifest.json" \
        "${data_root}/${arxiv_dataset_dir}/base.fbin" float32 \
        "${exact_data_root}/${arxiv_namespace}/${predicate}/${phase}"
    done
  done
}

generate_one() {
  local workload=$1
  local phase=$2
  local exact_manifest=$3
  "${python_bin}" "${repo_dir}/benchmarks/retrieve_workshop/exact_bitmap/generate_configs.py" \
    --exact-manifest "${exact_manifest}" --workload "${workload}" --phase "${phase}" \
    --index-marker "${exact_data_root}/markers/${workload}.index" \
    --output "${result_root}/configs/${phase}/${workload}"
}

generate_configs() {
  generate_one yfcc correctness "${exact_data_root}/yfcc/correctness_1000/manifest.json"
  generate_one yfcc throughput "${exact_data_root}/yfcc/throughput_10000/manifest.json"
  for predicate in em emis r; do
    generate_one "${predicate}" correctness \
      "${exact_data_root}/${arxiv_namespace}/${predicate}/correctness_1000/manifest.json"
    generate_one "${predicate}" throughput \
      "${exact_data_root}/${arxiv_namespace}/${predicate}/throughput_10000/manifest.json"
  done
}

ensure_binary() {
  if [[ ! -x "${bench_bin}" || ! -f "${build_libcuvs}" ]]; then
    (cd "${repo_dir}" && \
      ./build.sh bench-ann -n --limit-bench-ann=CUVS_BRUTE_FORCE_ANN_BENCH \
        --gpu-arch="${RETRIEVE_GPU_ARCH:-89-real}")
  fi
  if [[ ! -x "${bench_bin}" || ! -f "${build_libcuvs}" ]]; then
    echo "cuVS brute-force ANN benchmark build did not produce the expected files" >&2
    exit 1
  fi
}

capture_provenance() {
  "${python_bin}" "${repo_dir}/benchmarks/retrieve_workshop/exact_bitmap/capture_provenance.py" \
    --result-root "${result_root}" --repo "${repo_dir}" --stage "$1" \
    --bench-bin "${bench_bin}" --libcuvs "${build_libcuvs}"
}

run_group() {
  local phase=$1
  local workload=$2
  local repetitions=$3
  local min_time=$4
  local config_dir="${result_root}/configs/${phase}/${workload}"
  local raw_dir="${result_root}/raw/${phase}/${workload}"
  local log_dir="${result_root}/logs/${phase}/${workload}"
  local memory_dir="${result_root}/provenance/gpu_memory/${phase}/${workload}"
  mkdir -p "${raw_dir}" "${log_dir}" "${memory_dir}"
  local exact_manifest
  exact_manifest=$("${python_bin}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["exact_manifest"])' \
    "${config_dir}/manifest.json")
  mapfile -t configs < <("${python_bin}" -c \
    'import json,sys; print(*[x["config"] for x in json.load(open(sys.argv[1]))["configs"]], sep="\n")' \
    "${config_dir}/manifest.json")
  local shard=0
  for config in "${configs[@]}"; do
    local tag
    tag=$(printf 'shard_%02d' "${shard}")
    if [[ -e "${raw_dir}/${tag}.json" && "${RETRIEVE_EXACT_OVERWRITE:-0}" != 1 ]]; then
      echo "refusing to overwrite ${raw_dir}/${tag}.json; set RETRIEVE_EXACT_OVERWRITE=1 explicitly" >&2
      exit 1
    fi
    local memory_args=()
    if [[ -n "${RETRIEVE_EXACT_AVAILABLE_BYTES:-}" ]]; then
      memory_args+=(--available-bytes "${RETRIEVE_EXACT_AVAILABLE_BYTES}")
    fi
    "${python_bin}" \
      "${repo_dir}/benchmarks/retrieve_workshop/exact_bitmap/gpu_memory_preflight.py" \
      --exact-manifest "${exact_manifest}" --shard-number "${shard}" \
      --output "${memory_dir}/${tag}.json" "${memory_args[@]}"
    env LD_PRELOAD="${build_libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
      "${bench_bin}" --search --mode=throughput --threads=1 \
      --data_prefix=/ --index_prefix=/ \
      --benchmark_repetitions="${repetitions}" --benchmark_min_time="${min_time}" \
      --benchmark_min_warmup_time=0.001 --benchmark_report_aggregates_only=false \
      --benchmark_out_format=json --benchmark_out="${raw_dir}/${tag}.json" "${config}" \
      2>&1 | tee "${log_dir}/${tag}.log"
    shard=$((shard + 1))
  done
}

run_phase() {
  local phase=$1
  local repetitions=$2
  local min_time=$3
  capture_provenance "${phase}"
  for workload in yfcc em emis r; do
    run_group "${phase}" "${workload}" "${repetitions}" "${min_time}"
  done
}

analyze_phases() {
  local arguments=()
  for phase in "$@"; do
    arguments+=(--phase "${phase}")
  done
  "${python_bin}" "${repo_dir}/benchmarks/retrieve_workshop/exact_bitmap/analyze_exact.py" \
    --result-root "${result_root}" "${arguments[@]}"
}

run_smoke() {
  ensure_binary
  capture_provenance smoke
  local sparse_fixture="${result_root}/smoke_fixture/sparse"
  local dense_fixture="${result_root}/smoke_fixture/dense"
  "${python_bin}" "${repo_dir}/benchmarks/retrieve_workshop/exact_bitmap/make_smoke_fixture.py" \
    --output "${sparse_fixture}" --mode sparse
  "${python_bin}" "${repo_dir}/benchmarks/retrieve_workshop/exact_bitmap/make_smoke_fixture.py" \
    --output "${dense_fixture}" --mode dense
  generate_one smoke_sparse smoke "${sparse_fixture}/manifest.json"
  generate_one smoke_dense smoke "${dense_fixture}/manifest.json"
  run_group smoke smoke_sparse 1 0.001s
  run_group smoke smoke_dense 1 0.001s
  analyze_phases smoke
}

case "${stage}" in
  build) ensure_binary ;;
  prepare) prepare_data ;;
  configs) generate_configs ;;
  smoke) run_smoke ;;
  correctness)
    ensure_binary
    generate_configs
    run_phase correctness 1 0.001s
    analyze_phases correctness
    ;;
  throughput)
    ensure_binary
    generate_configs
    run_phase throughput 3 0.001s
    analyze_phases throughput
    ;;
  analyze) analyze_phases correctness throughput ;;
  all)
    ensure_binary
    prepare_data
    generate_configs
    run_phase correctness 1 0.001s
    run_phase throughput 3 0.001s
    analyze_phases correctness throughput
    ;;
  *)
    echo "usage: $0 {build|prepare|configs|smoke|correctness|throughput|analyze|all}" >&2
    exit 2
    ;;
esac
