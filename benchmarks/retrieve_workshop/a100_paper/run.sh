#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")"; pwd)
repo_dir=$(cd "${script_dir}/../../.."; pwd)
profile=${RETRIEVE_DATASET_PROFILE:-"${script_dir}/profiles/a100_yfcc10m_arxiv_large.json"}
python_bin=${PYTHON:-/home/ubuntu/micromamba/envs/cuvs/bin/python}
data_root=${RETRIEVE_DATA_ROOT:-/home/ubuntu/cuvs-filter/datasets}
run_tag=${RETRIEVE_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
run_root=${RETRIEVE_A100_RUN_ROOT:-/home/ubuntu/retrieve_workshop_runs/a100_gpu_${run_tag}}
raw_arxiv=${RETRIEVE_ARXIV_LARGE_RAW:-"${data_root}/arxiv-for-fanns-large"}
yfcc_source=${RETRIEVE_YFCC_SOURCE:-}
bench_bin=${RETRIEVE_BENCH_BIN:-"${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"}
libcuvs=${RETRIEVE_LIBCUVS:-"${repo_dir}/cpp/build/libcuvs.so"}
stage=${1:-all}

export RETRIEVE_DATASET_PROFILE="${profile}"
export RETRIEVE_DATA_ROOT="${data_root}"
export RETRIEVE_BENCH_BIN="${bench_bin}"
export RETRIEVE_LIBCUVS="${libcuvs}"
export PYTHON="${python_bin}"
export RETRIEVE_GPU_ARCH=80-real
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p "$(dirname "${run_root}")"
exec 9>"${run_root}.lock"
flock -n 9 || { echo "another A100 paper stage owns ${run_root}.lock" >&2; exit 2; }
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
mark_done() { mkdir -p "${run_root}/.done"; date -u +%Y-%m-%dT%H:%M:%SZ >"$(done_marker "$1")"; }

preflight() {
  is_done preflight && return
  local extra=()
  test "${RETRIEVE_ALLOW_NON_A100:-0}" = 1 && extra+=(--allow-non-a100)
  "${python_bin}" "${script_dir}/preflight.py" --repo "${repo_dir}" \
    --data-root "${data_root}" --run-root "${run_root}" --profile "${profile}" "${extra[@]}"
  mark_done preflight
}

download_arxiv() {
  "${python_bin}" "${script_dir}/download_arxiv_large.py" --output "${raw_arxiv}"
}

build_binaries() {
  is_done build && return
  (cd "${repo_dir}" && env PARALLEL_LEVEL=${PARALLEL_LEVEL:-12} \
    ./build.sh libcuvs tests bench-ann -n \
      --limit-tests=NEIGHBORS_ANN_CAGRA_FILTER_BITMAP_TEST \
      --limit-bench-ann="CUVS_CAGRA_ANN_BENCH;CUVS_BRUTE_FORCE_ANN_BENCH" \
      --gpu-arch=80-real)
  test -x "${bench_bin}" && test -f "${libcuvs}"
  mark_done build
}

test_gate() {
  is_done test && return
  env LD_PRELOAD="${libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
    "${repo_dir}/cpp/build/gtests/NEIGHBORS_ANN_CAGRA_FILTER_BITMAP_TEST" --gtest_color=no
  # These legacy synthetic suites construct the default medium-dataset fixture. Keep the
  # production A100 profile from leaking into their subprocesses; the A100-specific suite below
  # supplies and validates the large-dataset profile explicitly.
  env -u RETRIEVE_DATASET_PROFILE MPLBACKEND=Agg \
    "${python_bin}" "${repo_dir}/benchmarks/retrieve_workshop/gpu_graph/test_pipeline.py"
  env -u RETRIEVE_DATASET_PROFILE MPLBACKEND=Agg \
    "${python_bin}" "${repo_dir}/benchmarks/retrieve_workshop/matched_recall/test_matched_recall.py"
  env -u RETRIEVE_DATASET_PROFILE MPLBACKEND=Agg \
    "${python_bin}" "${repo_dir}/benchmarks/retrieve_workshop/exact_bitmap/test_pipeline.py"
  MPLBACKEND=Agg "${python_bin}" "${script_dir}/test_pipeline.py"
  mark_done test
}

link_yfcc() {
  local target="${data_root}/yfcc-10M"
  mkdir -p "${target}"
  if test -n "${yfcc_source}"; then
    declare -A files=(
      [base.10M.u8bin]=base.10M.u8bin
      [query.public.100K.u8bin]=query.public.100K.u8bin
      [base.metadata.10M.spmat]=base.metadata.10M.spmat
      [query.metadata.public.100K.spmat]=query.metadata.public.100K.spmat
      [GT.public.ibin]=GT.public.ibin
    )
    for destination in "${!files[@]}"; do
      local source="${yfcc_source}/${files[${destination}]}"
      test -f "${source}" || { echo "missing YFCC source ${source}" >&2; exit 2; }
      if ! test -e "${target}/${destination}"; then
        ln -s "$(realpath "${source}")" "${target}/${destination}"
      fi
    done
  fi
  for file in base.10M.u8bin query.public.100K.u8bin base.metadata.10M.spmat \
    query.metadata.public.100K.spmat GT.public.ibin; do
    test -f "${target}/${file}" || {
      echo "missing ${target}/${file}; set RETRIEVE_YFCC_SOURCE to the BIG-ann YFCC directory" >&2
      exit 2
    }
  done
}

prepare_yfcc_phase() {
  local phase=$1 limit=$2 shard=$3
  local output="${data_root}/navix_bitmap/yfcc/${phase}_${limit}"
  test -f "${output}/manifest.json" && return
  "${python_bin}" "${repo_dir}/benchmarks/favor/navix_bitmap/prepare_bitmaps.py" sparse \
    --base-metadata "${data_root}/yfcc-10M/base.metadata.10M.spmat" \
    --query-metadata "${data_root}/yfcc-10M/query.metadata.public.100K.spmat" \
    --query-vectors "${data_root}/yfcc-10M/query.public.100K.u8bin" \
    --groundtruth "${data_root}/yfcc-10M/GT.public.ibin" --vector-dtype uint8 \
    --output "${output}" --limit "${limit}" --shard-size "${shard}"
}

prepare_data() {
  is_done prepare && return
  link_yfcc
  prepare_yfcc_phase correctness 1000 1000
  prepare_yfcc_phase throughput 10000 2048
  "${python_bin}" "${script_dir}/prepare_arxiv_large.py" \
    --source "${raw_arxiv}" --data-root "${data_root}" --reuse-valid
  mark_done prepare
}

build_one_graph() {
  local config=$1 index=$2
  if test -s "${index}" && test "${RETRIEVE_REBUILD_GRAPHS:-0}" != 1; then
    echo "reuse graph ${index}"
    return
  fi
  if test -e "${index}"; then
    echo "refusing to overwrite ${index}; remove it or choose a new data root" >&2
    exit 2
  fi
  env LD_PRELOAD="${libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
    "${bench_bin}" --build --data_prefix="${data_root}" --index_prefix="${data_root}" "${config}"
  test -s "${index}"
}

build_graphs() {
  is_done graphs && return
  build_one_graph "${repo_dir}/benchmarks/retrieve_workshop/gpu_graph/graph_build_configs/yfcc_g64_ig128.json" \
    "${data_root}/yfcc-10M/cagra_g64_ig128.index"
  build_one_graph "${repo_dir}/benchmarks/retrieve_workshop/gpu_graph/graph_build_configs/arxiv_large_g64_ig128.json" \
    "${data_root}/arxiv-for-fanns-large/cagra_g64_ig128.index"
  sha256sum "${data_root}/yfcc-10M/cagra_g64_ig128.index" \
    "${data_root}/arxiv-for-fanns-large/cagra_g64_ig128.index" \
    >"${run_root}/provenance/graph_sha256.txt"
  mark_done graphs
}

run_gpu_stage() {
  local mode=$1
  is_done "gpu_${mode}" && return
  env RETRIEVE_RESULT_ROOT="${run_root}/gpu_graph" \
    "${repo_dir}/benchmarks/retrieve_workshop/gpu_graph/run_gpu_graph.sh" "${mode}"
  mark_done "gpu_${mode}"
}

run_matched() {
  is_done matched && return
  env RETRIEVE_RESULT_ROOT="${run_root}/matched_recall" \
    "${repo_dir}/benchmarks/retrieve_workshop/matched_recall/run_matched_recall.sh" all
  mark_done matched
}

run_exact() {
  is_done exact && return
  env RETRIEVE_EXACT_RESULT_ROOT="${run_root}/exact_bitmap" \
    RETRIEVE_EXACT_DATA_ROOT="${data_root}/retrieve_workshop/exact_bitmap_a100" \
    RETRIEVE_ARXIV_BITMAP_NAMESPACE=arxiv-large \
    RETRIEVE_ARXIV_DATASET_DIR=arxiv-for-fanns-large \
    RETRIEVE_HASH_CACHE="${run_root}/provenance/input_hash_cache.json" \
    "${repo_dir}/benchmarks/retrieve_workshop/exact_bitmap/run_exact.sh" all
  mark_done exact
}

run_resource() {
  is_done resource && return
  env RETRIEVE_RESOURCE_WORK_ROOT="${run_root}/resource_work" \
    "${repo_dir}/benchmarks/retrieve_workshop/resource_work/run.sh" all
  mark_done resource
}

run_mechanism_diagnostics() {
  is_done mechanism_diagnostics && return
  local root="${run_root}/mechanism_diagnostics"
  mkdir -p "${root}/raw" "${root}/analysis"
  "${python_bin}" "${script_dir}/mechanism_diagnostics.py" generate \
    --data-root "${data_root}" --output "${root}/config.json" --diagnostics "${root}/captures"
  env RETRIEVE_PROVENANCE_MAX_QUERIES=1024 RETRIEVE_PROVENANCE_REPETITIONS=1 \
    RETRIEVE_PROVENANCE_TIMING="untimed schema-9 diagnostic; never throughput evidence" \
    "${python_bin}" "${repo_dir}/benchmarks/retrieve_workshop/gpu_graph/capture_provenance.py" \
      --result-root "${root}" --repo "${repo_dir}" --stage diagnostics \
      --bench-bin "${bench_bin}" --libcuvs "${libcuvs}" --data-root "${data_root}"
  env LD_PRELOAD="${libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
    "${bench_bin}" --search --mode=throughput --threads=1 \
    --data_prefix="${data_root}" --index_prefix="${data_root}" \
    --benchmark_repetitions=1 --benchmark_min_time=0.001s \
    --benchmark_min_warmup_time=0.001 --benchmark_report_aggregates_only=false \
    --benchmark_out_format=json --benchmark_out="${root}/raw/results.json" "${root}/config.json"
  "${python_bin}" "${script_dir}/mechanism_diagnostics.py" summarize \
    --diagnostics "${root}/captures" --output "${root}/analysis/mechanism_summary.json"
  mark_done mechanism_diagnostics
}

run_dataset_stats() {
  is_done dataset_stats && return
  mkdir -p "${run_root}/dataset_stats"
  "${python_bin}" "${repo_dir}/benchmarks/retrieve_workshop/evidence/summarize_bitmap_selectivity.py" \
    --workload "yfcc=${data_root}/navix_bitmap/yfcc/throughput_10000/manifest.json" \
    --workload "arxiv_em=${data_root}/navix_bitmap/arxiv-large/em/throughput_10000/manifest.json" \
    --workload "arxiv_emis=${data_root}/navix_bitmap/arxiv-large/emis/throughput_10000/manifest.json" \
    --workload "arxiv_r=${data_root}/navix_bitmap/arxiv-large/r/throughput_10000/manifest.json" \
    --output "${run_root}/dataset_stats/workload_selectivity_summary.json"
  mark_done dataset_stats
}

analyze_all() {
  env RETRIEVE_RESULT_ROOT="${run_root}/gpu_graph" \
    "${repo_dir}/benchmarks/retrieve_workshop/gpu_graph/run_gpu_graph.sh" analyze
  env RETRIEVE_RESULT_ROOT="${run_root}/matched_recall" \
    "${repo_dir}/benchmarks/retrieve_workshop/matched_recall/run_matched_recall.sh" analyze
  env RETRIEVE_EXACT_RESULT_ROOT="${run_root}/exact_bitmap" \
    RETRIEVE_EXACT_DATA_ROOT="${data_root}/retrieve_workshop/exact_bitmap_a100" \
    RETRIEVE_ARXIV_BITMAP_NAMESPACE=arxiv-large RETRIEVE_ARXIV_DATASET_DIR=arxiv-for-fanns-large \
    "${repo_dir}/benchmarks/retrieve_workshop/exact_bitmap/run_exact.sh" analyze
  env RETRIEVE_RESOURCE_WORK_ROOT="${run_root}/resource_work" \
    "${repo_dir}/benchmarks/retrieve_workshop/resource_work/run.sh" analyze
}

bundle() {
  env MPLBACKEND=Agg "${python_bin}" "${script_dir}/bundle.py" \
    --run-root "${run_root}" --profile "${profile}"
}

case "${stage}" in
  preflight) preflight ;;
  download-arxiv) download_arxiv ;;
  build) build_binaries ;;
  test) test_gate ;;
  prepare) prepare_data ;;
  build-graphs) build_graphs ;;
  correctness) run_gpu_stage correctness ;;
  b0) run_gpu_stage b0 ;;
  matched-recall) run_matched ;;
  exact) run_exact ;;
  resource-work) run_resource ;;
  diagnostics) run_mechanism_diagnostics ;;
  dataset-stats) run_dataset_stats ;;
  analyze) analyze_all ;;
  bundle) bundle ;;
  all)
    preflight
    build_binaries
    test_gate
    prepare_data
    build_graphs
    run_gpu_stage correctness
    run_gpu_stage b0
    run_matched
    run_exact
    run_resource
    run_mechanism_diagnostics
    run_dataset_stats
    analyze_all
    bundle
    ;;
  *)
    echo "usage: $0 {preflight|download-arxiv|build|test|prepare|build-graphs|correctness|b0|matched-recall|exact|resource-work|diagnostics|dataset-stats|analyze|bundle|all}" >&2
    exit 2
    ;;
esac

printf '%s\n' "${run_root}"
