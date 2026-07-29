#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/../.."; pwd)
result_root=${1:-"${repo_dir}/benchmarks/favor/results_retention_safe_multi_confirmed"}

run_dataset() {
  local dataset_name=$1
  local result_prefix=$2
  local plot_title=$3
  local query_file=$4
  local base_file=$5
  local dtype=$6
  local subset_size=$7
  local itopk_values=$8

  FAVOR_BATCH_SIZES=1 \
  FAVOR_SELECTIVITIES="1 10 50 90" \
  FAVOR_ITOPK_VALUES="${itopk_values}" \
  FAVOR_SEARCH_WIDTHS=1 \
  FAVOR_SEARCH_ALGO=multi_cta \
  FAVOR_BENCHMARK_MODES="default favor_retention_safe" \
  FAVOR_PENALTY_LAMBDAS=1 \
  FAVOR_BENCHMARK_REPETITIONS=1 \
  FAVOR_BENCHMARK_MIN_TIME=0.2s \
  FAVOR_BENCHMARK_WARMUP_TIME=0.1 \
    "${repo_dir}/benchmarks/favor/run_benchmarks.sh" \
      "${repo_dir}/datasets" \
      "${result_root}/${result_prefix}" \
      "${dataset_name}" \
      "${result_prefix}" \
      "${plot_title}" \
      "${query_file}" \
      "${base_file}" \
      "${dtype}" \
      "${subset_size}"

  python "${repo_dir}/benchmarks/favor/plot_results.py" \
    --result-dir "${result_root}/${result_prefix}" \
    --result-prefix "${result_prefix}" \
    --plot-title "${plot_title}" \
    --selectivities 1 10 50 90 \
    --latency-derived-qps \
    --latency-batch-size 1 \
    --latency-unit us \
    --cta-mode MULTI_CTA \
    --target-recall 0.99 \
    --zero-y
}

run_dataset sift-128-euclidean sift SIFT-1M query.fbin base.fbin float 0 \
  "32 64 128 256 512"
run_dataset gist-960-euclidean gist GIST-1M query.fbin base.fbin float 0 \
  "32 64 128 256 512 640 768 1024"
run_dataset bigann-1M bigann1m BIGANN-1M query.public.10K.u8bin base.10M.u8bin uint8 1000000 \
  "32 64 128 256 512"
run_dataset bigann-10M bigann10m BIGANN-10M query.public.10K.u8bin base.10M.u8bin uint8 0 \
  "32 64 128 256 512"
run_dataset msturing-1M msturing1m MSTuring-1M query.fbin base.fbin float 0 \
  "32 64 128 256 512 640 768 1024"
run_dataset msturing-10M msturing10m MSTuring-10M query.fbin base.fbin float 0 \
  "32 64 128 256 512 1024 1536"
