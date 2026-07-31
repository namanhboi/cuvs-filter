#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/../.."; pwd)
source_root="${repo_dir}/benchmarks/favor/results_retention_safe_single_full"
result_root=${1:-"${repo_dir}/benchmarks/favor/results_automatic_retention_single_full"}

extract_matching_cells() {
  python - "$1" "$2" <<'PY'
import json
import sys

def cells(path):
    payload = json.load(open(path, encoding="utf-8"))
    return {
        (
            int(row["itopk"]),
            int(row["search_width"]),
            int(row.get("max_iterations", 0)),
            int(row.get("thread_block_size", 0)),
        )
        for row in payload["index"][0]["search_params"]
        if row.get("favor_penalty_mode") == "cagra_retention_safe"
    }

latency = cells(sys.argv[1])
throughput = cells(sys.argv[2])
if not latency:
    raise SystemExit(f"no retention-safe cells in {sys.argv[1]}")
if latency != throughput:
    raise SystemExit("latency and throughput source tuning cells differ")
print(" ".join(":".join(map(str, cell)) for cell in sorted(latency)))
PY
}

run_dataset() {
  local source_key=$1
  local dataset_name=$2
  local result_prefix=$3
  local plot_title=$4
  local query_file=$5
  local base_file=$6
  local dtype=$7
  local subset_size=$8

  for selectivity in 1 10 50 90; do
    local encoded_selectivity
    printf -v encoded_selectivity "%02d" "${selectivity}"
    local source_config_dir="${source_root}/${source_key}/configs"
    local latency_config="${source_config_dir}/${result_prefix}_s${encoded_selectivity}_nq10.json"
    local throughput_config="${source_config_dir}/${result_prefix}_s${encoded_selectivity}_nq10000.json"
    local cells
    cells=$(extract_matching_cells "${latency_config}" "${throughput_config}")

    FAVOR_BATCH_SIZES="10 10000" \
    FAVOR_SELECTIVITIES="${selectivity}" \
    FAVOR_SEARCH_CELLS="${cells}" \
    FAVOR_SEARCH_ALGO=single_cta \
    FAVOR_BENCHMARK_MODES=favor_retention_safe \
    FAVOR_PENALTY_LAMBDAS=1 \
    FAVOR_RETENTION_FRACTIONS=0 \
    FAVOR_BENCHMARK_REPETITIONS=1 \
    FAVOR_BENCHMARK_MIN_TIME=0.2s \
    FAVOR_BENCHMARK_WARMUP_TIME=0.1 \
      "${repo_dir}/benchmarks/favor/run_benchmarks.sh" \
        "${repo_dir}/datasets" \
        "${result_root}/${source_key}" \
        "${dataset_name}" \
        "${result_prefix}" \
        "${plot_title}" \
        "${query_file}" \
        "${base_file}" \
        "${dtype}" \
        "${subset_size}"
  done
}

run_dataset sift sift-128-euclidean sift SIFT-1M query.fbin base.fbin float 0
run_dataset gist gist-960-euclidean gist GIST-1M query_10000.fbin base.fbin float 0
run_dataset bigann1m bigann-1M bigann1m BIGANN-1M \
  query.public.10K.u8bin base.10M.u8bin uint8 1000000
run_dataset bigann10m bigann-10M bigann10m BIGANN-10M \
  query.public.10K.u8bin base.10M.u8bin uint8 0
run_dataset msturing1m msturing-1M msturing1m MSTuring-1M query.fbin base.fbin float 0
run_dataset msturing10m msturing-10M msturing10m MSTuring-10M query.fbin base.fbin float 0
