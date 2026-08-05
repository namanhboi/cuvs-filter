#!/bin/bash
set -euo pipefail
ulimit -c 0 || true

repo_dir=$(cd "$(dirname "$0")/../.."; pwd)
repo_parent=$(dirname "${repo_dir}")
analyzer="${repo_dir}/benchmarks/favor/analyze_multi_cta_retention_regression.py"
data_root="${repo_dir}/datasets"
result_root=${1:-"${repo_dir}/benchmarks/favor/results_multi_cta_retention_regression"}
worktree_root=${FAVOR_REGRESSION_WORKTREE_ROOT:-"${repo_parent}/cuvs-filter-regression-worktrees"}
baseline_label=${FAVOR_REGRESSION_BASELINE_LABEL:-old_79ca}
baseline_ref=${FAVOR_REGRESSION_BASELINE_REF:-79ca9e6c}
candidate_label=${FAVOR_REGRESSION_CANDIDATE_LABEL:-current_f080}
candidate_ref=${FAVOR_REGRESSION_CANDIDATE_REF:-f08044ac}
iterations=${FAVOR_REGRESSION_ITERATIONS:-1000}
repetitions=${FAVOR_REGRESSION_REPETITIONS:-5}
build_enabled=${FAVOR_REGRESSION_BUILD:-1}

if ! [[ "${iterations}" =~ ^[1-9][0-9]*$ ]]; then
  echo "FAVOR_REGRESSION_ITERATIONS must be a positive integer" >&2
  exit 1
fi
if ! [[ "${repetitions}" =~ ^[1-9][0-9]*$ ]]; then
  echo "FAVOR_REGRESSION_REPETITIONS must be a positive integer" >&2
  exit 1
fi
if [[ "${build_enabled}" != 0 && "${build_enabled}" != 1 ]]; then
  echo "FAVOR_REGRESSION_BUILD must be 0 or 1" >&2
  exit 1
fi

mkdir -p "${worktree_root}" "${result_root}/configs" "${result_root}/raw"

ensure_worktree() {
  local label=$1
  local ref=$2
  local path="${worktree_root}/${label}"
  local expected
  expected=$(git -C "${repo_dir}" rev-parse "${ref}^{commit}")
  if [[ -e "${path}/.git" ]]; then
    local actual
    actual=$(git -C "${path}" rev-parse HEAD)
    if [[ "${actual}" != "${expected}" ]]; then
      echo "Existing worktree ${path} is ${actual}, expected ${expected}" >&2
      exit 1
    fi
  else
    git -C "${repo_dir}" worktree add --detach "${path}" "${expected}"
  fi
  printf '%s\n' "${path}"
}

ensure_build() {
  local label=$1
  local path=$2
  local bench="${path}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"
  local library="${path}/cpp/build/libcuvs.so"
  if [[ -x "${bench}" && -f "${library}" ]]; then
    echo "Reusing complete ${label} build at ${path}"
    return
  fi
  if [[ "${build_enabled}" != 1 ]]; then
    echo "Missing ${label} build and FAVOR_REGRESSION_BUILD=0" >&2
    exit 1
  fi
  echo "Building ${label} at ${path}"
  (
    cd "${path}"
    ./build.sh libcuvs bench-ann -n --gpu-arch=89-real \
      --limit-bench-ann=CUVS_CAGRA_ANN_BENCH
  )
}

baseline_path=$(ensure_worktree "${baseline_label}" "${baseline_ref}")
candidate_path=$(ensure_worktree "${candidate_label}" "${candidate_ref}")
baseline_commit=$(git -C "${baseline_path}" rev-parse HEAD)
candidate_commit=$(git -C "${candidate_path}" rev-parse HEAD)

ensure_build "${baseline_label}" "${baseline_path}"
ensure_build "${candidate_label}" "${candidate_path}"

python "${analyzer}" generate \
  --config-dir "${result_root}/configs" \
  --data-root "${data_root}"
python "${analyzer}" manifest \
  --result-root "${result_root}" \
  --baseline "${baseline_label}" \
  --baseline-commit "${baseline_commit}" \
  --baseline-binary "${baseline_path}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH" \
  --baseline-library "${baseline_path}/cpp/build/libcuvs.so" \
  --candidate "${candidate_label}" \
  --candidate-commit "${candidate_commit}" \
  --candidate-binary "${candidate_path}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH" \
  --candidate-library "${candidate_path}/cpp/build/libcuvs.so" \
  --iterations "${iterations}" \
  --repetitions "${repetitions}"

run_build() {
  local label=$1
  local path=$2
  local raw_dir="${result_root}/raw/${label}"
  local bench="${path}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"
  local library="${path}/cpp/build/libcuvs.so"
  mkdir -p "${raw_dir}"
  for config in "${result_root}"/configs/*.json; do
    local key
    key=$(basename "${config}" .json)
    local raw="${raw_dir}/${key}.json"
    if python "${analyzer}" validate \
      --build "${label}" \
      --dataset-key "${key}" \
      --config "${config}" \
      --raw "${raw}" \
      --iterations "${iterations}" \
      --repetitions "${repetitions}" >/dev/null 2>&1; then
      echo "Reusing validated ${raw}"
      continue
    fi
    echo "Running ${label} ${key}"
    local log="${raw_dir}/${key}.log"
    set +e
    env LD_PRELOAD="${library}" "${bench}" \
      --search \
      --mode=latency \
      --threads=1 \
      --data_prefix="${data_root}" \
      --index_prefix="${data_root}" \
      --benchmark_repetitions="${repetitions}" \
      --benchmark_min_time="${iterations}x" \
      --benchmark_min_warmup_time=0.1 \
      --benchmark_enable_random_interleaving=true \
      --benchmark_report_aggregates_only=false \
      --benchmark_out_format=json \
      --benchmark_out="${raw}" \
      "${config}" >"${log}" 2>&1
    local status=$?
    set -e
    python "${analyzer}" validate \
      --build "${label}" \
      --dataset-key "${key}" \
      --config "${config}" \
      --raw "${raw}" \
      --iterations "${iterations}" \
      --repetitions "${repetitions}"
    if [[ "${status}" != 0 ]]; then
      echo "${label} ${key}: benchmark exited ${status} after writing a complete result" >&2
    fi
  done
}

if [[ ! -f "${result_root}/gpu_preflight.csv" ]]; then
  nvidia-smi --query-gpu=name,compute_cap,temperature.gpu,utilization.gpu,clocks.sm,clocks.mem,power.draw \
    --format=csv >"${result_root}/gpu_preflight.csv"
fi

run_build "${baseline_label}" "${baseline_path}"
run_build "${candidate_label}" "${candidate_path}"

python "${analyzer}" analyze \
  --result-root "${result_root}" \
  --baseline "${baseline_label}" \
  --candidate "${candidate_label}" \
  --iterations "${iterations}" \
  --repetitions "${repetitions}"
