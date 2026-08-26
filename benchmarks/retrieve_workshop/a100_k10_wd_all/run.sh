#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")"; pwd)
repo_dir=$(cd "${script_dir}/../../.."; pwd)
python_bin=${PYTHON:-/home/ubuntu/micromamba/envs/cuvs/bin/python}
data_root=${RETRIEVE_DATA_ROOT:-/data/retrieve_data}
reference_root=${RETRIEVE_A100_K10_RUN_ROOT:?set RETRIEVE_A100_K10_RUN_ROOT to the completed k=10 max_queries=2048 run}
result_root=${RETRIEVE_A100_K10_WD_ALL_RUN_ROOT:-"${reference_root}/k10_wd_all"}
profile=${RETRIEVE_DATASET_PROFILE:-"${repo_dir}/benchmarks/retrieve_workshop/a100_paper/profiles/a100_yfcc10m_arxiv_large.json"}
bench_bin=${RETRIEVE_BENCH_BIN:-"${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"}
libcuvs=${RETRIEVE_LIBCUVS:-"${repo_dir}/cpp/build/libcuvs.so"}
selected=${RETRIEVE_K10_REFERENCE_SELECTED:-"${reference_root}/matched_recall/analysis/selected_points.csv"}
selected_provenance=${RETRIEVE_K10_REFERENCE_PROVENANCE:-"${reference_root}/matched_recall/analysis/provenance.json"}
reference_bundle=${RETRIEVE_A100_REFERENCE_BUNDLE:-"${reference_root}/navix_refined_bundle/paper_gpu_bundle"}
merged_bundle=${RETRIEVE_A100_WD_MERGED_ROOT:-"${result_root}/merged_paper_gpu_bundle/paper_gpu_bundle"}
stage=${1:-all}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHON="${python_bin}"
export RETRIEVE_DATA_ROOT="${data_root}"
export RETRIEVE_DATASET_PROFILE="${profile}"
export RETRIEVE_PROVENANCE_K=10
export RETRIEVE_PROVENANCE_MAX_QUERIES=2048
export RETRIEVE_MATCHED_K=10
export RETRIEVE_HASH_CACHE="${result_root}/provenance/input_hash_cache.json"

mkdir -p "$(dirname "${result_root}")"
exec 9>"${result_root}.lock"
flock -n 9 || { echo "another all-workload W*D stage owns ${result_root}.lock" >&2; exit 2; }
mkdir -p "${result_root}/logs" "${result_root}/provenance"
exec > >(tee -a "${result_root}/logs/orchestrator.log") 2>&1

require_inputs() {
  local paths=("${profile}" "${selected}" "${selected_provenance}")
  while IFS= read -r path; do paths+=("${path}"); done < <(
    "${python_bin}" - "${profile}" "${data_root}" <<'PY'
import json, pathlib, sys
profile=json.loads(pathlib.Path(sys.argv[1]).read_text())
root=pathlib.Path(sys.argv[2])
for spec in profile["datasets"].values():
    print(root / spec["base_file"])
    print(root / spec["index_file"])
    bitmap=root / spec["bitmap_directory"]
    print(bitmap / "correctness_1000/manifest.json")
    print(bitmap / "throughput_10000/manifest.json")
PY
  )
  for path in "${paths[@]}"; do
    test -e "${path}" || { echo "missing all-workload W*D input: ${path}" >&2; exit 2; }
  done
}

preflight() {
  require_inputs
  "${python_bin}" - "${profile}" <<'PY'
import json, pathlib, sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text())
if int(p["max_queries"]) != 2048:
    raise SystemExit("W*D rerun requires max_queries=2048")
if set(p["datasets"]) != {"yfcc", "em", "emis", "r"}:
    raise SystemExit("profile must define yfcc, em, emis, and r")
for workload, spec in p["datasets"].items():
    if int(spec["graph_degree"]) != 64 or int(spec["intermediate_graph_degree"]) != 128:
        raise SystemExit(f"{workload} must use the degree-64/intermediate-128 graph")
PY
  local gpu
  gpu=$(nvidia-smi --query-gpu=name,memory.total,mig.mode.current --format=csv,noheader)
  printf '%s\n' "${gpu}"
  test "$(printf '%s\n' "${gpu}" | wc -l)" -eq 1 || {
    echo "preflight requires exactly one visible GPU" >&2; exit 2;
  }
  printf '%s\n' "${gpu}" | grep -q 'NVIDIA A100 80GB' || {
    echo "preflight requires an A100 80GB" >&2; exit 2;
  }
  printf '%s\n' "${gpu}" | grep -q 'Disabled' || {
    echo "preflight requires non-MIG mode" >&2; exit 2;
  }
}

build_binaries() {
  (cd "${repo_dir}" && env PARALLEL_LEVEL=${PARALLEL_LEVEL:-12} ./build.sh libcuvs tests bench-ann -n --limit-tests=NEIGHBORS_ANN_CAGRA_FILTER_BITMAP_TEST --limit-bench-ann="CUVS_CAGRA_ANN_BENCH" --gpu-arch=80-real)
}

test_gate() {
  env LD_PRELOAD="${libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
    "${repo_dir}/cpp/build/gtests/NEIGHBORS_ANN_CAGRA_FILTER_BITMAP_TEST" \
    --gtest_filter='*BitmapSeededNavixSupportsSeedCapAboveResultWidth*:*BitmapSeededNavixSupportsSeedCapBelowResultWidth*' \
    --gtest_color=no
  env MPLBACKEND=Agg "${python_bin}" "${script_dir}/test_pipeline.py"
  env -u RETRIEVE_DATASET_PROFILE \
    RETRIEVE_MATCHED_METHODS=default_cagra,default_cagra_accumulator,navix_reference \
    RETRIEVE_MATCHED_NAVIX_SEED_POLICY=k RETRIEVE_MATCHED_ALLOW_SHALLOW_NAVIX=0 \
    "${python_bin}" "${repo_dir}/benchmarks/retrieve_workshop/matched_recall/test_matched_recall.py"
}

capture_provenance() {
  local root=$1 stage_name=$2
  mkdir -p "${root}/provenance"
  "${python_bin}" "${repo_dir}/benchmarks/retrieve_workshop/gpu_graph/capture_provenance.py" \
    --result-root "${root}" --repo "${repo_dir}" --stage "${stage_name}" \
    --bench-bin "${bench_bin}" --libcuvs "${libcuvs}" --data-root "${data_root}"
}

raw_complete() {
  local manifest=$1 raw_dir=$2
  "${python_bin}" - "${manifest}" "${raw_dir}" <<'PY'
import collections, json, pathlib, sys
m=json.loads(pathlib.Path(sys.argv[1]).read_text())
raw=pathlib.Path(sys.argv[2])
repetitions=int(m["repetitions"])
searches=len(m["search_points"])
for shard in m["configs"]:
    path=raw / f"shard_{int(shard['shard_index']):02d}.json"
    if not path.is_file(): raise SystemExit(1)
    payload=json.loads(path.read_text())
    rows=[row for row in payload.get("benchmarks",[]) if row.get("run_type")=="iteration"]
    counts=collections.Counter(int(row.get("repetition_index",-1)) for row in rows)
    if len(rows)!=searches*repetitions or counts!=collections.Counter({i:searches for i in range(repetitions)}) or any(row.get("error_occurred") or row.get("skipped") for row in rows):
        raise SystemExit(1)
PY
}

run_manifest() {
  local experiment_root=$1 manifest=$2
  local group workload phase repetitions min_time raw_dir
  read -r group workload phase repetitions < <("${python_bin}" - "${manifest}" <<'PY'
import json,pathlib,sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text())
print(p["group"],p["workload"],p["phase"],p["repetitions"])
PY
  )
  min_time=0.10s
  test "${phase}" = correctness && min_time=0.01s
  raw_dir="${experiment_root}/raw/${group}/${workload}"
  if raw_complete "${manifest}" "${raw_dir}"; then
    echo "resume: retain complete ${group}/${workload}"
    return
  fi
  if test -d "${raw_dir}" && find "${raw_dir}" -type f -print -quit | grep -q .; then
    echo "refusing partial group ${group}/${workload}; remove only ${raw_dir} before retrying" >&2
    exit 2
  fi
  mkdir -p "${raw_dir}"
  while IFS=$'\t' read -r shard config; do
    env LD_PRELOAD="${libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
      "${bench_bin}" --search --mode=throughput --threads=1 \
      --data_prefix="${data_root}" --index_prefix="${data_root}" \
      --benchmark_repetitions="${repetitions}" --benchmark_min_time="${min_time}" \
      --benchmark_min_warmup_time=0.01 --benchmark_enable_random_interleaving=true \
      --benchmark_report_aggregates_only=false --benchmark_out_format=json \
      --benchmark_out="${raw_dir}/shard_${shard}.json" "${config}"
  done < <("${python_bin}" - "${manifest}" <<'PY'
import json,pathlib,sys
for row in json.loads(pathlib.Path(sys.argv[1]).read_text())["configs"]:
    print(f"{int(row['shard_index']):02d}\t{row['config']}")
PY
  )
}

run_tree() {
  local experiment_root=$1
  while IFS= read -r manifest; do run_manifest "${experiment_root}" "${manifest}"; done < <(
    find "${experiment_root}/configs" -mindepth 3 -maxdepth 3 -name manifest.json | sort
  )
}

run_frontier() {
  "${python_bin}" "${script_dir}/workflow.py" initialize --root "${result_root}" \
    --data-root "${data_root}" --reference-selected "${selected}" \
    --reference-provenance "${selected_provenance}"
  capture_provenance "${result_root}/frontier" wd_frontier
  run_tree "${result_root}/frontier"
  env MPLBACKEND=Agg "${python_bin}" "${repo_dir}/benchmarks/retrieve_workshop/gpu_graph/analyze_gpu_graph.py" \
    --result-root "${result_root}/frontier" --output "${result_root}/frontier/analysis" \
    --require-group correctness --require-group b0
  "${python_bin}" "${script_dir}/workflow.py" validate-frontier --root "${result_root}"
}

run_matched() {
  env RETRIEVE_RESULT_ROOT="${result_root}/matched_recall" \
    RETRIEVE_MATCHED_METHODS=navix_reference \
    RETRIEVE_MATCHED_NAVIX_SEED_POLICY=wd RETRIEVE_MATCHED_ALLOW_SHALLOW_NAVIX=1 \
    RETRIEVE_MATCHED_BASELINE_SUMMARY="${result_root}/frontier/analysis/summary_points.csv" \
    RETRIEVE_MATCHED_BASELINE_PROVENANCE="${result_root}/frontier/analysis/provenance.json" \
    "${repo_dir}/benchmarks/retrieve_workshop/matched_recall/run_matched_recall.sh" all
}

run_controls() {
  local selected_wd="${result_root}/matched_recall/analysis/selected_points.csv"
  "${python_bin}" "${script_dir}/workflow.py" create-controls --root "${result_root}" \
    --data-root "${data_root}" --selected "${selected_wd}"
  capture_provenance "${result_root}/controls" wd_seed_controls
  run_tree "${result_root}/controls"
}

run_diagnostics() {
  local selected_wd="${result_root}/matched_recall/analysis/selected_points.csv"
  "${python_bin}" "${script_dir}/workflow.py" create-diagnostics --root "${result_root}" \
    --data-root "${data_root}" --selected "${selected_wd}"
  capture_provenance "${result_root}/diagnostics" wd_diagnostics
  mkdir -p "${result_root}/diagnostics/raw" "${result_root}/diagnostics/resources"
  for workload in yfcc em emis r; do
    local raw="${result_root}/diagnostics/raw/${workload}.json"
    local resource_raw="${result_root}/diagnostics/raw/${workload}_resources.json"
    local resource_log="${result_root}/diagnostics/resources/${workload}.log"
    if ! test -s "${raw}"; then
      env LD_PRELOAD="${libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
        "${bench_bin}" --search --mode=throughput --threads=1 \
        --data_prefix="${data_root}" --index_prefix="${data_root}" \
        --benchmark_repetitions=1 --benchmark_min_time=0.001s \
        --benchmark_min_warmup_time=0.001 --benchmark_report_aggregates_only=false \
        --benchmark_out_format=json --benchmark_out="${raw}" \
        "${result_root}/diagnostics/configs/${workload}.json"
    fi
    if ! test -s "${resource_raw}" || ! test -s "${resource_log}"; then
      env CUVS_CAGRA_LOG_RESOURCES=1 LD_PRELOAD="${libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
        "${bench_bin}" --search --mode=throughput --threads=1 \
        --data_prefix="${data_root}" --index_prefix="${data_root}" \
        --benchmark_repetitions=1 --benchmark_min_time=0.001s \
        --benchmark_min_warmup_time=0.001 --benchmark_report_aggregates_only=false \
        --benchmark_out_format=json --benchmark_out="${resource_raw}" \
        "${result_root}/diagnostics/resource_configs/${workload}.json" 2>"${resource_log}"
    fi
  done
}

analyze() {
  env MPLBACKEND=Agg "${python_bin}" "${script_dir}/workflow.py" analyze --root "${result_root}"
}

bundle() {
  "${python_bin}" "${script_dir}/workflow.py" bundle --root "${result_root}"
}

merge() {
  test -e "${reference_bundle}" || { echo "missing reviewed reference bundle: ${reference_bundle}" >&2; exit 2; }
  local wd_bundle="${result_root}/paper_gpu_bundle_k10_wd_all"
  test -d "${wd_bundle}" || { echo "run bundle before merge" >&2; exit 2; }
  "${python_bin}" "${script_dir}/workflow.py" merge --reference-bundle "${reference_bundle}" \
    --wd-bundle "${wd_bundle}" --output "${merged_bundle}"
}

case "${stage}" in
  preflight) preflight ;;
  build) build_binaries ;;
  test) test_gate ;;
  frontier) run_frontier ;;
  matched) run_matched ;;
  controls) run_controls ;;
  diagnostics) run_diagnostics ;;
  analyze) analyze ;;
  bundle) bundle ;;
  merge) merge ;;
  all)
    preflight
    build_binaries
    test_gate
    capture_provenance "${result_root}" wd_all
    run_frontier
    run_matched
    run_controls
    run_diagnostics
    analyze
    bundle
    ;;
  *) echo "usage: $0 {preflight|build|test|frontier|matched|controls|diagnostics|analyze|bundle|merge|all}" >&2; exit 2 ;;
esac

printf '%s\n' "${result_root}"
