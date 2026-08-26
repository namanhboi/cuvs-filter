#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")"; pwd)
repo_dir=$(cd "${script_dir}/../../.."; pwd)
python_bin=${PYTHON:-/home/ubuntu/micromamba/envs/cuvs/bin/python}
data_root=${RETRIEVE_DATA_ROOT:-/data/retrieve_data}
reference_root=${RETRIEVE_A100_K10_RUN_ROOT:?set RETRIEVE_A100_K10_RUN_ROOT to the completed k=10 max_queries=2048 run}
result_root=${RETRIEVE_A100_K10_WD_SEED_RUN_ROOT:-"${reference_root}/k10_wd_seed_ablation"}
profile=${RETRIEVE_DATASET_PROFILE:-"${repo_dir}/benchmarks/retrieve_workshop/a100_paper/profiles/a100_yfcc10m_arxiv_large.json"}
bench_bin=${RETRIEVE_BENCH_BIN:-"${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"}
libcuvs=${RETRIEVE_LIBCUVS:-"${repo_dir}/cpp/build/libcuvs.so"}
selected=${RETRIEVE_K10_REFERENCE_SELECTED:-"${reference_root}/matched_recall/analysis/selected_points.csv"}
selected_provenance=${RETRIEVE_K10_REFERENCE_PROVENANCE:-"${reference_root}/matched_recall/analysis/provenance.json"}
stage=${1:-all}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHON="${python_bin}"
export RETRIEVE_DATA_ROOT="${data_root}"
export RETRIEVE_DATASET_PROFILE="${profile}"
export RETRIEVE_PROVENANCE_K=10
export RETRIEVE_PROVENANCE_MAX_QUERIES=2048
export RETRIEVE_HASH_CACHE="${result_root}/provenance/input_hash_cache.json"

mkdir -p "$(dirname "${result_root}")"
exec 9>"${result_root}.lock"
flock -n 9 || { echo "another k=10 W*D seed-ablation stage owns ${result_root}.lock" >&2; exit 2; }
mkdir -p "${result_root}/logs" "${result_root}/state"
exec > >(tee -a "${result_root}/logs/orchestrator.log") 2>&1

require_inputs() {
  local paths=(
    "${bench_bin}"
    "${libcuvs}"
    "${profile}"
    "${selected}"
    "${selected_provenance}"
    "${data_root}/yfcc-10M/base.10M.u8bin"
    "${data_root}/yfcc-10M/cagra_g64_ig128.index"
    "${data_root}/navix_bitmap/yfcc/correctness_1000/manifest.json"
    "${data_root}/navix_bitmap/yfcc/throughput_10000/manifest.json"
  )
  for path in "${paths[@]}"; do
    test -e "${path}" || { echo "missing k=10 W*D seed-ablation input: ${path}" >&2; exit 2; }
  done
}

preflight() {
  require_inputs
  "${python_bin}" - "${profile}" <<'PY'
import json
import pathlib
import sys

profile = json.loads(pathlib.Path(sys.argv[1]).read_text())
if int(profile["max_queries"]) != 2048:
    raise SystemExit("k=10 W*D seed ablation requires max_queries=2048")
spec = profile["datasets"]["yfcc"]
if int(spec["graph_degree"]) != 64:
    raise SystemExit("k=10 W*D seed ablation requires the YFCC degree-64 graph")
PY
  nvidia-smi --query-gpu=name,memory.total,mig.mode.current --format=csv,noheader
}

build_binaries() {
  (cd "${repo_dir}" && env PARALLEL_LEVEL=${PARALLEL_LEVEL:-12} \
    ./build.sh libcuvs tests bench-ann -n \
      --limit-tests=NEIGHBORS_ANN_CAGRA_FILTER_BITMAP_TEST \
      --limit-bench-ann="CUVS_CAGRA_ANN_BENCH" \
      --gpu-arch=80-real)
}

test_gate() {
  env LD_PRELOAD="${libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
    "${repo_dir}/cpp/build/gtests/NEIGHBORS_ANN_CAGRA_FILTER_BITMAP_TEST" \
      --gtest_filter='*BitmapSeededNavixSupportsSeedCapAboveResultWidth*:*BitmapSeededNavixSupportsSeedCapBelowResultWidth*' \
      --gtest_color=no
  env MPLBACKEND=Agg "${python_bin}" "${script_dir}/test_pipeline.py"
}

initialize() {
  "${python_bin}" "${script_dir}/wd_seed_ablation.py" initialize \
    --root "${result_root}" --data-root "${data_root}" \
    --reference-selected "${selected}" \
    --reference-provenance "${selected_provenance}"
  mkdir -p "${result_root}/provenance"
  "${python_bin}" "${repo_dir}/benchmarks/retrieve_workshop/gpu_graph/capture_provenance.py" \
    --result-root "${result_root}" --repo "${repo_dir}" --stage k10_wd_seed_ablation \
    --bench-bin "${bench_bin}" --libcuvs "${libcuvs}" --data-root "${data_root}"
}

raw_complete() {
  local manifest=$1
  local raw_dir=$2
  "${python_bin}" - "${manifest}" "${raw_dir}" <<'PY'
import collections
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
raw_dir = pathlib.Path(sys.argv[2])
repetitions = int(manifest["repetitions"])
searches = len(manifest["search_points"])
for shard in manifest["configs"]:
    path = raw_dir / f"shard_{int(shard['shard_index']):02d}.json"
    if not path.is_file():
        raise SystemExit(1)
    payload = json.loads(path.read_text())
    rows = [row for row in payload.get("benchmarks", []) if row.get("run_type") == "iteration"]
    counts = collections.Counter(int(row.get("repetition_index", -1)) for row in rows)
    if (
        len(rows) != searches * repetitions
        or counts != collections.Counter({index: searches for index in range(repetitions)})
        or any(row.get("error_occurred") or row.get("skipped") for row in rows)
    ):
        raise SystemExit(1)
PY
}

run_manifest() {
  local manifest=$1
  local group phase repetitions min_time raw_dir
  group=$("${python_bin}" - "${manifest}" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["group"])
PY
  )
  read -r phase repetitions < <("${python_bin}" - "${manifest}" <<'PY'
import json, pathlib, sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text())
print(p["phase"], p["repetitions"])
PY
  )
  min_time=0.10s
  test "${phase}" = correctness && min_time=0.01s
  raw_dir="${result_root}/raw/${group}/yfcc"
  if raw_complete "${manifest}" "${raw_dir}"; then
    echo "resume: retain complete group ${group}"
    return
  fi
  if test -d "${raw_dir}" && find "${raw_dir}" -type f -print -quit | grep -q .; then
    echo "refusing partial group ${group}; remove only ${raw_dir} before retrying" >&2
    exit 2
  fi
  mkdir -p "${raw_dir}"
  while IFS=$'\t' read -r shard config; do
    env LD_PRELOAD="${libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
      "${bench_bin}" --search --mode=throughput --threads=1 \
      --data_prefix="${data_root}" --index_prefix="${data_root}" \
      --benchmark_repetitions="${repetitions}" \
      --benchmark_min_time="${min_time}" --benchmark_min_warmup_time=0.01 \
      --benchmark_enable_random_interleaving=true \
      --benchmark_report_aggregates_only=false --benchmark_out_format=json \
      --benchmark_out="${raw_dir}/shard_${shard}.json" "${config}"
  done < <("${python_bin}" - "${manifest}" <<'PY'
import json, pathlib, sys
for row in json.loads(pathlib.Path(sys.argv[1]).read_text())["configs"]:
    print(f"{int(row['shard_index']):02d}\t{row['config']}")
PY
  )
}

run_pending() {
  while IFS= read -r manifest; do
    local group
    group=$("${python_bin}" - "${manifest}" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["group"])
PY
    )
    test -d "${result_root}/raw/${group}/yfcc" && raw_complete "${manifest}" "${result_root}/raw/${group}/yfcc" && continue
    run_manifest "${manifest}"
  done < <(find "${result_root}/configs" -path '*/yfcc/manifest.json' -type f | sort)
}

calibrate() {
  initialize
  run_pending
  local round
  for round in $(seq 0 47); do
    "${python_bin}" "${script_dir}/wd_seed_ablation.py" plan-next \
      --root "${result_root}" --data-root "${data_root}"
    if "${python_bin}" - "${result_root}/state/calibration_state.json" <<'PY'
import json, pathlib, sys
raise SystemExit(0 if json.loads(pathlib.Path(sys.argv[1]).read_text())["complete"] else 1)
PY
    then
      break
    fi
    run_pending
  done
  "${python_bin}" - "${result_root}/state/calibration_state.json" <<'PY'
import json, pathlib, sys
if not json.loads(pathlib.Path(sys.argv[1]).read_text())["complete"]:
    raise SystemExit("calibration exceeded 48 adaptive rounds")
PY
}

run_experiment() {
  calibrate
  "${python_bin}" "${script_dir}/wd_seed_ablation.py" prepare-finalists \
    --root "${result_root}" --data-root "${data_root}"
  run_pending
  "${python_bin}" "${script_dir}/wd_seed_ablation.py" prepare-controls \
    --root "${result_root}" --data-root "${data_root}"
  run_pending
  env MPLBACKEND=Agg "${python_bin}" "${script_dir}/wd_seed_ablation.py" analyze \
    --root "${result_root}"
  "${python_bin}" "${script_dir}/wd_seed_ablation.py" generate-diagnostics \
    --root "${result_root}" --data-root "${data_root}"
  run_pending
  env MPLBACKEND=Agg "${python_bin}" "${script_dir}/wd_seed_ablation.py" analyze \
    --root "${result_root}" --require-diagnostics
}

bundle() {
  "${python_bin}" "${script_dir}/wd_seed_ablation.py" bundle --root "${result_root}"
}

case "${stage}" in
  preflight) preflight ;;
  build) build_binaries ;;
  test) test_gate ;;
  initialize) initialize ;;
  run) run_experiment ;;
  analyze) env MPLBACKEND=Agg "${python_bin}" "${script_dir}/wd_seed_ablation.py" analyze --root "${result_root}" --require-diagnostics ;;
  bundle) bundle ;;
  all)
    preflight
    build_binaries
    test_gate
    run_experiment
    bundle
    ;;
  *)
    echo "usage: $0 {preflight|build|test|initialize|run|analyze|bundle|all}" >&2
    exit 2
    ;;
esac

printf '%s\n' "${result_root}"
