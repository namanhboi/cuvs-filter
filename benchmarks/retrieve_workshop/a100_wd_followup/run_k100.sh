#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")"; pwd)
repo_dir=$(cd "${script_dir}/../../.."; pwd)
python_bin=${PYTHON:-/home/ubuntu/micromamba/envs/cuvs/bin/python}
data_root=${RETRIEVE_DATA_ROOT:-/data/retrieve_data}
profile=${RETRIEVE_DATASET_PROFILE:-"${repo_dir}/benchmarks/retrieve_workshop/a100_k100/profiles/a100_yfcc10m_arxiv_large_k100.json"}
reference_bundle=${RETRIEVE_A100_K100_REFERENCE_BUNDLE:-"${HOME}/a100_k100_matched_paper_gpu_bundle.tar.gz"}
run_tag=${RETRIEVE_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
run_root=${RETRIEVE_A100_K100_WD_RUN_ROOT:-"/data/retrieve_workshop_runs/a100_k100_wd_${run_tag}"}
bench_bin=${RETRIEVE_BENCH_BIN:-"${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"}
libcuvs=${RETRIEVE_LIBCUVS:-"${repo_dir}/cpp/build/libcuvs.so"}
bundle_root=${RETRIEVE_A100_K100_WD_BUNDLE:-"${run_root}/paper_gpu_bundle_k100_wd"}
bundle_archive=${RETRIEVE_A100_K100_WD_ARCHIVE:-"${HOME}/$(basename "${run_root}")_paper_gpu_bundle.tar.gz"}
stage=${1:-all}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHON="${python_bin}"
export MPLBACKEND=${MPLBACKEND:-Agg}
export RETRIEVE_DATA_ROOT="${data_root}"
export RETRIEVE_DATASET_PROFILE="${profile}"
export RETRIEVE_BENCH_BIN="${bench_bin}"
export RETRIEVE_LIBCUVS="${libcuvs}"
export RETRIEVE_PROVENANCE_K=100
export RETRIEVE_PROVENANCE_MAX_QUERIES=2048
export RETRIEVE_RESUME_COMPLETE=1
export RETRIEVE_HASH_CACHE="${run_root}/provenance/input_hash_cache.json"

mkdir -p "$(dirname "${run_root}")"
exec 9>"${run_root}.lock"
flock -n 9 || { echo "another k=100 W*D run owns ${run_root}.lock" >&2; exit 2; }
mkdir -p "${run_root}/logs" "${run_root}/state"
exec > >(tee -a "${run_root}/logs/orchestrator.log") 2>&1

done_marker() { printf '%s/.done/%s\n' "${run_root}" "$1"; }
is_done() { test -f "$(done_marker "$1")"; }
mark_done() {
  mkdir -p "${run_root}/.done"
  date -u +%Y-%m-%dT%H:%M:%SZ >"$(done_marker "$1")"
}

pin_contract() {
  "${python_bin}" - "${run_root}" "${repo_dir}" "${profile}" "${reference_bundle}" <<'PY'
import hashlib
import json
import pathlib
import subprocess
import sys

root, repo, profile, reference = map(pathlib.Path, sys.argv[1:])

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

payload = {
    "schema_version": 1,
    "experiment": "a100_k100_navix_wd_frontier",
    "git_head": subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip(),
    "k": 100,
    "max_queries": 2048,
    "graph_degree": 64,
    "seed_policy": "search_width * graph_degree",
    "profile": str(profile.resolve()),
    "profile_sha256": digest(profile),
    "reference_bundle": str(reference.resolve()),
    "reference_sha256": digest(reference),
}
destination = root / "state/contract.json"
if destination.exists():
    if json.loads(destination.read_text()) != payload:
        raise SystemExit("immutable k=100 W*D run contract changed")
else:
    destination.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

require_inputs() {
  test -f "${profile}" || { echo "missing profile: ${profile}" >&2; exit 2; }
  test -f "${reference_bundle}" || {
    echo "missing reviewed k=100 reference bundle: ${reference_bundle}" >&2
    exit 2
  }
  "${python_bin}" - "${profile}" "${data_root}" <<'PY'
import json
import pathlib
import sys

profile = json.loads(pathlib.Path(sys.argv[1]).read_text())
root = pathlib.Path(sys.argv[2])
if int(profile["max_queries"]) != 2048:
    raise SystemExit("k=100 W*D follow-up requires max_queries=2048")
if set(profile["datasets"]) != {"yfcc", "em", "emis", "r"}:
    raise SystemExit("profile must define yfcc, em, emis, and r")
for workload, spec in profile["datasets"].items():
    if int(spec["graph_degree"]) != 64:
        raise SystemExit(f"{workload} must use graph degree 64")
    required = [
        root / spec["base_file"],
        root / spec["index_file"],
        root / spec["bitmap_directory"] / "correctness_1000/manifest.json",
        root / spec["bitmap_directory"] / "throughput_10000/manifest.json",
    ]
    for path in required:
        if not path.is_file():
            raise SystemExit(f"missing required input: {path}")
PY
}

preflight() {
  is_done preflight && return
  require_inputs
  "${python_bin}" "${repo_dir}/benchmarks/retrieve_workshop/a100_paper/preflight.py" \
    --repo "${repo_dir}" --data-root "${data_root}" --run-root "${run_root}" \
    --profile "${profile}" --minimum-free-gib "${RETRIEVE_WD_MINIMUM_FREE_GIB:-5}"
  pin_contract
  mark_done preflight
}

build_binaries() {
  is_done build && return
  (cd "${repo_dir}" && env PARALLEL_LEVEL=${PARALLEL_LEVEL:-12} \
    ./build.sh libcuvs tests bench-ann -n \
      --limit-tests=NEIGHBORS_ANN_CAGRA_FILTER_BITMAP_TEST \
      --limit-bench-ann=CUVS_CAGRA_ANN_BENCH --gpu-arch=80-real)
  test -x "${bench_bin}" && test -f "${libcuvs}"
  mark_done build
}

test_gate() {
  is_done test && return
  env LD_PRELOAD="${libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
    "${repo_dir}/cpp/build/gtests/NEIGHBORS_ANN_CAGRA_FILTER_BITMAP_TEST" \
    --gtest_filter='*BitmapSeededDefaultSupportsIndependentSeedCaps*:*BitmapSeededNavixSupportsSeedCapAboveResultWidth*:*BitmapSeededNavixSupportsSeedCapBelowResultWidth*' \
    --gtest_color=no
  env -u RETRIEVE_DATASET_PROFILE "${python_bin}" \
    "${repo_dir}/benchmarks/retrieve_workshop/gpu_graph/test_pipeline.py"
  env -u RETRIEVE_DATASET_PROFILE "${python_bin}" \
    "${script_dir}/test_pipeline.py"
  mark_done test
}

run_frontier() {
  is_done frontier && return
  env RETRIEVE_RESULT_ROOT="${run_root}/frontier" \
    "${repo_dir}/benchmarks/retrieve_workshop/gpu_graph/run_gpu_graph.sh" all \
      --k=100 --cartesian-b0 --methods=navix_reference --seed-policy=wd
  mark_done frontier
}

run_matched() {
  is_done matched && return
  test -f "${run_root}/frontier/analysis/summary_points.csv" || {
    echo "run the W*D frontier before matched recall" >&2
    exit 2
  }
  env RETRIEVE_RESULT_ROOT="${run_root}/matched_recall" \
    RETRIEVE_MATCHED_K=100 \
    RETRIEVE_MATCHED_METHODS=navix_reference \
    RETRIEVE_MATCHED_NAVIX_SEED_POLICY=wd \
    RETRIEVE_MATCHED_ALLOW_SHALLOW_NAVIX=1 \
    RETRIEVE_MATCHED_BASELINE_SUMMARY="${run_root}/frontier/analysis/summary_points.csv" \
    RETRIEVE_MATCHED_BASELINE_PROVENANCE="${run_root}/frontier/analysis/provenance.json" \
    "${repo_dir}/benchmarks/retrieve_workshop/matched_recall/run_matched_recall.sh" all
  mark_done matched
}

capture_controls_provenance() {
  "${python_bin}" "${repo_dir}/benchmarks/retrieve_workshop/gpu_graph/capture_provenance.py" \
    --result-root "${run_root}/controls" --repo "${repo_dir}" \
    --stage k100_seed_policy_controls --bench-bin "${bench_bin}" \
    --libcuvs "${libcuvs}" --data-root "${data_root}"
}

raw_complete() {
  local raw=$1 config=$2
  "${python_bin}" - "${raw}" "${config}" <<'PY'
import collections
import json
import pathlib
import sys

raw = json.loads(pathlib.Path(sys.argv[1]).read_text())
config = json.loads(pathlib.Path(sys.argv[2]).read_text())
searches = sum(len(index["search_params"]) for index in config["index"])
rows = [row for row in raw.get("benchmarks", []) if row.get("run_type") == "iteration"]
counts = collections.Counter(int(row.get("repetition_index", -1)) for row in rows)
valid = (
    searches == 2
    and len(rows) == searches * 3
    and counts == collections.Counter({0: searches, 1: searches, 2: searches})
    and all(not row.get("error_occurred") and not row.get("skipped") for row in rows)
)
raise SystemExit(0 if valid else 1)
PY
}

run_control_manifest() {
  local manifest=$1
  local group workload
  read -r group workload < <("${python_bin}" - "${manifest}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(payload["group"], payload["workload"])
PY
  )
  local destination="${run_root}/controls/raw/${group}/${workload}"
  mkdir -p "${destination}"
  while IFS=$'\t' read -r shard config; do
    local output="${destination}/shard_${shard}.json"
    if test -e "${output}"; then
      if raw_complete "${output}" "${config}"; then
        echo "resume: retain complete ${output}"
        continue
      fi
      echo "incomplete control output; remove only ${output} and rerun" >&2
      exit 2
    fi
    env LD_PRELOAD="${libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
      "${bench_bin}" --search --mode=throughput --threads=1 \
      --data_prefix="${data_root}" --index_prefix="${data_root}" \
      --benchmark_repetitions=3 --benchmark_min_time=0.10s \
      --benchmark_min_warmup_time=0.01 \
      --benchmark_enable_random_interleaving=true \
      --benchmark_report_aggregates_only=false \
      --benchmark_out_format=json --benchmark_out="${output}" "${config}"
    raw_complete "${output}" "${config}"
  done < <("${python_bin}" - "${manifest}" <<'PY'
import json
import pathlib
import sys

for row in json.loads(pathlib.Path(sys.argv[1]).read_text())["configs"]:
    print(f"{int(row['shard_index']):02d}\t{row['config']}")
PY
  )
}

run_controls() {
  is_done controls && return
  local selected="${run_root}/matched_recall/analysis/selected_points.csv"
  test -f "${selected}" || { echo "run matched recall before paired controls" >&2; exit 2; }
  if ! test -f "${run_root}/controls/state/contract.json"; then
    "${python_bin}" "${script_dir}/workflow.py" create-k100-controls \
      --root "${run_root}" --data-root "${data_root}" \
      --reference-bundle "${reference_bundle}" --new-selected "${selected}"
  fi
  capture_controls_provenance
  while IFS= read -r manifest; do
    run_control_manifest "${manifest}"
  done < <(find "${run_root}/controls/configs" -mindepth 3 -maxdepth 3 -name manifest.json | sort)
  mark_done controls
}

analyze() {
  "${python_bin}" "${script_dir}/workflow.py" analyze-k100 \
    --root "${run_root}" --reference-bundle "${reference_bundle}"
}

bundle() {
  if ! test -d "${bundle_root}"; then
    "${python_bin}" "${script_dir}/workflow.py" bundle-k100 \
      --root "${run_root}" --reference-bundle "${reference_bundle}" \
      --output "${bundle_root}"
  fi
  if ! test -f "${bundle_archive}"; then
    tar -C "$(dirname "${bundle_root}")" -czf "${bundle_archive}" "$(basename "${bundle_root}")"
  fi
  sha256sum "${bundle_archive}"
}

case "${stage}" in
  preflight) preflight ;;
  build) build_binaries ;;
  test) test_gate ;;
  frontier) run_frontier ;;
  matched) run_matched ;;
  controls) run_controls ;;
  analyze) analyze ;;
  bundle) bundle ;;
  all)
    preflight
    build_binaries
    test_gate
    run_frontier
    run_matched
    run_controls
    analyze
    bundle
    ;;
  *)
    echo "usage: $0 {preflight|build|test|frontier|matched|controls|analyze|bundle|all}" >&2
    exit 2
    ;;
esac

printf '%s\n' "${run_root}"
printf '%s\n' "${bundle_archive}"
