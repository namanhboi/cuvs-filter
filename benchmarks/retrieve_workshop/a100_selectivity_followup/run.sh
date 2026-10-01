#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")"; pwd)
repo_dir=$(cd "${script_dir}/../../.."; pwd)
python_bin=${PYTHON:-/home/ubuntu/micromamba/envs/cuvs/bin/python}
data_root=${RETRIEVE_DATA_ROOT:-/data/retrieve_data}
stage=${1:-real}
if [[ "${stage}" == negative ]]; then
  run_root=${RETRIEVE_SELECTIVITY_RUN_ROOT:-${HOME}/a100_selectivity_negative_$(date -u +%Y%m%dT%H%M%SZ)}
else
  run_root=${RETRIEVE_SELECTIVITY_RUN_ROOT:-/data/retrieve_workshop_runs/a100_selectivity_$(date -u +%Y%m%dT%H%M%SZ)}
fi

if [[ "${stage}" == tight-match ]]; then
  exec bash "${script_dir}/run_tight.sh" "${2:-all}"
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MPLBACKEND=Agg
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp}
unset RETRIEVE_DATASET_PROFILE
if [[ "${stage}" == check-data ]]; then
  "${python_bin}" "${script_dir}/workflow.py" check-data --data-root "${data_root}"
  exit 0
fi
if [[ "${stage}" == real || "${stage}" == low || "${stage}" == correlation ||
      "${stage}" == negative ||
      "${stage}" == matched-control || "${stage}" == all ]]; then
  "${python_bin}" "${script_dir}/workflow.py" check-data --data-root "${data_root}"
fi
mkdir -p "${run_root}/logs" "${run_root}/state"
exec 9>"${run_root}.lock"
flock -n 9 || { echo "another process owns ${run_root}.lock" >&2; exit 2; }
exec > >(tee -a "${run_root}/logs/orchestrator.log") 2>&1

invoke() {
  "${python_bin}" "${script_dir}/workflow.py" "$1" --root "${run_root}" \
    --data-root "${data_root}" --cohort-prefix "${2:-}"
}

build() {
  if [[ ! -f "${run_root}/state/build.done" ]]; then
    (cd "${repo_dir}" && env PARALLEL_LEVEL=${PARALLEL_LEVEL:-12} \
      ./build.sh libcuvs bench-ann -n \
      '--limit-bench-ann=CUVS_CAGRA_ANN_BENCH;CUVS_BRUTE_FORCE_ANN_BENCH' \
      --gpu-arch=80-real)
    "${python_bin}" "${script_dir}/test_pipeline.py"
    "${python_bin}" "${repo_dir}/benchmarks/retrieve_workshop/gpu_graph/test_pipeline.py"
    date -u +%Y-%m-%dT%H:%M:%SZ >"${run_root}/state/build.done"
  fi
  test -x "${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"
  test -x "${repo_dir}/cpp/build/bench/ann/CUVS_BRUTE_FORCE_ANN_BENCH"
  test -f "${repo_dir}/cpp/build/libcuvs.so"
}

bundle() {
  "${python_bin}" "${script_dir}/workflow.py" bundle --root "${run_root}" \
    --bundle-path "${RETRIEVE_SELECTIVITY_BUNDLE:-${HOME}/$(basename "${run_root}")_results.tar.gz}"
}

record_contract() {
  "${python_bin}" - "${run_root}" "${repo_dir}" "${data_root}" <<'PY'
import hashlib
import json
import pathlib
import subprocess
import sys

root, repo, data = map(pathlib.Path, sys.argv[1:])
def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

graph_binary = repo / "cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"
exact_binary = repo / "cpp/build/bench/ann/CUVS_BRUTE_FORCE_ANN_BENCH"
library = repo / "cpp/build/libcuvs.so"
payload = {
    "schema_version": 1,
    "study": "a100_arxiv_selectivity_and_correlation_k10",
    "git_head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
    "k": 10,
    "max_queries": 2048,
    "target_recall": 0.95,
    "data_root": str(data.resolve()),
    "graph_binary": str(graph_binary.resolve()),
    "graph_binary_sha256": sha256(graph_binary),
    "exact_binary_sha256": sha256(exact_binary),
    "libcuvs_sha256": sha256(library),
    "source_manifest_sha256": {
        name: sha256(data / "navix_bitmap/arxiv-large" / name / "throughput_10000/manifest.json")
        for name in ("em", "emis", "r")
    },
}
path = root / "state/contract.json"
if path.exists():
    if json.loads(path.read_text()) != payload:
        raise SystemExit(f"run contract drifted: {path}")
else:
    path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

real() {
  invoke prepare-real
  invoke graph real_
  invoke exact-control real_
  invoke analyze
}

low() {
  invoke prepare-synthetic-low
  invoke generate-gt synthetic_low_
  invoke graph synthetic_low_
  invoke exact-control synthetic_low_
  invoke analyze
}

correlation() {
  invoke prepare-nearest
  invoke prepare-synthetic-correlation
  invoke generate-gt synthetic_correlation_
  invoke graph synthetic_correlation_
  invoke exact-control synthetic_correlation_
  invoke analyze
}

negative() {
  nearest_file=${RETRIEVE_SELECTIVITY_NEAREST_FILE:-${RETRIEVE_SELECTIVITY_REFERENCE_ROOT:?set RETRIEVE_SELECTIVITY_REFERENCE_ROOT to the completed selectivity run}/state/nearest_1024.ibin}
  test -f "${nearest_file}" || { echo "missing exact nearest IDs: ${nearest_file}" >&2; exit 2; }
  "${python_bin}" "${script_dir}/workflow.py" prepare-synthetic-negative \
    --root "${run_root}" --data-root "${data_root}" --nearest-file "${nearest_file}"
  "${python_bin}" "${script_dir}/workflow.py" validate-synthetic-negative \
    --root "${run_root}" --data-root "${data_root}" --nearest-file "${nearest_file}"
  invoke generate-gt synthetic_correlation_
  invoke graph synthetic_correlation_
  invoke exact-control synthetic_correlation_
  invoke analyze
}

case "${stage}" in
  build) build ;;
  real) build; record_contract; real ;;
  low) build; record_contract; low ;;
  correlation) build; record_contract; correlation ;;
  negative) build; record_contract; negative; bundle ;;
  matched-control) build; record_contract; invoke matched-control; invoke analyze ;;
  analyze) invoke analyze ;;
  bundle) bundle ;;
  all) build; record_contract; real; low; correlation; bundle ;;
  *) echo "usage: $0 {check-data|build|real|low|correlation|negative|matched-control|tight-match|analyze|bundle|all}" >&2; exit 2 ;;
esac

printf '%s\n' "${run_root}"
