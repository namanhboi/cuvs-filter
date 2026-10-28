#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON:-/home/ubuntu/micromamba/envs/cuvs/bin/python}" \
  "${script_dir}/primary_ivf.py" "$@"
