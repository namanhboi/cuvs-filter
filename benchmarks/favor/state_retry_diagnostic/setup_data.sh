#!/bin/bash
# Retired FAVOR workflow. Historical results remain available for inspection.
printf '%s\n' 'FAVOR is retired; use benchmarks/retrieve_workshop for Base, Retain, and NaviX.' >&2
exit 64
set -euo pipefail

script_dir=$(cd "$(dirname "$0")"; pwd)
exec "${script_dir}/../multi_seed_b0/setup_data.sh" "$@"
