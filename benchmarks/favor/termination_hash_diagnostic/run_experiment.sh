#!/bin/bash
# Retired FAVOR workflow. Historical results remain available for inspection.
printf '%s\n' 'FAVOR is retired; use benchmarks/retrieve_workshop for Base, Retain, and NaviX.' >&2
exit 64
set -euo pipefail

echo "This historical experiment is retired: the recall-unsafe forgetful-hash runtime path was removed." >&2
echo "Use FAVOR_TERMINATION_HASH_DIAGNOSTIC.org and results/ for its retained negative evidence." >&2
exit 2
