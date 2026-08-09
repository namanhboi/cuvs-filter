#!/usr/bin/env python3
"""Return success when a NaviX point in a benchmark JSON reaches the target recall."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--target", type=float, default=0.95)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text())
    recalls = [
        float(row.get("Recall", 0.0))
        for row in payload.get("benchmarks", [])
        if "navix_mode=" in str(row.get("label", "")) and "error_occurred" not in row
    ]
    raise SystemExit(0 if recalls and max(recalls) >= args.target else 1)


if __name__ == "__main__":
    main()
