#!/usr/bin/env python3
"""Merge a targeted benchmark extension into an existing result/config pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def append_unique(primary: list[dict], extension: list[dict]) -> list[dict]:
    existing = {json.dumps(row, sort_keys=True) for row in primary}
    return primary + [
        row for row in extension if json.dumps(row, sort_keys=True) not in existing
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("primary_result", type=Path)
    parser.add_argument("extension_result", type=Path)
    parser.add_argument("primary_config", type=Path)
    parser.add_argument("extension_config", type=Path)
    args = parser.parse_args()

    primary_result = json.loads(args.primary_result.read_text())
    extension_result = json.loads(args.extension_result.read_text())
    primary_result["benchmarks"] = append_unique(
        primary_result["benchmarks"], extension_result["benchmarks"]
    )
    args.primary_result.write_text(json.dumps(primary_result, indent=2) + "\n")

    primary_config = json.loads(args.primary_config.read_text())
    extension_config = json.loads(args.extension_config.read_text())
    primary_config["index"][0]["search_params"] = append_unique(
        primary_config["index"][0]["search_params"],
        extension_config["index"][0]["search_params"],
    )
    args.primary_config.write_text(json.dumps(primary_config, indent=2) + "\n")


if __name__ == "__main__":
    main()
