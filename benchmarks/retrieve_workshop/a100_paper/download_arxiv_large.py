#!/usr/bin/env python3
"""Explicitly download the official SPCL ArXiv-large benchmark inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

FILES = (
    "database_vectors.fvecs",
    "query_vectors.fvecs",
    "database_attributes.jsonl",
    "em_query_attributes.jsonl",
    "r_query_attributes.jsonl",
    "emis_query_attributes.jsonl",
    "ground_truth_em.ivecs",
    "ground_truth_r.ivecs",
    "ground_truth_emis.ivecs",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "install huggingface_hub in the cuVS environment first"
        ) from exc
    args.output.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = hf_hub_download(
            repo_id="SPCL/arxiv-for-fanns-large",
            filename=name,
            repo_type="dataset",
            local_dir=str(args.output),
        )
        print(path, flush=True)


if __name__ == "__main__":
    main()
