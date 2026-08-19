#!/usr/bin/env python3
"""Run the frozen native-CPU context experiment without rebuilding any graph."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import shlex
import struct
import subprocess
import sys
import time
from typing import Any

EF_SEARCH = (10, 15, 20, 25, 30, 50, 100, 250, 500, 750)
WORKLOADS = ("yfcc", "em", "emis", "r")
METHODS = (
    "faiss_navix",
    "acorn_1",
    "acorn_gamma",
    "acorn_1_navix_seeded",
    "acorn_gamma_navix_seeded",
)
PLOTTED_METHODS = ("faiss_navix", "acorn_1", "acorn_gamma")
SEEDED_METHODS = ("acorn_1_navix_seeded", "acorn_gamma_navix_seeded")

DEFAULT_DATA_ROOT = pathlib.Path("/home/ubuntu/cuvs-filter/datasets")
DEFAULT_ARTIFACT_ROOT = pathlib.Path("/home/ubuntu/navix_cpu_artifacts")
DEFAULT_FAISS_REPO = pathlib.Path("/home/ubuntu/faiss-navix-native-benchmark")
DEFAULT_ACORN_REPO = pathlib.Path("/home/ubuntu/ACORN-gamma-benchmark")

ACORN_GAMMA = {
    "em": (16, 10, 24, 160),
    "emis": (16, 15, 24, 240),
    "r": (32, 12, 24, 384),
    "yfcc": (64, 30, 64, 1920),
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sampled_file_fingerprint(path: pathlib.Path, sample_bytes: int = 1024 * 1024) -> str:
    """Fast identity aid for multi-GB graphs; this is explicitly not a full-file checksum."""
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", size))
    with path.open("rb") as stream:
        digest.update(stream.read(sample_bytes))
        if size > sample_bytes:
            stream.seek(max(0, size - sample_bytes))
            digest.update(stream.read(sample_bytes))
    return digest.hexdigest()


def command_output(argv: list[str]) -> str:
    try:
        return subprocess.run(
            argv, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def git_provenance(path: pathlib.Path) -> dict[str, Any]:
    status = command_output(["git", "-C", str(path), "status", "--porcelain=v1"])
    diff = command_output(["git", "-C", str(path), "diff", "--binary"])
    return {
        "path": str(path.resolve()),
        "head": command_output(["git", "-C", str(path), "rev-parse", "HEAD"]),
        "branch": command_output(["git", "-C", str(path), "branch", "--show-current"]),
        "dirty": bool(status and not status.startswith("unavailable:")),
        "status": status.splitlines(),
        "working_tree_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
    }


def path_provenance(path: pathlib.Path, *, graph: bool = False) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if graph:
        result["sampled_sha256"] = sampled_file_fingerprint(path)
        result["sampled_sha256_definition"] = "sha256(le64(size) || first_1MiB || last_1MiB)"
        build_csv = pathlib.Path(str(path) + ".build.csv")
        if not build_csv.is_file():
            raise FileNotFoundError(f"missing graph build provenance: {build_csv}")
        result["build_csv_path"] = str(build_csv.resolve())
        result["build_csv_sha256"] = sha256_file(build_csv)
        with build_csv.open(newline="") as stream:
            result["build_metadata"] = list(csv.DictReader(stream))
    else:
        result["sha256"] = sha256_file(path)
    return result


def matrix_header(path: pathlib.Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        payload = stream.read(8)
    if len(payload) != 8:
        raise ValueError(f"matrix header is truncated: {path}")
    return struct.unpack("<II", payload)


def validate_matrix_size(path: pathlib.Path, element_bytes: int) -> tuple[int, int]:
    rows, columns = matrix_header(path)
    expected = 8 + rows * columns * element_bytes
    if path.stat().st_size != expected:
        raise ValueError(f"matrix size/header mismatch: {path}")
    return rows, columns


def bitmap_header(path: pathlib.Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        payload = stream.read(40)
    if len(payload) != 40:
        raise ValueError(f"bitmap header is truncated: {path}")
    magic, version, word_bits, rows, cols, words = struct.unpack("<8sIIQQQ", payload)
    if magic != b"CUVSBMAP" or version != 1 or word_bits != 32:
        raise ValueError(f"unsupported bitmap header: {path}")
    expected_words = (rows * cols + 31) // 32
    expected_bytes = 40 + words * 4
    if words != expected_words or path.stat().st_size != expected_bytes:
        raise ValueError(f"bitmap size/header mismatch: {path}")
    return rows, cols


def manifest_path(data_root: pathlib.Path, workload: str) -> pathlib.Path:
    if workload == "yfcc":
        return data_root / "navix_bitmap/yfcc/throughput_10000/manifest.json"
    return data_root / f"navix_bitmap/arxiv/{workload}/throughput_10000/manifest.json"


def load_manifest(data_root: pathlib.Path, workload: str) -> dict[str, Any]:
    path = manifest_path(data_root, workload)
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1 or payload.get("query_rows") != 10_000:
        raise ValueError(f"unexpected manifest schema/query count: {path}")
    expected_counts = [2048, 2048, 2048, 2048, 1808] if workload == "yfcc" else [10_000]
    shards = payload.get("shards", [])
    if [int(item["query_count"]) for item in shards] != expected_counts:
        raise ValueError(f"unexpected shard layout in {path}")
    cursor = 0
    for index, item in enumerate(shards):
        if int(item["first_query"]) != cursor:
            raise ValueError(f"non-contiguous shard {index} in {path}")
        cursor += int(item["query_count"])
        directory = pathlib.Path(item["directory"])
        query = pathlib.Path(item.get("query", directory / "query.bin"))
        gt = pathlib.Path(item.get("groundtruth", directory / "groundtruth.ibin"))
        bitmap = pathlib.Path(item.get("bitmap", directory / "filter.bitmap"))
        for candidate in (query, gt, bitmap):
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
        query_element_bytes = 1 if workload == "yfcc" else 4
        query_rows, _ = validate_matrix_size(query, query_element_bytes)
        gt_rows, gt_cols = validate_matrix_size(gt, 4)
        bitmap_rows, bitmap_cols = bitmap_header(bitmap)
        if query_rows != item["query_count"] or gt_rows != query_rows or bitmap_rows != query_rows:
            raise ValueError(f"query/GT/bitmap row mismatch in shard {index} of {path}")
        if gt_cols < 10 or bitmap_cols != payload["base_rows"]:
            raise ValueError(f"GT width or bitmap columns are invalid in shard {index} of {path}")
        item["resolved_query"] = str(query.resolve())
        item["resolved_groundtruth"] = str(gt.resolve())
        item["resolved_bitmap"] = str(bitmap.resolve())
    if cursor != 10_000:
        raise ValueError(f"manifest does not cover exactly 10,000 queries: {path}")
    payload["manifest_path"] = str(path.resolve())
    payload["manifest_sha256"] = sha256_file(path)
    return payload


def graph_configuration(artifact_root: pathlib.Path, method: str, workload: str) -> dict[str, Any]:
    seeded = method.endswith("_navix_seeded")
    base_method = method.removesuffix("_navix_seeded") if seeded else method
    if base_method == "faiss_navix":
        if workload == "yfcc":
            graph = artifact_root / "graphs/faiss_hnsw_yfcc_M64_efc200.index"
            M, efc = 64, 200
        else:
            graph = artifact_root / "graphs/faiss_hnsw_arxiv_M32_efc200.index"
            M, efc = 32, 200
        return {"path": graph, "M": M, "ef_construction": efc}
    if base_method == "acorn_1":
        key = "yfcc" if workload == "yfcc" else "arxiv"
        graph = artifact_root / f"graphs/acorn_1_{key}_M32_efc200.index"
        return {"path": graph, "M": 32, "gamma": 1, "M_beta": 32, "ef_construction": 200}
    if base_method == "acorn_gamma":
        M, gamma, M_beta, efc = ACORN_GAMMA[workload]
        graph = artifact_root / (
            f"graphs/acorn_gamma_{workload}_M{M}_g{gamma}_b{M_beta}_efc{efc}.index"
        )
        return {
            "path": graph,
            "M": M,
            "gamma": gamma,
            "M_beta": M_beta,
            "ef_construction": efc,
        }
    raise ValueError(method)


def workload_base(data_root: pathlib.Path, workload: str) -> tuple[pathlib.Path, str, str]:
    if workload == "yfcc":
        return data_root / "yfcc-10M/base.10M.u8bin", "u8", "u8"
    return data_root / "arxiv-for-fanns-medium/base.fbin", "f32", "f32"


def expected_threads(method: str, workload: str) -> int:
    return 16 if method == "faiss_navix" and workload != "yfcc" else 32


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    manifests = {workload: load_manifest(args.data_root, workload) for workload in WORKLOADS}
    binaries = {
        "faiss_navix": args.faiss_repo / "build/benchs/bench_navix_bitmap",
        "acorn": args.acorn_repo / "build/benchs/bench_acorn_bitmap",
    }
    for path in binaries.values():
        if not path.is_file() or not os.access(path, os.X_OK):
            raise FileNotFoundError(f"native runner is missing or not executable: {path}")

    bases: dict[str, dict[str, Any]] = {}
    for workload in WORKLOADS:
        base, dtype, qtype = workload_base(args.data_root, workload)
        if not base.is_file():
            raise FileNotFoundError(base)
        rows, dimensions = validate_matrix_size(base, 1 if dtype == "u8" else 4)
        if rows != manifests[workload]["base_rows"]:
            raise ValueError(f"base/manifest row mismatch for {workload}")
        bases[workload] = {
            "path": str(base.resolve()), "dtype": dtype, "qtype": qtype,
            "rows": rows, "dimensions": dimensions, "bytes": base.stat().st_size,
        }

    graph_configs: dict[str, dict[str, Any]] = {}
    unique_graphs: dict[str, pathlib.Path] = {}
    for method in METHODS:
        for workload in WORKLOADS:
            config = graph_configuration(args.artifact_root, method, workload)
            graph = config.pop("path")
            if not graph.is_file():
                raise FileNotFoundError(graph)
            key = f"{method}/{workload}"
            graph_configs[key] = {**config, "path": str(graph.resolve()), "bytes": graph.stat().st_size}
            unique_graphs[str(graph.resolve())] = graph

            build_csv = pathlib.Path(str(graph) + ".build.csv")
            if not build_csv.is_file():
                raise FileNotFoundError(build_csv)
            with build_csv.open(newline="") as stream:
                build_rows = list(csv.DictReader(stream))
            if len(build_rows) != 1:
                raise ValueError(f"expected one build-provenance row in {build_csv}")
            build = build_rows[0]
            base_method = method.removesuffix("_navix_seeded")
            if build.get("method") != base_method:
                raise ValueError(f"graph method mismatch in {build_csv}")
            expected_numeric = {
                "rows": bases[workload]["rows"], "dimensions": bases[workload]["dimensions"],
                "M": config["M"], "ef_construction": config["ef_construction"],
                "index_bytes": graph.stat().st_size,
            }
            if base_method.startswith("acorn"):
                expected_numeric.update({"gamma": config["gamma"], "M_beta": config["M_beta"]})
            for field, expected in expected_numeric.items():
                if int(build[field]) != expected:
                    raise ValueError(
                        f"graph provenance mismatch for {field} in {build_csv}: "
                        f"{build[field]} != {expected}"
                    )

    return {
        "manifests": manifests,
        "bases": bases,
        "binaries": {key: str(value.resolve()) for key, value in binaries.items()},
        "graph_configs": graph_configs,
        "unique_graphs": unique_graphs,
    }


def hardware_provenance() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "uname": command_output(["uname", "-a"]),
        "lscpu_json": command_output(["lscpu", "--json"]),
        "numactl_hardware": command_output(["numactl", "--hardware"]),
        "meminfo": pathlib.Path("/proc/meminfo").read_text(),
    }


def append_event(path: pathlib.Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def native_command(
    args: argparse.Namespace,
    preflight_data: dict[str, Any],
    method: str,
    workload: str,
    shard: dict[str, Any],
    output: pathlib.Path,
) -> tuple[list[str], dict[str, str]]:
    base = preflight_data["bases"][workload]
    config = preflight_data["graph_configs"][f"{method}/{workload}"]
    threads = expected_threads(method, workload)
    chunk = 512 if workload == "yfcc" else 10_000
    common = [
        "--base", base["path"], "--dtype", base["dtype"],
        "--queries", shard["resolved_query"], "--qtype", base["qtype"],
        "--ground-truth", shard["resolved_groundtruth"],
        "--bitmap", shard["resolved_bitmap"], "--index", config["path"],
        "--csv", str(output), "--chunk", str(chunk),
        "--ef-search", ",".join(map(str, EF_SEARCH)),
        "--M", str(config["M"]), "--ef-construction", str(config["ef_construction"]),
        "--threads", str(threads), "--build-threads", "24",
    ]
    if method == "faiss_navix":
        argv = [preflight_data["binaries"]["faiss_navix"], *common]
    else:
        base_method = method.removesuffix("_navix_seeded")
        filtered_seeds = 10 if method.endswith("_navix_seeded") else 0
        argv = [
            preflight_data["binaries"]["acorn"], *common,
            "--method", base_method, "--gamma", str(config["gamma"]),
            "--M-beta", str(config["M_beta"]), "--filtered-seeds", str(filtered_seeds),
        ]
    environment = {
        "OMP_NUM_THREADS": str(threads),
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
    }
    return argv, environment


def write_exclusive(path: pathlib.Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def execute(args: argparse.Namespace, preflight_data: dict[str, Any]) -> pathlib.Path:
    run_id = args.run_id or f"cpu_context_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}_{os.getpid()}"
    args.results_root.mkdir(parents=True, exist_ok=True)
    run_root = args.results_root / run_id
    run_root.mkdir(parents=False, exist_ok=False)

    graph_provenance = {
        path: path_provenance(graph, graph=True)
        for path, graph in sorted(preflight_data["unique_graphs"].items())
    }
    created = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": utc_now(),
        "invocation": sys.argv,
        "run_config": {
            "repetitions": 3,
            "ef_search": list(EF_SEARCH),
            "workloads": list(WORKLOADS),
            "methods": list(METHODS),
            "plotted_methods": list(PLOTTED_METHODS),
            "seeded_methods": list(SEEDED_METHODS),
            "query_count": 10_000,
            "yfcc_shard_counts": [2048, 2048, 2048, 2048, 1808],
            "arxiv_shard_counts": [10_000],
            "chunks": {"yfcc": 512, "arxiv": 10_000},
            "threads": {"faiss_yfcc": 32, "faiss_arxiv": 16, "acorn": 32},
        },
        "hardware": hardware_provenance(),
        "repositories": {
            "cuvs_filter": git_provenance(pathlib.Path(__file__).resolve().parents[3]),
            "faiss_navix": git_provenance(args.faiss_repo),
            "acorn": git_provenance(args.acorn_repo),
        },
        "binaries": {
            name: path_provenance(pathlib.Path(path))
            for name, path in preflight_data["binaries"].items()
        },
        "graphs": graph_provenance,
        "graph_configurations": preflight_data["graph_configs"],
        "bases": preflight_data["bases"],
        "dataset_manifests": preflight_data["manifests"],
        "timing_semantics": (
            "native CPU search call only; packed-bitmap to byte-mask conversion is outside timing"
        ),
        "comparison_warning": (
            "CPU and GPU hardware, graphs, and software differ; absolute CPU/GPU speedup claims are invalid"
        ),
    }
    write_exclusive(run_root / "run.created.json", created)
    events = run_root / "events.jsonl"
    events.touch(exist_ok=False)

    try:
        command_index = 0
        for repetition in range(1, 4):
            rep_root = run_root / f"rep_{repetition:02d}"
            rep_root.mkdir(exist_ok=False)
            for method in METHODS:
                for workload in WORKLOADS:
                    for shard_index, shard in enumerate(preflight_data["manifests"][workload]["shards"]):
                        command_index += 1
                        output_dir = rep_root / "raw" / method / workload
                        output_dir.mkdir(parents=True, exist_ok=True)
                        first = int(shard["first_query"])
                        last = first + int(shard["query_count"])
                        output = output_dir / f"shard_{first:05d}_{last:05d}.csv"
                        log = output.with_suffix(".log")
                        if output.exists() or log.exists():
                            raise FileExistsError(output)
                        argv, environment = native_command(
                            args, preflight_data, method, workload, shard, output
                        )
                        command_id = f"cmd_{command_index:04d}"
                        started = utc_now()
                        append_event(events, {
                            "event": "command_started", "command_id": command_id,
                            "repetition": repetition, "method": method, "workload": workload,
                            "shard_index": shard_index, "started_utc": started,
                            "argv": argv, "shell_rendering": shlex.join(argv),
                            "environment": environment, "csv": str(output), "log": str(log),
                        })
                        process_environment = os.environ.copy()
                        process_environment.update(environment)
                        start_monotonic = time.monotonic()
                        with log.open("x", encoding="utf-8") as stream:
                            result = subprocess.run(
                                argv, env=process_environment, stdout=stream,
                                stderr=subprocess.STDOUT, text=True, check=False,
                            )
                        elapsed = time.monotonic() - start_monotonic
                        append_event(events, {
                            "event": "command_completed", "command_id": command_id,
                            "ended_utc": utc_now(), "wall_seconds": elapsed,
                            "returncode": result.returncode, "csv_exists": output.is_file(),
                        })
                        if result.returncode != 0 or not output.is_file():
                            raise RuntimeError(
                                f"{command_id} failed with return code {result.returncode}; see {log}"
                            )
        write_exclusive(run_root / "run.completed.json", {
            "schema_version": 1, "run_id": run_id, "completed_utc": utc_now(),
            "commands_completed": command_index,
        })
    except BaseException as error:
        write_exclusive(run_root / "run.failed.json", {
            "schema_version": 1, "run_id": run_id, "failed_utc": utc_now(),
            "error_type": type(error).__name__, "error": str(error),
        })
        raise
    return run_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="validate inputs without running")
    parser.add_argument("--results-root", type=pathlib.Path,
                        default=pathlib.Path("/home/ubuntu/retrieve_cpu_context_results"))
    parser.add_argument("--run-id", help="new, unique result directory name")
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--artifact-root", type=pathlib.Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--faiss-repo", type=pathlib.Path, default=DEFAULT_FAISS_REPO)
    parser.add_argument("--acorn-repo", type=pathlib.Path, default=DEFAULT_ACORN_REPO)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = preflight(args)
    if args.preflight:
        print(
            f"preflight PASS: {len(data['unique_graphs'])} graphs, "
            f"{len(METHODS)} methods, {len(WORKLOADS)} workloads, 10,000 queries/workload"
        )
        return
    root = execute(args, data)
    print(root)


if __name__ == "__main__":
    main()
