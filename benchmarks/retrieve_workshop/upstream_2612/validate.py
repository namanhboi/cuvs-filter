#!/usr/bin/env python3
"""Replay frozen 2,048-query configurations for the pinned upstream integration.

All generated files are external to the checkouts. Never retune parameters, rewrite
paper results, advance main, or push from this runner.
"""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import re
import shlex
import shutil
import statistics
import subprocess
import tarfile
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKLOADS = ("yfcc", "em", "emis", "r")
RATES = ("0p0001", "0p0003", "0p001", "0p003", "0p01")
METHODS = ("default_cagra", "default_cagra_accumulator", "navix_reference")
BINARIES = {
    "cuvs_cagra": "CUVS_CAGRA_ANN_BENCH",
    "cuvs_ivf_flat": "CUVS_IVF_FLAT_ANN_BENCH",
    "cuvs_brute_force": "CUVS_BRUTE_FORCE_ANN_BENCH",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def absolute(path: str, data_root: Path) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else data_root / p)


def point_matches(sp: dict, row: dict) -> bool:
    return sp.get("bitmap_method") == row["method"] and all(
        int(sp[key]) == int(row[key])
        for key in ("itopk", "search_width", "max_iterations")
    )


def source_point(
    candidates: list,
    row: dict,
    k: int,
    workload: str,
    wd: bool,
    parameter_source: Path | None = None,
) -> dict:
    for path, payload in candidates:
        if (
            path.parent.name != workload
            or payload["search_basic_param"]["k"] != k
        ):
            continue
        for index in payload["index"]:
            for point in index["search_params"]:
                if not point_matches(point, row):
                    continue
                if wd and row["method"] == "navix_reference":
                    if (
                        point.get("navix_seed_cap")
                        != int(row["search_width"]) * 64
                    ):
                        continue
                selected = copy.deepcopy(payload)
                selected["index"] = [copy.deepcopy(index)]
                selected["index"][0]["search_params"] = [copy.deepcopy(point)]
                return {
                    "source": str(path),
                    "source_sha256": sha(path),
                    "config": selected,
                }
    # Calibration archives preserve scalar search settings even when their disposable
    # per-round JSON files were omitted. Reconstruct those scalars over the same frozen
    # method/width template, and bind the decision to the archived CSV, never to retuning.
    if parameter_source is not None:
        for path, payload in candidates:
            if (
                path.parent.name != workload
                or payload["search_basic_param"]["k"] != k
            ):
                continue
            for index in payload["index"]:
                for point in index["search_params"]:
                    if point.get("bitmap_method") != row["method"] or int(
                        point["search_width"]
                    ) != int(row["search_width"]):
                        continue
                    selected = copy.deepcopy(payload)
                    selected["index"] = [copy.deepcopy(index)]
                    restored = copy.deepcopy(point)
                    restored.update(
                        {
                            key: int(row[key])
                            for key in (
                                "itopk",
                                "search_width",
                                "max_iterations",
                            )
                        }
                    )
                    selected["index"][0]["search_params"] = [restored]
                    return {
                        "source": str(path),
                        "source_sha256": sha(path),
                        "config": selected,
                        "parameter_source": str(parameter_source),
                        "parameter_source_sha256": sha(parameter_source),
                    }
    raise ValueError(f"missing frozen configuration: {workload} k={k} {row}")


def normalize_config(
    config: dict,
    data_root: Path,
    integrated: bool,
    index_dir: Path,
    representation: str,
    case_id: str,
    allowlist_dir: Path,
) -> dict:
    result = copy.deepcopy(config)
    ds = result["dataset"]
    for key in ("base_file", "query_file", "groundtruth_neighbors_file"):
        if key in ds:
            ds[key] = absolute(ds[key], data_root)
    if "filter" in ds:
        ds["filter"]["file"] = absolute(ds["filter"]["file"], data_root)
    result["search_basic_param"]["batch_size"] = 2048
    for index in result["index"]:
        index["file"] = absolute(index["file"], data_root)
        if integrated and index["algo"] == "cuvs_cagra":
            corpus = "yfcc" if ds["dtype"] == "uint8" else "arxiv"
            index["file"] = str(index_dir / f"{corpus}_cagra.index")
        elif integrated and index["algo"] == "cuvs_ivf_flat":
            index["file"] = str(index_dir / Path(index["file"]).name)
        for point in index["search_params"]:
            point.pop("benchmark_output_neighbors_file", None)
            point["validation_case_id"] = case_id
            if index["algo"] == "cuvs_cagra":
                if point.get("filter_mode", "default") != "default":
                    raise ValueError(
                        "retired FAVOR configuration in validation"
                    )
                point["max_queries"] = 2048
                point.setdefault("rand_xor_mask", 0x128394)
                point.setdefault("hashmap_max_fill_rate", 0.5)
                if integrated:
                    if "favor_udf_passing_accumulator" in point:
                        point["passing_result_accumulator"] = point.pop(
                            "favor_udf_passing_accumulator"
                        )
                    point["filter_representation"] = representation
                    cohort = case_id.split("__")[0]
                    archive = allowlist_dir / f"{cohort}.allowids"
                    if representation == "roaring" and archive.exists():
                        point["roaring_allowlist_file"] = str(archive)
    return result


def prepare(args) -> None:
    root, run, paper = args.root, args.runs, args.paper
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            "validation manifest already exists; run or analyze it"
        )
    records = []
    candidates = []
    for name in (
        "a100_k10_wd_all_20260826T233255Z",
        "a100_maxq2048_20260824T065013Z",
        "a100_k100_wd_20260827T031138Z",
        "a100_k100_20260825T232747Z",
    ):
        for p in sorted(
            (run / name).glob(
                "matched_recall/configs/final_candidates/*/shard_00.json"
            )
        ):
            candidates.append((p, json.loads(p.read_text())))
    for k in (10, 100):
        suffix = "_k100" if k == 100 else ""
        rows = read_csv(paper / f"current_gpu_matched_recall{suffix}.csv")
        if len(rows) != 12:
            raise ValueError("incomplete primary paper selection")
        for row in rows:
            item = source_point(candidates, row, k, row["workload"], wd=True)
            records.append(
                item
                | {
                    "cohort": row["workload"],
                    "k": k,
                    "method": row["method"],
                    "selection": "frozen_paper",
                    "paper_recall": row["recall_median"],
                }
            )
    ivf_root = run / "a100_ivf_bitmap_20261014T024338Z"
    for row in read_csv(args.ivf_report / "matched_recall.csv"):
        if row["method"] != "ivf_flat":
            continue
        cohort, k, index_id = row["workload"], int(row["k"]), row["index_id"]
        paths = list(
            (
                ivf_root
                / "configs/final"
                / index_id
                / f"{cohort}_k{k}"
                / "selected"
            ).glob("shard_00.json")
        )
        if not paths:
            paths = list(
                (
                    ivf_root
                    / "configs/calibration"
                    / index_id
                    / f"{cohort}_k{k}"
                ).glob("**/shard_00.json")
            )
        append_ivf(records, paths, row, cohort, k)

    sparse_root = run / "a100_selectivity_20260928T070431Z"
    references = paper / "selectivity_followup_sources"
    selected = read_csv(references / "matched_selected.csv")
    observed = []
    for row in read_csv(references / "graph_summary.csv"):
        observed.append(
            row
            | {
                "cohort": row["workload"],
                "recall": row["recall_median"],
                "qps": row["qps_median"],
            }
        )
    observed.extend(read_csv(references / "calibration_points.csv"))
    sparse_candidates = []
    for p in sorted((sparse_root / "graph/configs").glob("*/*/shard_00.json")):
        if p.parent.name.startswith("synthetic_low_"):
            sparse_candidates.append((p, json.loads(p.read_text())))
    # Copy only bounded JSON members; original archives and results stay unchanged.
    for archive_name in (
        "a100_selectivity_matched_002_20260928T094801Z_results.tar.gz",
    ):
        archive = args.paper.parent.parent / archive_name
        if not archive.exists():
            continue
        with tarfile.open(archive) as source:
            for member in source:
                relative = Path(member.name)
                if (
                    not member.isfile()
                    or member.size > 4 * 1024 * 1024
                    or ".." in relative.parts
                    or relative.is_absolute()
                ):
                    continue
                if (
                    relative.name != "shard_00.json"
                    or "/configs/" not in member.name
                    or not relative.parent.name.startswith("synthetic_low_")
                ):
                    continue
                content = source.extractfile(member).read()
                p = root / "archived-configs" / archive.stem / relative
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(content)
                sparse_candidates.insert(0, (p, json.loads(content)))
    sparse_ivf = run / "a100_ivf_flat_sparse_20261015T025433Z"
    sparse_flat = read_csv(sparse_ivf / "analysis/comparison.csv")
    for rate in RATES:
        cohort = f"synthetic_low_{rate}_random"
        for method in METHODS:
            row = next(
                x
                for x in selected
                if x["cohort"] == cohort and x["method"] == method
            )
            if row["status"] != "matched":
                row = max(
                    (
                        x
                        for x in observed
                        if x["cohort"] == cohort and x["method"] == method
                    ),
                    key=lambda x: (float(x["recall"]), float(x["qps"])),
                )
                selection = "max_observed_recall"
            else:
                selection = "frozen_matched"
            source_csv = references / (
                "matched_selected.csv"
                if selection == "frozen_matched"
                else "calibration_points.csv"
            )
            item = source_point(
                sparse_candidates,
                row,
                10,
                cohort,
                wd=False,
                parameter_source=source_csv,
            )
            records.append(
                item
                | {
                    "cohort": cohort,
                    "k": 10,
                    "method": method,
                    "selection": selection,
                }
            )
        row = next(
            x
            for x in sparse_flat
            if x.get("workload", x.get("cohort")) == cohort
            and x["method"] == "ivf_flat"
        )
        paths = list(
            (
                sparse_ivf
                / "configs/final"
                / row["index_id"]
                / f"{cohort}_k10"
                / "selected"
            ).glob("shard_00.json")
        )
        append_ivf(records, paths, row, cohort, 10)
        p = sparse_root / "exact/configs" / cohort / "control.json"
        records.append(
            {
                "source": str(p),
                "source_sha256": sha(p),
                "config": json.loads(p.read_text()),
                "cohort": cohort,
                "k": 10,
                "method": "exact",
                "selection": "exact_control",
            }
        )
    if len(records) != 57:
        raise ValueError(
            f"expected 32 primary and 25 sparse cases, got {len(records)}"
        )

    groups = {}
    native = []
    inputs = {}
    for record in records:
        case_id = f"{record['cohort']}__k{record['k']}__{record['method']}"
        record["id"] = case_id
        original = record["config"]
        for key in ("base_file", "query_file", "groundtruth_neighbors_file"):
            p = Path(absolute(original["dataset"][key], args.data))
            st = p.stat()
            inputs[str(p)] = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
        p = Path(absolute(original["dataset"]["filter"]["file"], args.data))
        st = p.stat()
        inputs[str(p)] = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
        for variant, integrated, representation in (
            ("old", False, "bitmap"),
            ("integrated", True, "bitmap"),
            ("roaring", True, "roaring"),
        ):
            if variant == "roaring" and record["method"] not in METHODS:
                continue
            config = normalize_config(
                original,
                args.data,
                integrated,
                args.index_dir,
                representation,
                case_id,
                args.allowlist_dir,
            )
            group_id = f"{variant}/{record['cohort']}_k{record['k']}/{config['index'][0]['algo']}"
            if group_id in groups:
                group = groups[group_id]
                assert group["config"]["dataset"] == config["dataset"]
                group["config"]["index"][0]["search_params"].extend(
                    config["index"][0]["search_params"]
                )
                group["cases"].append(case_id)
            else:
                groups[group_id] = {
                    "id": group_id,
                    "variant": variant,
                    "config": config,
                    "cases": [case_id],
                }
        if record["method"] == "default_cagra":
            config = normalize_config(
                original,
                args.data,
                True,
                args.index_dir,
                "bitmap",
                case_id,
                args.allowlist_dir,
            )
            native.append({"id": case_id, "config": config})
    jobs = []
    for group in groups.values():
        p = root / "configs" / (group["id"] + ".json")
        save(p, group.pop("config"))
        jobs.append(group | {"config": str(p), "config_sha256": sha(p)})
    for item in native:
        p = root / "native/configs" / (item["id"] + ".json")
        save(p, item.pop("config"))
        item["config"] = str(p)
    manifest = {
        "schema_version": 1,
        "query_count": 2048,
        "repetitions": 3,
        "minimum_time": "0.10s",
        "minimum_warmup": "0.10",
        "jobs": jobs,
        "native": native,
        "sources": records,
        "inputs": inputs,
        "old_repo": str(args.old_repo),
        "integrated_repo": str(args.integrated_repo),
        "control_repo": str(args.control_repo),
        "validation_dir": str(args.validation_dir),
        "environment": str(args.environment),
    }
    save(manifest_path, manifest)
    print(
        json.dumps(
            {
                "cases": len(records),
                "processes": len(jobs),
                "native_cases": len(native),
                "manifest": str(manifest_path),
            }
        )
    )


def append_ivf(records, paths, row, cohort, k):
    for p in paths:
        payload = json.loads(p.read_text())
        for index in payload["index"]:
            for sp in index["search_params"]:
                if int(sp["nprobe"]) == int(row["nprobe"]):
                    chosen = copy.deepcopy(payload)
                    chosen["index"] = [copy.deepcopy(index)]
                    chosen["index"][0]["search_params"] = [copy.deepcopy(sp)]
                    records.append(
                        {
                            "source": str(p),
                            "source_sha256": sha(p),
                            "config": chosen,
                            "cohort": cohort,
                            "k": k,
                            "method": "ivf_flat",
                            "selection": "frozen_matched",
                        }
                    )
                    return
    raise ValueError(f"missing frozen IVF point {cohort} k={k}: {row}")


def read_measurements(path: Path, expected: list[str]) -> list[dict]:
    payload = json.loads(path.read_text())
    grouped = defaultdict(list)
    for row in payload.get("benchmarks", []):
        if row.get("run_type") == "aggregate":
            continue
        if row.get("error_occurred") or row.get("skipped"):
            raise ValueError(
                f"benchmark failure: {row.get('error_message', row)}"
            )
        match = re.search(
            r'(?:^|#)validation_case_id="([^"]+)"', row.get("label", "")
        )
        if not match:
            raise ValueError("benchmark did not emit its frozen case identity")
        if match.group(1).endswith("__exact"):
            if (
                float(row.get("NativeL2CutoffValidated", 0)) != 1
                or float(row.get("NativeL2CutoffRecall", 0)) < 1 - 1e-12
            ):
                raise ValueError(
                    "exact control did not pass native-distance validation"
                )
            if any(
                float(row.get(k, 0)) != 0
                for k in ("NativeL2CutoffErrors", "NativeL2StrictPrefixErrors")
            ):
                raise ValueError("exact control has native-distance errors")
        if int(row.get("n_queries", -1)) != 2048:
            raise ValueError("incorrect timed query count")
        for key in (
            "FilterViolations",
            "InvalidSentinelErrors",
            "SentinelOrderErrors",
        ):
            if float(row.get(key, 0)) != 0:
                raise ValueError(f"invalid output: {key}")
        grouped[match.group(1)].append(row)
    if set(grouped) != set(expected):
        raise ValueError("missing or unexpected benchmark cases")
    result = []
    for key, rows in grouped.items():
        if len(rows) != 3 or {
            int(r.get("repetition_index", -1)) for r in rows
        } != {0, 1, 2}:
            raise ValueError("incomplete repetitions")
        recalls = [float(r["ValidGTRecall"]) for r in rows]
        qps = [float(r["items_per_second"]) for r in rows]
        if (
            any(not math.isfinite(v) for v in recalls + qps)
            or min(qps) <= 0
            or min(recalls) < 0
            or max(recalls) > 1
        ):
            raise ValueError("invalid recall or throughput")
        result.append(
            {
                "id": key,
                "recall": statistics.median(recalls),
                "recall_min": min(recalls),
                "recall_max": max(recalls),
                "qps": statistics.median(qps),
                "qps_min": min(qps),
                "qps_max": max(qps),
                "repetitions": 3,
                "raw": str(path),
            }
        )
    return result


def await_gates(validation: Path, wait: bool) -> None:
    names = (
        "control-compile",
        "integrated-compile",
        "convert-paper-indexes",
        "convert-yfcc-extra",
        "native-driver-compile",
    )
    while True:
        states = {
            n: json.loads((validation / f"{n}.state.json").read_text())[
                "status"
            ]
            for n in names
        }
        if any(v == "failed" for v in states.values()):
            raise RuntimeError(f"validation prerequisite failed: {states}")
        if all(v == "passed" for v in states.values()):
            return
        if not wait:
            raise RuntimeError(f"prerequisites still running: {states}")
        time.sleep(5)


def execute(command: list, cwd: Path, log: Path, lock_fd: int) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        subprocess.run(
            command,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            check=True,
            pass_fds=(lock_fd,),
        )


def compile_native(args) -> None:
    """Compile each public-API probe with its own checkout's headers and ABI."""
    validation = args.validation_dir
    state = {
        "status": "running",
        "steps": [],
        "source_sha256": sha(HERE / "native_control.cu"),
    }
    state_path = validation / "native-driver-compile.state.json"
    save(state_path, state)
    for name, repo in (
        ("control", args.control_repo),
        ("integration", args.integrated_repo),
    ):
        commands = json.loads(
            (repo / "cpp/build/compile_commands.json").read_text()
        )
        entry = next(
            e
            for e in commands
            if e["file"].endswith("/cuvs_benchmark.cu")
            and "-DCUVS_ANN_BENCH_USE_CUVS_CAGRA=" in e["command"]
        )
        command = [
            str(HERE / "native_control.cu") if p == entry["file"] else p
            for p in shlex.split(entry["command"])
            if not p.startswith("-DANN_BENCH_BUILD_MAIN")
        ]
        for flag in ("-o", "-MF", "-MT"):
            if flag in command:
                command[command.index(flag) + 1] = str(
                    validation / f"native-{name}.o"
                )
        command = ["micromamba", "run", "-p", str(args.environment), *command]
        log = validation / f"native-{name}-compile.log"
        record = {
            "command": command,
            "cwd": entry["directory"],
            "log": str(log),
            "status": "running",
        }
        state["steps"].append(record)
        save(state_path, state)
        with log.open("w") as stream:
            code = subprocess.run(
                command,
                cwd=entry["directory"],
                stdout=stream,
                stderr=subprocess.STDOUT,
            ).returncode
        record.update(returncode=code, status="failed" if code else "passed")
        if code:
            state["status"] = "failed"
            save(state_path, state)
            raise RuntimeError(f"native control compilation failed: {log}")
        save(state_path, state)
    state["status"] = "passed"
    save(state_path, state)


def hash_inputs(args) -> None:
    manifest = json.loads((args.root / "manifest.json").read_text())
    output = args.root / "input_checksums.json"
    values = json.loads(output.read_text()) if output.exists() else {}
    for path, identity in manifest["inputs"].items():
        st = Path(path).stat()
        if (
            st.st_size != identity["bytes"]
            or st.st_mtime_ns != identity["mtime_ns"]
        ):
            raise ValueError("frozen input changed: " + path)
        if path not in values or any(
            values[path].get(k) != v for k, v in identity.items()
        ):
            digest = sha(Path(path))
            after = Path(path).stat()
            if (after.st_size, after.st_mtime_ns) != (
                st.st_size,
                st.st_mtime_ns,
            ):
                raise ValueError("input changed while hashing: " + path)
            values[path] = identity | {"sha256": digest}
            save(output, values)
    print(json.dumps({"verified_input_checksums": len(values)}))


def run(args) -> None:
    root = args.root
    manifest = json.loads((root / "manifest.json").read_text())
    validation = Path(manifest["validation_dir"])
    progress = {
        "status": "waiting_for_builds_and_correctness",
        "completed": [],
        "total": len(manifest["jobs"]),
        "native_completed": [],
        "native_total": 2 * len(manifest["native"]),
    }
    prior_failure = root / "failure.json"
    if prior_failure.exists():
        target = root / "failures" / f"{time.time_ns()}.json"
        target.parent.mkdir(exist_ok=True)
        prior_failure.replace(target)
    save(root / "progress.json", progress)
    await_gates(validation, args.wait)
    hash_inputs(args)
    for path, identity in manifest["inputs"].items():
        st = Path(path).stat()
        if (
            st.st_size != identity["bytes"]
            or st.st_mtime_ns != identity["mtime_ns"]
        ):
            raise ValueError("frozen input changed: " + path)
    repositories = {
        "old": Path(manifest["old_repo"]),
        "integrated": Path(manifest["integrated_repo"]),
        "control": Path(manifest["control_repo"]),
    }
    environment = Path(manifest["environment"])
    provenance = {
        "manifest_sha256": sha(root / "manifest.json"),
        "repositories": {},
        "binaries": {},
    }
    for name, repo in repositories.items():
        provenance["repositories"][name] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    for variant in ("old", "integrated"):
        repo = repositories[variant]
        for binary in BINARIES.values():
            p = repo / "cpp/build/bench/ann" / binary
            provenance["binaries"][str(p)] = sha(p)
        provenance["binaries"][str(repo / "cpp/build/libcuvs.so")] = sha(
            repo / "cpp/build/libcuvs.so"
        )
    save(root / "provenance.json", provenance)
    for name, prefix in (
        ("old", Path(manifest["old_repo"]).parent / "micromamba/envs/cuvs"),
        ("integrated", environment),
    ):
        details = subprocess.check_output(
            ["micromamba", "list", "--explicit", "-p", str(prefix)], text=True
        )
        (root / f"environment-{name}.txt").write_text(details)
    (root / "gpu.txt").write_text(
        subprocess.check_output(["nvidia-smi", "-q"], text=True)
    )
    with (validation / "gpu.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        progress["status"] = "native_correctness"
        save(root / "progress.json", progress)
        # These native upstream smoke tests are separate from the custom bitmap/UDF suite.
        for variant, targets in (
            (
                "control",
                [
                    "CORE_ROARING_ALLOWLIST_TEST",
                    "NEIGHBORS_ANN_CAGRA_FILTER_UDF_TEST",
                    "NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST",
                    "NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST",
                ],
            ),
            (
                "integrated",
                [
                    "NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST",
                    "NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST",
                ],
            ),
        ):
            for target in targets:
                out = root / "correctness" / f"{variant}-{target}.json"
                if target == "NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST":
                    # The initial library builds have finished before this point. Never run
                    # concurrent Ninja instances against the same build directory.
                    execute(
                        [
                            "micromamba",
                            "run",
                            "-p",
                            str(environment),
                            "cmake",
                            "--build",
                            str(repositories[variant] / "cpp/build"),
                            "--parallel",
                            "4",
                            "--target",
                            target,
                        ],
                        root,
                        out.with_suffix(".build.log"),
                        lock.fileno(),
                    )
                command = [
                    "micromamba",
                    "run",
                    "-p",
                    str(environment),
                    "timeout",
                    "1800",
                    str(repositories[variant] / "cpp/build/gtests" / target),
                    "--gtest_output=json:" + str(out),
                ]
                if target == "NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST":
                    command.append("--gtest_filter=*MultiPartition*")
                execute(command, root, out.with_suffix(".log"), lock.fileno())
                tests = json.loads(out.read_text())
                if tests["tests"] == 0 or tests.get("failures", 0):
                    raise ValueError("empty or failing native GPU smoke suite")
        progress["status"] = "benchmarking"
        save(root / "progress.json", progress)
        for job in manifest["jobs"]:
            variant = "old" if job["variant"] == "old" else "integrated"
            config = Path(job["config"])
            if sha(config) != job["config_sha256"]:
                raise ValueError("generated configuration changed")
            payload = json.loads(config.read_text())
            algo = payload["index"][0]["algo"]
            binary = (
                repositories[variant] / "cpp/build/bench/ann" / BINARIES[algo]
            )
            raw = root / "raw" / (job["id"] + ".json")
            raw.parent.mkdir(parents=True, exist_ok=True)
            command = [
                "micromamba",
                "run",
                "-p",
                str(Path(manifest["old_repo"]).parent / "micromamba/envs/cuvs")
                if variant == "old"
                else str(environment),
                str(binary),
                "--search",
                "--mode=throughput",
                "--threads=1",
                "--data_prefix=/data/retrieve_data",
                "--index_prefix=/data/retrieve_data",
                "--benchmark_repetitions=3",
                "--benchmark_min_time=0.10s",
                "--benchmark_min_warmup_time=0.10",
                "--benchmark_enable_random_interleaving=true",
                "--benchmark_report_aggregates_only=false",
                "--benchmark_out_format=json",
                "--benchmark_out=" + str(raw),
                str(config),
            ]
            signature = {
                "config_sha256": job["config_sha256"],
                "binary_sha256": provenance["binaries"][str(binary)],
                "library_sha256": provenance["binaries"][
                    str(repositories[variant] / "cpp/build/libcuvs.so")
                ],
                "manifest_sha256": provenance["manifest_sha256"],
            }
            marker = raw.with_suffix(".complete.json")
            reusable = (
                raw.exists()
                and marker.exists()
                and json.loads(marker.read_text()).get("signature")
                == signature
            )
            if not reusable:
                if raw.exists():
                    archived = (
                        root
                        / "attempts"
                        / str(time.time_ns())
                        / raw.relative_to(root)
                    )
                    archived.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(raw, archived)
                execute(command, root, raw.with_suffix(".log"), lock.fileno())
            read_measurements(raw, job["cases"])
            save(
                marker,
                {
                    "signature": signature,
                    "command": command,
                    "raw_sha256": sha(raw),
                },
            )
            progress["completed"].append(job["id"])
            save(root / "progress.json", progress)
        build_native_drivers(manifest, root, lock.fileno())
        progress["status"] = "native_public_api"
        save(root / "progress.json", progress)
        for native in manifest["native"]:
            for variant in ("control", "integrated"):
                prefix = root / "native" / variant / native["id"]
                prefix.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    "micromamba",
                    "run",
                    "-p",
                    str(environment),
                    str(root / "bin" / f"native-{variant}"),
                    native["config"],
                    str(prefix),
                ]
                execute(
                    command, root, prefix.with_suffix(".log"), lock.fileno()
                )
                progress["native_completed"].append(
                    f"{variant}/{native['id']}"
                )
                save(root / "progress.json", progress)
        progress["status"] = "benchmarks_complete"
        save(root / "progress.json", progress)
    analyze(args)


def build_native_drivers(manifest, root: Path, lock_fd: int) -> None:
    env, validation = (
        Path(manifest["environment"]),
        Path(manifest["validation_dir"]),
    )
    (root / "bin").mkdir(exist_ok=True)
    for name, repo in (
        ("control", manifest["control_repo"]),
        ("integrated", manifest["integrated_repo"]),
    ):
        object_name = "integration" if name == "integrated" else "control"
        build = Path(repo) / "cpp/build"
        command = [
            "micromamba",
            "run",
            "-p",
            str(env),
            str(env / "bin/nvcc"),
            str(validation / f"native-{object_name}.o"),
            "-o",
            str(root / "bin" / f"native-{name}"),
            "-L" + str(build),
            "-lcuvs",
            "-L" + str(env / "lib"),
            "-lrmm",
            "-lrapids_logger",
            "-lkvikio",
            "-Xlinker=-rpath",
            "-Xlinker=" + str(build),
            "-Xlinker=-rpath",
            "-Xlinker=" + str(env / "lib"),
        ]
        execute(
            command, root, root / "bin" / f"native-{name}-link.log", lock_fd
        )


def comparison(old: dict, new: dict) -> dict:
    delta = new["recall"] - old["recall"]
    ratio = new["qps"] / old["qps"]
    return {
        "recall_delta": delta,
        "qps_ratio": ratio,
        "recall_flag": abs(delta) > 0.002 + 1e-12,
        "qps_flag": ratio < 0.9,
    }


def analyze(args) -> None:
    root = args.root
    manifest = json.loads((root / "manifest.json").read_text())
    values = {}
    for job in manifest["jobs"]:
        for row in read_measurements(
            root / "raw" / (job["id"] + ".json"), job["cases"]
        ):
            values[(job["variant"], row["id"])] = row
    rows = []
    for case in manifest["sources"]:
        key = case["id"]
        old, integrated = values[("old", key)], values[("integrated", key)]
        row = {
            "id": key,
            "cohort": case["cohort"],
            "k": case["k"],
            "method": case["method"],
            "old_recall": old["recall"],
            "integrated_recall": integrated["recall"],
            "old_qps": old["qps"],
            "integrated_qps": integrated["qps"],
            **comparison(old, integrated),
        }
        if ("roaring", key) in values:
            roaring = values[("roaring", key)]
            row.update(
                roaring_recall=roaring["recall"],
                roaring_qps=roaring["qps"],
                roaring_qps_ratio=roaring["qps"] / integrated["qps"],
                representation_recall_delta=roaring["recall"]
                - integrated["recall"],
            )
        rows.append(row)
    native_rows = []
    for case in manifest["native"]:
        key = case["id"]
        pair = {}
        for variant in ("control", "integrated"):
            prefix = root / "native" / variant / key
            value = json.loads(prefix.with_suffix(".json").read_text())
            value["qps"] = statistics.median(
                r["qps"] for r in value["repetitions"]
            )
            value["ids_sha256"] = sha(prefix.with_suffix(".ids"))
            pair[variant] = value
        native_rows.append(
            {
                "id": key,
                **comparison(pair["control"], pair["integrated"]),
                "identical_ids": pair["control"]["ids_sha256"]
                == pair["integrated"]["ids_sha256"],
                "control": pair["control"],
                "integrated": pair["integrated"],
            }
        )
    flags = [
        r
        for r in rows
        if r["recall_flag"]
        or r["qps_flag"]
        or abs(r.get("representation_recall_delta", 0)) > 1e-8
    ]
    native_flags = [
        r
        for r in native_rows
        if r["recall_flag"] or r["qps_flag"] or not r["identical_ids"]
    ]
    save(
        root / "comparison.json",
        {
            "rows": rows,
            "native": native_rows,
            "flagged": flags,
            "native_flagged": native_flags,
            "ready_for_review": not flags and not native_flags,
        },
    )
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (root / "comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Upstream integration validation",
        "",
        "All benchmark rows use the same first 2,048 queries and three repetitions.",
        "Settings are frozen paper selections or maximum-observed-recall configurations; no retuning is performed.",
        "",
        "| Dataset | k | Method | Old recall | Integrated recall | Old QPS | Integrated QPS | QPS ratio | Roaring QPS |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['cohort']} | {r['k']} | {r['method']} | {r['old_recall']:.6f} | {r['integrated_recall']:.6f} | {r['old_qps']:.1f} | {r['integrated_qps']:.1f} | {r['qps_ratio']:.3f} | {r.get('roaring_qps', 0):.1f} |"
        )
    lines.extend(
        [
            "",
            f"Threshold flags requiring investigation: {len(flags)} benchmark rows; {len(native_flags)} native-control rows.",
            "Flags are not automatically waived. Native-control timing is separate from cuVS-bench timing.",
            "No push or main-branch update is performed by this runner.",
        ]
    )
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "flags": len(flags),
                "native_flags": len(native_flags),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("prepare", "compile-native", "hash-inputs", "run", "analyze"),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/cuvs-upstream-validation/benchmarks"),
    )
    parser.add_argument(
        "--runs", type=Path, default=Path("/data/retrieve_workshop_runs")
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path("/home/ubuntu/paper_selectivity_revision/data"),
    )
    parser.add_argument(
        "--ivf-report",
        type=Path,
        default=Path("/home/ubuntu/retrieve_ivf_flat_report"),
    )
    parser.add_argument(
        "--data", type=Path, default=Path("/data/retrieve_data")
    )
    parser.add_argument(
        "--old-repo", type=Path, default=Path("/home/ubuntu/cuvs-filter")
    )
    parser.add_argument(
        "--integrated-repo",
        type=Path,
        default=Path("/data/cuvs-upstream-integration"),
    )
    parser.add_argument(
        "--control-repo",
        type=Path,
        default=Path("/data/cuvs-upstream-control"),
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path("/data/cuvs-upstream-validation"),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("/data/cuvs-upstream-validation/indices"),
    )
    parser.add_argument(
        "--allowlist-dir",
        type=Path,
        default=Path("/data/cuvs-upstream-validation/allowlists"),
    )
    parser.add_argument(
        "--environment",
        type=Path,
        default=Path("/data/cuvs-upstream-envs/cuvs-2612"),
    )
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    try:
        {
            "prepare": prepare,
            "compile-native": compile_native,
            "hash-inputs": hash_inputs,
            "run": run,
            "analyze": analyze,
        }[args.action](args)
    except Exception as error:
        if args.action == "run":
            save(args.root / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
