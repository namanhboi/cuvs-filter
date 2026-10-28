#!/usr/bin/env python3
"""Frozen primary-workload batch-1 comparison: three CAGRA methods and IVF-Flat."""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import math
import os
import re
import shutil
import statistics
import struct
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "upstream_2612"))
import validate as frozen  # noqa: E402

WORKLOADS = ("yfcc", "em", "emis", "r")
METHODS = (*frozen.METHODS, "ivf_flat")
LABELS = dict(zip(METHODS, ("Base", "Retain", "NaviX", "IVF-Flat")))
REPS, QUERIES, GATE, FLOOR = 3, 10_000, 1000, 128
DRIFT = 0.0001


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    frozen.save(Path(path), value)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return frozen.sha(Path(path))


def csv_rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(dict.fromkeys(k for r in rows for k in r)),
            lineterminator="\n",
        )
        w.writeheader()
        w.writerows(rows)


def identity(path, known=None):
    p = Path(path)
    st = p.stat()
    item = dict(path=str(p), bytes=st.st_size, mtime_ns=st.st_mtime_ns)
    previous = (known or {}).get(str(p), {})
    if all(
        previous.get(k) == item[k] for k in ("bytes", "mtime_ns")
    ) and previous.get("sha256"):
        item.update(
            sha256=previous["sha256"], hash_origin="inherited_frozen_input"
        )
    else:
        item.update(sha256=sha(p), hash_origin="read_current_bytes")
    if previous.get("sha256") and item["sha256"] != previous["sha256"]:
        raise ValueError(f"Input bytes differ from the frozen experiment: {p}")
    if (p.stat().st_size, p.stat().st_mtime_ns) != (
        st.st_size,
        st.st_mtime_ns,
    ):
        raise ValueError(f"Input changed while hashing: {p}")
    return item


def validate_shards(shards):
    cursor = 0
    for s in shards:
        if s["first_query"] != cursor or s["query_count"] <= 0:
            raise ValueError("Noncontiguous query shards")
        cursor += s["query_count"]
    if cursor != QUERIES or [s["query_count"] for s in shards] != [
        2048
    ] * 4 + [1808]:
        raise ValueError(
            "Expected all 10,000 queries in their five original shards"
        )


def selected_settings(config, row):
    """Bind source table scalars to the actual frozen search configuration."""
    sp = config["index"][0]["search_params"][0]
    fields = (
        ("nprobe",)
        if row["method"] == "ivf_flat"
        else ("itopk", "search_width", "max_iterations")
    )
    if any(int(sp[k]) != int(row[k]) for k in fields):
        raise ValueError(f"Source settings disagree with selection: {row}")
    if row["method"] != "ivf_flat" and sp.get("algo") != "single_cta":
        raise ValueError("Frozen CAGRA point is not single-CTA")


def gate_inputs(root, ds):
    """Create only a prefix for the paired gate; leave original files immutable."""
    out = copy.deepcopy(ds)
    directory = root / "inputs/gate" / ds["name"]
    directory.mkdir(parents=True, exist_ok=True)
    dtype_bytes = 1 if ds["dtype"] == "uint8" else 4
    for key, width in (
        ("query_file", dtype_bytes),
        ("groundtruth_neighbors_file", 4),
    ):
        p = Path(ds[key])
        with p.open("rb") as f:
            rows, cols = struct.unpack("<II", f.read(8))
            if rows < GATE:
                raise ValueError("Too few gate queries")
            data = f.read(GATE * cols * width)
        target = directory / p.name
        target.write_bytes(struct.pack("<II", GATE, cols) + data)
        out[key] = str(target)
    with Path(ds["filter"]["file"]).open("rb") as f:
        magic, version, bits, rows, cols, _ = struct.unpack(
            "<8sIIQQQ", f.read(40)
        )
        if (
            (magic, version, bits) != (b"CUVSBMAP", 1, 32)
            or cols % 32
            or rows < GATE
        ):
            raise ValueError("Gate requires the original aligned dense bitmap")
        words = GATE * cols // 32
        target = directory / "filter.bitmap"
        with target.open("wb") as output:
            output.write(
                struct.pack(
                    "<8sIIQQQ", magic, version, bits, GATE, cols, words
                )
            )
            remaining = words * 4
            while remaining:
                data = f.read(min(8 << 20, remaining))
                if not data:
                    raise ValueError("Truncated gate bitmap")
                output.write(data)
                remaining -= len(data)
        out["filter"]["file"] = str(target)
    return out


def job_config(template, points, ds, root, stage, shard, rep):
    result = copy.deepcopy(template)
    count, first = shard["query_count"], shard["first_query"]
    result["dataset"] = copy.deepcopy(ds)
    serial = stage in ("gate_serial", "trace")
    result["search_basic_param"]["batch_size"] = 1 if serial else count
    search = []
    cases = []
    for point in points:
        sp = copy.deepcopy(point["search"])
        case = point["id"]
        sp.update(validation_case_id=case, n_queries=1 if serial else count)
        output = (
            root
            / "neighbors"
            / stage
            / case
            / f"shard_{shard['index']:02d}.ibin"
        )
        if stage != "trace":
            sp["benchmark_output_neighbors_file"] = str(output)
        trace = (
            root
            / "traces"
            / stage
            / case
            / f"rep_{rep}/shard_{shard['index']:02d}.csv"
        )
        if serial:
            sp.update(
                max_queries=1,
                benchmark_latency_trace_schema=2,
                benchmark_latency_trace_file=str(trace),
                benchmark_latency_workload=point["workload"],
                benchmark_latency_method=point["method"],
                benchmark_latency_repetition=rep,
                benchmark_latency_shard_index=shard["index"],
                benchmark_latency_global_query_offset=first,
                benchmark_latency_start_query=(rep * count) // REPS
                if stage == "trace"
                else 0,
                benchmark_latency_warmup_queries=32,
                benchmark_latency_timer_floor_samples=FLOOR,
                benchmark_latency_reference_recall=point["source_recall"],
                benchmark_latency_target_recall=point["target"],
                benchmark_latency_source_max_queries=2048,
            )
        search.append(sp)
        cases.append(
            dict(
                id=case,
                trace=str(trace) if serial else None,
                neighbors=str(output) if stage != "trace" else None,
            )
        )
    result["index"][0]["search_params"] = search
    return result, cases


def prepare(args):
    root = args.root.resolve()
    if (root / "manifest.json").exists():
        raise FileExistsError(
            "Run already prepared; use run/status/analyze to resume"
        )
    root.mkdir(parents=True, exist_ok=True)
    upstream = read(args.validation / "manifest.json")
    ivf = read(args.ivf_run / "state/manifest.json")
    paper = {
        k: {
            (r["workload"], r["method"]): r
            for r in csv_rows(
                args.paper
                / (
                    "current_gpu_matched_recall"
                    + ("_k100" if k == 100 else "")
                    + ".csv"
                )
            )
        }
        for k in (10, 100)
    }
    flat = {
        (r["workload"], int(r["k"])): r
        for r in csv_rows(args.report / "matched_recall.csv")
        if r["method"] == "ivf_flat"
    }
    known = read(args.validation / "input_checksums.json")
    known.update({r["path"]: r for r in ivf["input_files"]})
    inputs = set()
    for p in (
        args.validation / "manifest.json",
        args.ivf_run / "state/manifest.json",
        args.report / "matched_recall.csv",
        args.paper / "current_gpu_matched_recall.csv",
        args.paper / "current_gpu_matched_recall_k100.csv",
    ):
        inputs.add(str(p.resolve()))
    points, configs = [], {}
    for source in upstream["sources"]:
        w, k, method = source["cohort"], source["k"], source["method"]
        if w not in WORKLOADS or method not in METHODS:
            continue
        selection = flat[w, k] if method == "ivf_flat" else paper[k][w, method]
        selected_settings(source["config"], selection)
        if method == "ivf_flat" and selection["status"] != "measured":
            raise ValueError(
                "Need an actual measured IVF configuration, not an interpolated point"
            )
        cfg = frozen.normalize_config(
            source["config"],
            args.data,
            True,
            args.indices,
            "bitmap",
            source["id"],
            Path("/absent"),
        )
        for sp in cfg["index"][0]["search_params"]:
            sp.pop("roaring_allowlist_file", None)
        index = Path(cfg["index"][0]["file"])
        conversion = read(Path(str(index) + ".json"))
        if not conversion["verified_components"]:
            raise ValueError("Index copy was not independently verified")
        if sha(source["source"]) != source["source_sha256"]:
            raise ValueError(
                f"Original configuration changed: {source['source']}"
            )
        inputs.update((str(index), str(index) + ".json", source["source"]))
        target = 0.8 if w == "yfcc" else 0.95
        recall = float(
            selection["recall"]
            if method == "ivf_flat"
            else selection["recall_median"]
        )
        points.append(
            dict(
                id=source["id"],
                workload=w,
                k=k,
                method=method,
                target=target,
                source_recall=recall,
                source_matched=target <= recall <= target + 0.002,
                source_config=source["source"],
                source_sha256=source["source_sha256"],
                index=str(index),
                build=cfg["index"][0]["build_param"],
                search=cfg["index"][0]["search_params"][0],
            )
        )
        configs[source["id"]] = cfg
    expected = {
        f"{w}__k{k}__{m}"
        for w in WORKLOADS
        for k in (10, 100)
        for m in METHODS
    }
    if {p["id"] for p in points} != expected:
        raise ValueError("Incomplete frozen operating-point matrix")
    runtime = root / "runtime"
    runtime.mkdir()
    binaries = {}
    for algo, binary in list(frozen.BINARIES.items())[:2]:
        target = runtime / binary
        shutil.copy2(args.build / "bench/ann" / binary, target)
        if b"gpu_latency_ns" not in target.read_bytes():
            raise ValueError("Rebuild benchmark with schema-2 dual timings")
        binaries[algo] = str(target)
    shutil.copy2(args.build / "libcuvs.so", runtime / "libcuvs.so")
    env = os.environ | {"LD_LIBRARY_PATH": f"{runtime}:{args.environment}/lib"}
    links = {
        a: subprocess.check_output(["ldd", b], env=env, text=True)
        for a, b in binaries.items()
    }
    if any(
        f"libcuvs.so => {runtime}/libcuvs.so" not in s or "not found" in s
        for s in links.values()
    ):
        raise ValueError("Runtime libraries did not resolve to the snapshot")
    jobs = []
    for w in WORKLOADS:
        for k in (10, 100):
            shards = ivf["inputs"][f"{w}_k{k}"]["shards"]
            validate_shards(shards)
            for shard in shards:
                inputs.update(
                    shard[f]
                    for f in ("query_file", "groundtruth_file", "bitmap_file")
                )
            for algo, methods in (
                ("cuvs_cagra", frozen.METHODS),
                ("cuvs_ivf_flat", ("ivf_flat",)),
            ):
                group = [
                    p
                    for p in points
                    if p["workload"] == w
                    and p["k"] == k
                    and p["method"] in methods
                ]
                template = configs[group[0]["id"]]
                inputs.add(template["dataset"]["base_file"])
                for stage in (
                    "gate_batch",
                    "gate_serial",
                    "baseline",
                    "trace",
                ):
                    for rep in range(REPS if stage == "trace" else 1):
                        for i, s in enumerate(
                            shards
                            if stage in ("baseline", "trace")
                            else shards[:1]
                        ):
                            ds = copy.deepcopy(template["dataset"])
                            ds["name"] = f"batch1-{w}-k{k}-shard{i}"
                            ds.update(
                                query_file=s["query_file"],
                                groundtruth_neighbors_file=s[
                                    "groundtruth_file"
                                ],
                            )
                            ds["filter"]["file"] = s["bitmap_file"]
                            count = s["query_count"]
                            if stage.startswith("gate_"):
                                directory = root / "inputs/gate" / ds["name"]
                                if (
                                    stage == "gate_batch"
                                    and not directory.exists()
                                ):
                                    gate_ds = gate_inputs(root, ds)
                                else:
                                    gate_ds = copy.deepcopy(ds)
                                    for key in (
                                        "query_file",
                                        "groundtruth_neighbors_file",
                                    ):
                                        gate_ds[key] = str(
                                            directory / Path(ds[key]).name
                                        )
                                    gate_ds["filter"]["file"] = str(
                                        directory / "filter.bitmap"
                                    )
                                ds, count = gate_ds, GATE
                                inputs.update(
                                    (
                                        ds["query_file"],
                                        ds["groundtruth_neighbors_file"],
                                        ds["filter"]["file"],
                                    )
                                )
                            sh = dict(
                                index=i,
                                first_query=s["first_query"],
                                query_count=count,
                            )
                            cfg, cases = job_config(
                                template, group, ds, root, stage, sh, rep
                            )
                            job_id = f"{stage}/{w}_k{k}/{algo}/rep_{rep}/shard_{i:02d}"
                            config = root / "configs" / f"{job_id}.json"
                            save(config, cfg)
                            jobs.append(
                                dict(
                                    id=job_id,
                                    stage=stage,
                                    workload=w,
                                    k=k,
                                    algo=algo,
                                    repetition=rep,
                                    shard=i,
                                    first=s["first_query"],
                                    count=count,
                                    config=str(config),
                                    raw=str(root / "raw" / f"{job_id}.json"),
                                    cases=cases,
                                    config_sha256=sha(config),
                                )
                            )
    # Complete each workload's traces early; group corpus/index loads within each stage.
    order = {
        s: i
        for i, s in enumerate(
            ("gate_batch", "gate_serial", "baseline", "trace")
        )
    }
    jobs.sort(
        key=lambda j: (
            order[j["stage"]],
            WORKLOADS.index(j["workload"]),
            j["k"],
            j["algo"],
            j["repetition"],
            j["shard"],
        )
    )
    input_records = [identity(p, known) for p in sorted(inputs)]
    manifest = dict(
        schema=1,
        created_at=now(),
        points=points,
        jobs=jobs,
        input_files=input_records,
        binaries=binaries,
        runtime=[identity(p) for p in sorted(runtime.iterdir())],
        environment=str(args.environment),
        linked_libraries=links,
        source_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
        ).strip(),
        runner_sha256=sha(__file__),
        benchmark_header_sha256=sha(
            HERE.parents[2] / "cpp/bench/ann/src/common/benchmark.hpp"
        ),
        contracts=dict(
            queries=QUERIES,
            repetitions=REPS,
            timed_calls=QUERIES * REPS * 32,
            cagra_mode="single_cta",
            filter="dense_bitmap",
            drift_limit=DRIFT,
            host="API call through event completion",
            gpu="CUDA event interval",
            floor_subtraction=False,
        ),
    )
    save(root / "manifest.json", manifest)
    write_csv(
        root / "configurations.csv",
        [
            dict(
                id=p["id"],
                workload=p["workload"],
                k=p["k"],
                method=p["method"],
                target=p["target"],
                source_recall=p["source_recall"],
                source_matched=p["source_matched"],
                index=p["index"],
                search_json=json.dumps(p["search"], sort_keys=True),
            )
            for p in points
        ],
    )
    print(
        f"Prepared {len(points)} points, {len(jobs)} processes; {QUERIES * REPS * 32:,} timed single-query calls",
        flush=True,
    )


def trace_rows(job, case):
    rows = csv_rows(case["trace"])
    queries = [r for r in rows if r["record_kind"] == "query"]
    floors = [r for r in rows if r["record_kind"] == "timer_floor"]
    if (
        len(queries) != job["count"]
        or len(floors) != FLOOR
        or len(rows) != len(queries) + len(floors)
    ):
        raise ValueError("Incomplete trace")
    if sorted(int(r["query_global_id"]) for r in queries) != list(
        range(job["first"], job["first"] + job["count"])
    ):
        raise ValueError("Trace does not cover every query exactly once")
    if sorted(int(r["query_local_id"]) for r in queries) != list(
        range(job["count"])
    ) or [int(r["call_order"]) for r in queries] != list(range(job["count"])):
        raise ValueError("Invalid local query IDs or call order")
    for r in rows:
        if (
            int(r["schema_version"]),
            int(r["k"]),
            r["workload"],
            r["method"],
            int(r["repetition"]),
            int(r["shard_index"]),
            int(r["max_queries"]),
        ) != (
            2,
            job["k"],
            job["workload"],
            case["id"].split("__")[-1],
            job["repetition"],
            job["shard"],
            1,
        ):
            raise ValueError("Trace identity/schema mismatch")
        if (
            int(r["host_latency_ns"]) <= 0
            or int(r["gpu_latency_ns"]) < 0
            or (r["record_kind"] == "query" and int(r["gpu_latency_ns"]) == 0)
        ):
            raise ValueError("Invalid timing sample")
    return queries, floors


def observations(job):
    rows = [
        r
        for r in read(job["raw"])["benchmarks"]
        if r.get("run_type") != "aggregate"
    ]
    found = {}
    for r in rows:
        if r.get("error_occurred") or r.get("skipped"):
            raise ValueError(f"Benchmark failed: {r.get('error_message', r)}")
        match = re.search(
            r'(?:^|#)validation_case_id="([^"]+)"', r.get("label", "")
        )
        if not match or match[1] in found:
            raise ValueError("Missing or duplicate point identity")
        case_id = match[1]
        for key in (
            "FilterViolations",
            "InvalidSentinelErrors",
            "SentinelOrderErrors",
            "InvalidSentinelDistanceErrors",
        ):
            if key not in r or float(r[key]) != 0:
                raise ValueError(
                    f"{case_id}: invalid output {key}={r.get(key)}"
                )
        # Base historically emits duplicate valid slots; its distinct-ID recall is the
        # source experiment's contract. Keep that diagnostic, never count duplicates as hits.
        if (
            not case_id.endswith("__default_cagra")
            and float(r["DuplicateOutputQueries"]) != 0
        ):
            raise ValueError(f"{case_id}: duplicate output")
        serial = job["stage"] in ("trace", "gate_serial")
        if int(r["n_queries"]) != (1 if serial else job["count"]):
            raise ValueError("Wrong search batch size")
        if serial and int(r["total_queries"]) != job["count"]:
            raise ValueError("Incomplete serialized pass")
        weight = round(float(r["ValidGTFraction"]) * job["count"] * job["k"])
        recall = float(r["ValidGTRecall"])
        if not weight or not math.isfinite(recall) or not 0 <= recall <= 1:
            raise ValueError("Invalid recall")
        found[case_id] = dict(
            recall=recall,
            valid_gt=weight,
            matching=round(recall * weight),
            duplicate_query_rate=float(r["DuplicateOutputQueries"]),
            qps=float(r["items_per_second"]),
        )
    if set(found) != {c["id"] for c in job["cases"]}:
        raise ValueError("Missing or extra benchmark cases")
    for case in job["cases"]:
        if case["trace"]:
            trace_rows(job, case)
        if case["neighbors"]:
            p = Path(case["neighbors"])
            if p.stat().st_size != 8 + job["count"] * job["k"] * 4:
                raise ValueError("Neighbor output has incorrect dimensions")
    return found


def output_hashes(job):
    paths = [job["raw"]] + [
        p for c in job["cases"] for p in (c["trace"], c["neighbors"]) if p
    ]
    return {p: sha(p) for p in paths}


def completed(job, signature):
    marker = Path(job["raw"] + ".complete.json")
    if not marker.exists():
        return False
    saved = read(marker)
    if (
        saved["signature"] != signature
        or saved["config_sha256"] != sha(job["config"])
        or saved["outputs"] != output_hashes(job)
    ):
        raise ValueError(f"Saved job changed; refusing reuse: {job['id']}")
    observations(job)
    return True


def check_inputs(manifest):
    for item in manifest["input_files"] + manifest["runtime"]:
        p = Path(item["path"])
        st = p.stat()
        if st.st_size != item["bytes"] or st.st_mtime_ns != item["mtime_ns"]:
            raise ValueError(f"Frozen file changed: {p}")
    for item in manifest["runtime"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Frozen runtime changed")
    for job in manifest["jobs"]:
        if sha(job["config"]) != job["config_sha256"]:
            raise ValueError("Frozen configuration changed")


def gate_report(root, manifest):
    import numpy as np

    obs, outputs = {}, {}
    for j in manifest["jobs"]:
        if j["stage"] not in ("gate_batch", "gate_serial"):
            continue
        for key, row in observations(j).items():
            obs[j["stage"], key] = row
        for c in j["cases"]:
            outputs[j["stage"], c["id"]] = np.fromfile(
                c["neighbors"], dtype="<u4", offset=8
            ).reshape(GATE, j["k"])
    result = []
    for p in manifest["points"]:
        a, b = (obs[s, p["id"]] for s in ("gate_batch", "gate_serial"))
        delta = abs(a["recall"] - b["recall"])
        sets = (
            sum(
                set(x.tolist()) == set(y.tolist())
                for x, y in zip(
                    outputs["gate_batch", p["id"]],
                    outputs["gate_serial", p["id"]],
                )
            )
            / GATE
        )
        result.append(
            dict(
                id=p["id"],
                batch_recall=a["recall"],
                serial_recall=b["recall"],
                recall_delta=delta,
                identical_output_set_fraction=sets,
                passed=delta <= DRIFT + 1e-12,
            )
        )
    save(
        root / "analysis/gate.json",
        dict(rows=result, passed=all(r["passed"] for r in result)),
    )
    serial_jobs = [j for j in manifest["jobs"] if j["stage"] == "gate_serial"]
    measured_search_seconds = 0.0
    overhead_seconds = 0.0
    for j in serial_jobs:
        search_seconds = sum(
            sum(int(q["host_latency_ns"]) for q in trace_rows(j, c)[0]) / 1e9
            for c in j["cases"]
        )
        measured_search_seconds += search_seconds
        elapsed = read(j["raw"] + ".complete.json")["seconds"]
        overhead_seconds += max(0, elapsed - search_seconds)
    save(
        root / "analysis/eta.json",
        dict(
            estimated_main_trace_seconds=measured_search_seconds
            * QUERIES
            * REPS
            / GATE
            + overhead_seconds * 15,
            estimated_search_only_seconds=measured_search_seconds
            * QUERIES
            * REPS
            / GATE,
            basis="All 32 paired-gate points; extrapolate search samples and observed per-process setup. Approximate; excludes baselines.",
            recorded_at=now(),
        ),
    )
    write_csv(root / "analysis/gate.csv", result)
    if not all(r["passed"] for r in result):
        raise ValueError(
            "Paired gate recall drift exceeds 0.0001; inspect analysis/gate.json"
        )


def run(args):
    root = args.root.resolve()
    manifest = read(root / "manifest.json")
    check_inputs(manifest)
    signature = sha(root / "manifest.json")
    if sha(__file__) != manifest["runner_sha256"]:
        raise ValueError("Runner changed; record a revision before resuming")
    (root / "logs").mkdir(exist_ok=True)
    with (root / "run.lock").open("w") as own, args.gpu_lock.open("a") as gpu:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu, fcntl.LOCK_EX)
        finished = []
        start = time.monotonic()
        for job in manifest["jobs"]:
            if args.stage != "all" and job["stage"] != args.stage:
                continue
            if job["stage"] == "baseline":
                gate_report(root, manifest)
            if completed(job, signature):
                finished.append(job["id"])
                continue
            state = dict(
                status="running",
                started_at=now(),
                pid=os.getpid(),
                active=job["id"],
                completed=len(finished),
                total=len(manifest["jobs"]),
                elapsed_seconds=time.monotonic() - start,
            )
            save(root / "progress.json", state)
            raw = Path(job["raw"])
            raw.parent.mkdir(parents=True, exist_ok=True)
            existing = [
                p
                for p in [raw]
                + [
                    Path(p)
                    for c in job["cases"]
                    for p in (c["trace"], c["neighbors"])
                    if p
                ]
                if p.exists()
            ]
            for p in existing:
                archived = (
                    root
                    / "attempts"
                    / str(time.time_ns())
                    / p.relative_to(root)
                )
                archived.parent.mkdir(parents=True, exist_ok=True)
                p.rename(archived)
            command = [
                "taskset",
                "-c",
                str(min(os.sched_getaffinity(0))),
                manifest["binaries"][job["algo"]],
                "--search",
                "--mode=latency",
                "--threads=1",
                "--data_prefix=/data/retrieve_data",
                "--index_prefix=/data/retrieve_data",
                "--benchmark_repetitions=1",
                "--benchmark_min_time=1x",
                "--benchmark_min_warmup_time=0",
                "--benchmark_enable_random_interleaving=false",
                "--benchmark_report_aggregates_only=false",
                "--benchmark_out_format=json",
                f"--benchmark_out={raw}",
                job["config"],
            ]
            log = root / "logs" / (job["id"].replace("/", "_") + ".log")
            save(
                raw.with_suffix(".command.json"),
                dict(command=command, started_at=now()),
            )
            env = os.environ | {
                "LD_LIBRARY_PATH": f"{root}/runtime:{manifest['environment']}/lib",
                "CUDA_VISIBLE_DEVICES": "0",
                "OMP_NUM_THREADS": "1",
            }
            before = time.monotonic()
            with log.open("w") as f:
                subprocess.run(
                    command,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    check=True,
                )
            result = observations(job)
            save(
                Path(str(raw) + ".complete.json"),
                dict(
                    signature=signature,
                    config_sha256=sha(job["config"]),
                    outputs=output_hashes(job),
                    seconds=time.monotonic() - before,
                    observations=result,
                    finished_at=now(),
                ),
            )
            finished.append(job["id"])
            print(
                f"{len(finished)}/{len(manifest['jobs'])} {job['id']} ({time.monotonic() - before:.1f}s)",
                flush=True,
            )
            if (
                job["stage"] == "trace"
                and job["repetition"] == 2
                and job["shard"] == 4
            ):
                analyze(args, partial=True)
        if args.stage == "all":
            analyze(args)
        save(
            root / "progress.json",
            dict(
                status="complete" if args.stage == "all" else "stage_complete",
                stage=args.stage,
                completed=len(finished),
                total=len(manifest["jobs"]),
                finished_at=now(),
            ),
        )


def aggregate(rows):
    return sum(r["matching"] for r in rows) / sum(r["valid_gt"] for r in rows)


def stats(values):
    values = sorted(values)
    return dict(
        mean_us=statistics.fmean(values) / 1000,
        std_us=statistics.pstdev(values) / 1000,
        **{
            f"p{q}_us": values[max(0, math.ceil(q / 100 * len(values)) - 1)]
            / 1000
            for q in (50, 95, 99)
        },
        max_us=max(values) / 1000,
    )


def analyze(args, partial=False):
    import numpy as np

    root = args.root.resolve()
    m = read(root / "manifest.json")
    base, measured, samples, floors = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(dict),
        defaultdict(list),
    )
    for j in m["jobs"]:
        if (
            j["stage"] not in ("baseline", "trace")
            or not Path(j["raw"] + ".complete.json").exists()
        ):
            continue
        obs = observations(j)
        for c in j["cases"]:
            key = c["id"]
            if j["stage"] == "baseline":
                base[key].append(obs[key])
                continue
            measured[key, j["repetition"]].append(obs[key])
            qs, fs = trace_rows(j, c)
            for q in qs:
                sample_key = (key, int(q["query_global_id"]))
                if j["repetition"] in samples[sample_key]:
                    raise ValueError("Duplicate query/repetition")
                samples[sample_key][j["repetition"]] = (
                    int(q["host_latency_ns"]),
                    int(q["gpu_latency_ns"]),
                )
            floors[key].extend(
                (int(q["host_latency_ns"]), int(q["gpu_latency_ns"]))
                for q in fs
            )
    summaries, query_rows = [], []
    arrays = {}
    for p in m["points"]:
        key = p["id"]
        if len(base[key]) != 5 or any(
            len(measured[key, r]) != 5 for r in range(REPS)
        ):
            if partial:
                continue
            raise ValueError(f"Incomplete workload: {key}")
        per_query = []
        for q in range(QUERIES):
            rep = samples[key, q]
            if set(rep) != {0, 1, 2}:
                raise ValueError(f"Incomplete samples for {key} query {q}")
            med = [
                statistics.median(rep[r][col] for r in range(REPS))
                for col in (0, 1)
            ]
            per_query.append(med)
            query_rows.append(
                dict(
                    id=key,
                    query_id=q,
                    **{
                        f"{metric}_rep{r}_ns": rep[r][col]
                        for col, metric in enumerate(("host", "gpu"))
                        for r in range(REPS)
                    },
                    host_median_ns=med[0],
                    gpu_median_ns=med[1],
                )
            )
        values = np.array(per_query)
        arrays[key] = values
        recall = [aggregate(measured[key, r]) for r in range(REPS)]
        baseline = aggregate(base[key])
        delta = abs(statistics.median(recall) - baseline)
        matched = (
            p["source_matched"]
            and min(recall) >= p["target"] - 1e-12
            and max(recall) <= p["target"] + 0.002 + 1e-12
        )
        row = dict(
            id=key,
            workload=p["workload"],
            k=p["k"],
            method=p["method"],
            queries=QUERIES,
            passes=REPS,
            target=p["target"],
            source_recall=p["source_recall"],
            source_matched=p["source_matched"],
            baseline_recall=baseline,
            recall_median=statistics.median(recall),
            recall_min=min(recall),
            recall_max=max(recall),
            recall_delta=delta,
            recall_preserved=delta <= DRIFT + 1e-12,
            status="target_matched"
            if matched
            else (
                "source_maximum_unmatched"
                if not p["source_matched"]
                else "serialized_out_of_window"
            ),
            duplicate_query_rate_max=max(
                r["duplicate_query_rate"]
                for rep in range(REPS)
                for r in measured[key, rep]
            ),
        )
        if len(floors[key]) != 15 * FLOOR:
            raise ValueError("Incomplete floor samples")
        for col, metric in enumerate(("host", "gpu")):
            row.update(
                {
                    f"{metric}_{name}": v
                    for name, v in stats(values[:, col].tolist()).items()
                }
            )
            row[f"{metric}_floor_p50_us"] = (
                statistics.median(f[col] for f in floors[key]) / 1000
            )
        summaries.append(row)
    out = root / "analysis"
    write_csv(out / "latency_summary.csv", summaries)
    write_csv(out / "per_query_samples.csv", query_rows)
    save(
        out / "latency_summary.json",
        dict(
            complete=len(summaries) == 32,
            rows=summaries,
            queries_per_method=QUERIES,
            passes=REPS,
            recall_drift_flags=[
                r["id"] for r in summaries if not r["recall_preserved"]
            ],
        ),
    )
    lines = [
        "# Batch-1 latency: frozen QPS–recall operating points",
        "",
        "Dense bitmaps; CAGRA SINGLE_CTA; all 10,000 queries; three passes.",
        "Statistics summarize each query's median across passes. Times are microseconds. No timer-floor subtraction.",
        "CUDA timing is a stream-event interval, not summed kernel execution time. Unmatched rows are excluded from matched-recall comparisons.",
        "",
        "| Workload | k | Method | Recall | Status | Host mean | Host p50 | Host p95 | Host p99 | GPU mean | GPU p50 | GPU p95 | GPU p99 |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summaries:
        lines.append(
            f"| {r['workload']} | {r['k']} | {LABELS[r['method']]} | {r['recall_median']:.6f} | {r['status']} | "
            + " | ".join(
                f"{r[f'{metric}_{stat}_us']:.2f}"
                for metric in ("host", "gpu")
                for stat in ("mean", "p50", "p95", "p99")
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"Completed selections: {len(summaries)}/32. See CSV/JSON for max, standard deviation, timer floor, duplicate diagnostics, and recall drift flags.",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    if summaries:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for k in (10, 100):
            for col, metric in enumerate(("host", "gpu")):
                fig, axes = plt.subplots(
                    1, 4, figsize=(15, 3.5), layout="constrained"
                )
                for ax, w in zip(axes, WORKLOADS):
                    for method in METHODS:
                        key = f"{w}__k{k}__{method}"
                        if key in arrays:
                            x = np.sort(arrays[key][:, col]) / 1000
                            ax.plot(
                                x,
                                np.arange(1, len(x) + 1) / len(x),
                                label=LABELS[method],
                            )
                    ax.set(
                        xscale="log",
                        title=f"{w.upper()}, k={k}",
                        xlabel=f"{metric.capitalize()} latency (µs)",
                        ylabel="CDF",
                    )
                    ax.grid(alpha=0.2)
                handles, labels = next(
                    (
                        ax.get_legend_handles_labels()
                        for ax in axes
                        if ax.lines
                    ),
                    ([], []),
                )
                if handles:
                    fig.legend(
                        handles, labels, loc="outside lower center", ncol=4
                    )
                for ext in ("pdf", "png"):
                    fig.savefig(out / f"{metric}_cdf_k{k}.{ext}", dpi=180)
                plt.close(fig)
    if not partial and any(not r["recall_preserved"] for r in summaries):
        raise ValueError(
            "Full serialized recall differs from merged batched reference; see saved flags"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "run", "analyze", "status"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--stage",
        choices=("all", "gate_batch", "gate_serial", "baseline", "trace"),
        default="all",
    )
    p.add_argument(
        "--validation",
        type=Path,
        default=Path("/data/cuvs-upstream-validation/benchmarks"),
    )
    p.add_argument(
        "--ivf-run",
        type=Path,
        default=Path(
            "/data/retrieve_workshop_runs/a100_ivf_bitmap_20261014T024338Z"
        ),
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("/home/ubuntu/retrieve_ivf_flat_report"),
    )
    p.add_argument(
        "--paper",
        type=Path,
        default=Path("/home/ubuntu/paper_selectivity_revision/data"),
    )
    p.add_argument("--data", type=Path, default=Path("/data/retrieve_data"))
    p.add_argument(
        "--indices",
        type=Path,
        default=Path("/data/cuvs-upstream-validation/indices"),
    )
    p.add_argument(
        "--build",
        type=Path,
        default=Path("/data/cuvs-upstream-integration/cpp/build"),
    )
    p.add_argument(
        "--environment",
        type=Path,
        default=Path("/data/cuvs-upstream-envs/cuvs-2612"),
    )
    p.add_argument(
        "--gpu-lock",
        type=Path,
        default=Path("/data/cuvs-upstream-validation/gpu.lock"),
    )
    args = p.parse_args()
    try:
        if args.action == "prepare":
            prepare(args)
        elif args.action == "run":
            run(args)
        elif args.action == "analyze":
            analyze(args)
        else:
            print(json.dumps(read(args.root / "progress.json"), indent=2))
    except Exception as e:
        if args.action in ("prepare", "run"):
            save(args.root / "failure.json", dict(error=repr(e), at=now()))
            state = (
                read(args.root / "progress.json")
                if (args.root / "progress.json").exists()
                else {}
            )
            save(
                args.root / "progress.json",
                state | dict(status="failed", error=repr(e), failed_at=now()),
            )
        raise


if __name__ == "__main__":
    main()
