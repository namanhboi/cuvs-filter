#!/usr/bin/env python3
"""Reproduce the paper's bitmap workloads with native cuVS IVF search."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import statistics
import struct
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
WORKLOADS = ("yfcc", "em", "emis", "r")
METHODS = ("ivf_flat", "ivf_sq", "ivf_pq", "ivf_pq_refined")
TARGETS = {"yfcc": 0.8, "em": 0.95, "emis": 0.95, "r": 0.95}
ERROR_COUNTERS = (
    "FilterViolations",
    "DuplicateOutputQueries",
    "InvalidSentinelErrors",
    "SentinelOrderErrors",
    "InvalidSentinelDistanceErrors",
)
SCHEMA = 1


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def matrix_header(path, item_bytes):
    with Path(path).open("rb") as source:
        rows, cols = struct.unpack("<II", source.read(8))
    if Path(path).stat().st_size != 8 + rows * cols * item_bytes:
        raise ValueError(f"Invalid matrix size: {path}")
    return rows, cols


def bitmap_header(path, rows, cols):
    with Path(path).open("rb") as source:
        magic, version, word_bits, actual_rows, actual_cols, words = (
            struct.unpack("<8sIIQQQ", source.read(40))
        )
    expected = (rows * cols + 31) // 32
    if (magic, version, word_bits, actual_rows, actual_cols, words) != (
        b"CUVSBMAP",
        1,
        32,
        rows,
        cols,
        expected,
    ) or Path(path).stat().st_size != 40 + words * 4:
        raise ValueError(f"Invalid bitmap header: {path}")


def index_specs():
    output = []
    for corpus in ("yfcc", "arxiv"):
        for kind in ("flat", "sq", "pq"):
            for nlist in (1024, 2048, 4096):
                dims = (32, 64) if corpus == "yfcc" else (64, 128)
                for pq_dim in dims if kind == "pq" else (0,):
                    build = dict(nlist=nlist, niter=25)
                    if kind == "sq":
                        build["max_train_points_per_cluster"] = 256
                    else:
                        build["ratio"] = 10
                    if kind == "pq":
                        build.update(
                            pq_dim=pq_dim, pq_bits=8, codebook_kind="subspace"
                        )
                    output.append(
                        dict(
                            id=f"{corpus}_{kind}_n{nlist}"
                            + (f"_m{pq_dim}" if pq_dim else ""),
                            corpus=corpus,
                            kind=kind,
                            build=build,
                            dtype="uint8"
                            if corpus == "yfcc" and kind != "sq"
                            else "float",
                        )
                    )
    return output


def point_id(point):
    return f"{point['index_id']}_p{point['nprobe']}_f{point['refine_ratio']}"


def coarse_points(spec):
    ratios = (1, 2, 4, 8, 16, 32, 64) if spec["kind"] == "pq" else (1,)
    return [
        dict(index_id=spec["id"], nprobe=2**exponent, refine_ratio=ratio)
        for exponent in range(int(math.log2(spec["build"]["nlist"])) + 1)
        for ratio in ratios
    ]


def method_for(spec, ratio):
    return (
        "ivf_pq_refined"
        if spec["kind"] == "pq" and ratio > 1
        else f"ivf_{spec['kind']}"
    )


def prepare(args):
    root = args.run_root
    manifest_path = root / "state/manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["schema_version"] != SCHEMA:
            raise ValueError("Unsupported run schema")
        for source in manifest["input_files"]:
            if identity(source["path"]) != {
                key: source[key] for key in ("path", "bytes", "mtime_ns")
            }:
                raise ValueError(
                    f"Input changed since preparation: {source['path']}"
                )
        for source in manifest["runtime_files"]:
            if digest(source["path"]) != source["sha256"]:
                raise ValueError(
                    f"Runtime changed since preparation: {source['path']}"
                )
        return manifest
    if shutil.disk_usage(root.parent).free < 180 * 1024**3:
        raise ValueError(
            "At least 180 GiB free space is required for this index sweep"
        )
    root.mkdir(parents=True, exist_ok=True)
    data = args.data_root
    sources = set()
    inputs = {}
    conversions = read_json(
        data
        / "retrieve_workshop/exact_bitmap_a100/yfcc/throughput_10000/manifest.json"
    )
    float_shards = {row["first_query"]: row for row in conversions["shards"]}
    converted_base = Path(conversions["base_file"])
    if conversions["conversion"] != "uint8_to_float32_coordinate_exact":
        raise ValueError("YFCC conversion must preserve coordinates exactly")
    import numpy as np

    original_base = data / "yfcc-10M/base.10M.u8bin"
    original = np.memmap(
        original_base,
        dtype=np.uint8,
        offset=8,
        mode="r",
        shape=(10000000, 192),
    )
    converted = np.memmap(
        converted_base,
        dtype=np.float32,
        offset=8,
        mode="r",
        shape=(10000000, 192),
    )
    print(
        f"[{utc()}] Checking the existing YFCC uint8-to-float conversion",
        flush=True,
    )
    for first in range(0, 10000000, 65536):
        if not np.array_equal(
            original[first : first + 65536], converted[first : first + 65536]
        ):
            raise ValueError(
                "YFCC base conversion differs from original coordinates"
            )
    del original, converted
    for k in (10, 100):
        for workload in WORKLOADS:
            prefix = "navix_bitmap" if k == 10 else "navix_bitmap_k100"
            directory = (
                "yfcc" if workload == "yfcc" else f"arxiv-large/{workload}"
            )
            source_manifest = (
                data / prefix / directory / "throughput_10000/manifest.json"
            )
            source = read_json(source_manifest)
            sources.add(source_manifest)
            size, dim = (
                (10000000, 192) if workload == "yfcc" else (2735264, 4096)
            )
            if source["base_rows"] != size or source["query_rows"] != 10000:
                raise ValueError(f"Dataset mismatch: {source_manifest}")
            base = (
                original_base
                if workload == "yfcc"
                else data / "arxiv-for-fanns-large/base.fbin"
            )
            if matrix_header(base, 1 if workload == "yfcc" else 4) != (
                size,
                dim,
            ):
                raise ValueError(f"Base shape mismatch: {base}")
            sources.add(base)
            sources.add(converted_base)
            shards = []
            first = 0
            for shard in source["shards"]:
                count = shard["query_count"]
                if shard["first_query"] != first or count != min(
                    2048, 10000 - first
                ):
                    raise ValueError(
                        f"Incomplete or reordered shards: {source_manifest}"
                    )
                directory = Path(shard["directory"])
                query = directory / "query.bin"
                gt = directory / "groundtruth.ibin"
                bitmap = Path(shard["bitmap"])
                if matrix_header(query, 1 if workload == "yfcc" else 4) != (
                    count,
                    dim,
                ):
                    raise ValueError(f"Query shape mismatch: {query}")
                gt_rows, gt_k = matrix_header(gt, 4)
                if gt_rows != count or gt_k < k:
                    raise ValueError(f"Ground truth shape mismatch: {gt}")
                bitmap_header(bitmap, count, size)
                float_query = (
                    Path(float_shards[first]["query_file"])
                    if workload == "yfcc"
                    else query
                )
                if workload == "yfcc":
                    if matrix_header(float_query, 4) != (count, dim):
                        raise ValueError("YFCC float query shape differs")
                    a = np.memmap(
                        query,
                        dtype=np.uint8,
                        offset=8,
                        mode="r",
                        shape=(count, dim),
                    )
                    b = np.memmap(
                        float_query,
                        dtype=np.float32,
                        offset=8,
                        mode="r",
                        shape=(count, dim),
                    )
                    if not np.array_equal(a, b):
                        raise ValueError("YFCC query conversion differs")
                    del a, b
                # Verify valid GT entries obey exactly the query's bitmap, including k=100 padding.
                ids = np.memmap(
                    gt,
                    dtype=np.uint32,
                    offset=8,
                    mode="r",
                    shape=(count, gt_k),
                )[:, :k]
                words = np.memmap(bitmap, dtype=np.uint32, offset=40, mode="r")
                for q in range(count):
                    valid = ids[q][ids[q] < size].astype(np.uint64)
                    if len(np.unique(valid)) != len(valid):
                        raise ValueError(
                            f"Duplicate valid GT IDs: {gt}, query {q}"
                        )
                    bits = q * size + valid
                    if not np.all((words[bits >> 5] >> (bits & 31)) & 1):
                        raise ValueError(
                            f"Ground truth fails bitmap: {gt}, query {q}"
                        )
                del ids, words
                sources.update((query, gt, bitmap, float_query))
                shards.append(
                    dict(
                        first_query=first,
                        query_count=count,
                        query_file=str(query),
                        float_query_file=str(float_query),
                        groundtruth_file=str(gt),
                        bitmap_file=str(bitmap),
                    )
                )
                first += count
            if first != 10000 or len(shards) != 5:
                raise ValueError("Expected five shards and 10,000 queries")
            inputs[f"{workload}_k{k}"] = dict(
                workload=workload,
                k=k,
                base_file=str(base),
                float_base_file=str(
                    converted_base if workload == "yfcc" else base
                ),
                size=size,
                dim=dim,
                shards=shards,
                source_manifest=str(source_manifest),
            )
    runtime = [args.build_dir / "libcuvs.so"]
    for kind in ("flat", "sq", "pq"):
        binary = (
            args.build_dir / "bench/ann" / f"CUVS_IVF_{kind.upper()}_ANN_BENCH"
        )
        if not os.access(binary, os.X_OK):
            raise ValueError(f"Missing executable: {binary}")
        runtime.append(binary)
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    if "A100" not in gpu or "81920" not in gpu:
        raise ValueError(f"Expected A100 80GB: {gpu}")
    reference_rows = []
    references = []
    for k, name in (
        (10, "current_gpu_matched_recall.csv"),
        (100, "current_gpu_matched_recall_k100.csv"),
    ):
        source = args.paper_data / name
        destination = root / "references" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        references.append(
            {
                **identity(source),
                "sha256": digest(source),
                "snapshot": str(destination),
            }
        )
        with source.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["method"] not in (
                    "default_cagra_accumulator",
                    "navix_reference",
                ):
                    continue
                if (
                    int(row["queries_per_repetition"]) != 10000
                    or int(row["repetitions"]) != 3
                ):
                    raise ValueError(
                        "Paper reference does not follow the 10,000-query, three-repetition contract"
                    )
                if float(row["target_recall"]) != TARGETS[row["workload"]]:
                    raise ValueError("Paper reference target differs")
                reference_rows.append(
                    {
                        **row,
                        "k": k,
                        "method": "retain"
                        if row["method"] == "default_cagra_accumulator"
                        else "navix",
                        "source_csv": str(destination),
                    }
                )
    if len(reference_rows) != 16:
        raise ValueError("Expected 16 saved Retain/NaviX reference cells")
    provenance = []
    for name in (
        "matched_recall_provenance.json",
        "matched_recall_k100_provenance.json",
    ):
        source = args.paper_data / name
        destination = root / "references" / name
        shutil.copyfile(source, destination)
        provenance.append({**identity(source), "sha256": digest(source)})
    print(f"[{utc()}] Hashing {len(sources)} unique input files", flush=True)
    unique = {str(Path(path).resolve()): path for path in sources}
    files = [
        {**identity(path), "sha256": digest(path)} for path in unique.values()
    ]
    specs = index_specs()
    for spec in specs:
        spec["file"] = str(root / "indices" / (spec["id"] + ".index"))
    manifest = dict(
        schema_version=SCHEMA,
        created_utc=utc(),
        run_root=str(root),
        repo=str(REPO),
        build_dir=str(args.build_dir),
        data_root=str(data),
        gpu=gpu,
        inputs=inputs,
        input_files=files,
        runtime_files=[{**identity(p), "sha256": digest(p)} for p in runtime],
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        working_diff_sha256=hashlib.sha256(
            subprocess.check_output(["git", "diff"], cwd=REPO)
        ).hexdigest(),
        workflow_sha256=digest(__file__),
        targets=TARGETS,
        target_window=0.002,
        batch_size=2048,
        repetitions=3,
        minimum_time=args.min_time,
        minimum_warmup_time=0.01,
        indexes=specs,
        references=references,
        reference_provenance=provenance,
        reference_rows=reference_rows,
        timing_contract="Timed complete GPU search, including PQ GPU reranking; preparation and bitmap upload excluded",
        recall_contract="ValidGTRecall weighted by valid GT slots; QPS=10000/sum(shard seconds) per repetition",
        interpolation_contract="Adjacent measured probes, same index/reranking setting; linear interpolation of seconds per query",
    )
    write_json(manifest_path, manifest)
    shutil.copyfile(__file__, root / "state/workflow_initial.py")
    write_csv(root / "references/baselines.csv", reference_rows)
    return manifest


def make_config(spec, inp, shard, points, building=False):
    use_float = spec["dtype"] == "float"
    dataset = dict(
        name=f"paper-{inp['workload']}-k{inp['k']}-q{shard['first_query']}",
        base_file=inp["float_base_file"] if use_float else inp["base_file"],
        query_file=shard["float_query_file"]
        if use_float
        else shard["query_file"],
        groundtruth_neighbors_file=shard["groundtruth_file"],
        distance="euclidean",
        dtype=spec["dtype"],
        filter=dict(kind="bitmap", file=shard["bitmap_file"]),
    )
    entries = []
    for point in points:
        params = dict(
            nprobe=point["nprobe"], n_queries=shard["query_count"], k=inp["k"]
        )
        if spec["kind"] == "pq":
            params.update(
                refine_ratio=point["refine_ratio"],
                internalDistanceDtype="float",
                smemLutDtype="half",
            )
            if point["refine_ratio"] > 1:
                params["dataset_memory_type"] = "device"
        entries.append(
            dict(
                name="ivf-" + point_id(point),
                algo=f"cuvs_ivf_{spec['kind']}",
                file=spec["file"],
                build_param=spec["build"],
                search_params=[params],
            )
        )
    if building:
        entries = entries[:1]
    return dict(
        dataset=dataset,
        search_basic_param=dict(k=inp["k"], batch_size=shard["query_count"]),
        index=entries,
    )


def validate_raw(path, points, query_count, k, repetitions):
    rows = [
        row
        for row in read_json(path).get("benchmarks", [])
        if row.get("run_type", "iteration") == "iteration"
    ]
    expected = {
        (point_id(point), rep)
        for point in points
        for rep in range(repetitions)
    }
    seen = set()
    for row in rows:
        if row.get("error_occurred"):
            raise ValueError(
                f"Benchmark error in {path}: {row.get('error_message')}"
            )
        identity_key = (
            row["name"].split("/")[0].removeprefix("ivf-"),
            row.get("repetition_index", 0),
        )
        if identity_key not in expected or identity_key in seen:
            raise ValueError(
                f"Unexpected/duplicate benchmark row: {identity_key}, {path}"
            )
        seen.add(identity_key)
        if (
            row["k"] != k
            or row["n_queries"] != query_count
            or row["threads"] != 1
        ):
            raise ValueError(f"Runtime batch/k/thread mismatch: {path}")
        if (
            int(row["iterations"]) < 1
            or row["total_queries"] != query_count * row["iterations"]
        ):
            raise ValueError(f"Incomplete query batch: {path}")
        for name in ERROR_COUNTERS:
            if row.get(name) != 0:
                raise ValueError(
                    f"Missing/nonzero correctness counter {name}: {path}"
                )
        for name in (
            "ValidGTRecall",
            "ValidGTFraction",
            "items_per_second",
            "Latency",
        ):
            if name not in row or not math.isfinite(row[name]):
                raise ValueError(f"Missing/nonfinite {name}: {path}")
        if (
            not 0 <= row["ValidGTRecall"] <= 1
            or not 0 < row["ValidGTFraction"] <= 1
            or row["items_per_second"] <= 0
        ):
            raise ValueError(f"Invalid recall/QPS: {path}")
    if seen != expected:
        raise ValueError(
            f"Incomplete raw report {path}: missing {expected - seen}"
        )
    return rows


def aggregate(raw_shards, points, k, repetitions):
    groups = defaultdict(list)
    for count, path in raw_shards:
        for row in validate_raw(path, points, count, k, repetitions):
            name = row["name"].split("/")[0].removeprefix("ivf-")
            groups[(name, row.get("repetition_index", 0))].append((count, row))
    output = []
    for point in points:
        reps = []
        for rep in range(repetitions):
            members = groups[(point_id(point), rep)]
            if (
                len(members) != 5
                or sum(count for count, row in members) != 10000
            ):
                raise ValueError(
                    "A point must include all five shards and 10,000 queries"
                )
            valid_slots = sum(
                count * k * row["ValidGTFraction"] for count, row in members
            )
            matches = sum(
                count * k * row["ValidGTFraction"] * row["ValidGTRecall"]
                for count, row in members
            )
            seconds = sum(
                count / row["items_per_second"] for count, row in members
            )
            reps.append(
                dict(
                    repetition_index=rep,
                    queries=10000,
                    valid_gt_slots=valid_slots,
                    recall=matches / valid_slots,
                    seconds=seconds,
                    qps=10000 / seconds,
                )
            )
        output.append(
            {
                **point,
                "point_id": point_id(point),
                "repetitions": repetitions,
                "recall_median": statistics.median(
                    rep["recall"] for rep in reps
                ),
                "recall_min": min(rep["recall"] for rep in reps),
                "recall_max": max(rep["recall"] for rep in reps),
                "qps_median": statistics.median(rep["qps"] for rep in reps),
                "qps_min": min(rep["qps"] for rep in reps),
                "qps_max": max(rep["qps"] for rep in reps),
                "seconds_median": statistics.median(
                    rep["seconds"] for rep in reps
                ),
                "repetition_rows": reps,
            }
        )
    return output


def adjacent_brackets(rows, target, adjacent=False):
    families = defaultdict(list)
    for row in rows:
        families[(row["index_id"], row["refine_ratio"])].append(row)
    pairs = []
    for members in families.values():
        members = sorted(members, key=lambda row: row["nprobe"])
        for a, b in zip(members, members[1:]):
            if adjacent and b["nprobe"] - a["nprobe"] != 1:
                continue
            low, high = sorted((a, b), key=lambda row: row["recall_median"])
            if low["recall_median"] < target <= high["recall_median"]:
                pairs.append((low, high))
    return pairs


def refinement_proposals(rows, target):
    seen = {point_id(row) for row in rows}
    output = {}
    for a, b in adjacent_brackets(rows, target):
        lo, hi = sorted((a["nprobe"], b["nprobe"]))
        if hi - lo > 1:
            fraction = (target - a["recall_median"]) / (
                b["recall_median"] - a["recall_median"]
            )
            predicted = round(
                a["nprobe"] + fraction * (b["nprobe"] - a["nprobe"])
            )
            for probe in {
                (lo + hi) // 2,
                predicted - 1,
                predicted,
                predicted + 1,
            }:
                if not lo < probe < hi:
                    continue
                point = dict(
                    index_id=a["index_id"],
                    nprobe=probe,
                    refine_ratio=a["refine_ratio"],
                )
                if point_id(point) not in seen:
                    output[point_id(point)] = point
    return list(output.values())


def within_window(row, target):
    return (
        row["recall_min"] >= target
        and row["recall_median"] <= target + 0.002 + 1e-12
    )


def interpolate(a, b, target):
    if (a["index_id"], a["refine_ratio"]) != (
        b["index_id"],
        b["refine_ratio"],
    ):
        raise ValueError(
            "Interpolation endpoints must use the same index and reranking setting"
        )
    if abs(a["nprobe"] - b["nprobe"]) != 1:
        raise ValueError(
            "Interpolation endpoints must have adjacent probe counts"
        )
    low, high = sorted((a, b), key=lambda row: row["recall_median"])
    if not low["recall_median"] < target <= high["recall_median"]:
        raise ValueError("Interpolation would extrapolate")
    weight = (target - low["recall_median"]) / (
        high["recall_median"] - low["recall_median"]
    )
    seconds = (1 - weight) * low["seconds_median"] + weight * high[
        "seconds_median"
    ]
    return dict(
        status="interpolated",
        recall=target,
        qps=10000 / seconds,
        interpolation_weight=weight,
        lower_point_id=low["point_id"],
        upper_point_id=high["point_id"],
        lower_recall=low["recall_median"],
        upper_recall=high["recall_median"],
        lower_qps=low["qps_median"],
        upper_qps=high["qps_median"],
    )


def finalists(rows, target):
    selected = {}
    matched = sorted(
        (row for row in rows if within_window(row, target)),
        key=lambda row: -row["qps_median"],
    )[:3]
    for row in matched:
        selected[row["point_id"]] = row
    if not matched:
        brackets = sorted(
            adjacent_brackets(rows, target, adjacent=True),
            key=lambda pair: -interpolate(*pair, target)["qps"],
        )[:3]
        for pair in brackets:
            for row in pair:
                selected[row["point_id"]] = row
    maximum = max(
        rows, key=lambda row: (row["recall_median"], row["qps_median"])
    )
    selected[maximum["point_id"]] = maximum
    return [
        dict(
            index_id=row["index_id"],
            nprobe=row["nprobe"],
            refine_ratio=row["refine_ratio"],
        )
        for row in selected.values()
    ]


class Runner:
    def __init__(self, args, manifest):
        self.args, self.manifest, self.root = args, manifest, args.run_root
        self.specs = {spec["id"]: spec for spec in manifest["indexes"]}
        self.started = time.monotonic()
        self.completed = 0

    def progress(self, stage, detail, **extra):
        calibrated = len(
            list((self.root / "state/calibrated").glob("*/*.json"))
        )
        builds = len(list((self.root / "state/builds").glob("*.json")))
        status = dict(
            updated_utc=utc(),
            state="running",
            stage=stage,
            detail=detail,
            completed_commands=self.completed,
            elapsed_seconds=time.monotonic() - self.started,
            completed_indexes=builds,
            total_indexes=len(self.specs),
            completed_calibrations=calibrated,
            total_calibrations=96,
            **extra,
        )
        write_json(self.root / "state/progress.json", status)
        print(f"[{utc()}] {stage}: {detail}", flush=True)

    def invoke(self, command, log_path, stage, detail):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.progress(stage, detail)
        environment = dict(
            os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="12"
        )
        lib = str(self.args.build_dir / "libcuvs.so")
        environment["LD_PRELOAD"] = lib + (
            ":" + environment["LD_PRELOAD"]
            if environment.get("LD_PRELOAD")
            else ""
        )
        start = time.monotonic()
        with log_path.open("w") as output:
            output.write(json.dumps(command) + "\n")
            output.flush()
            subprocess.run(
                command,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
            )
        elapsed = time.monotonic() - start
        self.completed += 1
        with (self.root / "logs/timing.jsonl").open("a") as output:
            output.write(
                json.dumps(
                    dict(
                        stage=stage,
                        detail=detail,
                        elapsed_seconds=elapsed,
                        completed_utc=utc(),
                    )
                )
                + "\n"
            )

    def binary(self, spec):
        return str(
            self.args.build_dir
            / "bench/ann"
            / f"CUVS_IVF_{spec['kind'].upper()}_ANN_BENCH"
        )

    def build(self, spec):
        marker = self.root / "state/builds" / (spec["id"] + ".json")
        if marker.exists():
            recorded = read_json(marker)
            if (
                identity(spec["file"]) != recorded["index_file"]
                or recorded["build"] != spec["build"]
            ):
                raise ValueError(f"Index changed: {spec['id']}")
            return
        if Path(spec["file"]).exists():
            raise ValueError(
                f"Incomplete index without successful build marker: {spec['file']}"
            )
        inp = self.manifest["inputs"][
            f"{'yfcc' if spec['corpus'] == 'yfcc' else 'em'}_k10"
        ]
        point = dict(index_id=spec["id"], nprobe=1, refine_ratio=1)
        config_path = self.root / "configs/build" / (spec["id"] + ".json")
        write_json(
            config_path,
            make_config(spec, inp, inp["shards"][0], [point], building=True),
        )
        output = self.root / "raw/build" / (spec["id"] + ".json")
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.binary(spec),
            "--build",
            f"--data_prefix={self.args.data_root}",
            f"--index_prefix={self.root}",
            "--benchmark_repetitions=1",
            "--benchmark_min_time=0.001s",
            "--benchmark_min_warmup_time=0",
            "--benchmark_out_format=json",
            f"--benchmark_out={output}",
            str(config_path),
        ]
        self.invoke(
            command,
            self.root / "logs/build" / (spec["id"] + ".log"),
            "build",
            spec["id"],
        )
        report = read_json(output)["benchmarks"]
        if not report or any(row.get("error_occurred") for row in report):
            raise ValueError(f"Index build failed: {output}")
        write_json(
            marker,
            dict(
                index_file=identity(spec["file"]),
                build=spec["build"],
                created_utc=utc(),
                raw=str(output),
            ),
        )

    def measure(self, spec, workload, k, points, phase, group, repetitions):
        # A refined search retains the original vectors on the GPU. Run plain PQ in its own
        # process so its memory budget and timings describe the compressed index alone.
        if spec["kind"] == "pq":
            plain = [point for point in points if point["refine_ratio"] == 1]
            refined = [point for point in points if point["refine_ratio"] > 1]
            if plain and refined:
                return self.measure(
                    spec,
                    workload,
                    k,
                    plain,
                    phase,
                    group + "_plain",
                    repetitions,
                ) + self.measure(
                    spec,
                    workload,
                    k,
                    refined,
                    phase,
                    group + "_refined",
                    repetitions,
                )
        directory = (
            self.root
            / "measurements"
            / phase
            / spec["id"]
            / f"{workload}_k{k}"
        )
        completion = directory / (group + ".json")
        request = dict(points=points, phase=phase, repetitions=repetitions)
        if completion.exists():
            saved = read_json(completion)
            if saved["request"] != request:
                raise ValueError(f"Measurement request changed: {completion}")
            for count, path in saved["raw_shards"]:
                validate_raw(path, points, count, k, repetitions)
            return saved["rows"]
        inp = self.manifest["inputs"][f"{workload}_k{k}"]
        raw_shards = []
        for shard_number, shard in enumerate(inp["shards"]):
            stem = (
                Path(phase)
                / spec["id"]
                / f"{workload}_k{k}"
                / group
                / f"shard_{shard_number:02}"
            )
            config_path = self.root / "configs" / stem.with_suffix(".json")
            report = self.root / "raw" / stem.with_suffix(".json")
            config = make_config(spec, inp, shard, points)
            if config_path.exists() and read_json(config_path) != config:
                raise ValueError(f"Immutable config changed: {config_path}")
            write_json(config_path, config)
            if report.exists():
                validate_raw(
                    report, points, shard["query_count"], k, repetitions
                )
            else:
                report.parent.mkdir(parents=True, exist_ok=True)
                temporary = report.with_suffix(".partial.json")
                command = [
                    self.binary(spec),
                    "--search",
                    "--mode=throughput",
                    "--threads=1",
                    f"--data_prefix={self.args.data_root}",
                    f"--index_prefix={self.root}",
                    f"--benchmark_repetitions={repetitions}",
                    f"--benchmark_min_time={self.manifest['minimum_time']}",
                    "--benchmark_min_warmup_time=0.01",
                    "--benchmark_enable_random_interleaving=true",
                    "--benchmark_report_aggregates_only=false",
                    "--benchmark_out_format=json",
                    f"--benchmark_out={temporary}",
                    str(config_path),
                ]
                self.invoke(
                    command,
                    self.root / "logs" / stem.with_suffix(".log"),
                    phase,
                    f"{spec['id']} {workload} k={k} {group} shard={shard_number} points={len(points)}",
                )
                validate_raw(
                    temporary, points, shard["query_count"], k, repetitions
                )
                temporary.replace(report)
            raw_shards.append((shard["query_count"], str(report)))
        rows = aggregate(raw_shards, points, k, repetitions)
        for row in rows:
            row.update(
                workload=workload,
                k=k,
                phase=phase,
                method=method_for(spec, row["refine_ratio"]),
            )
        write_json(
            completion,
            dict(
                request=request,
                rows=rows,
                raw_shards=raw_shards,
                completed_utc=utc(),
            ),
        )
        return rows

    def calibration(self, spec):
        workloads = (
            ("yfcc",) if spec["corpus"] == "yfcc" else ("em", "emis", "r")
        )
        for workload in workloads:
            for k in (10, 100):
                rows = self.measure(
                    spec,
                    workload,
                    k,
                    coarse_points(spec),
                    "calibration",
                    "coarse",
                    1,
                )
                round_number = 0
                while proposed := refinement_proposals(
                    rows, TARGETS[workload]
                ):
                    rows.extend(
                        self.measure(
                            spec,
                            workload,
                            k,
                            proposed,
                            "calibration",
                            f"refine_{round_number:02}",
                            1,
                        )
                    )
                    round_number += 1
                    if round_number > 16:
                        raise ValueError("Probe refinement did not converge")
                write_json(
                    self.root
                    / "state/calibrated"
                    / spec["id"]
                    / f"{workload}_k{k}.json",
                    rows,
                )
                self.progress(
                    "calibrated",
                    f"{spec['id']} {workload} k={k}: {len(rows)} points",
                )
                self.report()

    def load_rows(self, phase):
        paths = sorted((self.root / "measurements" / phase).glob("*/*/*.json"))
        rows = [row for path in paths for row in read_json(path)["rows"]]
        unique = {
            (row["workload"], row["k"], row["point_id"]): row for row in rows
        }
        return list(unique.values())

    def final(self, workloads=WORKLOADS):
        selection_path = (
            self.root
            / "state"
            / ("final_selection_" + "_".join(workloads) + ".json")
        )
        if selection_path.exists():
            selections = read_json(selection_path)
        else:
            selections = []
            rows = self.load_rows("calibration")
            for workload in workloads:
                for k in (10, 100):
                    for method in METHODS:
                        cells = [
                            row
                            for row in rows
                            if (row["workload"], row["k"], row["method"])
                            == (workload, k, method)
                        ]
                        if not cells:
                            raise ValueError(
                                f"Missing calibration: {workload} k={k} {method}"
                            )
                        selections.append(
                            dict(
                                workload=workload,
                                k=k,
                                method=method,
                                points=finalists(cells, TARGETS[workload]),
                            )
                        )
            write_json(selection_path, selections)
        batches = defaultdict(dict)
        for selection in selections:
            for point in selection["points"]:
                batches[
                    (point["index_id"], selection["workload"], selection["k"])
                ][point_id(point)] = point
        for (index_id, workload, k), selected in batches.items():
            points = sorted(selected.values(), key=point_id)
            self.measure(
                self.specs[index_id],
                workload,
                k,
                points,
                "final",
                "selected",
                3,
            )
            self.report()

    def report(self, complete=False):
        calibration, final = (
            self.load_rows("calibration"),
            self.load_rows("final"),
        )
        write_csv(
            self.root / "analysis/calibration_points.csv",
            [
                {
                    key: value
                    for key, value in row.items()
                    if key != "repetition_rows"
                }
                for row in calibration
            ],
        )
        write_csv(
            self.root / "analysis/final_points.csv",
            [
                {
                    key: value
                    for key, value in row.items()
                    if key != "repetition_rows"
                }
                for row in final
            ],
        )
        comparison = []
        refs = {
            (row["workload"], row["k"], row["method"]): row
            for row in self.manifest["reference_rows"]
        }
        for workload in WORKLOADS:
            for k in (10, 100):
                for method in METHODS:
                    cells = [
                        row
                        for row in final
                        if (row["workload"], row["k"], row["method"])
                        == (workload, k, method)
                    ]
                    if not cells:
                        continue
                    target = TARGETS[workload]
                    maximum = max(
                        cells,
                        key=lambda row: (
                            row["recall_median"],
                            row["qps_median"],
                        ),
                    )
                    row = dict(
                        workload=workload,
                        k=k,
                        method=method,
                        target=target,
                        max_observed_recall=maximum["recall_median"],
                        qps_at_max_recall=maximum["qps_median"],
                        max_recall_point_id=maximum["point_id"],
                        source="new_ivf_final_measurements",
                    )
                    matched = [
                        cell for cell in cells if within_window(cell, target)
                    ]
                    if matched:
                        best = max(
                            matched, key=lambda cell: cell["qps_median"]
                        )
                        row.update(
                            status="measured",
                            recall=best["recall_median"],
                            recall_min=best["recall_min"],
                            qps=best["qps_median"],
                            qps_min=best["qps_min"],
                            qps_max=best["qps_max"],
                            point_id=best["point_id"],
                            index_id=best["index_id"],
                            nprobe=best["nprobe"],
                            refine_ratio=best["refine_ratio"],
                        )
                    elif pairs := adjacent_brackets(
                        cells, target, adjacent=True
                    ):
                        row.update(
                            max(
                                (interpolate(*pair, target) for pair in pairs),
                                key=lambda item: item["qps"],
                            )
                        )
                    else:
                        status = (
                            "unreached"
                            if maximum["recall_median"] < target
                            else "unmatched"
                        )
                        row.update(status=status, recall="", qps="")
                    for name in ("retain", "navix"):
                        reference = refs[(workload, k, name)]
                        row[f"{name}_recall"] = float(
                            reference["recall_median"]
                        )
                        row[f"{name}_qps"] = float(reference["qps_median"])
                        matched_reference = (
                            reference["within_target_window"].lower() == "true"
                            and float(reference["recall_min"]) >= target
                        )
                        row[f"{name}_status"] = (
                            "measured" if matched_reference else "unreached"
                        )
                        row[f"speedup_vs_{name}"] = (
                            row["qps"] / float(reference["qps_median"])
                            if row["qps"] and matched_reference
                            else ""
                        )
                    comparison.append(row)
        write_csv(self.root / "analysis/comparison.csv", comparison)
        write_json(
            self.root / "analysis/comparison.json",
            dict(complete=complete, updated_utc=utc(), rows=comparison),
        )
        self.markdown(comparison, refs, complete)
        if complete:
            if len(comparison) != 32:
                raise ValueError("Expected all 32 IVF comparison cells")
            self.plots(calibration, final, refs)

    def markdown(self, rows, references, complete):
        lines = [
            "# Bitmap IVF comparison with saved paper measurements",
            "",
            f"Run: `{self.root}`",
            "",
            "Complete."
            if complete
            else "Partial results; tuning and final measurements are still running.",
            "",
            "10,000 queries; batches of 2,048; three final repetitions; median QPS. Recall uses valid ground-truth IDs.",
            "",
            "An asterisk marks an interpolated estimate. `max R @ Q` means the target was not matched; R is maximum observed recall and Q is its measured QPS.",
            "",
        ]
        indexed = {
            (row["workload"], row["k"], row["method"]): row for row in rows
        }
        for k in (10, 100):
            lines.extend(
                [
                    f"## k={k}",
                    "",
                    "| Workload | Target | Retain | NaviX | IVF-Flat | IVF-SQ | IVF-PQ | IVF-PQ + GPU reranking |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                    "| | | Recall / QPS | Recall / QPS | Recall / QPS | Recall / QPS | Recall / QPS | Recall / QPS |",
                ]
            )
            for workload in WORKLOADS:
                cells = [workload.upper(), f"{TARGETS[workload]:.3f}"]
                for method in ("retain", "navix"):
                    ref = references[(workload, k, method)]
                    text = f"{float(ref['recall_median']):.4f} / {float(ref['qps_median']):,.1f}"
                    cells.append(
                        text
                        if ref["within_target_window"].lower() == "true"
                        else "max " + text
                    )
                for method in METHODS:
                    row = indexed.get((workload, k, method))
                    if row is None:
                        cells.append("pending")
                    elif row["qps"]:
                        star = "*" if row["status"] == "interpolated" else ""
                        cells.append(
                            f"{row['recall']:.4f} / {row['qps']:,.1f}{star}"
                        )
                    else:
                        cells.append(
                            f"max {row['max_observed_recall']:.4f} @ {row['qps_at_max_recall']:,.1f}"
                        )
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
        lines.extend(
            [
                "## Speedups at matched recall",
                "",
                "| Workload | k | Method | Status | vs Retain | vs NaviX |",
                "|---|---:|---|---|---:|---:|",
            ]
        )
        for row in rows:
            ratios = [
                f"{row[f'speedup_vs_{method}']:.2f}×"
                if row[f"speedup_vs_{method}"]
                else "—"
                for method in ("retain", "navix")
            ]
            lines.append(
                f"| {row['workload'].upper()} | {row['k']} | {row['method']} | {row['status']} | {' | '.join(ratios)} |"
            )
        (self.root / "analysis/comparison.md").write_text(
            "\n".join(lines) + "\n"
        )

    def plots(self, calibration, final, references):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = dict(
            ivf_flat="C0", ivf_sq="C1", ivf_pq="C2", ivf_pq_refined="C3"
        )
        for k in (10, 100):
            fig, axes = plt.subplots(
                2, 2, figsize=(11, 7), constrained_layout=True
            )
            for workload, ax in zip(WORKLOADS, axes.flat):
                for method in METHODS:
                    rows = [
                        row
                        for row in calibration
                        if (row["workload"], row["k"], row["method"])
                        == (workload, k, method)
                    ]
                    ax.scatter(
                        [row["recall_median"] for row in rows],
                        [row["qps_median"] for row in rows],
                        s=10,
                        alpha=0.35,
                        color=colors[method],
                        label=method,
                    )
                for name, marker, color in (
                    ("retain", "s", "black"),
                    ("navix", "*", "purple"),
                ):
                    ref = references[(workload, k, name)]
                    ax.scatter(
                        float(ref["recall_median"]),
                        float(ref["qps_median"]),
                        marker=marker,
                        s=85,
                        color=color,
                        label=f"{name} (saved paper point)",
                    )
                ax.axvspan(
                    TARGETS[workload],
                    TARGETS[workload] + 0.002,
                    color="grey",
                    alpha=0.2,
                )
                ax.set(
                    title=workload.upper(),
                    xlabel=f"Recall@{k}",
                    ylabel="QPS",
                    yscale="log",
                )
                ax.grid(alpha=0.2)
            axes.flat[0].legend(fontsize=7)
            for extension in ("pdf", "png"):
                fig.savefig(
                    self.root / "analysis" / f"qps_recall_k{k}.{extension}",
                    dpi=180,
                )
            plt.close(fig)

    def run(self, stage):
        if stage in ("all", "build", "calibrate"):
            for number, spec in enumerate(self.manifest["indexes"], 1):
                self.progress(
                    "index",
                    f"{number}/{len(self.manifest['indexes'])}: {spec['id']}",
                    index_number=number,
                )
                self.build(spec)
                if stage != "build":
                    self.calibration(spec)
                if stage == "all" and (
                    number == len(self.manifest["indexes"])
                    or self.manifest["indexes"][number]["corpus"]
                    != spec["corpus"]
                ):
                    self.final(
                        ("yfcc",)
                        if spec["corpus"] == "yfcc"
                        else ("em", "emis", "r")
                    )
        if stage == "final":
            self.final()
        if stage in ("all", "final", "analyze"):
            cells = {
                (row["workload"], row["k"], row["method"])
                for row in self.load_rows("final")
            }
            self.report(complete=len(cells) == 32)
        self.progress("complete", stage)
        progress = read_json(self.root / "state/progress.json")
        progress["state"] = (
            "complete" if stage in ("all", "final") else "stage_complete"
        )
        write_json(self.root / "state/progress.json", progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("prepare", "build", "calibrate", "final", "analyze", "all"),
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, default=Path("/data/retrieve_data")
    )
    parser.add_argument("--build-dir", type=Path, default=REPO / "cpp/build")
    parser.add_argument(
        "--paper-data",
        type=Path,
        default=Path("/home/ubuntu/paper_selectivity_revision/data"),
    )
    parser.add_argument("--min-time", default="0.10s")
    args = parser.parse_args()
    for name in ("run_root", "data_root", "build_dir", "paper_data"):
        setattr(args, name, getattr(args, name).resolve())
    args.run_root.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(args.run_root) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            manifest = prepare(args)
            if args.stage != "prepare":
                Runner(args, manifest).run(args.stage)
            else:
                write_json(
                    args.run_root / "state/progress.json",
                    dict(state="prepared", updated_utc=utc()),
                )
        except Exception as error:
            write_json(
                args.run_root / "state/progress.json",
                dict(state="failed", updated_utc=utc(), error=str(error)),
            )
            raise


if __name__ == "__main__":
    main()
