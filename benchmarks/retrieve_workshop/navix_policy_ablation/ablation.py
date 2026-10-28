#!/usr/bin/env python3
"""Paired NaviX policy ablation on the frozen 2,048-query primary workloads."""

from __future__ import annotations
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

MODES = ("adaptive_kuzu", "adaptive_blind")
SOURCE = Path(
    "/data/retrieve_workshop_runs/a100_batch1_primary_ivf_20261028T073000Z"
)
ENV = Path("/data/cuvs-upstream-envs/cuvs-2612")
BUILD = Path("/data/cuvs-upstream-integration/cpp/build")


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(p)


def load(p):
    return json.loads(Path(p).read_text())


def csv_out(p, rows):
    if not rows:
        return
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        w = csv.DictWriter(
            f, fieldnames=list(dict.fromkeys(k for r in rows for k in r))
        )
        w.writeheader()
        w.writerows(rows)


def within(recall, target):
    return target - 1e-12 <= recall <= target + 0.002 + 1e-12


def b0(frontier, w):
    return (
        frontier // w + 5
    )  # Both degree-64 primary graphs need five reachability steps.


def legal(frontier, w, m, k):
    return (
        k <= frontier <= 512
        and w in (1, 2)
        and m >= 0
        and ((frontier + 31) // 32 * 32 + 128 * w * (m or b0(frontier, w)))
        <= 524288
    )


def key(sp):
    return (
        sp["navix_mode"],
        sp["itopk"],
        sp["search_width"],
        sp["max_iterations"],
    )


def fingerprint(sp):
    mode, frontier, w, m = key(sp)
    return (
        mode,
        (frontier + 31) // 32 * 32,
        w,
        m or b0(frontier, w),
        sp["navix_seed_cap"],
    )


def point(template, mode, frontier, w, m, *, tune=False):
    sp = copy.deepcopy(template)
    for name in list(sp):
        if name.startswith("benchmark_") or name.startswith(
            "search_diagnostics_"
        ):
            del sp[name]
    sp.update(
        navix_mode=mode, itopk=frontier, search_width=w, max_iterations=m
    )
    if tune:
        sp["navix_seed_cap"] = 64 * w
    sp["validation_case_id"] = (
        f"{mode}_L{frontier}_W{w}_M{m}_S{sp['navix_seed_cap']}"
    )
    return sp


def prepare(root, source, build):
    if (root / "manifest.json").exists():
        return
    manifest = load(source / "manifest.json")
    cases = []
    for p in manifest["points"]:
        if p["method"] != "navix_reference":
            continue
        workload, k = p["workload"], p["k"]
        config_path = (
            source
            / f"configs/baseline/{workload}_k{k}/cuvs_cagra/rep_0/shard_00.json"
        )
        config = load(config_path)
        config["index"][0]["search_params"] = [p["search"]]
        cases.append(
            dict(
                id=f"{workload}_k{k}",
                workload=workload,
                k=k,
                target=0.8 if workload == "yfcc" else 0.95,
                config=config,
            )
        )
    if len(cases) != 8:
        raise ValueError("Expected eight frozen NaviX selections")
    runtime = root / "runtime"
    runtime.mkdir(parents=True)
    for src in (
        build / "bench/ann/CUVS_CAGRA_ANN_BENCH",
        build / "libcuvs.so",
    ):
        shutil.copy2(src.resolve(), runtime / src.name)
    inputs = {}
    for case in cases:
        cfg = case["config"]
        ds = cfg["dataset"]
        for name in (
            ds["base_file"],
            ds["query_file"],
            ds["groundtruth_neighbors_file"],
            ds["filter"]["file"],
            cfg["index"][0]["file"],
        ):
            if name not in inputs:
                stat = Path(name).stat()
                inputs[name] = dict(
                    bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    sha256=sha(name),
                )
    save(
        root / "manifest.json",
        dict(
            cases=cases,
            source_manifest_sha256=sha(source / "manifest.json"),
            runner_sha256=sha(__file__),
            inputs=inputs,
            runtime={p.name: sha(p) for p in runtime.iterdir()},
            source_head=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            created=time.time(),
        ),
    )


def validate(raw, points, reps):
    grouped = {p["validation_case_id"]: [] for p in points}
    for row in raw.get("benchmarks", []):
        if row.get("run_type") == "aggregate":
            continue
        if row.get("error_occurred") or row.get("skipped"):
            raise ValueError(f"Benchmark failed: {row}")
        match = re.search(
            r'(?:^|#)validation_case_id="([^"]+)"', row.get("label", "")
        )
        if not match or match[1] not in grouped:
            raise ValueError("Unexpected benchmark identity")
        if int(row.get("n_queries", -1)) != 2048:
            raise ValueError("Wrong query count")
        for counter in (
            "FilterViolations",
            "InvalidSentinelErrors",
            "SentinelOrderErrors",
            "DuplicateOutputQueries",
        ):
            if counter not in row or row[counter] != 0:
                raise ValueError(
                    f"Invalid or missing correctness counter {counter}"
                )
        grouped[match[1]].append(row)
    out = []
    for sp in points:
        rows = grouped[sp["validation_case_id"]]
        if len(rows) != reps or {
            int(r["repetition_index"]) for r in rows
        } != set(range(reps)):
            raise ValueError("Incomplete repetitions")
        recalls = [float(r["ValidGTRecall"]) for r in rows]
        qps = [float(r["items_per_second"]) for r in rows]
        if (
            not all(math.isfinite(x) for x in recalls + qps)
            or min(qps) <= 0
            or min(recalls) < 0
            or max(recalls) > 1
        ):
            raise ValueError("Invalid measurement")
        out.append(
            dict(
                search=sp,
                recall=statistics.median(recalls),
                recalls=recalls,
                qps=statistics.median(qps),
                qps_min=min(qps),
                qps_max=max(qps),
            )
        )
    return out


def job(root, case, stage, points, reps=1, diagnostic=False):
    if not points:
        return []
    directory = root / "jobs" / case["id"] / stage
    directory.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(case["config"])
    config["search_basic_param"]["batch_size"] = 2048
    for sp in points:
        sp["max_queries"] = sp["n_queries"] = 2048
        if diagnostic:
            sp.update(
                search_diagnostics_output=str(
                    directory / "captures" / sp["validation_case_id"]
                ),
                search_diagnostics_groundtruth=config["dataset"][
                    "groundtruth_neighbors_file"
                ],
                search_diagnostics_dataset=case["id"],
                search_diagnostics_variant=sp["validation_case_id"],
            )
    config["index"][0]["search_params"] = points
    signature = hashlib.sha256(
        json.dumps([config, reps], sort_keys=True).encode()
    ).hexdigest()
    marker = directory / "complete.json"
    if marker.exists():
        record = load(marker)
        if (
            record["signature"] != signature
            or sha(directory / "raw.json") != record["raw_sha256"]
        ):
            raise ValueError(f"Completed job changed: {directory}")
        return validate(load(directory / "raw.json"), points, reps)
    if (directory / "raw.json").exists():
        (directory / "raw.json").rename(
            directory / f"partial-{time.time_ns()}.json"
        )
    save(directory / "config.json", config)
    save(
        root / "progress.json",
        dict(
            case=case["id"],
            stage=stage,
            points=len(points),
            updated=time.time(),
            status="running",
        ),
    )
    command = [
        str(root / "runtime/CUVS_CAGRA_ANN_BENCH"),
        "--search",
        "--mode=throughput",
        "--threads=1",
        "--data_prefix=/data/retrieve_data",
        "--index_prefix=/data/retrieve_data",
        f"--benchmark_repetitions={reps}",
        "--benchmark_min_time=" + ("1x" if reps == 1 else "0.2s"),
        "--benchmark_min_warmup_time=0.01",
        "--benchmark_enable_random_interleaving=true",
        "--benchmark_report_aggregates_only=false",
        "--benchmark_out_format=json",
        f"--benchmark_out={directory}/raw.json",
        str(directory / "config.json"),
    ]
    env = os.environ | {
        "LD_LIBRARY_PATH": f"{root}/runtime:{ENV}/lib",
        "CUVS_CAGRA_LOG_RESOURCES": "1" if diagnostic else "0",
    }
    with (directory / "log.txt").open("w") as log:
        subprocess.run(
            command,
            env=env,
            cwd=directory,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    result = validate(load(directory / "raw.json"), points, reps)
    save(
        marker,
        dict(signature=signature, raw_sha256=sha(directory / "raw.json")),
    )
    print(f"{case['id']}: {stage} complete ({len(points)} points)", flush=True)
    return result


def tune(root, case, template):
    k, target = case["k"], case["target"]
    measured = {}
    seen = set()

    def evaluate(stage, candidates):
        pending = []
        for mode, frontier, w, m in sorted(set(candidates)):
            sp = point(template, mode, frontier, w, m, tune=True)
            fp = fingerprint(sp)
            if legal(frontier, w, m, k) and fp not in seen:
                seen.add(fp)
                pending.append(sp)
        for row in job(root, case, stage, pending):
            measured[key(row["search"])] = row

    anchors = sorted(
        {
            k,
            *[
                frontier
                for frontier in (32, 64, 128, 256, 512)
                if frontier >= k
            ],
        }
    )
    evaluate(
        "anchors",
        [
            (mode, frontier, w, 0)
            for mode in MODES
            for frontier in anchors
            for w in (1, 2)
        ],
    )
    candidates = []
    for mode in MODES:
        for w in (1, 2):
            rows = sorted(
                (r for t, r in measured.items() if t[0] == mode and t[2] == w),
                key=lambda r: r["search"]["itopk"],
            )
            for a, b in zip(rows, rows[1:]):
                if any(
                    (a["recall"] - bound) * (b["recall"] - bound) <= 0
                    for bound in (target, target + 0.002)
                ):
                    candidates += [
                        (mode, frontier, w, 0)
                        for frontier in range(
                            a["search"]["itopk"] + 1, b["search"]["itopk"]
                        )
                    ]
    evaluate("refine_l", candidates)
    # At each W refine the nearest below/above frontiers, plus minimum and maximum L.
    cells = set()
    for mode in MODES:
        for w in (1, 2):
            local = [
                r for t, r in measured.items() if t[0] == mode and t[2] == w
            ]
            cells.update((mode, frontier, w) for frontier in (k, 512))
            for side in (False, True):
                subset = [r for r in local if (r["recall"] >= target) == side]
                if subset:
                    r = min(subset, key=lambda r: abs(r["recall"] - target))
                    cells.add((mode, r["search"]["itopk"], w))
    candidates = []
    for mode, frontier, w in sorted(cells):
        cap = min(7569, (524288 - (frontier + 31) // 32 * 32) // (128 * w))
        candidates += [(mode, frontier, w, 1)]
        if measured[(mode, frontier, w, 0)]["recall"] >= target:
            continue
        m = b0(frontier, w)
        while m < cap:
            m = min(cap, m * 2)
            candidates.append((mode, frontier, w, m))
    evaluate("iteration_anchors", candidates)
    # Resolve every observed target crossing down to integer iteration caps.
    for round_index in range(14):
        candidates = []
        for mode, frontier, w in sorted(cells):
            rows = sorted(
                (
                    r
                    for t, r in measured.items()
                    if t[:3] == (mode, frontier, w)
                ),
                key=lambda r: r["search"]["max_iterations"] or b0(frontier, w),
            )
            for a, b in zip(rows, rows[1:]):
                ma = a["search"]["max_iterations"] or b0(frontier, w)
                mb = b["search"]["max_iterations"] or b0(frontier, w)
                if mb - ma > 1 and any(
                    (a["recall"] - bound) * (b["recall"] - bound) <= 0
                    for bound in (target, target + 0.002)
                ):
                    candidates.append((mode, frontier, w, (ma + mb) // 2))
        before = len(measured)
        evaluate(f"iteration_refine_{round_index:02}", candidates)
        if len(measured) == before:
            break
    candidates = []
    for r in list(measured.values()):
        if within(r["recall"], target):
            mode, frontier, w, m = key(r["search"])
            candidates += [
                (mode, ll, w, m)
                for ll in range(
                    max(k, frontier - 4), min(512, frontier + 4) + 1
                )
            ]
    evaluate("target_neighborhood", candidates)
    csv_out(
        root / "analysis" / f"{case['id']}_calibration.csv",
        [
            dict(
                mode=t[0],
                L=t[1],
                W=t[2],
                iterations=t[3],
                recall=r["recall"],
                qps=r["qps"],
            )
            for t, r in measured.items()
        ],
    )
    selected = []
    for mode in MODES:
        local = [r for t, r in measured.items() if t[0] == mode]
        matches = sorted(
            [r for r in local if within(r["recall"], target)],
            key=lambda r: -r["qps"],
        )
        finalists = matches[:4] or [
            max(local, key=lambda r: (r["recall"], r["qps"]))
        ]
        confirmed = job(
            root,
            case,
            "confirm_" + mode,
            [copy.deepcopy(r["search"]) for r in finalists],
            3,
        )
        good = [
            r
            for r in confirmed
            if all(within(x, target) for x in r["recalls"])
        ]
        winner = (
            max(good, key=lambda r: r["qps"])
            if good
            else max(confirmed, key=lambda r: (r["recall"], r["qps"]))
        )
        winner.update(
            matched=bool(good),
            mode=mode,
            case=case["id"],
            target=target,
            maximum_observed_recall=max(r["recall"] for r in local),
            qps_at_maximum_observed_recall=max(
                local, key=lambda r: (r["recall"], r["qps"])
            )["qps"],
            closest_below=max(
                (r["recall"] for r in local if r["recall"] < target),
                default=None,
            ),
            closest_above=min(
                (r["recall"] for r in local if r["recall"] > target + 0.002),
                default=None,
            ),
        )
        selected.append(winner)
    return selected


def diagnostic_summary(root):
    summaries = []
    counters = (
        "iterations",
        "distance_evaluations",
        "graph_rows_read",
        "predicate_probes",
        "navix_one_hop_parents",
        "navix_directed_parents",
        "navix_blind_parents",
        "navix_second_hop_checks",
        "navix_bridge_rows_loaded",
        "navix_admitted_candidates",
    )
    for path in sorted(
        (root / "jobs").glob("*/diagnostics_*/captures/*/query_summary.csv")
    ):
        with path.open() as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 2048 or {int(r["query_id"]) for r in rows} != set(
            range(2048)
        ):
            raise ValueError(f"Incomplete diagnostic coverage: {path}")
        record = dict(
            case=path.parts[-5],
            stage=path.parts[-4],
            policy=path.parts[-2],
            queries=len(rows),
        )
        for counter in counters:
            record[counter + "_per_query"] = sum(
                int(r[counter]) for r in rows
            ) / len(rows)
        record["frontier_exhausted_fraction"] = sum(
            r["stop_reason"] == "3" for r in rows
        ) / len(rows)
        if (
            record["policy"].startswith("adaptive_blind")
            and record["navix_directed_parents_per_query"] != 0
        ):
            raise ValueError("Blind ablation executed a directed expansion")
        summaries.append(record)
    csv_out(root / "analysis/diagnostics.csv", summaries)


def report(root, fixed, selected):
    rows = []
    for stage, values in [("fixed", fixed), ("matched", selected)]:
        for r in values:
            sp = r["search"]
            rows.append(
                dict(
                    stage=stage,
                    case=r["case"],
                    mode=sp["navix_mode"],
                    L=sp["itopk"],
                    W=sp["search_width"],
                    iterations=sp["max_iterations"],
                    seed_cap=sp["navix_seed_cap"],
                    recall=r["recall"],
                    recall_min=min(r["recalls"]),
                    recall_max=max(r["recalls"]),
                    qps=r["qps"],
                    qps_min=r["qps_min"],
                    qps_max=r["qps_max"],
                    matched=r.get("matched", False),
                )
            )
    csv_out(root / "analysis/comparison.csv", rows)
    save(root / "analysis/results.json", dict(fixed=fixed, selected=selected))
    lines = [
        "# NaviX directed-expansion ablation",
        "",
        "Dense bitmaps; first 2,048 queries; single-CTA batched throughput. Three repetitions per reported point.",
        "Fixed settings replay the paper configuration; achieved subset recall is reported explicitly.",
        "Matched settings require all three recalls inside YFCC [0.800,0.802] or ArXiv [0.950,0.952].",
        "",
        "| Stage | Workload | Policy | Recall | QPS | Matched |",
        "|---|---|---|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['stage']} | {r['case']} | {r['mode']} | {r['recall']:.6f} | {r['qps']:.1f} | {r['matched']} |"
        )
    (root / "analysis/REPORT.md").write_text("\n".join(lines) + "\n")
    diagnostic_summary(root)
    if selected:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 4, figsize=(14, 6))
        for ax, case in zip(axes.flat, load(root / "manifest.json")["cases"]):
            calibration = root / "analysis" / f"{case['id']}_calibration.csv"
            if not calibration.exists():
                ax.set_visible(False)
                continue
            data = list(csv.DictReader(calibration.open()))
            for mode in MODES:
                rr = [r for r in data if r["mode"] == mode]
                ax.scatter(
                    [float(r["recall"]) for r in rr],
                    [float(r["qps"]) for r in rr],
                    s=8,
                    label=mode,
                )
            ax.axvspan(
                case["target"],
                case["target"] + 0.002,
                alpha=0.15,
                color="green",
            )
            ax.set(
                title=case["id"], xlabel="Recall", ylabel="QPS", yscale="log"
            )
        axes.flat[0].legend(fontsize=7)
        fig.tight_layout()
        for suffix in ("pdf", "png"):
            fig.savefig(root / f"analysis/recall_qps.{suffix}", dpi=160)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "run", "status"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--build", type=Path, default=BUILD)
    args = p.parse_args()
    root = args.root.resolve()
    if args.stage == "prepare":
        prepare(root, args.source, args.build)
        return
    if args.stage == "status":
        print(json.dumps(load(root / "progress.json"), indent=2))
        return
    manifest = load(root / "manifest.json")
    if manifest["runner_sha256"] != sha(__file__):
        raise ValueError("Runner changed after preparation")
    for name, digest in manifest["runtime"].items():
        if sha(root / "runtime" / name) != digest:
            raise ValueError("Runtime changed")
    for name, record in manifest["inputs"].items():
        st = Path(name).stat()
        if (st.st_size, st.st_mtime_ns) != (
            record["bytes"],
            record["mtime_ns"],
        ):
            raise ValueError(f"Input changed: {name}")
    with (
        (root / "run.lock").open("w") as own,
        Path("/data/cuvs-upstream-validation/gpu.lock").open("a") as gpu,
    ):
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu, fcntl.LOCK_EX)
        fixed = []
        for case in manifest["cases"]:
            template = case["config"]["index"][0]["search_params"][0]
            points = [
                point(
                    template,
                    mode,
                    template["itopk"],
                    template["search_width"],
                    template["max_iterations"],
                )
                for mode in MODES
            ]
            values = job(root, case, "fixed", points, 3)
            for r in values:
                r["case"] = case["id"]
            fixed += values
            report(root, fixed, [])
        selected = []
        for case in manifest["cases"]:
            template = case["config"]["index"][0]["search_params"][0]
            winners = tune(root, case, template)
            selected += winners
            for stage, points in [
                (
                    "fixed",
                    [r["search"] for r in fixed if r["case"] == case["id"]],
                ),
                ("matched", [r["search"] for r in winners]),
            ]:
                job(
                    root,
                    case,
                    "diagnostics_" + stage,
                    copy.deepcopy(points),
                    diagnostic=True,
                )
            report(root, fixed, selected)
        save(
            root / "progress.json",
            dict(status="complete", updated=time.time()),
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import sys

        if "--root" in sys.argv:
            failure_root = Path(sys.argv[sys.argv.index("--root") + 1])
            save(
                failure_root / "failure.json",
                dict(error=repr(exc), time=time.time()),
            )
        raise
