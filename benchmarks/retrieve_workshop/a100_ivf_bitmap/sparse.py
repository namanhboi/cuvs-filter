#!/usr/bin/env python3
"""IVF-Flat on the paper's five frozen low-selectivity ArXiv cohorts."""

import argparse
import csv
import fcntl
import json
import os
import shutil
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import workflow as core

RATES = ("0p0001", "0p0003", "0p001", "0p003", "0p01")
COHORTS = tuple(f"synthetic_low_{rate}_random" for rate in RATES)
LOWER, UPPER, INTERPOLATION_TARGET = 0.950, 0.952, 0.951
QUERY_COUNT = 2048
LABELS = {
    "ivf_flat": "IVF-Flat",
    "default_cagra": "Base",
    "default_cagra_accumulator": "Retain",
    "navix_reference": "NaviX",
    "exact": "Exact KNN",
}


def in_window(row):
    return row["recall_min"] >= LOWER and row["recall_max"] <= UPPER + 1e-12


def best_result(rows):
    if not rows:
        raise ValueError("No measured points")
    maximum = max(rows, key=lambda r: (r["recall_median"], r["qps_median"]))
    result = dict(
        status="unmatched",
        recall="",
        qps="",
        max_recall=maximum["recall_median"],
        qps_at_max_recall=maximum["qps_median"],
        max_recall_point_id=maximum["point_id"],
    )
    matches = [r for r in rows if in_window(r)]
    if matches:
        row = max(matches, key=lambda r: r["qps_median"])
        result.update(
            status="measured",
            recall=row["recall_median"],
            qps=row["qps_median"],
            point_id=row["point_id"],
            index_id=row["index_id"],
            nprobe=row["nprobe"],
            recall_min=row["recall_min"],
            recall_max=row["recall_max"],
        )
    else:
        brackets = core.adjacent_brackets(
            rows, INTERPOLATION_TARGET, adjacent=True
        )
        estimates = [
            core.interpolate(a, b, INTERPOLATION_TARGET, QUERY_COUNT)
            for a, b in brackets
        ]
        if estimates:
            result.update(max(estimates, key=lambda r: r["qps"]))
    return result


def select_finalists(rows):
    selected = {}
    matches = sorted(
        (r for r in rows if in_window(r)), key=lambda r: -r["qps_median"]
    )[:3]
    for row in matches:
        selected[row["point_id"]] = row
    if not matches:
        pairs = sorted(
            core.adjacent_brackets(rows, INTERPOLATION_TARGET, adjacent=True),
            key=lambda pair: -core.interpolate(
                *pair, INTERPOLATION_TARGET, QUERY_COUNT
            )["qps"],
        )[:3]
        for pair in pairs:
            for row in pair:
                selected[row["point_id"]] = row
    row = max(rows, key=lambda r: (r["recall_median"], r["qps_median"]))
    selected[row["point_id"]] = row
    return [
        {key: row[key] for key in ("index_id", "nprobe", "refine_ratio")}
        for row in selected.values()
    ]


def validate_sources(manifest):
    for source in manifest["files"]:
        current = core.identity(source["path"])
        if any(current[key] != source[key] for key in current):
            raise ValueError(
                f"Input/runtime identity changed: {source['path']}"
            )
        if (
            source.get("verify_sha256", True)
            and core.digest(source["path"]) != source["sha256"]
        ):
            raise ValueError(
                f"Input/runtime content changed: {source['path']}"
            )


def prepare(args):
    root = args.run_root
    contract = dict(
        reference_root=str(args.reference_root),
        index_source_root=str(args.index_source_root),
        paper_data=str(args.paper_data),
        build_dir=str(args.build_dir),
        minimum_time=args.min_time,
    )
    path = root / "state/manifest.json"
    if path.exists():
        manifest = core.read_json(path)
        if manifest["contract"] != contract:
            raise ValueError(
                "Sparse run arguments changed; use a new run root"
            )
        validate_sources(manifest)
        return manifest
    for source in (args.reference_root, args.index_source_root):
        if root == source or root.is_relative_to(source):
            raise ValueError(
                "Sparse output must be outside the immutable source runs"
            )
    source_manifest_path = args.index_source_root / "state/manifest.json"
    source_manifest = core.read_json(source_manifest_path)
    base = Path(source_manifest["inputs"]["em_k10"]["base_file"])
    base_record = next(
        r for r in source_manifest["input_files"] if r["path"] == str(base)
    )
    if core.identity(base) != {
        k: base_record[k] for k in ("path", "bytes", "mtime_ns")
    }:
        raise ValueError("Original base file changed since index construction")
    if core.matrix_header(base, 4) != (2735264, 4096):
        raise ValueError("Unexpected ArXiv base geometry")
    files = {
        str(base): {
            **base_record,
            "verify_sha256": False,
            "validation": "unchanged identity; SHA256 inherited from index-source preflight",
        }
    }

    def record(path):
        entry = {**core.identity(path), "sha256": core.digest(path)}
        files[entry["path"]] = entry
        return entry

    record(source_manifest_path)
    record(__file__)
    record(core.__file__)
    record(args.build_dir / "libcuvs.so")
    binary = args.build_dir / "bench/ann/CUVS_IVF_FLAT_ANN_BENCH"
    if not os.access(binary, os.X_OK):
        raise ValueError(f"Missing executable: {binary}")
    record(binary)
    indexes = [
        spec
        for spec in source_manifest["indexes"]
        if spec["corpus"] == "arxiv" and spec["kind"] == "flat"
    ]
    if sorted(spec["build"]["nlist"] for spec in indexes) != [
        1024,
        2048,
        4096,
    ]:
        raise ValueError("Expected the three original ArXiv IVF-Flat indexes")
    for spec in indexes:
        marker = (
            args.index_source_root / "state/builds" / (spec["id"] + ".json")
        )
        build = core.read_json(marker)
        if (
            core.identity(spec["file"]) != build["index_file"]
            or spec["build"] != build["build"]
        ):
            raise ValueError(f"Index/build provenance changed: {spec['id']}")
        record(marker)
        files[spec["file"]] = {
            **build["index_file"],
            "verify_sha256": False,
            "validation": "unchanged identity from successful source build",
        }
    references = root / "references"
    references.mkdir(parents=True, exist_ok=True)
    provenance = args.paper_data / "selectivity_followup_provenance.json"
    expected = {
        r["path"]: r["sha256"] for r in core.read_json(provenance)["sources"]
    }
    record(provenance)
    names = (
        "matched_selected.csv",
        "graph_summary.csv",
        "calibration_points.csv",
        "exact_control.csv",
    )
    for name in names:
        source = args.paper_data / "selectivity_followup_sources" / name
        entry = record(source)
        key = "data/selectivity_followup_sources/" + name
        if entry["sha256"] != expected[key]:
            raise ValueError(f"Paper reference hash mismatch: {source}")
        destination = references / name
        shutil.copyfile(source, destination)
        record(destination)
    inputs = {}
    for cohort in COHORTS:
        source = args.reference_root / "data" / cohort / "manifest.json"
        record(source)
        m = core.read_json(source)
        frozen = (
            args.paper_data
            / "selectivity_followup_sources/manifests"
            / (cohort + ".json")
        )
        if core.digest(source) != core.digest(frozen):
            raise ValueError(f"Cohort differs from paper source: {cohort}")
        if (m["base_rows"], m["query_rows"], len(m["shards"])) != (
            2735264,
            QUERY_COUNT,
            1,
        ):
            raise ValueError(f"Invalid sparse cohort geometry: {cohort}")
        shard = m["shards"][0]
        if (shard["first_query"], shard["query_count"]) != (0, QUERY_COUNT):
            raise ValueError(f"Invalid sparse shard: {cohort}")
        query = Path(shard["directory"]) / "query.bin"
        gt = Path(shard["directory"]) / "groundtruth.ibin"
        bitmap = Path(shard["bitmap"])
        if core.matrix_header(query, 4) != (
            QUERY_COUNT,
            4096,
        ) or core.matrix_header(gt, 4) != (QUERY_COUNT, 10):
            raise ValueError(f"Invalid query/GT shape: {cohort}")
        core.bitmap_header(bitmap, QUERY_COUNT, 2735264)
        ids = np.memmap(
            gt, mode="r", dtype=np.uint32, offset=8, shape=(QUERY_COUNT, 10)
        )
        words = np.memmap(bitmap, mode="r", dtype=np.uint32, offset=40)
        for q in range(QUERY_COUNT):
            valid = ids[q].astype(np.uint64)
            if np.any(valid >= 2735264) or len(np.unique(valid)) != 10:
                raise ValueError(
                    f"Invalid/duplicate GT in {cohort}, query {q}"
                )
            bits = q * 2735264 + valid
            if not np.all((words[bits >> 5] >> (bits & 31)) & 1):
                raise ValueError(
                    f"Ground truth violates bitmap in {cohort}, query {q}"
                )
        del ids, words
        for p in (query, gt, bitmap):
            record(p)
        inputs[cohort + "_k10"] = dict(
            workload=cohort,
            k=10,
            size=2735264,
            dim=4096,
            base_file=str(base),
            float_base_file=str(base),
            source_manifest=str(source),
            selectivity=m["metadata"]["global_selectivity"],
            actual_selectivity=shard["mean_selectivity"],
            passing_per_query=m["metadata"]["exact_passing"],
            shards=[
                dict(
                    first_query=0,
                    query_count=QUERY_COUNT,
                    query_file=str(query),
                    float_query_file=str(query),
                    groundtruth_file=str(gt),
                    bitmap_file=str(bitmap),
                )
            ],
        )
    exact = list(csv.DictReader((references / "exact_control.csv").open()))
    for cohort in COHORTS:
        rows = [r for r in exact if r["cohort"] == cohort]
        if len(rows) != 3 or any(
            float(r["native_l2_cutoff_recall"]) != 1 for r in rows
        ):
            raise ValueError(
                f"Exact control is incomplete/incorrect: {cohort}"
            )
    manifest = dict(
        schema_version=1,
        experiment="arxiv_sparse_ivf_flat",
        created_utc=core.utc(),
        contract=contract,
        indexes=indexes,
        inputs=inputs,
        files=list(files.values()),
        query_count=QUERY_COUNT,
        shard_count=1,
        k=10,
        repetitions=3,
        target=LOWER,
        upper=UPPER,
        interpolation_target=INTERPOLATION_TARGET,
        minimum_time=args.min_time,
        minimum_warmup_time=0.01,
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=core.REPO, text=True
        ).strip(),
        timing_contract="Complete GPU search; resident bitmap; index load and bitmap upload excluded",
    )
    core.write_json(path, manifest)
    return manifest


class SparseRunner(core.Runner):
    def progress(self, stage, detail, **extra):
        completed = list((self.root / "state/calibrated").glob("*/*.json"))
        timings = self.root / "logs/timing.jsonl"
        durations = defaultdict(float)
        if timings.exists():
            for line in timings.read_text().splitlines():
                row = json.loads(line)
                if row["stage"] == "calibration":
                    fields = row["detail"].split()
                    durations[(fields[0], fields[1])] += row["elapsed_seconds"]
        samples = [
            durations[(p.parent.name, p.stem.removesuffix("_k10"))]
            for p in completed
        ]
        samples = [s for s in samples if s > 0]
        status = dict(
            updated_utc=core.utc(),
            state="running",
            stage=stage,
            detail=detail,
            completed_calibrations=len(completed),
            total_calibrations=15,
            completed_commands=self.completed,
            elapsed_seconds=time.monotonic() - self.started,
            estimated_remaining_calibration_seconds=round(
                statistics.median(samples) * (15 - len(completed))
            )
            if samples
            else None,
            eta_scope="Calibration only; excludes final repetitions and analysis",
            **extra,
        )
        core.write_json(self.root / "state/progress.json", status)
        print(f"[{core.utc()}] {stage}: {detail}", flush=True)

    def calibrate(self):
        for spec in self.specs.values():
            for cohort in COHORTS:
                rows = self.measure(
                    spec,
                    cohort,
                    10,
                    core.coarse_points(spec),
                    "calibration",
                    "coarse",
                    1,
                )
                all_probes = next(
                    r for r in rows if r["nprobe"] == spec["build"]["nlist"]
                )
                if all_probes["recall_min"] < LOWER:
                    raise ValueError(
                        f"All-list search misses target: {spec['id']} {cohort}"
                    )
                core.write_json(
                    self.root
                    / "state/all_list_checks"
                    / spec["id"]
                    / (cohort + ".json"),
                    all_probes,
                )
                for iteration in range(17):
                    proposals = {}
                    for target in (LOWER, INTERPOLATION_TARGET, UPPER):
                        for point in core.refinement_proposals(rows, target):
                            proposals[core.point_id(point)] = point
                    if not proposals:
                        break
                    if iteration == 16:
                        raise ValueError(
                            "Sparse probe refinement failed to converge"
                        )
                    rows.extend(
                        self.measure(
                            spec,
                            cohort,
                            10,
                            list(proposals.values()),
                            "calibration",
                            f"refine_{iteration:02}",
                            1,
                        )
                    )
                core.write_json(
                    self.root
                    / "state/calibrated"
                    / spec["id"]
                    / (cohort + "_k10.json"),
                    rows,
                )
                self.progress(
                    "calibrated", f"{spec['id']} {cohort}: {len(rows)} points"
                )
                self.report()

    def finalize(self):
        required = [
            self.root / "state/calibrated" / spec / (c + "_k10.json")
            for spec in self.specs
            for c in COHORTS
        ]
        if not all(p.exists() for p in required):
            raise ValueError(
                "All 15 sparse calibration cases must complete before finals"
            )
        selection_path = self.root / "state/final_selection.json"
        if not selection_path.exists():
            rows = self.load_rows("calibration")
            core.write_json(
                selection_path,
                {
                    cohort: select_finalists(
                        [r for r in rows if r["workload"] == cohort]
                    )
                    for cohort in COHORTS
                },
            )
        for cohort, points in core.read_json(selection_path).items():
            grouped = defaultdict(list)
            for point in points:
                grouped[point["index_id"]].append(point)
            for index_id, selected in grouped.items():
                self.measure(
                    self.specs[index_id],
                    cohort,
                    10,
                    selected,
                    "final",
                    "selected",
                    3,
                )
                self.report()

    def report(self, complete=False):
        out = self.root / "analysis"
        out.mkdir(parents=True, exist_ok=True)
        calibration, final = (
            self.load_rows("calibration"),
            self.load_rows("final"),
        )
        for name, rows in (
            ("calibration_points", calibration),
            ("final_points", final),
        ):
            if rows:
                core.write_csv(
                    out / (name + ".csv"),
                    [
                        {k: v for k, v in r.items() if k != "repetition_rows"}
                        for r in rows
                    ],
                )
        references, curves = baseline_results(self.root)
        comparison = []
        for cohort in COHORTS:
            cells = [r for r in final if r["workload"] == cohort]
            if cells:
                comparison.append(
                    dict(
                        cohort=cohort,
                        method="ivf_flat",
                        **best_result(cells),
                        repetitions=3,
                        source="new_ivf_final",
                    )
                )
            comparison.extend(r for r in references if r["cohort"] == cohort)
        core.write_csv(out / "comparison.csv", comparison)
        lines = [
            "# Sparse ArXiv IVF-Flat comparison",
            "",
            "k=10; 2,048 original queries; resident bitmaps; matched Recall@10 [0.950, 0.952].",
            "IVF final and saved CAGRA/exact measurements use three repetitions. QPS is queries/s.",
            "An asterisk denotes latency interpolation at 0.951. Unmatched cells show maximum observed recall and its QPS.",
            "",
            "| Passing rate | Exact KNN | Base | Retain | NaviX | IVF-Flat |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for cohort in COHORTS:
            by_method = {
                r["method"]: r for r in comparison if r["cohort"] == cohort
            }
            cells = []
            for method in (
                "exact",
                "default_cagra",
                "default_cagra_accumulator",
                "navix_reference",
                "ivf_flat",
            ):
                row = by_method.get(method)
                if not row:
                    cell = "pending"
                elif row["status"] == "unmatched":
                    cell = f"max {row['max_recall']:.4f} @ {row['qps_at_max_recall']:,.1f}"
                else:
                    cell = f"{row['qps']:,.1f}" + (
                        "*" if row["status"] == "interpolated" else ""
                    )
                cells.append(cell)
            rate = (
                self.manifest["inputs"][cohort + "_k10"]["selectivity"] * 100
            )
            lines.append(f"| {rate:g}% | " + " | ".join(cells) + " |")
        lines += [
            "",
            "Exact KNN is a saved full-recall reference, not a matched-recall ANN point.",
            "Saved CAGRA data come from the paper's frozen selectivity sources; existing experiments and paper files are unchanged.",
        ]
        (out / "comparison.md").write_text("\n".join(lines) + "\n")
        curve_rows = curves + [
            dict(
                cohort=r["workload"],
                method="ivf_flat",
                recall=r["recall_median"],
                qps=r["qps_median"],
                repetitions=r["repetitions"],
                source=r["phase"],
                point_id=r["point_id"],
            )
            for r in calibration + final
        ]
        core.write_csv(out / "curve_points.csv", curve_rows)
        if complete:
            if any(
                not any(
                    r["cohort"] == c and r["method"] == "ivf_flat"
                    for r in comparison
                )
                for c in COHORTS
            ):
                raise ValueError("Missing final IVF results")
            plot_results(out, curve_rows, comparison)
            core.write_json(
                out / "provenance.json",
                dict(
                    manifest_sha256=core.digest(
                        self.root / "state/manifest.json"
                    ),
                    comparison_sha256=core.digest(out / "comparison.csv"),
                    curve_points_sha256=core.digest(out / "curve_points.csv"),
                    created_utc=core.utc(),
                ),
            )


def baseline_results(root):
    def read(name):
        with (root / "references" / name).open() as stream:
            return list(csv.DictReader(stream))

    graph = read("graph_summary.csv")
    selected = read("matched_selected.csv")
    calibration = read("calibration_points.csv")
    exact = read("exact_control.csv")
    output, curves = [], []
    for cohort in COHORTS:
        for method in (
            "default_cagra",
            "default_cagra_accumulator",
            "navix_reference",
        ):
            rows = [
                r
                for r in graph
                if r["workload"] == cohort and r["method"] == method
            ]
            if not rows or any(
                int(r["queries_per_repetition"]) != QUERY_COUNT
                or int(r["repetitions"]) != 3
                for r in rows
            ):
                raise ValueError(
                    f"Saved graph contract mismatch: {cohort} {method}"
                )
            maximum = max(
                rows,
                key=lambda r: (
                    float(r["recall_median"]),
                    float(r["qps_median"]),
                ),
            )
            match = next(
                r
                for r in selected
                if r["cohort"] == cohort and r["method"] == method
            )
            row = dict(
                cohort=cohort,
                method=method,
                status="unmatched",
                recall="",
                qps="",
                max_recall=float(maximum["recall_median"]),
                qps_at_max_recall=float(maximum["qps_median"]),
                repetitions=3,
                source="frozen_paper_selectivity",
            )
            if match["status"] == "matched":
                if (
                    not LOWER
                    <= float(match["recall_min"])
                    <= float(match["recall_max"])
                    <= UPPER + 1e-12
                ):
                    raise ValueError(
                        f"Saved match outside window: {cohort} {method}"
                    )
                row.update(
                    status="measured",
                    recall=float(match["recall_median"]),
                    qps=float(match["qps_median"]),
                )
            output.append(row)
            curves.extend(
                dict(
                    cohort=cohort,
                    method=method,
                    recall=float(r["recall_median"]),
                    qps=float(r["qps_median"]),
                    repetitions=3,
                    source="saved_graph",
                )
                for r in rows
            )
            if row["status"] == "measured":
                curves.append(
                    dict(
                        cohort=cohort,
                        method=method,
                        recall=row["recall"],
                        qps=row["qps"],
                        repetitions=3,
                        source="saved_matched",
                    )
                )
        rows = [r for r in exact if r["cohort"] == cohort]
        qps = statistics.median(float(r["qps"]) for r in rows)
        output.append(
            dict(
                cohort=cohort,
                method="exact",
                status="exact",
                recall=1.0,
                qps=qps,
                max_recall=1.0,
                qps_at_max_recall=qps,
                repetitions=3,
                source="frozen_exact_control",
            )
        )
        curves.append(
            dict(
                cohort=cohort,
                method="exact",
                recall=1.0,
                qps=qps,
                repetitions=3,
                source="saved_exact",
            )
        )
    curves.extend(
        dict(
            cohort=r["cohort"],
            method=r["method"],
            recall=float(r["recall"]),
            qps=float(r["qps"]),
            repetitions=1 if r["source"] != "reference" else 3,
            source="saved_calibration",
        )
        for r in calibration
        if r["cohort"] in COHORTS
    )
    return output, curves


def plot_results(out, curves, comparison):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for cohort in COHORTS:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
        for method, label in LABELS.items():
            rows = [
                r
                for r in curves
                if r["cohort"] == cohort and r["method"] == method
            ]
            if method == "exact":
                for ax in axes:
                    ax.axhline(
                        rows[0]["qps"], color="0.35", ls=":", label=label
                    )
                continue
            # Upper envelope: remove points dominated in both recall and QPS.
            frontier, best = [], -1
            for r in sorted(
                rows, key=lambda r: (r["recall"], r["qps"]), reverse=True
            ):
                if r["qps"] > best:
                    frontier.append(r)
                    best = r["qps"]
            frontier.reverse()
            for ax in axes:
                ax.plot(
                    [r["recall"] for r in frontier],
                    [r["qps"] for r in frontier],
                    ".-",
                    label=label,
                    ms=3,
                )
        for ax in axes:
            ax.axvspan(LOWER, UPPER, alpha=0.12, color="green")
            ax.set(xlabel="Recall@10", ylabel="Queries/s", yscale="log")
            ax.grid(alpha=0.25)
        axes[0].set_xlim(0, 1.005)
        axes[1].set_xlim(0.94, 0.96)
        axes[0].legend(fontsize=8)
        rate = (
            float(
                cohort.removeprefix("synthetic_low_")
                .removesuffix("_random")
                .replace("p", ".")
            )
            * 100
        )
        fig.suptitle(f"ArXiv-large — {rate:g}% passing; k=10; 2,048 queries")
        for extension in ("pdf", "png"):
            fig.savefig(out / f"{cohort}.{extension}", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("all", "prepare", "calibrate", "final", "analyze", "status"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path(
            "/data/retrieve_workshop_runs/a100_selectivity_20260928T070431Z"
        ),
    )
    parser.add_argument(
        "--index-source-root",
        type=Path,
        default=Path(
            "/data/retrieve_workshop_runs/a100_ivf_bitmap_20261014T024338Z"
        ),
    )
    parser.add_argument(
        "--paper-data",
        type=Path,
        default=Path("/home/ubuntu/paper_selectivity_revision/data"),
    )
    parser.add_argument(
        "--build-dir", type=Path, default=core.REPO / "cpp/build"
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("/data/retrieve_data")
    )
    parser.add_argument("--min-time", default="0.10s")
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.stage == "status":
        print(
            json.dumps(
                core.read_json(args.run_root / "state/progress.json"), indent=2
            )
        )
        return
    args.run_root.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(args.run_root) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            manifest = prepare(args)
            runner = SparseRunner(args, manifest)
            runner.progress(
                "prepared",
                "Five sparse cohorts; three reused IVF-Flat indexes",
            )
            if args.stage in ("all", "calibrate"):
                runner.calibrate()
            if args.stage in ("all", "final"):
                runner.finalize()
            runner.report(complete=args.stage in ("all", "final", "analyze"))
            runner.progress(
                "complete"
                if args.stage in ("all", "final", "analyze")
                else "stage_complete",
                args.stage,
            )
            p = core.read_json(args.run_root / "state/progress.json")
            p["state"] = p["stage"]
            core.write_json(args.run_root / "state/progress.json", p)
        except Exception as error:
            core.write_json(
                args.run_root / "state/progress.json",
                dict(state="failed", updated_utc=core.utc(), error=str(error)),
            )
            raise


if __name__ == "__main__":
    main()
