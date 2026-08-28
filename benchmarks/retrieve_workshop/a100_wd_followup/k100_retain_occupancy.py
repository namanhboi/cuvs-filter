#!/usr/bin/env python3
"""Generate and validate matched Base/Retain resource captures at k=100."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

WORKLOADS = ("yfcc", "em", "emis", "r")
BASE = "default_cagra"
RETAIN = "default_cagra_accumulator"
RESOURCE_METHODS = ("base", "retain")
RESOURCE_PREFIX = "CAGRA_KERNEL_RESOURCES "
K = 100
MAX_QUERIES = 2048
GRAPH_DEGREE = 64
EXPECTED_RETAIN_SMEM_DELTA = 4 + 2 * K * 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_selected_retain_points(bundle: Path) -> dict[str, dict]:
    with tarfile.open(bundle, "r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and member.name.endswith("/matched_recall/selected_points.csv")
        ]
        if len(members) != 1:
            raise ValueError(
                f"expected one matched_recall/selected_points.csv in {bundle}, "
                f"found {[member.name for member in members]}"
            )
        extracted = archive.extractfile(members[0])
        if extracted is None:
            raise ValueError(f"could not read {members[0].name} from {bundle}")
        rows = list(csv.DictReader(io.StringIO(extracted.read().decode("utf-8"))))

    selected: dict[str, dict] = {}
    for row in rows:
        if (
            row.get("phase") != "throughput"
            or row.get("method") != RETAIN
            or not truthy(row.get("selected"))
            or not truthy(row.get("paper_included"))
        ):
            continue
        workload = str(row.get("workload"))
        if workload not in WORKLOADS:
            continue
        if workload in selected:
            raise ValueError(f"duplicate selected Retain row for {workload}")
        point = {
            "requested_itopk": int(row["itopk"]),
            "search_width": int(row["search_width"]),
            "max_iterations": int(row["max_iterations"]),
            "resolved_iterations": int(row["resolved_iterations"]),
            "target_recall": float(row["target_recall"]),
            "target_reached": truthy(row["target_reached"]),
            "selection_rule": str(row["selection_rule"]),
        }
        if int(row["graph_degree"]) != GRAPH_DEGREE:
            raise ValueError(f"{workload} selected point is not degree {GRAPH_DEGREE}")
        if point["requested_itopk"] < K or point["search_width"] <= 0:
            raise ValueError(f"invalid selected Retain point for {workload}: {point}")
        selected[workload] = point
    if set(selected) != set(WORKLOADS):
        raise ValueError(
            f"selected Retain points do not cover all workloads: "
            f"missing {sorted(set(WORKLOADS) - set(selected))}"
        )
    return selected


def method_template(searches: list[dict], method: str) -> dict:
    matches = [row for row in searches if row.get("bitmap_method") == method]
    if not matches:
        raise ValueError(f"source configuration has no {method} search")
    return copy.deepcopy(matches[0])


def paired_search(template: dict, method: str, point: dict) -> dict:
    result = copy.deepcopy(template)
    for key in list(result):
        if (
            key.startswith(("favor_diagnostics_", "navix_", "cagra_"))
            or key == "benchmark_output_neighbors_file"
        ):
            result.pop(key)
    result.update(
        {
            "algo": "single_cta",
            "filter_mode": "default",
            "max_queries": MAX_QUERIES,
            "itopk": point["requested_itopk"],
            "search_width": point["search_width"],
            "max_iterations": point["max_iterations"],
            "favor_udf_passing_accumulator": method == RETAIN,
            "require_identity_source_indices": True,
            "bitmap_method": method,
            "k": K,
        }
    )
    return result


def write_immutable_json(path: Path, payload: dict) -> None:
    serialized = json.dumps(payload, indent=2) + "\n"
    if path.exists():
        if path.read_text() != serialized:
            raise FileExistsError(f"refusing to replace incompatible file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized)


def generate_configs(
    source_config_root: Path, reference_bundle: Path, output: Path
) -> dict:
    selected = read_selected_retain_points(reference_bundle)
    cases: list[dict] = []
    for workload in WORKLOADS:
        source_path = source_config_root / workload / "shard_00.json"
        if not source_path.is_file():
            raise FileNotFoundError(f"missing source configuration: {source_path}")
        payload = json.loads(source_path.read_text())
        if int(payload.get("search_basic_param", {}).get("k", -1)) != K:
            raise ValueError(f"{source_path} is not a k={K} configuration")
        indexes = payload.get("index", [])
        if len(indexes) != 1:
            raise ValueError(f"expected one index in {source_path}")
        index = indexes[0]
        if int(index.get("build_param", {}).get("graph_degree", -1)) != GRAPH_DEGREE:
            raise ValueError(f"{source_path} is not a degree-{GRAPH_DEGREE} CAGRA index")
        searches = index.get("search_params", [])
        point = selected[workload]
        base = paired_search(method_template(searches, BASE), BASE, point)
        retain = paired_search(method_template(searches, RETAIN), RETAIN, point)
        paired_payload = copy.deepcopy(payload)
        paired_payload["search_basic_param"]["k"] = K
        paired_payload["index"][0]["search_params"] = [base, retain]
        destination = output / f"{workload}.json"
        write_immutable_json(destination, paired_payload)
        cases.append(
            {
                "workload": workload,
                "config": str(destination.resolve()),
                "config_sha256": sha256(destination),
                "source_config": str(source_path.resolve()),
                "source_config_sha256": sha256(source_path),
                **point,
            }
        )

    manifest = {
        "schema_version": 1,
        "experiment": "a100_k100_retain_occupancy",
        "k": K,
        "max_queries": MAX_QUERIES,
        "graph_degree": GRAPH_DEGREE,
        "methods": [BASE, RETAIN],
        "workloads": list(WORKLOADS),
        "expected_retain_dynamic_smem_delta_bytes": EXPECTED_RETAIN_SMEM_DELTA,
        "selection_source": {
            "reference_bundle": str(reference_bundle.resolve()),
            "reference_bundle_sha256": sha256(reference_bundle),
            "member": "matched_recall/selected_points.csv",
        },
        "source_config_root": str(source_config_root.resolve()),
        "cases": cases,
    }
    write_immutable_json(output / "manifest.json", manifest)
    return manifest


def resource_tuple(record: dict) -> tuple:
    fields = (
        "method",
        "graph_degree",
        "itopk",
        "search_width",
        "threads_per_cta",
        "dynamic_smem_bytes",
        "static_smem_bytes",
        "registers_per_thread",
        "active_ctas_per_sm",
    )
    return tuple(str(record[field]) if field == "method" else int(record[field]) for field in fields)


def load_resources(path: Path) -> dict[str, dict]:
    grouped: dict[str, set[tuple]] = {method: set() for method in RESOURCE_METHODS}
    values: dict[tuple, dict] = {}
    for line in path.read_text().splitlines():
        position = line.find(RESOURCE_PREFIX)
        if position < 0:
            continue
        record = json.loads(line[position + len(RESOURCE_PREFIX) :])
        method = str(record.get("method"))
        if method not in grouped or bool(record.get("diagnostics")):
            continue
        key = resource_tuple(record)
        grouped[method].add(key)
        values[key] = record
    result: dict[str, dict] = {}
    for method, keys in grouped.items():
        if len(keys) != 1:
            raise ValueError(
                f"expected one consistent {method} resource tuple in {path}, got {sorted(keys)}"
            )
        result[method] = values[next(iter(keys))]
    return result


def label_value(label: str, key: str) -> str:
    match = re.search(rf'(?:^|#){re.escape(key)}="([^"]+)"', label)
    return match.group(1) if match else ""


def rounded_counter(record: dict, key: str) -> int:
    return round(float(record.get(key, -1)))


def load_raw(path: Path, case: dict) -> dict:
    payload = json.loads(path.read_text())
    context = payload.get("context", {})
    gpu_name = str(context.get("gpu_name", ""))
    if "A100" not in gpu_name:
        raise ValueError(f"resource capture did not run on an A100: {gpu_name!r}")
    if int(context.get("max_k", -1)) != K:
        raise ValueError(f"resource capture did not advertise max_k={K}")
    rows: dict[str, dict] = {}
    for record in payload.get("benchmarks", []):
        if record.get("run_type") != "iteration":
            continue
        method = label_value(str(record.get("label", "")), "bitmap_method")
        if method not in (BASE, RETAIN):
            continue
        if method in rows:
            raise ValueError(f"duplicate iteration row for {method} in {path}")
        if record.get("error_occurred") or record.get("skipped"):
            raise ValueError(f"benchmark error for {method} in {path}")
        expected = {
            "k": K,
            "max_queries": MAX_QUERIES,
            "itopk": int(case["requested_itopk"]),
            "search_width": int(case["search_width"]),
            "max_iterations": int(case["max_iterations"]),
        }
        for key, value in expected.items():
            if rounded_counter(record, key) != value:
                raise ValueError(
                    f"{method} {key} mismatch in {path}: "
                    f"{rounded_counter(record, key)} != {value}"
                )
        if rounded_counter(record, "n_queries") <= 0:
            raise ValueError(f"{method} processed no queries in {path}")
        rows[method] = record
    if set(rows) != {BASE, RETAIN}:
        raise ValueError(f"missing Base/Retain iteration rows in {path}")
    return {"gpu_name": gpu_name, "rows": rows}


def compare_case(case: dict, raw_path: Path, resource_path: Path) -> dict:
    errors: list[str] = []
    raw = load_raw(raw_path, case)
    resources = load_resources(resource_path)
    base = resources["base"]
    retain = resources["retain"]

    if int(base["graph_degree"]) != GRAPH_DEGREE or int(retain["graph_degree"]) != GRAPH_DEGREE:
        errors.append(f"graph degree is not {GRAPH_DEGREE}")
    if int(base["itopk"]) != int(retain["itopk"]):
        errors.append("Base and Retain use different internal L")
    if int(base["itopk"]) < int(case["requested_itopk"]):
        errors.append("internal L is smaller than requested L")
    if int(base["search_width"]) != int(case["search_width"]):
        errors.append("Base resource record has the wrong W")
    if int(retain["search_width"]) != int(case["search_width"]):
        errors.append("Retain resource record has the wrong W")

    equal_fields = (
        ("threads_per_cta", "threads per CTA"),
        ("static_smem_bytes", "static shared memory"),
        ("registers_per_thread", "registers per thread"),
        ("active_ctas_per_sm", "active CTAs per SM"),
    )
    for field, label in equal_fields:
        if int(base[field]) != int(retain[field]):
            errors.append(
                f"{label} changed: Base={int(base[field])}, Retain={int(retain[field])}"
            )
    delta = int(retain["dynamic_smem_bytes"]) - int(base["dynamic_smem_bytes"])
    if delta != EXPECTED_RETAIN_SMEM_DELTA:
        errors.append(
            f"dynamic shared-memory delta is {delta} B, expected "
            f"{EXPECTED_RETAIN_SMEM_DELTA} B"
        )
    if int(base["active_ctas_per_sm"]) <= 0:
        errors.append("CUDA reported a nonpositive active-CTA ceiling")

    return {
        "workload": case["workload"],
        "gpu_name": raw["gpu_name"],
        "requested_itopk": int(case["requested_itopk"]),
        "internal_itopk": int(base["itopk"]),
        "search_width": int(case["search_width"]),
        "max_iterations": int(case["max_iterations"]),
        "resolved_iterations": int(case["resolved_iterations"]),
        "base_threads_per_cta": int(base["threads_per_cta"]),
        "retain_threads_per_cta": int(retain["threads_per_cta"]),
        "base_registers_per_thread": int(base["registers_per_thread"]),
        "retain_registers_per_thread": int(retain["registers_per_thread"]),
        "base_dynamic_smem_bytes": int(base["dynamic_smem_bytes"]),
        "retain_dynamic_smem_bytes": int(retain["dynamic_smem_bytes"]),
        "retain_dynamic_smem_delta_bytes": delta,
        "base_static_smem_bytes": int(base["static_smem_bytes"]),
        "retain_static_smem_bytes": int(retain["static_smem_bytes"]),
        "base_active_ctas_per_sm": int(base["active_ctas_per_sm"]),
        "retain_active_ctas_per_sm": int(retain["active_ctas_per_sm"]),
        "passed": not errors,
        "errors": errors,
    }


def write_analysis(result_root: Path, report: dict) -> None:
    output = result_root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "k100_retain_occupancy.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    rows = report["rows"]
    fieldnames = [
        "workload",
        "gpu_name",
        "requested_itopk",
        "internal_itopk",
        "search_width",
        "max_iterations",
        "base_threads_per_cta",
        "retain_threads_per_cta",
        "base_registers_per_thread",
        "retain_registers_per_thread",
        "base_dynamic_smem_bytes",
        "retain_dynamic_smem_bytes",
        "retain_dynamic_smem_delta_bytes",
        "base_active_ctas_per_sm",
        "retain_active_ctas_per_sm",
        "passed",
        "errors",
    ]
    with (output / "k100_retain_occupancy.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["errors"] = "; ".join(row["errors"])
            writer.writerow(serialized)

    lines = [
        f"{report['status']}: k=100 Retain static occupancy validation",
        (
            "workload  requested L/W  internal L  registers (B/R)  "
            "dynamic smem B/R (+delta)  active CTAs/SM B/R"
        ),
    ]
    for row in rows:
        if "requested_itopk" not in row:
            lines.append(
                f"{row.get('workload', 'unknown')!s:8}  validation failed"
            )
            for error in row["errors"]:
                lines.append(f"  ERROR: {error}")
            continue
        lines.append(
            f"{row['workload']:8}  {row['requested_itopk']:4}/{row['search_width']}"
            f"          {row['internal_itopk']:4}        "
            f"{row['base_registers_per_thread']}/{row['retain_registers_per_thread']}"
            f"               {row['base_dynamic_smem_bytes']}/"
            f"{row['retain_dynamic_smem_bytes']} (+{row['retain_dynamic_smem_delta_bytes']})"
            f"          {row['base_active_ctas_per_sm']}/"
            f"{row['retain_active_ctas_per_sm']}"
        )
        for error in row["errors"]:
            lines.append(f"  ERROR: {error}")
    for error in report["global_errors"]:
        lines.append(f"ERROR: {error}")
    lines.extend(
        [
            "",
            (
                "Scope: cudaFuncGetAttributes plus "
                "cudaOccupancyMaxActiveBlocksPerMultiprocessor. This validates the "
                "static active-CTA ceiling, not achieved runtime occupancy or throughput."
            ),
        ]
    )
    (output / "k100_retain_occupancy.txt").write_text("\n".join(lines) + "\n")


def analyze(result_root: Path) -> dict:
    manifest_path = result_root / "configs" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("experiment") != "a100_k100_retain_occupancy"
        or int(manifest.get("k", -1)) != K
        or int(manifest.get("expected_retain_dynamic_smem_delta_bytes", -1))
        != EXPECTED_RETAIN_SMEM_DELTA
    ):
        raise ValueError(f"invalid occupancy manifest: {manifest_path}")

    global_errors: list[str] = []
    provenance = result_root / "provenance" / "run.json"
    if not provenance.is_file():
        global_errors.append(f"missing provenance: {provenance}")
    rows: list[dict] = []
    for case in manifest.get("cases", []):
        workload = str(case.get("workload"))
        try:
            config_path = Path(case["config"])
            if not config_path.is_file() or sha256(config_path) != case["config_sha256"]:
                raise ValueError(f"generated configuration changed: {config_path}")
            rows.append(
                compare_case(
                    case,
                    result_root / "raw" / f"{workload}.json",
                    result_root / "resources" / f"{workload}.log",
                )
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            rows.append(
                {
                    "workload": workload,
                    "passed": False,
                    "errors": [str(error)],
                }
            )
    if [row.get("workload") for row in rows] != list(WORKLOADS):
        global_errors.append("manifest cases are not exactly yfcc, em, emis, r in order")

    evidence = [manifest_path]
    if provenance.is_file():
        evidence.append(provenance)
    for workload in WORKLOADS:
        evidence.extend(
            [
                result_root / "configs" / f"{workload}.json",
                result_root / "raw" / f"{workload}.json",
                result_root / "resources" / f"{workload}.log",
            ]
        )
    missing_evidence = [str(path) for path in evidence if not path.is_file()]
    if missing_evidence:
        global_errors.append(f"missing evidence files: {missing_evidence}")
    report = {
        "schema_version": 1,
        "status": "PASS"
        if not global_errors and len(rows) == len(WORKLOADS) and all(row["passed"] for row in rows)
        else "FAIL",
        "experiment": "a100_k100_retain_occupancy",
        "k": K,
        "expected_retain_dynamic_smem_delta_bytes": EXPECTED_RETAIN_SMEM_DELTA,
        "validation": {
            "same_threads_per_cta": True,
            "same_registers_per_thread": True,
            "same_static_smem_bytes": True,
            "same_active_ctas_per_sm": True,
            "api": [
                "cudaFuncGetAttributes",
                "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
            ],
            "interpretation": (
                "static active-CTA ceiling; not achieved runtime occupancy or performance"
            ),
        },
        "rows": rows,
        "global_errors": global_errors,
        "evidence": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in evidence
            if path.is_file()
        ],
    }
    write_analysis(result_root, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--source-config-root", type=Path, required=True)
    generate.add_argument("--reference-bundle", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "generate":
        manifest = generate_configs(
            args.source_config_root.resolve(),
            args.reference_bundle.resolve(),
            args.output.resolve(),
        )
        print(args.output.resolve() / "manifest.json")
        print(json.dumps({"cases": len(manifest["cases"]), "k": K}))
    else:
        report = analyze(args.result_root.resolve())
        print(args.result_root.resolve() / "analysis")
        print(json.dumps({"status": report["status"], "rows": len(report["rows"])}))
        if report["status"] != "PASS":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
