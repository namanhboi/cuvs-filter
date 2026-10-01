#!/usr/bin/env python3
"""Measured tight-recall refinement for the selectivity follow-up.

The completed selectivity run is read-only. New configurations and measurements
live in a separate result root, and only fresh three-repetition finalists can
enter the matched-recall comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path

import workflow as study

TARGET = 0.950
HIGH = 0.951
ROUNDING_EPSILON = 1e-12
ANCHORS = (10, 16, 24, 32, 48, 64, 96, 128, 256, 512)
WIDTHS = (1, 2)
METHODS = study.METHODS
FINALISTS = 4
MIN_CALIBRATION_MATCHES = 1
REACH_STEPS = 5  # ceil(log_32(2,735,264))
# The completed calibration already covers the coarse L frontier.  A small
# neighborhood catches an unmeasured capacity transition without reopening a
# broad sweep over every requested-L value.
RESCUE_L_RADIUS = 4
RESCUE_PAIRS = frozenset({
    ("synthetic_correlation_0p10_positive", "navix_reference"),
    ("synthetic_correlation_0p25_random", "navix_reference"),
    ("synthetic_correlation_0p90_random", "navix_reference"),
})


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def cohorts(reference: Path) -> list[str]:
    names = sorted(
        p.parent.name
        for p in (reference / "data").glob("*/manifest.json")
        if p.parent.name.startswith(("real_", "synthetic_low_", "synthetic_correlation_"))
    )
    negative = {
        f"synthetic_correlation_{level.replace('.', 'p')}_negative"
        for level in study.CORRELATION_SELECTIVITIES
    }
    if len(names) != 21 and set(names) != negative:
        raise ValueError(f"expected 21 legacy cohorts or five negative cohorts, found {names}")
    return names


def point_key(row: object) -> tuple[str, str, int, int, int]:
    get = row.get if isinstance(row, dict) else lambda name: getattr(row, name)
    return (
        str(get("workload")),
        str(get("method")),
        int(get("itopk")),
        int(get("search_width")),
        int(get("max_iterations")),
    )


def automatic_iterations(itopk: int, width: int) -> int:
    if not 10 <= itopk <= 512 or width not in WIDTHS:
        raise ValueError(f"invalid L/W: {itopk}/{width}")
    return itopk // width + REACH_STEPS


def fingerprint(itopk: int, width: int, maximum: int) -> tuple[int, int, int]:
    """Execution identity: rounded capacity, width, and resolved iteration budget."""
    return ((itopk + 31) // 32 * 32, width, maximum or automatic_iterations(itopk, width))


def legal(method: str, itopk: int, width: int, maximum: int) -> bool:
    return (
        method in METHODS
        and 10 <= itopk <= 512
        and width in WIDTHS
        and maximum >= 0
        and (maximum or automatic_iterations(itopk, width))
        <= study.legal_hash_iterations(method, itopk, width)
    )


def search(row: tuple[str, str, int, int, int]) -> dict:
    _, method, itopk, width, maximum = row
    if not legal(method, itopk, width, maximum):
        raise ValueError(f"illegal search point: {row}")
    return study.followup_search_point(method, itopk, width, maximum)


def in_band(recalls: list[float]) -> bool:
    return len(recalls) == 3 and all(within(value) for value in recalls)


def within(value: float) -> bool:
    """Accept only binary floating-point roundoff at a closed endpoint."""
    return TARGET - ROUNDING_EPSILON <= value <= HIGH + ROUNDING_EPSILON


def initial_contract(root: Path, reference: Path, binary: Path, library: Path,
                     calibration_root: Path | None = None,
                     prior_finalists_root: Path | None = None) -> None:
    if root.resolve() == reference.resolve() or root.resolve().is_relative_to(reference.resolve()):
        raise ValueError("matched result root must be outside the immutable reference root")
    if calibration_root is not None and root.resolve() == calibration_root.resolve():
        raise ValueError("matched result root must differ from the calibration root")
    if prior_finalists_root is not None and root.resolve() == prior_finalists_root.resolve():
        raise ValueError("matched result root must differ from the prior finalists root")
    names = cohorts(reference)
    expected = {
        "schema_version": 1,
        "experiment": "a100_selectivity_matched_recall_001",
        "reference_root": str(reference.resolve()),
        "reference_contract_sha256": digest(reference / "state/contract.json"),
        "reference_graph_summary_sha256": digest(reference / "analysis/graph_summary.csv"),
        "reference_exact_control_sha256": digest(reference / "analysis/exact_control.csv"),
        "binary": str(binary.resolve()),
        "binary_sha256": digest(binary),
        "libcuvs": str(library.resolve()),
        "libcuvs_sha256": digest(library),
        "target": TARGET,
        "upper": HIGH,
        "repetitions": 3,
        "methods": list(METHODS),
        "widths": list(WIDTHS),
        "cohort_manifests": {
            name: digest(reference / "data" / name / "manifest.json") for name in names
        },
    }
    if calibration_root is not None:
        expected["calibration_root"] = str(calibration_root.resolve())
        expected["calibration_contract_sha256"] = digest(
            calibration_root / "state/contract.json"
        )
        expected["calibration_points_sha256"] = digest(
            calibration_root / "analysis/calibration_points.csv"
        )
    if prior_finalists_root is not None:
        expected["prior_finalists_root"] = str(prior_finalists_root.resolve())
        expected["prior_finalists_contract_sha256"] = digest(
            prior_finalists_root / "state/contract.json"
        )
        expected["prior_finalists_selected_sha256"] = digest(
            prior_finalists_root / "analysis/matched_selected.csv"
        )
    contract = root / "state/contract.json"
    contract.parent.mkdir(parents=True, exist_ok=True)
    if contract.exists() and json.loads(contract.read_text()) != expected:
        raise ValueError(f"matched run contract drifted: {contract}")
    if not contract.exists():
        study.atomic_json(contract, expected)
    for name in names:
        source = reference / "data" / name / "manifest.json"
        target = root / "data" / name / "manifest.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"cohort manifest drifted: {target}")
        if not target.exists():
            shutil.copyfile(source, target)
        manifest = json.loads(source.read_text())
        shard = Path(manifest["shards"][0]["directory"])
        if not all((shard / item).is_file() for item in ("query.bin", "filter.bitmap", "groundtruth.ibin")):
            raise FileNotFoundError(f"missing frozen cohort inputs: {shard}")


def baseline_rows(reference: Path) -> dict[tuple[str, str, int, int, int], dict]:
    result = {}
    with (reference / "analysis/graph_summary.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["method"] not in METHODS:
                continue
            key = point_key(row)
            result[key] = {
                "recall": float(row["recall_median"]),
                "qps": float(row["qps_median"]),
                "recalls": [float(row["recall_min"]), float(row["recall_median"]), float(row["recall_max"])],
                "source": "reference",
            }
    return result


def new_rows(root: Path, *, finalists: bool = False) -> dict[tuple[str, str, int, int, int], dict]:
    result = {}
    for path in sorted((root / "graph/configs").glob("*/*/manifest.json")):
        group = path.parent.parent.name
        is_finalist = group.startswith("finalists") or group.startswith("rescue_finalists")
        if is_finalist != finalists:
            continue
        manifest = json.loads(path.read_text())
        try:
            raw = study.graph_analysis.load_group(path, manifest, root / "graph/raw")
        except (FileNotFoundError, json.JSONDecodeError):
            # A stopped run may have a manifest and a partially written raw file.
            # The frozen plan below resumes that group; the benchmark helper
            # refuses to overwrite an incomplete existing result.
            if finalists:
                raise
            continue
        except ValueError as exc:
            if finalists or not str(exc).startswith(("incomplete raw files", "incomplete points")):
                raise
            continue
        by_key: dict[tuple[str, str, int, int, int], list] = defaultdict(list)
        for row in raw:
            by_key[point_key(row)].append(row)
        for key, rows in by_key.items():
            rows.sort(key=lambda row: row.repetition_index)
            if len(rows) != int(manifest["repetitions"]):
                raise ValueError(f"incomplete repetitions for {key} in {path}")
            result[key] = {
                "recall": statistics.median(row.recall for row in rows),
                "qps": statistics.median(row.qps for row in rows),
                "recalls": [row.recall for row in rows],
                "source": group,
                "underfilled": max(row.underfilled_queries for row in rows),
                "duplicates": max(row.duplicate_output_query_rate for row in rows),
            }
    return result


def observations(root: Path, reference: Path,
                 calibration_root: Path | None = None) -> dict[tuple[str, str, int, int, int], dict]:
    measured = baseline_rows(reference) | new_rows(calibration_root or root)
    if calibration_root is not None:
        measured |= new_rows(root)
    return measured


def planned(root: Path, stage: str, candidates: set[tuple[str, str, int, int, int]]) -> list[tuple[str, str, int, int, int]]:
    path = root / "state" / f"{stage}.json"
    rows = sorted(candidates)
    if path.exists():
        return [tuple(row) for row in json.loads(path.read_text())]
    study.atomic_json(path, rows)
    return rows


def run_stage(root: Path, data_root: Path, binary: Path, library: Path,
              stage: str, candidates: set[tuple[str, str, int, int, int]], repetitions: int = 1) -> None:
    rows = planned(root, stage, candidates)
    for cohort in sorted({row[0] for row in rows}):
        searches = [search(row) for row in rows if row[0] == cohort]
        if searches:
            print(f"{stage}: {cohort}: {len(searches)} points", flush=True)
            study.run_graph_group(root, data_root, cohort, stage, searches,
                                  binary, library, repetitions)


def missing(candidates: set[tuple[str, str, int, int, int]], measured: dict) -> set:
    return {row for row in candidates if row not in measured and legal(*row[1:])}


def anchors(names: list[str], measured: dict) -> set:
    candidates = {
        (name, method, itopk, width, 0)
        for name in names for method in METHODS for width in WIDTHS for itopk in ANCHORS
    }
    return missing(candidates, measured)


def refine_l(names: list[str], measured: dict) -> set:
    candidates = set()
    for name in names:
        for method in METHODS:
            for width in WIDTHS:
                local = sorted(
                    (key, value) for key, value in measured.items()
                    if key[0] == name and key[1] == method and key[3] == width and key[4] == 0
                )
                local.sort(key=lambda item: item[0][2])
                seen = {fingerprint(key[2], width, 0) for key, _ in local}
                ranges = []
                for (left, a), (right, b) in zip(local, local[1:]):
                    if (a["recall"] - TARGET) * (b["recall"] - TARGET) <= 0 or (
                        (a["recall"] - HIGH) * (b["recall"] - HIGH) <= 0
                    ):
                        ranges.append(range(left[2] + 1, right[2]))
                for key, value in local:
                    if within(value["recall"]):
                        ranges.append(range(max(10, key[2] - 4), min(512, key[2] + 4) + 1))
                for interval in ranges:
                    for itopk in interval:
                        identity = fingerprint(itopk, width, 0)
                        if identity not in seen:
                            seen.add(identity)
                            candidates.add((name, method, itopk, width, 0))
    return missing(candidates, measured)


def iteration_seeds(names: list[str], measured: dict) -> set:
    candidates = set()
    for name in names:
        for method in METHODS:
            for width in WIDTHS:
                local = [
                    (key, value) for key, value in measured.items()
                    if key[:2] == (name, method) and key[3] == width and key[4] == 0
                ]
                if not local:
                    continue
                below = sorted((item for item in local if item[1]["recall"] < TARGET),
                               key=lambda item: TARGET - item[1]["recall"])
                above = sorted((item for item in local if item[1]["recall"] > HIGH),
                               key=lambda item: item[1]["recall"] - HIGH)
                for key, value in below[:1] + above[:1]:
                    itopk = key[2]
                    auto = automatic_iterations(itopk, width)
                    limit = study.legal_hash_iterations(method, itopk, width)
                    if value["recall"] > HIGH:
                        sequence = range(1, auto) if auto <= 64 else [1, 2, 4, 8, 16, 32, auto // 2, auto - 1]
                    else:
                        sequence = []
                        step = auto
                        while step < limit:
                            step = min(limit, step * 2)
                            sequence.append(step)
                    for maximum in sequence:
                        if maximum != auto:
                            candidates.add((name, method, itopk, width, maximum))
    return missing(candidates, measured)


def refine_iterations(names: list[str], measured: dict) -> set:
    candidates = set()
    for name in names:
        for method in METHODS:
            # One measured in-window point is enough to enter strict three-repeat
            # confirmation. The confirmation stage retries other measured points
            # if the first one misses; exhaustive iteration gaps are unnecessary
            # for pairs that already have a viable calibration point.
            in_window = sum(
                key[:2] == (name, method)
                and within(value["recall"])
                and all(within(recall) for recall in value["recalls"])
                for key, value in measured.items()
            )
            if in_window >= MIN_CALIBRATION_MATCHES:
                continue
            for width in WIDTHS:
                by_l: dict[int, list[tuple[int, float]]] = defaultdict(list)
                for key, value in measured.items():
                    if key[:2] == (name, method) and key[3] == width:
                        by_l[key[2]].append((key[4] or automatic_iterations(key[2], width), value["recall"]))
                for itopk, series in by_l.items():
                    # Only iteration series actually seeded by this tuner are refined.
                    if len({item[0] for item in series}) < 2:
                        continue
                    series = sorted(set(series))
                    for (left_i, left_r), (right_i, right_r) in zip(series, series[1:]):
                        if right_i - left_i <= 1:
                            continue
                        crossed = (left_r - TARGET) * (right_r - TARGET) <= 0 or (
                            (left_r - HIGH) * (right_r - HIGH) <= 0
                        )
                        if crossed:
                            if right_i - left_i <= 64:
                                choices = range(left_i + 1, right_i)
                            else:
                                choices = [(left_i + right_i) // 2]
                            for maximum in choices:
                                if maximum != automatic_iterations(itopk, width):
                                    candidates.add((name, method, itopk, width, maximum))
    return missing(candidates, measured)


def finalists(names: list[str], measured: dict) -> set:
    result = set()
    for name in names:
        for method in METHODS:
            local = [
                (key, value) for key, value in measured.items()
                if key[:2] == (name, method)
                and within(value["recall"])
                and all(within(recall) for recall in value["recalls"])
            ]
            local.sort(key=lambda item: item[1]["qps"], reverse=True)
            result.update(key for key, _ in local[:FINALISTS])
    return result


def rescue_l_candidates(measured: dict) -> set:
    """Expand requested-L neighborhoods only for the known overshoot pairs."""
    candidates = set()
    for name, method in sorted(RESCUE_PAIRS):
        if any(key[:2] == (name, method) and within(value["recall"])
               for key, value in measured.items()):
            continue
        seen = {
            fingerprint(key[2], key[3], key[4])
            for key in measured
            if key[:2] == (name, method)
        }
        for width in WIDTHS:
            local = [
                (key, value) for key, value in measured.items()
                if key[:2] == (name, method) and key[3] == width
            ]
            near = [
                (key, value) for key, value in local
                if TARGET - 0.01 <= value["recall"] <= HIGH + 0.01
            ]
            centers = {key[2] for key, _ in near}
            for center in centers:
                for itopk in range(
                    max(10, center - RESCUE_L_RADIUS),
                    min(512, center + RESCUE_L_RADIUS) + 1,
                ):
                    identity = fingerprint(itopk, width, 0)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    candidates.add((name, method, itopk, width, 0))
    return missing(candidates, measured)


def rescue_iteration_candidates(measured: dict) -> set:
    """Exhaust every legal integer cap around each target recall crossing."""
    candidates = set()
    for name, method in sorted(RESCUE_PAIRS):
        if any(key[:2] == (name, method) and within(value["recall"])
               for key, value in measured.items()):
            continue
        seen = {
            fingerprint(key[2], key[3], key[4])
            for key in measured
            if key[:2] == (name, method)
        }
        by_cell: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)
        for key, value in measured.items():
            if key[:2] != (name, method):
                continue
            effective = key[4] or automatic_iterations(key[2], key[3])
            by_cell[key[2], key[3]].append((effective, value["recall"]))
        for (itopk, width), series in by_cell.items():
            near = [
                item for item in series
                if TARGET - 0.01 <= item[1] <= HIGH + 0.01
            ]
            if not near:
                continue
            below = [item[0] for item in series if item[1] < TARGET]
            above = [item[0] for item in series if item[1] > HIGH]
            if not below or not above:
                continue
            lower = max(1, max(below) - 4)
            upper = min(study.legal_hash_iterations(method, itopk, width), min(above) + 4)
            automatic = automatic_iterations(itopk, width)
            for maximum in range(lower, upper + 1):
                if maximum == automatic:
                    continue
                identity = fingerprint(itopk, width, maximum)
                if identity in seen or not legal(method, itopk, width, maximum):
                    continue
                seen.add(identity)
                candidates.add((name, method, itopk, width, maximum))
    return missing(candidates, measured)


def rescue_finalists(measured: dict) -> set:
    """Select fresh-confirmation candidates only for the rescue pairs."""
    result = set()
    for name, method in sorted(RESCUE_PAIRS):
        local = [
            (key, value) for key, value in measured.items()
            if key[:2] == (name, method)
            and within(value["recall"])
            and all(within(recall) for recall in value["recalls"])
        ]
        local.sort(key=lambda item: item[1]["qps"], reverse=True)
        result.update(key for key, _ in local[:FINALISTS])
    return result


def finalist_retries(names: list[str], measured: dict, confirmed: dict,
                     allowed_pairs: set[tuple[str, str]] | None = None) -> set:
    """Try remaining calibrated settings when the first finalist batch misses."""
    result = set()
    for name in names:
        for method in METHODS:
            if allowed_pairs is not None and (name, method) not in allowed_pairs:
                continue
            if any(key[:2] == (name, method) and in_band(value["recalls"])
                   for key, value in confirmed.items()):
                continue
            remaining = [
                (key, value) for key, value in measured.items()
                if key[:2] == (name, method)
                and key not in confirmed
                and within(value["recall"])
                and all(within(recall) for recall in value["recalls"])
            ]
            remaining.sort(key=lambda item: item[1]["qps"], reverse=True)
            result.update(key for key, _ in remaining[:FINALISTS])
    return result


def write_analysis(root: Path, reference: Path, names: list[str],
                   calibration_root: Path | None = None,
                   prior_finalists_root: Path | None = None) -> None:
    measured = observations(root, reference, calibration_root)
    final = new_rows(calibration_root or root, finalists=True)
    if prior_finalists_root is not None:
        final |= new_rows(prior_finalists_root, finalists=True)
    if calibration_root is not None:
        final |= new_rows(root, finalists=True)
    for stage in sorted((root / "state").glob("*.json")):
        if stage.stem in {"contract", "gpu_preflight"}:
            continue
        planned_rows = {tuple(row) for row in json.loads(stage.read_text())}
        is_finalist_stage = (
            stage.stem.startswith("finalists")
            or stage.stem.startswith("rescue_finalists")
        )
        observed = final if is_finalist_stage else measured
        absent = planned_rows - observed.keys()
        if absent:
            raise ValueError(f"incomplete {stage.stem}: {len(absent)} missing points")
    rows = []
    for name in names:
        for method in METHODS:
            local = [(key, value) for key, value in final.items() if key[:2] == (name, method)]
            good = [(key, value) for key, value in local if in_band(value["recalls"])]
            selected = max(good, key=lambda item: item[1]["qps"]) if good else None
            all_local = [(key, value) for key, value in (measured | final).items()
                         if key[:2] == (name, method)]
            below = max((v["recall"] for _, v in all_local if v["recall"] < TARGET), default=None)
            above = min((v["recall"] for _, v in all_local if v["recall"] > HIGH), default=None)
            rows.append({
                "cohort": name, "method": method,
                "status": "matched" if selected else "unmatched",
                "recall_min": min(selected[1]["recalls"]) if selected else "",
                "recall_median": selected[1]["recall"] if selected else "",
                "recall_max": max(selected[1]["recalls"]) if selected else "",
                "qps_median": selected[1]["qps"] if selected else "",
                "itopk": selected[0][2] if selected else "",
                "search_width": selected[0][3] if selected else "",
                "max_iterations": selected[0][4] if selected else "",
                "nearest_below": below if below is not None else "",
                "nearest_above": above if above is not None else "",
                "calibration_points": len(all_local),
                "finalist_points": len(local),
            })
    output = root / "analysis/matched_selected.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_pair = {(row["cohort"], row["method"]): row for row in rows}
    comparisons = []
    for name in names:
        for left, right in ((METHODS[0], METHODS[1]), (METHODS[0], METHODS[2]),
                            (METHODS[1], METHODS[2])):
            first, second = by_pair[name, left], by_pair[name, right]
            matched = first["status"] == second["status"] == "matched"
            comparisons.append({
                "cohort": name, "left_method": left, "right_method": right,
                "status": "matched" if matched else "unmatched",
                "left_recall": first["recall_median"] if matched else "",
                "right_recall": second["recall_median"] if matched else "",
                "left_qps": first["qps_median"] if matched else "",
                "right_qps": second["qps_median"] if matched else "",
                "right_over_left": (
                    float(second["qps_median"]) / float(first["qps_median"])
                    if matched else ""
                ),
            })
    with (root / "analysis/matched_comparisons.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    calibration = [
        {"cohort": key[0], "method": key[1], "itopk": key[2], "search_width": key[3],
         "max_iterations": key[4], "recall": value["recall"], "qps": value["qps"],
         "source": value["source"]}
        for key, value in sorted(measured.items())
    ]
    with (root / "analysis/calibration_points.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(calibration[0]))
        writer.writeheader()
        writer.writerows(calibration)
    print(json.dumps({"pairs": len(rows), "matched": sum(row["status"] == "matched" for row in rows),
                      "unmatched": sum(row["status"] == "unmatched" for row in rows)}, indent=2), flush=True)


def bundle(root: Path, archive: Path) -> None:
    if archive.exists():
        raise FileExistsError(archive)
    if not (root / "analysis/matched_selected.csv").is_file():
        raise ValueError("analyze before bundling")
    temporary = archive.with_name(f".{archive.name}.tmp.{os.getpid()}")
    with tarfile.open(temporary, "w:gz") as stream:
        for name in ("state", "data", "graph/configs", "graph/raw", "analysis"):
            stream.add(root / name, arcname=f"tight_match/{name}")
    os.replace(temporary, archive)
    print(json.dumps({"archive": str(archive), "sha256": digest(archive)}, indent=2), flush=True)


def main() -> None:
    global HIGH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("all", "calibrate", "rescue", "finalists", "analyze", "bundle"),
    )
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/data/retrieve_data"))
    parser.add_argument("--binary", type=Path, default=study.REPO / "cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH")
    parser.add_argument("--library", type=Path, default=study.REPO / "cpp/build/libcuvs.so")
    parser.add_argument("--bundle-path", type=Path)
    parser.add_argument("--upper", type=float, default=HIGH)
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--prior-finalists-root", type=Path)
    args = parser.parse_args()
    if not TARGET < args.upper <= 1:
        parser.error("--upper must be greater than 0.950 and at most 1")
    HIGH = args.upper
    root, reference = args.root.resolve(), args.reference_root.resolve()
    binary, library = args.binary.resolve(), args.library.resolve()
    calibration_root = args.calibration_root.resolve() if args.calibration_root else None
    prior_finalists_root = (
        args.prior_finalists_root.resolve() if args.prior_finalists_root else None
    )
    initial_contract(root, reference, binary, library, calibration_root, prior_finalists_root)
    names = cohorts(reference)
    if args.stage == "rescue":
        study.preflight_graph(root, args.data_root.resolve(), binary, library)
        measured = observations(root, reference, calibration_root)
        run_stage(
            root, args.data_root, binary, library, "rescue_l",
            rescue_l_candidates(measured),
        )
        for round_number in range(8):
            measured = observations(root, reference, calibration_root)
            points = rescue_iteration_candidates(measured)
            if not points:
                break
            run_stage(
                root, args.data_root, binary, library,
                f"rescue_iterations_{round_number:02d}", points,
            )
        measured = observations(root, reference, calibration_root)
        run_stage(
            root, args.data_root, binary, library, "rescue_finalists",
            rescue_finalists(measured), 3,
        )
        round_number = 0
        while True:
            measured = observations(root, reference, calibration_root)
            confirmed = new_rows(root, finalists=True)
            remaining = finalist_retries(
                names, measured, confirmed, set(RESCUE_PAIRS)
            )
            if not remaining:
                break
            run_stage(
                root, args.data_root, binary, library,
                f"rescue_finalists_retry_{round_number:03d}", remaining, 3,
            )
            round_number += 1
        write_analysis(root, reference, names, calibration_root, prior_finalists_root)
        print(root)
        return
    if args.stage in ("all", "calibrate"):
        study.preflight_graph(root, args.data_root.resolve(), binary, library)
        measured = observations(root, reference, calibration_root)
        run_stage(root, args.data_root, binary, library, "anchors", anchors(names, measured))
        measured = observations(root, reference, calibration_root)
        run_stage(root, args.data_root, binary, library, "refine_l", refine_l(names, measured))
        measured = observations(root, reference, calibration_root)
        run_stage(root, args.data_root, binary, library, "iter_seeds", iteration_seeds(names, measured))
        for round_number in range(12):
            measured = observations(root, reference, calibration_root)
            points = refine_iterations(names, measured)
            if not points:
                break
            run_stage(root, args.data_root, binary, library, f"iter_refine_{round_number:02d}", points)
    if args.stage in ("all", "finalists"):
        measured = observations(root, reference, calibration_root)
        run_stage(root, args.data_root, binary, library, "finalists", finalists(names, measured), 3)
        round_number = 0
        while True:
            confirmed = new_rows(root, finalists=True)
            remaining = finalist_retries(names, measured, confirmed)
            if not remaining:
                break
            run_stage(root, args.data_root, binary, library,
                      f"finalists_retry_{round_number:03d}", remaining, 3)
            round_number += 1
    if args.stage in ("all", "analyze"):
        write_analysis(root, reference, names, calibration_root, prior_finalists_root)
    if args.stage in ("all", "bundle"):
        bundle(root, (args.bundle_path or root.with_name(root.name + "_results.tar.gz")).resolve())
    print(root)


if __name__ == "__main__":
    main()
