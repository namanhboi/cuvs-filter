#!/usr/bin/env python3
"""Wait for a detached latency run, validate/export it, and optionally publish its snapshot."""

from __future__ import annotations

import argparse
import fcntl
import shutil
import subprocess
import time
from pathlib import Path

import collect_primary_ivf as collector
import primary_ivf as pipeline


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--publish", action="store_true")
    args = p.parse_args()
    root, repo = args.root.resolve(), args.repo.resolve()
    state = root / "finalize.state.json"
    with (root / "finalize.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pipeline.save(
            state,
            dict(status="waiting_for_measurements", started_at=pipeline.now()),
        )
        try:
            while True:
                supervisor = pipeline.read(root / "supervisor.state.json")
                if supervisor["status"] == "failed":
                    raise ValueError(
                        "Measurement worker failed; inspect failure.json and run.log"
                    )
                if supervisor["status"] == "passed":
                    break
                time.sleep(30)
            manifest = pipeline.read(root / "manifest.json")
            if (
                pipeline.sha(Path(pipeline.__file__))
                != manifest["runner_sha256"]
            ):
                pipeline.save(
                    state,
                    dict(
                        status="measurements_complete_export_pending",
                        reason="Runner source changed; use recorded source revision to export",
                        at=pipeline.now(),
                    ),
                )
                return
            pipeline.save(
                state,
                dict(status="validating_and_exporting", at=pipeline.now()),
            )
            bundle = collector.collect(root, root / "review_bundle")
            if not args.publish:
                pipeline.save(
                    state,
                    dict(
                        status="export_complete",
                        bundle=str(bundle),
                        at=pipeline.now(),
                    ),
                )
                return
            if (
                git(repo, "branch", "--show-current") != "main"
                or git(repo, "rev-parse", "HEAD") != args.expected_head
                or git(repo, "status", "--porcelain", "--untracked-files=no")
            ):
                pipeline.save(
                    state,
                    dict(
                        status="export_complete_commit_pending",
                        bundle=str(bundle),
                        reason="Repository advanced or has concurrent changes; original work is preserved",
                        at=pipeline.now(),
                    ),
                )
                return
            if (
                git(repo, "remote", "get-url", "origin")
                != "git@github.com:namanhboi/cuvs-filter.git"
            ):
                raise ValueError("Unexpected publication remote")
            relative = (
                Path("benchmarks/retrieve_workshop/per_query_latency/results")
                / root.name
            )
            output = repo / relative
            if output.exists():
                raise FileExistsError(
                    "Review snapshot already exists; refusing replacement"
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(bundle, output)
            handoff = Path(
                "benchmarks/retrieve_workshop/per_query_latency/PRIMARY_IVF_HANDOFF.md"
            )
            text = (repo / handoff).read_text()
            text += (
                f"\n## Completion — {pipeline.now()}\n\n"
                "All 32 operating points, all 10,000 queries, and all three passes completed.\n"
                "The paired gates and final recall-drift checks passed; the collector verified\n"
                "all configuration/output signatures before publication.\n\n"
                f"Results: [summary]({Path('results') / root.name / 'analysis/REPORT.md'}).\n"
                f"Full evidence and raw per-query traces remain at `{root}`.\n"
            )
            (repo / handoff).write_text(text)
            subprocess.run(
                ["git", "add", str(relative), str(handoff)],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                ["git", "diff", "--cached", "--check"], cwd=repo, check=True
            )
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Nam Anh Dang",
                    "-c",
                    "user.email=dangnamanh05@gmail.com",
                    "commit",
                    "--only",
                    "-m",
                    "Archive complete frozen batch-one CAGRA and IVF-Flat latency results",
                    "--",
                    str(relative),
                    str(handoff),
                ],
                cwd=repo,
                check=True,
            )
            commit = git(repo, "rev-parse", "HEAD")
            subprocess.run(
                ["git", "push", "origin", "main"], cwd=repo, check=True
            )
            pipeline.save(
                state,
                dict(
                    status="complete",
                    bundle=str(bundle),
                    commit=commit,
                    at=pipeline.now(),
                ),
            )
        except Exception as error:
            pipeline.save(
                state,
                dict(status="failed", error=repr(error), at=pipeline.now()),
            )
            raise


if __name__ == "__main__":
    main()
