import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import finalize_primary_ivf as finalizer
import primary_ivf as pipeline


class FinalizationSafetyTest(unittest.TestCase):
    def test_concurrent_repository_work_preserves_results_without_committing(
        self,
    ):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "run"
            repo = Path(d) / "repo"
            repo.mkdir()
            pipeline.save(
                root / "supervisor.state.json", dict(status="passed")
            )
            pipeline.save(
                root / "manifest.json",
                dict(runner_sha256=pipeline.sha(pipeline.__file__)),
            )
            args = [
                "finalize",
                "--root",
                str(root),
                "--repo",
                str(repo),
                "--expected-head",
                "previous",
                "--publish",
            ]
            with (
                patch("sys.argv", args),
                patch.object(
                    finalizer.collector,
                    "collect",
                    return_value=root / "review_bundle",
                ),
                patch.object(finalizer, "git", return_value="concurrent-work"),
                patch.object(finalizer.subprocess, "run") as mutate,
            ):
                finalizer.main()
            self.assertEqual(
                pipeline.read(root / "finalize.state.json")["status"],
                "export_complete_commit_pending",
            )
            mutate.assert_not_called()

    def test_failed_worker_does_not_export_or_publish(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "run"
            pipeline.save(
                root / "supervisor.state.json", dict(status="failed")
            )
            args = [
                "finalize",
                "--root",
                str(root),
                "--repo",
                d,
                "--expected-head",
                "previous",
                "--publish",
            ]
            with (
                patch("sys.argv", args),
                patch.object(finalizer.collector, "collect") as export,
            ):
                with self.assertRaisesRegex(ValueError, "worker failed"):
                    finalizer.main()
            self.assertEqual(
                pipeline.read(root / "finalize.state.json")["status"], "failed"
            )
            export.assert_not_called()


if __name__ == "__main__":
    unittest.main()
