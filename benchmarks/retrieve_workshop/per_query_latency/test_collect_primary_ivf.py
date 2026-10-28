import tempfile
import unittest
from pathlib import Path

import collect_primary_ivf as collector
import primary_ivf as pipeline


class PublicationGateTest(unittest.TestCase):
    def test_running_experiment_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "run"
            out = Path(d) / "published"
            pipeline.save(root / "manifest.json", {})
            pipeline.save(root / "progress.json", dict(status="running"))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                collector.collect(root, out)
            self.assertFalse(out.exists())

    def test_recall_drift_cannot_be_published_as_success(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "run"
            out = Path(d) / "published"
            pipeline.save(root / "manifest.json", {})
            pipeline.save(root / "progress.json", dict(status="complete"))
            pipeline.save(
                root / "analysis/latency_summary.json",
                dict(
                    complete=True,
                    rows=[dict(recall_preserved=False)] * 32,
                    recall_drift_flags=["case"],
                ),
            )
            with self.assertRaisesRegex(ValueError, "drift"):
                collector.collect(root, out)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
