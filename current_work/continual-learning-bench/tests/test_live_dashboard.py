import tempfile
import unittest
from pathlib import Path

from src.live_dashboard import _root_relative_path, start_live_dashboard_server


class LiveDashboardTests(unittest.TestCase):
    def test_root_relative_path_accepts_relative_paths(self):
        root_dir = Path("/tmp/example-root")
        manifest_path = Path("results/poker/live/run-1/manifest.json")

        self.assertEqual(
            _root_relative_path(manifest_path, root_dir),
            "results/poker/live/run-1/manifest.json",
        )

    def test_start_live_dashboard_server_accepts_relative_manifest_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            viewer_path = root_dir / "viewers" / "single_task_viewer.html"
            viewer_path.parent.mkdir(parents=True, exist_ok=True)
            viewer_path.write_text("<html></html>", encoding="utf-8")
            manifest_rel = Path("results/poker/live/run-1/manifest.json")
            manifest_abs = root_dir / manifest_rel
            manifest_abs.parent.mkdir(parents=True, exist_ok=True)
            manifest_abs.write_text("{}", encoding="utf-8")

            server, thread, viewer_url = start_live_dashboard_server(
                root_dir=root_dir,
                manifest_path=manifest_rel,
            )
            try:
                self.assertIn(
                    "artifact=results/poker/live/run-1/manifest.json",
                    viewer_url,
                )
            finally:
                server.shutdown()
                thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
