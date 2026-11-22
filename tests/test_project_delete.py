from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gr_tabs.tts_tab as tts_tab


class ProjectDeleteTests(unittest.TestCase):
    """Deleting project folders from the TTS batch list."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="lecta_delete_test_")
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        for name in ("Alpha", "Beta", "Gamma"):
            (self.data / name).mkdir()
            (self.data / name / "mp3").mkdir()
            (self.data / name / "mp3" / "001.mp3").write_bytes(b"x")
        # Internal folders must survive every deletion path
        for name in ("_archive", "tts_cache", "_audio_tools", "__pycache__"):
            (self.data / name).mkdir()
        (self.data / ".hidden").mkdir()

        self.patches = [
            mock.patch.object(tts_tab, "data_path", self.data),
            mock.patch.object(tts_tab, "_bg_status_path", lambda: self.root / "no_status.json"),
            mock.patch.object(tts_tab.gr, "Info"),
            mock.patch.object(tts_tab.gr, "Warning"),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def test_list_and_sanitize_ignore_internal_dirs(self):
        self.assertEqual(tts_tab._list_project_dirs(), ["Alpha", "Beta", "Gamma"])
        names = tts_tab._real_project_names(
            ["Alpha", "_archive", "tts_cache", "../data", ".hidden", "", None, "Alpha"]
        )
        self.assertEqual(names, ["Alpha"])

    def test_selected_deletion_requires_two_clicks(self):
        out1 = tts_tab.delete_selected_projects(["Alpha"], "idle", "Alpha", [], {})
        self.assertEqual(out1[6], "confirm")
        self.assertTrue((self.data / "Alpha").is_dir(), "first click must not delete")
        self.assertIn("Confirm", out1[5]["value"])

        out2 = tts_tab.delete_selected_projects(
            ["Alpha"], out1[6], "Alpha", out1[2], out1[3]
        )
        self.assertFalse((self.data / "Alpha").exists())
        self.assertTrue((self.data / "Beta").is_dir())
        self.assertTrue((self.data / "_archive").is_dir())
        self.assertTrue((self.data / "tts_cache").is_dir())
        self.assertEqual(out2[6], "idle")
        self.assertEqual(out2[0]["choices"], ["Beta", "Gamma"])
        self.assertEqual(out2[1]["value"], "")  # current project was deleted
        self.assertEqual(out2[2], [])
        self.assertEqual(out2[3], {})

    def test_delete_all_keeps_current_project_choice_if_alive(self):
        out1 = tts_tab.delete_all_projects("idle", "Gamma", [], {})
        self.assertEqual(out1[6], "confirm")
        self.assertEqual(len(list(self.data.iterdir())), 8)

        out2 = tts_tab.delete_all_projects(out1[6], "Gamma", out1[2], out1[3])
        self.assertEqual(tts_tab._list_project_dirs(), [])
        self.assertEqual(out2[1]["choices"], [])
        self.assertEqual(out2[1]["value"], "")
        self.assertTrue((self.data / "_archive").is_dir())
        self.assertTrue((self.data / "tts_cache").is_dir())

    def test_busy_background_job_blocks_deletion(self):
        with mock.patch.object(tts_tab, "_bg_job_running", return_value="Alpha"):
            out = tts_tab.delete_selected_projects(["Alpha"], "confirm", "Alpha", [], {})
            self.assertEqual(out[6], "idle")
            self.assertTrue((self.data / "Alpha").is_dir())
            out_all = tts_tab.delete_all_projects("confirm", "Alpha", [], {})
            self.assertEqual(out_all[6], "idle")
            self.assertTrue((self.data / "Alpha").is_dir())

    def test_stale_status_file_is_ignored(self):
        status = self.root / "stale_status.json"
        status.write_text('{"state": "running", "project": "Alpha"}', encoding="utf-8")
        import os
        import time

        old = time.time() - 60 * 60
        os.utime(status, (old, old))
        with mock.patch.object(tts_tab, "_bg_status_path", lambda: status):
            self.assertEqual(tts_tab._bg_job_running(), "")


if __name__ == "__main__":
    unittest.main()