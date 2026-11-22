from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gr_tabs.audio_tools_tab as tab
from _fixtures import build_speech_with_pauses


class CleanerTabTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="lecta_ui_test_")
        self.root = Path(self.tmp.name)
        self.a = build_speech_with_pauses(self.root / "a.mp3", pauses=(2.5, 0.0))
        self.b = build_speech_with_pauses(self.root / "b.mp3", pauses=(3.0, 0.0))
        self.data_patch = mock.patch.object(tab, "data_path", self.root / "data")
        self.data_patch.start()
        self.proj_patch = mock.patch.object(tab, "project_mp3_files", lambda p: [self.a])
        self.proj_patch.start()

    def tearDown(self):
        self.proj_patch.stop()
        self.data_patch.stop()
        self.tmp.cleanup()

    def test_resolve_sources_dedupes_and_accepts_folder(self):
        folder_list = f"{self.root}\n{self.a}\n\"{self.a}\""
        sources = tab.resolve_sources([str(self.a)], [], folder_list)
        self.assertEqual(len([s for s in sources if s.name == "a.mp3"]), 1)
        self.assertTrue(any(s.name == "b.mp3" for s in sources))

    def test_load_sources_returns_choices_and_selection(self):
        update_batch, update_picker, status = tab.load_sources([str(self.a)], [], "")
        self.assertIn("Loaded", status)
        self.assertEqual(update_picker["choices"], [(f"{self.a.name} — {self.a.parent}", str(self.a))])
        self.assertEqual(update_picker["value"], str(self.a))
        self.assertEqual(update_batch["value"], [str(self.a)])

    def test_batch_clean_creates_outputs_and_zip(self):
        outputs = list(tab.clean_selected(
            [str(self.a), str(self.b)], "compact", -38, 0.35, 110, 50, True, 128))
        final = outputs[-1]
        progress, report_text, table, archive, folder, folder_msg = final
        self.assertTrue(archive and Path(archive).is_file())
        self.assertTrue(Path(folder).is_dir())
        self.assertEqual(len(table), 2)
        self.assertTrue(all(row[1].startswith("✅") for row in table))
        self.assertIn("Files processed: 2", report_text)
        self.assertTrue(any(Path(folder).glob("*_compact.mp3")))

    def test_cancel_event_stops_batch(self):
        from libs.audio_tools import clean_audio_file as real_clean

        calls = []

        def clean_then_cancel(source, **kwargs):
            result = real_clean(source, **kwargs)
            calls.append(result)
            tab._CANCEL.set()  # simulate the Cancel button during the batch
            return result

        try:
            with mock.patch.object(tab, "clean_audio_file", clean_then_cancel):
                outputs = list(tab.clean_selected(
                    [str(self.a), str(self.b)], "compact", -38, 0.35, 110, 50, True, 128))
            self.assertEqual(len(calls), 1, "second file must not be processed after cancel")
            self.assertIn("cancelled", outputs[-1][0].lower())
        finally:
            tab._CANCEL.clear()

    def test_open_output_folder_handles_missing(self):
        self.assertIn("No output folder", tab.open_output_folder(""))


if __name__ == "__main__":
    unittest.main()
