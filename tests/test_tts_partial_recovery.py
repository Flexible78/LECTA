from __future__ import annotations

from concurrent.futures import Future
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from pydub import AudioSegment
import gr_tabs.tts_tab as tt
from libs.audio_tools import verify_audio_file


class PartialRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="lecta_partial_test_")
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name)
        self.base = self.data / "Fixture"
        (self.base / "xml").mkdir(parents=True)
        (self.base / "mp3").mkdir()
        (self.base / "xml" / "1.xml").write_text(
            '<root autor="Test" album="Test">'
            '<p>Первый test.</p><p>Второй test.</p><p>Третий test.</p></root>',
            encoding="utf-8")
        self.full = self.base / "mp3" / "Fixture.mp3"
        self.partial = self.base / "mp3" / "Fixture_PARTIAL.mp3"
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in (("data_path", self.data), ("LAST_BATCH_FAILURES", []),
                            ("stop_text_to_sp", False), ("_synthesis_completed", False)):
            self.stack.enter_context(mock.patch.object(tt, name, value))
        self.stack.enter_context(mock.patch.object(tt.AppConfig, "save_user_settings"))
        self.stack.enter_context(mock.patch.object(tt.AppConfig, "load_user_settings",
                                                  return_value=tt.AppConfig()))
        self.stack.enter_context(mock.patch.object(tt, "get_device", return_value="cpu"))
        self.stack.enter_context(mock.patch.object(tt, "get_gpu_temp", return_value=None))
        self.stack.enter_context(mock.patch.object(tt, "_poll_gpu_temp_thermal", return_value=(False, "")))
        self.stack.enter_context(mock.patch.object(tt, "_safe_audio", side_effect=lambda p: str(p)))
        self.stack.enter_context(mock.patch.object(tt.gr, "Info"))
        self.stack.enter_context(mock.patch.object(tt.gr, "Warning"))
        self.executor = self.stack.enter_context(mock.patch.object(tt, "_tts_executor"))

    def tone(self, freq=220):
        t = np.arange(12000, dtype=np.float32) / 24000
        return (24000, (0.25 * np.sin(2 * np.pi * freq * t)).astype(np.float32), 0.01)

    def run_pipeline(self, fail_at=None, repl=False, stop_when_ready=False):
        count = 0
        def submit(*args, **kwargs):
            nonlocal count
            count += 1
            f = Future()
            if count == fail_at:
                f.set_exception(RuntimeError("fixture chunk failure"))
            else:
                f.set_result(self.tone(220 * count))
            return f
        self.executor.submit.side_effect = submit
        rows = []
        for row in tt.tts("Fixture", repl, "test", 1, "", 128, 10, False, False, False):
            rows.append(row)
            if stop_when_ready and "Synthesizing chunk 0" in row[1]:
                tt.stop_text_to_sp = True
        return rows

    def test_third_chunk_failure_exports_valid_prefix_and_cause(self):
        rows = self.run_pipeline(fail_at=3)
        self.assertTrue(self.partial.is_file())
        self.assertFalse(self.full.exists())
        check = verify_audio_file(str(self.partial))
        self.assertTrue(check["ok"], check)
        self.assertGreater(check["info"]["duration"], 0.85)
        self.assertIn("chunk 3 of 3", rows[-1][1])
        self.assertIn("fixture chunk failure", rows[-1][1])
        self.assertEqual(rows[-1][3], str(self.partial))
        self.assertFalse(tt._synthesis_completed)
        self.assertTrue(tt.LAST_BATCH_FAILURES)

    def test_first_chunk_failure_does_not_export_silence(self):
        rows = self.run_pipeline(fail_at=1)
        self.assertFalse(self.partial.exists())
        self.assertFalse(self.full.exists())
        self.assertIn("fixture chunk failure", rows[-1][1])
        self.assertFalse(tt._synthesis_completed)

    def test_stop_recovers_ready_prefix_and_player_points_to_partial(self):
        rows = self.run_pipeline(stop_when_ready=True)
        self.assertTrue(self.partial.is_file())
        self.assertTrue(verify_audio_file(str(self.partial))["ok"])
        self.assertFalse(self.full.exists())
        self.assertIn("Stopped", rows[-1][1])
        self.assertEqual(rows[-1][3], str(self.partial))
        self.assertFalse(tt._synthesis_completed)

    def test_partial_export_failure_keeps_original_synthesis_cause(self):
        with mock.patch.object(tt, "_save_partial_audio", side_effect=OSError("disk fixture")):
            rows = self.run_pipeline(fail_at=3)
        self.assertIn("fixture chunk failure", rows[-1][1])
        self.assertIn("disk fixture", rows[-1][1])
        self.assertFalse(self.full.exists())
        self.assertFalse(tt._synthesis_completed)

    def test_conversion_failure_does_not_count_failed_audio_as_completed(self):
        original_tone = self.tone
        def bad_third(freq=220):
            if freq == 660:
                return (24000, np.zeros(12000, dtype=np.float32), 0.01)
            return original_tone(freq)
        with mock.patch.object(self, "tone", side_effect=bad_third):
            rows = self.run_pipeline()
        self.assertTrue(self.partial.is_file())
        self.assertIn("chunk 3 of 3", rows[-1][1])
        self.assertIn("silent audio", rows[-1][1])
        self.assertFalse(tt._synthesis_completed)

    def test_success_still_exports_full(self):
        rows = self.run_pipeline()
        self.assertTrue(self.full.is_file())
        self.assertTrue(verify_audio_file(str(self.full))["ok"])
        self.assertFalse(self.partial.exists())
        self.assertTrue(tt._synthesis_completed)
        self.assertIn("ALL FILES SUCCESSFULLY", rows[-1][1])

    def test_failure_preserves_existing_full_and_partial(self):
        audio = AudioSegment(self.tone()[1].__mul__(32767).astype(np.int16).tobytes(),
                             frame_rate=24000, sample_width=2, channels=1)
        tt._export_mp3_atomic(audio * 6, self.partial, bitrate=128)
        tt._export_mp3_atomic(audio, self.full, bitrate=128)
        old_full, old_partial = self.full.read_bytes(), self.partial.read_bytes()
        rows = self.run_pipeline(fail_at=3, repl=True)
        self.assertEqual(self.full.read_bytes(), old_full)
        self.assertEqual(self.partial.read_bytes(), old_partial)
        self.assertIn("fixture chunk failure", rows[-1][1])

    def test_failed_atomic_verification_preserves_target(self):
        self.full.write_bytes(b"previous output")
        audio = AudioSegment.silent(duration=500)
        with mock.patch.object(tt, "verify_audio_file", return_value={"ok": False, "errors": ["fixture"]}):
            with self.assertRaises(RuntimeError):
                tt._export_mp3_atomic(audio, self.full, bitrate=128)
        self.assertEqual(self.full.read_bytes(), b"previous output")
        self.assertEqual(list(self.full.parent.glob(".*tmp.mp3")), [])


if __name__ == "__main__":
    unittest.main()
