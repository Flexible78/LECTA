from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _fixtures import build_speech_with_pauses, tone_clip
from libs.audio_tools import (
    analyze_audio,
    clean_audio_file,
    default_cleaned_path,
    probe_audio,
    verify_audio_file,
)


class AudioToolsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="lecta_audio_test_")
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_silence_analysis_and_safe_clean(self):
        source = build_speech_with_pauses(self.root / "ספר-кириллица.mp3")
        before_bytes = source.read_bytes()
        analysis = analyze_audio(str(source), threshold_db=-38, min_silence_sec=0.35)
        self.assertTrue(analysis.info.ok, analysis.info.error)
        self.assertGreaterEqual(len(analysis.silences), 2)
        self.assertGreaterEqual(analysis.total_silence, 4.0)

        result = clean_audio_file(source, preset="compact", quality_kbps=128)
        self.assertEqual(result.status, "done", result.message)
        self.assertTrue(Path(result.output).is_file())
        self.assertEqual(source.read_bytes(), before_bytes, "source must remain untouched")
        self.assertGreater(result.removed, 2.0)
        check = verify_audio_file(result.output)
        self.assertTrue(check["ok"], check["errors"])
        self.assertLess(probe_audio(result.output).duration, analysis.info.duration)

    def test_collision_safe_name_and_repeat_clean(self):
        source = build_speech_with_pauses(self.root / "book.mp3")
        first = clean_audio_file(source)
        second = clean_audio_file(source)
        self.assertEqual(first.status, "done")
        self.assertEqual(second.status, "done")
        self.assertNotEqual(first.output, second.output)
        self.assertEqual(Path(first.output).name, "book_compact.mp3")
        self.assertEqual(default_cleaned_path(source).name, "book_compact_3.mp3")

    def test_invalid_file_rejected_without_output(self):
        bad = self.root / "empty.mp3"
        bad.write_bytes(b"not an mp3")
        result = clean_audio_file(bad)
        self.assertEqual(result.status, "failed")
        self.assertFalse(list(self.root.glob("empty_compact*.mp3")))

    def test_no_excessive_pause_is_non_destructive_skip(self):
        source = tone_clip(self.root / "short.mp3", seconds=1.0)
        result = clean_audio_file(source)
        self.assertEqual(result.status, "skipped")
        self.assertFalse(Path(str(source).replace(".mp3", "_compact.mp3")).exists())

    def test_generation_of_preview_pair(self):
        from libs.audio_tools import make_preview
        source = build_speech_with_pauses(self.root / "preview.mp3")
        payload = make_preview(source)
        self.assertTrue(payload["ok"], payload["message"])
        self.assertTrue(Path(payload["before"]).is_file())
        self.assertTrue(Path(payload["after"]).is_file())


if __name__ == "__main__":
    unittest.main()
