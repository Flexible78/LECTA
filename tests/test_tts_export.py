from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pydub import AudioSegment

import gr_tabs.tts_tab as tts_tab
from libs.audio_tools import verify_audio_file
from _fixtures import build_speech_with_pauses


class ExportPathTests(unittest.TestCase):
    def test_trim_and_normalize_handles_pydub_api(self):
        """Regression: pydub exposes ``max_dBFS``, not ``max_dbfs``."""
        audio = AudioSegment.from_file(
            build_speech_with_pauses(Path(tempfile.mkdtemp()) / "in.mp3", pauses=(0.4,))
        )
        out = tts_tab._trim_and_normalize_audio(audio)
        self.assertGreater(len(out), 0)
        self.assertLessEqual(abs(out.max_dBFS) >= 0, True)
        self.assertLessEqual(out.max_dBFS, -0.5)

    def test_export_is_verified_and_atomic(self):
        tmp = Path(tempfile.mkdtemp())
        audio = AudioSegment.from_file(
            build_speech_with_pauses(tmp / "in.mp3", pauses=(0.0,), clip_seconds=1.0)
        )
        target = tmp / "out.mp3"
        tts_tab._export_mp3_atomic(audio, target, bitrate=192, tags={"title": "t"})
        self.assertTrue(target.is_file())
        self.assertTrue(verify_audio_file(str(target))["ok"])
        leftovers = list(tmp.glob(".*tmp.mp3"))
        self.assertEqual(leftovers, [])

    def test_empty_audio_is_rejected(self):
        with self.assertRaises(RuntimeError):
            tts_tab._trim_and_normalize_audio(AudioSegment.empty())


if __name__ == "__main__":
    unittest.main()