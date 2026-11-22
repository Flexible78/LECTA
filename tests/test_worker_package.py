from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import worker
from libs.audio_tools import probe_audio, verify_audio_file
from _fixtures import build_speech_with_pauses


class WorkerPackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="lecta_pkg_test_")
        self.root = Path(self.tmp.name)
        self.project = "Книга-測試"
        self.base = self.root / "data" / self.project
        self.mp3_dir = self.base / "mp3"
        self.xml_dir = self.base / "xml"
        self.mp3_dir.mkdir(parents=True)
        self.xml_dir.mkdir(parents=True)
        (self.base / f"{self.project}.fb2").write_text(
            "<?xml version='1.0'?><FictionBook><body><section><p>Test book text.</p></section></body></FictionBook>",
            encoding="utf-8",
        )
        (self.xml_dir / "1.xml").write_text(
            "<speak><p>Test book text.</p></speak>", encoding="utf-8")
        self.mp3 = build_speech_with_pauses(self.mp3_dir / "1.mp3", pauses=(0.0,), clip_seconds=1.2)
        self.mp3_2 = build_speech_with_pauses(self.mp3_dir / "2.mp3", pauses=(0.0,), clip_seconds=1.0)
        self.original_data_path = worker.data_path
        worker.data_path = self.root / "data"
        worker.STATUS_FILE = self.root / "status.json"

    def tearDown(self):
        worker.data_path = self.original_data_path
        self.tmp.cleanup()

    def test_atomic_three_file_package_and_numeric_order(self):
        out = worker.build_package(self.project)
        self.assertIsNotNone(out)
        files = list(Path(out).iterdir())
        self.assertEqual(len(files), 3, [p.name for p in files])
        audio = Path(out) / f"{self.project}.mp3"
        self.assertTrue(verify_audio_file(str(audio))["ok"])
        expected = probe_audio(str(self.mp3)).duration + probe_audio(str(self.mp3_2)).duration
        self.assertAlmostEqual(probe_audio(str(audio)).duration, expected, delta=1.5)
        self.assertTrue((Path(out) / f"{self.project}.txt").read_text(encoding="utf-8").strip())

    def test_refuses_failed_tts_package(self):
        with self.assertRaisesRegex(RuntimeError, "TTS failed"):
            worker.build_package(self.project, failures=[f"{self.project}::1::synthesis failed"])
        self.assertFalse(list(self.root.glob("**/*аудиокнига")))

    def test_failed_mp3_combine_does_not_publish(self):
        broken = self.mp3_dir / "3.mp3"
        broken.write_bytes(b"invalid")
        # A corrupt chapter causes combined-output validation to fail and staging is removed.
        with self.assertRaises(RuntimeError):
            worker.build_package(self.project)
        self.assertFalse(list(self.root.glob("**/.lecta_pkg_*")))


if __name__ == "__main__":
    unittest.main()
