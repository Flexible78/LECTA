from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gr_tabs.tts_tab as tts_tab


class FileListTests(unittest.TestCase):
    def test_corrupt_mp3_does_not_break_project_table(self):
        with tempfile.TemporaryDirectory(prefix="lecta_files_test_") as tmp:
            data = Path(tmp)
            mp3 = data / "Proj" / "mp3"
            mp3.mkdir(parents=True)
            (mp3 / "001.mp3").write_bytes(b"not an mp3")
            with mock.patch.object(tts_tab, "data_path", data):
                rows = tts_tab.get_files_list("Proj")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], "001.mp3")
            self.assertEqual(rows[0][3], "—")


if __name__ == "__main__":
    unittest.main()