"""Voice list must never disappear.

`speakers_list()` returns [] until a TTS model is loaded, and every handler
used to index `sp_list[0]` unconditionally. The IndexError left the voice
dropdown blank on a fresh start, which reads as "voice selection is gone".
"""
from __future__ import annotations

import unittest
from unittest import mock

from config import AppConfig
from gr_tabs import tts_tab as tab


class VoiceListTests(unittest.TestCase):
    def test_empty_model_list_does_not_raise(self):
        with mock.patch.object(tab.synth, "speakers_list", return_value=[]):
            upd = tab.change_tts_model(None)
        self.assertIsInstance(upd, dict)
        self.assertEqual(list(upd.get("choices") or []), [])

    def test_empty_list_keeps_the_saved_voice_as_a_choice(self):
        cfg = mock.Mock(spk_sel="Misha24-10")
        with mock.patch.object(tab.synth, "speakers_list", return_value=[]), \
             mock.patch.object(AppConfig, "load_user_settings", return_value=cfg):
            upd = tab.change_tts_model(None)
        self.assertEqual(list(upd.get("choices") or []), ["Misha24-10"])
        self.assertEqual(upd.get("value"), "Misha24-10")

    def test_loaded_model_fills_the_list(self):
        cfg = mock.Mock(spk_sel="")
        with mock.patch.object(tab.synth, "speakers_list", return_value=["ann", "bob"]), \
             mock.patch.object(AppConfig, "load_user_settings", return_value=cfg):
            upd = tab.change_tts_model(6)
        self.assertEqual(list(upd.get("choices") or []), ["ann", "bob"])
        self.assertIn(upd.get("value"), ("ann", "bob"))

    def test_saved_voice_wins_when_it_exists(self):
        cfg = mock.Mock(spk_sel="bob")
        with mock.patch.object(tab.synth, "speakers_list", return_value=["ann", "bob"]), \
             mock.patch.object(AppConfig, "load_user_settings", return_value=cfg):
            upd = tab.change_tts_model(6)
        self.assertEqual(upd.get("value"), "bob")

    def test_broken_speaker_query_still_returns_an_update(self):
        with mock.patch.object(tab.synth, "speakers_list", side_effect=RuntimeError("boom")):
            upd = tab.change_tts_model(6)
        self.assertIsInstance(upd, dict)


if __name__ == "__main__":
    unittest.main()
