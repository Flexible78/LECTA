from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

import config
import libs.multilingual_router as router


class SettingsPersistenceTests(unittest.TestCase):
    def test_defaults_present_and_roundtrip(self):
        fields = config.AppConfig.__dataclass_fields__
        for name in ("generation_preset", "pause_intra_ms", "pause_sentence_ms",
                     "pause_paragraph_ms", "pause_language_ms", "pause_learning_ms",
                     "explicit_break_max_ms", "edge_trim_ms", "micro_fade_ms",
                     "lead_silence_ms", "cleaner_preset", "cleaner_quality_kbps",
                     "cleaner_trim_edges"):
            self.assertIn(name, fields)
        tmp = Path(self.id().replace(".", "_") + ".json")
        try:
            with mock.patch.object(config, "USER_SETTINGS_FILE", tmp):
                config.AppConfig.save_user_settings({"pause_language_ms": 210,
                                                     "cleaner_preset": "natural"})
                loaded = config.AppConfig.load_user_settings()
                self.assertEqual(loaded.pause_language_ms, 210)
                self.assertEqual(loaded.cleaner_preset, "natural")
                self.assertEqual(loaded.generation_preset, "compact")
        finally:
            tmp.unlink(missing_ok=True)

    def test_unknown_keys_ignored(self):
        tmp = Path(self.id().replace(".", "_") + ".json")
        try:
            with mock.patch.object(config, "USER_SETTINGS_FILE", tmp):
                config.AppConfig.save_user_settings({"totally_unknown": 1})
                self.assertNotIn("totally_unknown", json.loads(tmp.read_text(encoding="utf-8")))
        finally:
            tmp.unlink(missing_ok=True)

    def test_router_policy_matches_saved_defaults(self):
        router.apply_generation_settings(
            {"generation_preset": "compact",
             "pause_intra_ms": 40, "pause_sentence_ms": 280,
             "pause_paragraph_ms": 450, "pause_language_ms": 160,
             "pause_learning_ms": 900, "explicit_break_max_ms": 1400,
             "edge_trim_ms": 30, "micro_fade_ms": 8})
        settings = router.generation_settings()
        self.assertEqual(settings["pause_sentence_ms"], 280)
        self.assertEqual(settings["explicit_break_max_ms"], 1400)

    def test_cleaner_ui_persists_its_defaults(self):
        import gr_tabs.audio_tools_tab as tab
        tmp = Path(self.id().replace(".", "_") + ".json")
        try:
            with mock.patch.object(config, "USER_SETTINGS_FILE", tmp):
                tab._persist_cleaner_defaults("natural", False, 256)
                loaded = config.AppConfig.load_user_settings()
                self.assertEqual(loaded.cleaner_preset, "natural")
                self.assertEqual(loaded.cleaner_quality_kbps, 256)
                self.assertFalse(loaded.cleaner_trim_edges)
        finally:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
