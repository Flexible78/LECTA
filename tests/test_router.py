from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

import libs.multilingual_router as router


class RouterTests(unittest.TestCase):
    def test_explicit_break_is_deciseconds_and_capped(self):
        router.apply_generation_settings(preset="compact")
        self.assertEqual(router.explicit_break_ms("5"), 500)
        self.assertEqual(router.explicit_break_ms("30"), 1400)
        self.assertEqual(router.explicit_break_ms("30", cap_ms=250), 250)

    def test_pause_preset_and_custom_settings(self):
        settings = router.apply_generation_settings(preset="natural")
        self.assertEqual(settings["pause_language_ms"], 240)
        custom = router.apply_generation_settings({"pause_language_ms": 123, "micro_fade_ms": 4})
        self.assertEqual(custom["pause_language_ms"], 123)
        self.assertEqual(custom["micro_fade_ms"], 4)
        router.apply_generation_settings(preset="compact")

    def test_local_synthesis_failure_is_not_replaced_by_silence(self):
        def broken(*args, **kwargs):
            return 24000, None
        with self.assertRaisesRegex(RuntimeError, "returned no audio"):
            router.process_multilingual_text(
                "Это короткая русская строка для проверки.", broken,
                use_edge_en=False, use_edge_he=False, use_edge_ru=False,
                dict_mode=False,
            )

    def test_edge_synthesis_failure_is_not_replaced_by_silence(self):
        with mock.patch.object(router, "get_edge_audio", side_effect=RuntimeError("network down")):
            with self.assertRaisesRegex(RuntimeError, "Cloud speech synthesis failed"):
                router.process_multilingual_text(
                    "שלום", lambda *args, **kwargs: (24000, np.ones(1000, np.float32)),
                    use_edge_en=False, use_edge_he=True, use_edge_ru=False,
                    dict_mode=False,
                )

    def test_gap_policy_respects_sentence_language_and_paragraph(self):
        units = router._split_units("Первое предложение. Second sentence!\nשלום")
        self.assertGreaterEqual(len(units), 3)
        self.assertEqual(router._gap_between(units[0], units[1]), router.PAUSE_LANGUAGE_MS)
        self.assertEqual(router._gap_between(units[1], units[2]),
                         max(router.PAUSE_PARAGRAPH_MS, router.PAUSE_LANGUAGE_MS))

    def test_internal_pause_longer_than_limit_is_shortened(self):
        """A model's own breath at a comma must not exceed the configured limit."""
        sr = 24000
        speech = np.full(sr // 2, 0.2, dtype=np.float32)
        long_air = np.zeros(int(sr * 1.2), dtype=np.float32)
        audio = np.concatenate([speech, long_air, speech])

        result = router.squeeze_internal_silence(audio, sr, max_gap_ms=250)

        self.assertLess(len(result), len(audio))
        self.assertAlmostEqual(len(result) / sr, 0.5 + 0.25 + 0.5, delta=0.05)
        # Speech is never cut: the energy of the kept part is unchanged.
        self.assertAlmostEqual(float(np.sum(result.astype(np.float64) ** 2)),
                               float(np.sum(audio.astype(np.float64) ** 2)), delta=1e-3)

    def test_short_internal_pause_is_left_alone(self):
        sr = 24000
        speech = np.full(sr // 2, 0.2, dtype=np.float32)
        short_air = np.zeros(int(sr * 0.15), dtype=np.float32)
        audio = np.concatenate([speech, short_air, speech])
        self.assertEqual(len(router.squeeze_internal_silence(audio, sr, max_gap_ms=250)),
                         len(audio))

    def test_trim_removes_low_level_tail_after_speech(self):
        """Engine hiss below the -48 dBFS floor must not survive as dead air."""
        sr = 24000
        speech = np.full(sr // 2, 0.3, dtype=np.float32)
        hiss = np.full(int(sr * 0.8), 0.003, dtype=np.float32)  # ~ -50 dBFS
        trimmed = router.gentle_trim_and_fade(np.concatenate([speech, hiss]), sr,
                                              pad_sec=0.03, fade_sec=0.008)
        self.assertIsNotNone(trimmed)
        self.assertAlmostEqual(len(trimmed) / sr, 0.5 + 0.03, delta=0.03)

    def test_trim_keeps_quiet_but_real_continuation(self):
        sit = 24000
        speech = np.full(sit // 2, 0.3, dtype=np.float32)
        quiet_word = np.full(int(sit * 0.3), 0.01, dtype=np.float32)  # ~ -40 dBFS
        trimmed = router.gentle_trim_and_fade(np.concatenate([speech, quiet_word]), sit,
                                              pad_sec=0.0, fade_sec=0.0)
        self.assertIsNotNone(trimmed)
        self.assertGreater(len(trimmed) / sit, 0.75)


if __name__ == "__main__":
    unittest.main()
