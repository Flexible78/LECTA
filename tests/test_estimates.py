"""Tests for the UI time estimates used in tooltips and the help overlay."""
from __future__ import annotations

import unittest

from libs import estimates


class EstimateTests(unittest.TestCase):
    def test_time_grows_with_steps(self):
        slow = estimates.tts_seconds(1000, 32)
        fast = estimates.tts_seconds(1000, 8)
        self.assertGreater(slow, fast)
        # 4x the steps -> roughly 4x the synthesis time (plus fixed accent cost).
        self.assertLess(fast, slow / 2.5)

    def test_accent_cost_is_included_and_can_be_disabled(self):
        with_accents = estimates.tts_seconds(1000, 12, accents=True)
        without = estimates.tts_seconds(1000, 12, accents=False)
        self.assertAlmostEqual(with_accents - without, estimates.ACCENT_SECONDS_PER_KCHAR, delta=0.01)

    def test_cloud_is_not_gpu_bound(self):
        cloud = estimates.tts_seconds(1000, 32, cloud=True)
        local = estimates.tts_seconds(1000, 32, cloud=False)
        self.assertLess(cloud, local)

    def test_empty_text_costs_nothing(self):
        self.assertEqual(estimates.tts_seconds(0, 12), 0.0)

    def test_humanize(self):
        self.assertEqual(estimates.humanize_seconds(12), "12 s")
        self.assertEqual(estimates.humanize_seconds(600), "10 min")
        self.assertEqual(estimates.humanize_seconds(3540), "59 min")
        self.assertEqual(estimates.humanize_seconds(4500), "1 h 15 min")
        self.assertEqual(estimates.humanize_seconds(7200), "2 h")

    def test_steps_table_is_monotonic_and_complete(self):
        rows = estimates.steps_table_rows()
        self.assertEqual([r[0] for r in rows], [4, 8, 10, 14, 20, 32])
        seconds = [estimates.tts_seconds(estimates.TYPICAL_BOOK_CHARS, r[0]) for r in rows]
        self.assertEqual(seconds, sorted(seconds))
        self.assertTrue(all(r[1] and r[2] for r in rows))

    def test_hint_html_mentions_steps_and_time(self):
        html = estimates.steps_hint_html(14)
        self.assertIn("14 steps", html)
        self.assertIn("1000 characters", html)
        self.assertIn("lecta-hint", html)


if __name__ == "__main__":
    unittest.main()
