"""UI asset tests: help overlay, workflow guide, petrol theme tokens."""
from __future__ import annotations

import unittest

from libs import ui_assets as ui


class HelpOverlayTests(unittest.TestCase):
    def test_help_button_and_panel_present(self):
        html = ui.help_overlay_html()
        self.assertIn('id="lecta-help-btn"', html)
        self.assertIn('class="lecta-help-backdrop"', html)
        self.assertIn("data-lecta-help-close", html)

    def test_help_explains_every_major_setting(self):
        html = ui.help_overlay_html()
        for needle in ("F5-TTS inference steps", "Max pause inside a phrase",
                       "Language-switch gap", "bitrate", "ECO", "Ctrl+Enter"):
            self.assertIn(needle, html, needle)

    def test_help_contains_measured_time_table(self):
        html = ui.help_overlay_html()
        # rows come from libs.estimates
        self.assertGreaterEqual(html.count("<tr><td><code>"), 4)
        self.assertIn("300 000", html)

    def test_help_toggle_script_is_injected_in_head(self):
        self.assertIn("lecta-help-open", ui.custom_head)
        self.assertIn("lecta-help-btn", ui.custom_head)

    def test_open_state_survives_gradio_css_scoping(self):
        # Gradio rewrites custom selectors (`.gradio-container ... .contain xxx`),
        # which outranks `.lecta-help-open .lecta-help-backdrop`; toggling must
        # therefore set an inline display value as well.
        self.assertIn("back.style.display = open ? 'block' : ''", ui.custom_head)

    def test_sidebar_hint_is_css_only(self):
        # The JS mutator had to go: it appended a badge into Gradio's label,
        # Gradio re-rendered the label, the observer fired again and the page
        # grew until the main thread stalled.
        css = ui.custom_css
        self.assertIn(".sidebar label .md", css)
        self.assertIn(".sidebar label:hover > span.md", css)
        self.assertIn("label > div::after", css)  # the (i) marker
        self.assertIn("return;", ui.custom_head)  # script stays disabled

    def test_tabs_styled_for_gradio6_markup(self):
        # Gradio 6 uses div.tab-wrapper[role=tablist] > div[role=tab] > button;
        # the old `.tab-nav button` selectors matched nothing, so every tab
        # lost its background.
        css = ui.custom_css
        self.assertIn('[role="tab"].selected', css)
        self.assertIn(".tab-wrapper", css)

    def test_buttons_always_have_a_fill(self):
        css = ui.custom_css
        self.assertIn("button.secondary", css)
        self.assertIn("button.primary", css)
        # the tinted-outline variant made half the buttons look background-less
        self.assertNotIn("rgba(217, 153, 47, 0.13) !important", css)


class WorkflowGuideTests(unittest.TestCase):
    def test_five_clickable_steps(self):
        html = ui.workflow_steps_html(2)
        self.assertEqual(html.count("data-step="), 5)
        self.assertEqual(html.count("data-goto="), 5)
        self.assertEqual(html.count("role='button'"), 5)
        self.assertIn("class='lecta-step active'", html)   # current stage
        self.assertIn("class='lecta-step done'", html)     # stage 1 is behind us
        for title in ("Add a book", "Analyze", "Pick a voice", "Generate", "Done"):
            self.assertIn(f"<b>{title}</b>", html)

    def test_steps_target_real_tabs(self):
        html = ui.workflow_steps_html(1)
        for target in ("upload", "analyze", "voice", "tts", "files"):
            self.assertIn(f"data-goto='{target}'", html)
        self.assertIn("id='lecta-steps'", html)

    def test_stage_follows_the_pipeline_in_the_head_script(self):
        js = ui.custom_head
        self.assertIn("window.lectaSetStep", js)
        self.assertIn("classList.toggle('active'", js)
        # clicking a step must switch tabs, not just change colour
        self.assertIn("clickTab('analyze')", js)
        self.assertIn("clickTab('tts')", js)


class ThemeTests(unittest.TestCase):
    def test_petrol_palette_tokens(self):
        self.assertIn("--lecta-bg: #05171a", ui.custom_css)
        self.assertIn("--lecta-surface: #0a2126", ui.custom_css)
        self.assertIn("--lecta-accent: #d9992f", ui.custom_css)  # deeper brass

    def test_no_legacy_slate_backgrounds_left(self):
        for legacy in ("#111827", "#0b1120", "#1e293b", "#334155"):
            self.assertNotIn(legacy, ui.custom_css, legacy)

    def test_hover_tooltip_styles(self):
        self.assertIn(".lecta-tip", ui.custom_css)
        self.assertIn(".lecta-info-hidden", ui.custom_css)
        self.assertIn(".lecta-hint", ui.custom_css)
        self.assertIn(".lecta-steps", ui.custom_css)


if __name__ == "__main__":
    unittest.main()
