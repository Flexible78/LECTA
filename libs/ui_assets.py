# libs/ui_assets.py — CSS, JavaScript и HTML-генераторы прогресс-баров
import json

custom_css = """
/* ====================================================================
   LECTA design system — petrol edition
   Surfaces: deep petroleum teal | Accent: amber  | Data: aqua / mint / rose
   ==================================================================== */
:root {
  --lecta-bg: #05171a;
  --lecta-bg-glow: #0a2a31;
  --lecta-surface: #0a2126;
  --lecta-surface-2: #0e2b31;
  --lecta-surface-3: #123840;
  --lecta-surface-4: #174650;
  --lecta-border: #1d4a53;
  --lecta-border-strong: #2a6570;
  --lecta-text: #e8f4f4;
  --lecta-text-dim: #8fb0b3;
  /* Deeper brass: bright enough to stay the CTA colour, dark enough that
     near-black text on it clears WCAG AA (>= 7:1). */
  --lecta-accent: #d9992f;
  --lecta-accent-2: #b3751f;
  --lecta-accent-soft: rgba(217, 153, 47, 0.14);
  --lecta-on-accent: #06171b;
  --lecta-info: #4cc9c1;
  --lecta-ok: #3fb98d;
  --lecta-err: #e05a6f;
  --lecta-radius: 14px;
  --lecta-radius-sm: 9px;
  --lecta-shadow: 0 10px 30px rgba(1, 12, 15, 0.55);
  --lecta-shadow-soft: 0 2px 10px rgba(1, 12, 15, 0.35);
  --lecta-focus: 0 0 0 3px rgba(217, 153, 47, 0.35);
}

/* --- Gradio's own theme variables -------------------------------------
   Gradio ships a light palette by default: dropdowns stay white, "info"
   badges come out cream (#fef3c7) and grey text on them is unreadable.
   The head script forces dark mode, and these overrides put every built-in
   component on the petrol palette instead of amber-on-cream. */
.gradio-container,
body.dark,
.dark {
  --background-fill-primary: var(--lecta-surface) !important;
  --background-fill-secondary: var(--lecta-surface-2) !important;
  --body-background-fill: var(--lecta-bg) !important;
  --block-background-fill: var(--lecta-surface-2) !important;
  --block-border-color: var(--lecta-border) !important;
  --block-label-background-fill: transparent !important;
  --panel-background-fill: var(--lecta-surface) !important;
  --input-background-fill: #061a1e !important;
  --input-border-color: var(--lecta-border) !important;
  --body-text-color: var(--lecta-text) !important;
  --body-text-color-subdued: var(--lecta-text-dim) !important;
  --border-color-primary: var(--lecta-border) !important;
  --border-color-accent: var(--lecta-accent-2) !important;
  --color-accent: var(--lecta-accent) !important;
  --color-accent-soft: var(--lecta-accent-soft) !important;
  --color-accent-soft-text: var(--lecta-text) !important;
  --color-text-link: var(--lecta-info) !important;
  --slider-color: var(--lecta-accent) !important;
  --checkbox-background-color-selected: var(--lecta-accent) !important;
  --checkbox-border-color-selected: var(--lecta-accent-2) !important;
  --table-even-background-fill: var(--lecta-surface-2) !important;
  --table-odd-background-fill: var(--lecta-surface) !important;
  --button-primary-background-fill: var(--lecta-accent) !important;
  --button-primary-background-fill-hover: var(--lecta-accent-2) !important;
  --button-primary-text-color: var(--lecta-on-accent) !important;
  --button-secondary-background-fill: var(--lecta-surface-4) !important;
  --button-secondary-background-fill-hover: #1c5560 !important;
  --button-secondary-text-color: var(--lecta-text) !important;
  --button-secondary-border-color: var(--lecta-border-strong) !important;
  --link-text-color: var(--lecta-info) !important;
  --layout-gap: 8px !important;
}

/* Open dropdown lists and their rows. */
body .gradio-container ul.options,
body .gradio-container .options,
body .gradio-container .wrap-inner,
body .gradio-container .secondary-wrap {
  background: var(--lecta-surface) !important;
  color: var(--lecta-text) !important;
}
body .gradio-container .options .item:hover,
body .gradio-container .options .item.selected {
  background: var(--lecta-surface-4) !important;
  color: var(--lecta-text) !important;
}
/* --- App shell --- */
html, body {
  background:
    radial-gradient(1100px 520px at 12% -8%, var(--lecta-bg-glow) 0%, transparent 60%),
    radial-gradient(900px 420px at 88% 0%, #0a2026 0%, transparent 55%),
    linear-gradient(180deg, #061a1e 0%, var(--lecta-bg) 45%) !important;
  color: var(--lecta-text) !important;
}

gradio-app, .gradio-container {
  background:
    radial-gradient(1100px 520px at 12% -8%, var(--lecta-bg-glow) 0%, transparent 60%),
    radial-gradient(900px 420px at 88% 0%, #0a2026 0%, transparent 55%),
    linear-gradient(180deg, #061a1e 0%, var(--lecta-bg) 45%) !important;
  color: var(--lecta-text) !important;
}
.gradio-container {
  width: 100% !important;
  max-width: 100% !important;
  margin: 0 auto !important;
  padding: 12px !important;
}
/* Sidebar: Gradio 6 renders it as a FIXED overlay panel with its own
   collapse toggle. It must stay out of the normal flex flow — forcing it
   into the flow (position:relative + fixed width) squeezes the content
   and leaves dead space when collapsed. Just skin the panel and let the
   native collapse work: content automatically expands when it slides out. */
.gradio-container .sidebar {
  /* NEVER set overflow here: the toggle button lives OUTSIDE the panel edge,
     and any non-visible overflow clips it — the menu becomes impossible
     to reopen. Inner scrolling is handled by .sidebar-content itself. */
  overflow: visible !important;
  background: var(--lecta-surface) !important;
  border-right: 1px solid var(--lecta-border) !important;
}
/* Make sure the collapse/reopen toggle is always visible and tappable */
.gradio-container .sidebar .toggle-button {
  opacity: 1 !important;
  visibility: visible !important;
  z-index: 1002 !important;
  background: var(--lecta-surface-3) !important;
  border: 1px solid var(--lecta-border) !important;
  color: var(--lecta-text) !important;
}
/* Fix overlay: ensure content area doesn't get obscured */
.gradio-container .contain > .wrap > .tabs, 
.gradio-container .contain > .wrap > div {
  position: relative !important;
  z-index: 1 !important;
}
.gradio-container h1 { letter-spacing: -0.02em !important; }
.gradio-container h1, .gradio-container h2, .gradio-container h3 { color: var(--lecta-text) !important; }

/* --- Header: compact, pinned to the top --- */
.gradio-container {
  padding-top: 6px !important;
}
#lecta-header {
  margin: 0 0 4px 0 !important;
  padding: 0 !important;
}
#lecta-header .prose, #lecta-header div {
  margin: 0 !important;
  padding: 0 !important;
  min-height: 0 !important;
}
#lecta-header h1 { font-size: 20px !important; margin: 0 !important; }
#lecta-header p { margin: 1px 0 0 0 !important; line-height: 1.3 !important; }
/* Remove stray vertical gap above the first content block */
.gradio-container .contain > div:first-child,
.gradio-container .main .wrap > div:first-child { margin-top: 0 !important; }

/* --- Sidebar: compact --- */
.gradio-container .sidebar {
  gap: 4px !important;
  padding: 8px 10px !important;
}
.gradio-container .sidebar .block,
.gradio-container .sidebar .form {
  padding: 2px 4px !important;
  margin: 0 !important;
}
.gradio-container .sidebar .gap { gap: 4px !important; }
.gradio-container .sidebar label > span,
.gradio-container .sidebar span[data-testid=block-info] {
  font-size: 11px !important;
  line-height: 1.25 !important;
}
.gradio-container .sidebar label > span[data-testid="block-info"],
.gradio-container .sidebar span.info,
.gradio-container .sidebar label small {
  font-size: 10px !important;
  line-height: 1.2 !important;
  color: #6f9295 !important;
}
.gradio-container .sidebar button {
  min-height: 30px !important;
  padding: 4px 8px !important;
  font-size: 12px !important;
}
.gradio-container .sidebar hr { margin: 4px 0 !important; }
.gradio-container .sidebar p { margin: 2px 0 !important; font-size: 12px !important; }

/* --- Tab bar: sticky, obvious active state --- */
.tab-nav, .tabs > .tab-nav {
  position: sticky !important;
  top: 0 !important;
  z-index: 40 !important;
  gap: 4px !important;
  padding: 6px !important;
  border: 1px solid var(--lecta-border) !important;
  border-radius: var(--lecta-radius) !important;
  background: rgba(8, 32, 37, 0.92) !important;
  backdrop-filter: blur(8px) !important;
  box-shadow: var(--lecta-shadow) !important;
  overflow-x: auto !important;
}
.tab-nav button {
  border: 1px solid transparent !important;
  border-radius: var(--lecta-radius-sm) !important;
  padding: 9px 16px !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  color: var(--lecta-text-dim) !important;
  background: transparent !important;
  transition: background 0.15s ease, color 0.15s ease, transform 0.1s ease !important;
  white-space: nowrap !important;
}
.tab-nav button:hover { color: var(--lecta-text) !important; background: var(--lecta-surface-3) !important; }
.tab-nav button.selected {
  color: var(--lecta-on-accent) !important;
  background: linear-gradient(180deg, var(--lecta-accent), var(--lecta-accent-2)) !important;
  box-shadow: 0 2px 10px rgba(226, 163, 60, 0.28) !important;
}

/* --- Cards / blocks --- */
.block, .form, .gr-box, .panel {
  background: var(--lecta-surface) !important;
  border: 1px solid var(--lecta-border) !important;
  border-radius: var(--lecta-radius) !important;
}
.block-card {
  background: var(--lecta-surface-2) !important;
  border: 1px solid var(--lecta-border) !important;
  border-radius: var(--lecta-radius) !important;
  padding: 14px !important;
  box-shadow: var(--lecta-shadow) !important;
}
.block-card h3 {
  color: var(--lecta-info) !important;
  font-size: 12px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  margin-bottom: 10px !important;
}

/* --- Labels and helper text --- */
label > span, .gr-form label {
  color: var(--lecta-text-dim) !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  letter-spacing: 0.03em !important;
}

/* Gradio paints every block label as an amber "highlighter" chip. On the
   petrol background that reads as a smudge and the subdued grey text on it
   fails contrast badly, so the label becomes a quiet caption again. */
body .gradio-container span[data-testid="block-info"],
body .gradio-container label > span:first-child {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  color: var(--lecta-text) !important;
  font-size: 12px !important;
  font-weight: 600 !important;
}

/* --- Inputs --- */
input[type=text], input[type=number], input[type=password], textarea, .gr-input, select {
  background: #061a1e !important;
  color: var(--lecta-text) !important;
  border: 1px solid var(--lecta-border) !important;
  border-radius: var(--lecta-radius-sm) !important;
}
input[type=text]:focus, input[type=number]:focus, textarea:focus, select:focus {
  border-color: var(--lecta-accent) !important;
  box-shadow: var(--lecta-focus) !important;
  outline: none !important;
}
textarea[readonly] { min-height: 40px !important; color: var(--lecta-text-dim) !important; }

/* --- Buttons --- */
button.primary, .gr-button-primary {
  background: linear-gradient(180deg, var(--lecta-accent), var(--lecta-accent-2)) !important;
  color: var(--lecta-on-accent) !important;
  border: none !important;
  font-weight: 700 !important;
  border-radius: var(--lecta-radius-sm) !important;
  box-shadow: 0 4px 14px rgba(226, 163, 60, 0.24) !important;
}
button.secondary, .gr-button-secondary {
  background: var(--lecta-surface-3) !important;
  color: var(--lecta-text) !important;
  border: 1px solid var(--lecta-border-strong) !important;
  border-radius: var(--lecta-radius-sm) !important;
  font-weight: 600 !important;
}
button:hover:not(:disabled) { transform: translateY(-1px) !important; filter: brightness(1.06) !important; }
button:active:not(:disabled) { transform: translateY(0) !important; filter: brightness(0.96) !important; }
button:focus-visible { box-shadow: var(--lecta-focus) !important; outline: none !important; }
button:disabled { opacity: 0.5 !important; cursor: not-allowed !important; }
#tts_btn button, #batch_tts_btn button, #parse_btn button, #fb2_gen_btn button, #demo_tts_btn button {
  min-height: 46px !important;
  font-size: 15px !important;
  letter-spacing: 0.02em !important;
}
#stop_btn button {
  background: linear-gradient(180deg, #fb7185, #e11d48) !important;
  color: #ffffff !important;
  border: none !important;
  font-weight: 700 !important;
}

/* --- Accordions --- */
.gr-accordion, details {
  border: 1px solid var(--lecta-border) !important;
  border-radius: var(--lecta-radius) !important;
  background: var(--lecta-surface) !important;
}
.gr-accordion > .label-wrap, details > summary { font-weight: 600 !important; color: var(--lecta-text) !important; }

/* --- Tables --- */
.gr-dataframe table, table { border-collapse: collapse !important; }
.gr-dataframe thead th, table thead th {
  background: var(--lecta-surface-3) !important;
  color: var(--lecta-text-dim) !important;
  text-transform: uppercase !important;
  font-size: 11px !important;
  letter-spacing: 0.06em !important;
}
.gr-dataframe tbody tr:nth-child(even) { background: rgba(255, 255, 255, 0.02) !important; }
.gr-dataframe tbody tr:hover { background: rgba(226, 163, 60, 0.10) !important; }

/* --- Sliders and checkboxes --- */
input[type=range] { accent-color: var(--lecta-accent) !important; }
input[type=checkbox], input[type=radio] { accent-color: var(--lecta-accent) !important; }

/* --- Custom HTML progress bars ---
   Gradio may override inline width on divs inside gr.HTML, so the fill
   width is delivered through a CSS custom property with !important. */
.lecta-pb-fill {
  width: var(--pb-pct, 0%) !important;
  background-image: linear-gradient(90deg, var(--lecta-accent-2), var(--lecta-accent)) !important;
  transition: width 0.25s ease !important;
}

/* Gradio built-in progress bar (model loading) */
.progress-level { background-color: #061a1e !important; border-radius: 6px !important; }
.progress-level > div[style*=width], .progress-bar {
  background-color: var(--lecta-accent) !important;
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent) !important;
  background-size: 1rem 1rem !important;
  animation: progress-stripes 1s linear infinite !important;
}
@keyframes progress-stripes { from { background-position: 1rem 0; } to { background-position: 0 0; } }
.progress-text, .eta-text, .progress-info {
  color: var(--lecta-info) !important;
  font-weight: 700 !important;
  opacity: 1 !important;
  display: inline-block !important;
}
.meta-text, .eta-level { text-align: right !important; display: block !important; width: 100% !important; }

/* --- Scrollbars --- */
* { scrollbar-color: var(--lecta-border-strong) transparent; scrollbar-width: thin; }
*::-webkit-scrollbar { width: 10px; height: 10px; }
*::-webkit-scrollbar-track { background: transparent; }
*::-webkit-scrollbar-thumb { background: var(--lecta-border-strong); border-radius: 8px; }
*::-webkit-scrollbar-thumb:hover { background: var(--lecta-accent) !important; }

/* --- Tighter vertical rhythm between rows/blocks --- */
.gradio-container .main .gap,
.gradio-container .wrap .gap,
.gradio-container .contain .gap {
  gap: 8px !important;
}
.gradio-container .main .gap > *,
.gradio-container .wrap .gap > * { margin-top: 0 !important; margin-bottom: 0 !important; }

/* --- Misc --- */
footer { display: none !important; }
.gradio-container .prose a { color: var(--lecta-info) !important; }
@media (max-width: 900px) {
  .tab-nav button { padding: 8px 10px !important; font-size: 13px !important; }
  .gradio-container { padding: 8px !important; }
}
@media (prefers-reduced-motion: reduce) {
  button:hover:not(:disabled) { transform: none !important; }
  .lecta-pb-fill, .progress-bar { transition: none !important; animation: none !important; }
}

/* ====================================================================
   Ergonomics layer: help overlay, hover hints, workflow stepper, badges
   ==================================================================== */

/* --- Inline hint under a control --- */
.lecta-hint {
  margin: 2px 0 6px 0;
  padding: 6px 10px;
  border-radius: var(--lecta-radius-sm);
  border: 1px dashed var(--lecta-border-strong);
  background: rgba(76, 201, 193, 0.06);
  color: var(--lecta-text-dim);
  font-size: 12px;
  line-height: 1.45;
}
.lecta-hint b { color: var(--lecta-info); font-weight: 700; }

/* --- Badges --- */
.lecta-badge {
  display: inline-block;
  padding: 1px 8px;
  border-radius: 999px;
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  border: 1px solid var(--lecta-border-strong);
  color: var(--lecta-text-dim);
}
.lecta-badge.ok { color: #07181b; background: var(--lecta-ok); border-color: transparent; }
.lecta-badge.warn { color: #07181b; background: var(--lecta-accent); border-color: transparent; }
.lecta-badge.info { color: #07181b; background: var(--lecta-info); border-color: transparent; }

/* --- Selected radio / checkbox chips -------------------------------------
   Gradio paints the selected chip with the primary hue but keeps white text,
   which is unreadable on amber (2.1:1). Force the dark on-accent colour and
   inherit it into the inner text nodes. */
.gradio-container label.selected,
.gradio-container label.selected > span,
.gradio-container .wrap label.selected,
.gradio-container .wrap label.selected *,
.gradio-container input[type="radio"]:checked + label,
.gradio-container input[type="radio"]:checked + label *,
.gradio-container input[type="checkbox"]:checked + label,
.gradio-container input[type="checkbox"]:checked + label * {
  color: var(--lecta-on-accent) !important;
  -webkit-text-fill-color: var(--lecta-on-accent) !important;
}
.gradio-container label.selected,
.gradio-container .wrap label.selected,
.gradio-container input[type="radio"]:checked + label,
.gradio-container input[type="checkbox"]:checked + label {
  background: linear-gradient(180deg, var(--lecta-accent) 0%, var(--lecta-accent-2) 100%) !important;
  border-color: var(--lecta-accent-2) !important;
  font-weight: 700 !important;
  box-shadow: var(--lecta-shadow-soft);
}
.gradio-container label.selected:hover,
.gradio-container input[type="radio"]:checked + label:hover,
.gradio-container input[type="checkbox"]:checked + label:hover {
  filter: brightness(1.06);
}

/* --- Hover hints (\u24d8 markers everywhere in the UI) --- */
.lecta-tip {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 15px; height: 15px;
  margin-left: 5px;
  border-radius: 50%;
  border: 1px solid var(--lecta-border-strong);
  color: var(--lecta-info);
  font-size: 10px;
  font-weight: 700;
  line-height: 1;
  cursor: help;
  vertical-align: middle;
  background: rgba(76, 201, 193, 0.08);
}
.lecta-tip::after {
  content: attr(data-tip);
  position: absolute;
  top: calc(100% + 8px);
  left: -6px;
  z-index: 2000;
  width: max-content;
  max-width: 300px;
  padding: 8px 10px;
  border-radius: var(--lecta-radius-sm);
  border: 1px solid var(--lecta-border-strong);
  background: #06191d;
  color: var(--lecta-text);
  font-size: 12px;
  font-weight: 500;
  line-height: 1.45;
  text-align: left;
  white-space: normal;
  box-shadow: var(--lecta-shadow);
  opacity: 0;
  transform: translateY(-3px);
  pointer-events: none;
  transition: opacity 0.13s ease, transform 0.13s ease;
}
.lecta-tip:hover::after, .lecta-tip:focus::after { opacity: 1; transform: translateY(0); }
/* In the narrow sidebar the hint opens to the right of the icon. */
.gradio-container .sidebar .lecta-tip::after {
  top: calc(100% + 8px); left: auto; right: -4px; max-width: 250px;
}
/* Sidebar helper text moves into the tooltip (see the enhancer script). */
.gradio-container .sidebar .lecta-info-hidden { display: none !important; }

/* --- Sidebar helper text ------------------------------------------------
   Gradio renders `info=` as a paragraph inside the control label. In the
   narrow sidebar that text pushes the controls apart, so it becomes a
   floating card on hover / keyboard focus. Pure CSS on purpose: mutating
   Gradio's DOM from JS caused re-render loops that froze the page. */
body .gradio-container .sidebar label { position: relative; }
body .gradio-container .sidebar label > div::after {
  content: "i";
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 15px; height: 15px;
  margin-left: 5px;
  border-radius: 50%;
  border: 1px solid var(--lecta-border-strong);
  background: rgba(76, 201, 193, 0.08);
  color: var(--lecta-info);
  font-size: 10px;
  font-weight: 700;
  line-height: 1;
  vertical-align: middle;
  cursor: help;
}
body .gradio-container .sidebar label > span.md,
body .gradio-container .sidebar label .md {
  display: block;
  position: absolute;
  left: 0;
  right: 0;
  top: calc(100% + 4px);
  z-index: 70;
  max-height: 0;
  opacity: 0;
  overflow: hidden;
  padding: 0 10px;
  margin: 0;
  border-radius: var(--lecta-radius-sm);
  border: 1px solid var(--lecta-border-strong);
  background: #06191d;
  box-shadow: var(--lecta-shadow);
  pointer-events: none;
  transition: opacity 0.13s ease, max-height 0.13s ease, padding 0.13s ease;
}
body .gradio-container .sidebar label > span.md p,
body .gradio-container .sidebar label .md p {
  margin: 0 !important;
  color: var(--lecta-text) !important;
  font-size: 12px !important;
  line-height: 1.45 !important;
}
body .gradio-container .sidebar label:hover > span.md,
body .gradio-container .sidebar label:hover .md,
body .gradio-container .sidebar label:focus-within > span.md,
body .gradio-container .sidebar label:focus-within .md {
  max-height: 260px;
  opacity: 1;
  padding: 8px 10px;
  pointer-events: auto;
}

/* --- Workflow stepper --- */
.lecta-steps {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: stretch;
  margin: 0 0 6px 0;
}
.lecta-step {
  flex: 1 1 150px;
  min-width: 140px;
  display: flex;
  gap: 8px;
  align-items: flex-start;
  padding: 8px 10px;
  border-radius: var(--lecta-radius-sm);
  border: 1px solid var(--lecta-border);
  background: linear-gradient(180deg, var(--lecta-surface-2), var(--lecta-surface));
  cursor: pointer;
  transition: border-color 0.14s ease, background 0.14s ease, transform 0.14s ease;
}
.lecta-step:hover {
  border-color: var(--lecta-border-strong);
  background: linear-gradient(180deg, var(--lecta-surface-3), var(--lecta-surface-2));
}
.lecta-step:focus-visible {
  outline: none;
  border-color: var(--lecta-accent);
  box-shadow: var(--lecta-focus);
}
/* the stage the user is on right now */
.lecta-step.active {
  border-color: var(--lecta-accent);
  background: linear-gradient(180deg, rgba(217, 153, 47, 0.16), rgba(217, 153, 47, 0.05));
  box-shadow: 0 2px 14px rgba(217, 153, 47, 0.18);
}
/* already behind us */
.lecta-step.done { opacity: 0.72; }
.lecta-step.done .lecta-step-num {
  background: linear-gradient(180deg, var(--lecta-ok), #2f8f6d);
  color: #04120e;
}
.lecta-step.done .lecta-step-num::after { content: "✓"; }
.lecta-step.done .lecta-step-num { font-size: 0; }
.lecta-step.done .lecta-step-num::after { font-size: 12px; font-weight: 800; }
.lecta-step-num {
  flex: 0 0 20px;
  height: 20px;
  border-radius: 50%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  font-weight: 800;
  color: var(--lecta-on-accent);
  background: linear-gradient(180deg, var(--lecta-accent), var(--lecta-accent-2));
}
.lecta-step-body { font-size: 12px; line-height: 1.35; color: var(--lecta-text-dim); }
.lecta-step-body b { display: block; color: var(--lecta-text); font-size: 12.5px; }
/* brief emphasis after a programmatic jump */
.lecta-flash { animation: lecta-flash 1.1s ease; }
@keyframes lecta-flash {
  0% { box-shadow: 0 0 0 0 rgba(217, 153, 47, 0.55); }
  100% { box-shadow: 0 0 0 14px rgba(217, 153, 47, 0); }
}

/* --- Help button (top right) --- */
#lecta-help-btn {
  position: fixed;
  top: 12px;
  right: 16px;
  z-index: 6000;
  width: 34px; height: 34px;
  padding: 0;
  border-radius: 50%;
  cursor: pointer;
  font-size: 17px;
  font-weight: 800;
  line-height: 1;
  color: var(--lecta-on-accent);
  background: linear-gradient(180deg, var(--lecta-accent), var(--lecta-accent-2));
  border: 1px solid rgba(255, 255, 255, 0.18);
  box-shadow: 0 6px 20px rgba(226, 163, 60, 0.35);
  transition: transform 0.12s ease, filter 0.12s ease;
}
#lecta-help-btn:hover { transform: translateY(-1px) scale(1.04); filter: brightness(1.08); }
#lecta-help-btn:focus-visible { outline: none; box-shadow: var(--lecta-focus); }

/* --- Buttons ------------------------------------------------------------
   Every button must look like a button: a solid fill plus a border. Amber is
   reserved for the run actions, but even those stay fully filled. */
body .gradio-container button.secondary,
body .gradio-container button.lg {
  background: var(--lecta-surface-4) !important;
  color: var(--lecta-text) !important;
  border: 1px solid var(--lecta-border-strong) !important;
  box-shadow: 0 1px 0 rgba(255, 255, 255, 0.04) inset !important;
}
body .gradio-container button.secondary:hover,
body .gradio-container button.lg:hover {
  background: #1c5560 !important;
  border-color: var(--lecta-info) !important;
}
body .gradio-container button.primary,
body .gradio-container button.lg.primary {
  background: linear-gradient(180deg, var(--lecta-accent) 0%, var(--lecta-accent-2) 100%) !important;
  color: var(--lecta-on-accent) !important;
  border: 1px solid #9d6415 !important;
  font-weight: 700 !important;
  box-shadow: 0 2px 10px rgba(217, 153, 47, 0.26) !important;
}
body .gradio-container button.primary:hover,
body .gradio-container button.lg.primary:hover {
  filter: brightness(1.08);
}
#tts_btn, #tts_btn button,
#batch_tts_btn, #batch_tts_btn button,
#parse_btn, #parse_btn button,
#demo_tts_btn, #demo_tts_btn button,
#save_xml_btn, #save_xml_btn button,
#fb2_gen_btn, #fb2_gen_btn button {
  background: linear-gradient(180deg, #e6a93c 0%, var(--lecta-accent-2) 100%) !important;
  color: var(--lecta-on-accent) !important;
  border: 1px solid #9d6415 !important;
  font-weight: 800 !important;
  box-shadow: 0 3px 14px rgba(217, 153, 47, 0.32) !important;
}
#tts_btn:hover, #batch_tts_btn:hover, #parse_btn:hover,
#demo_tts_btn:hover, #save_xml_btn:hover, #fb2_gen_btn:hover {
  filter: brightness(1.07);
}

/* --- Tabs (Gradio 6 markup) ---------------------------------------------
   Gradio 6 renders `div.tabs > div.tab-wrapper[role=tablist] >
   div.tab-container[role=tab] > button`, so the old `.tab-nav button`
   selectors stopped matching and the tabs lost their background entirely. */
body .gradio-container .tabs .tab-wrapper {
  gap: 4px !important;
  padding: 6px !important;
  border: 1px solid var(--lecta-border) !important;
  border-radius: var(--lecta-radius) !important;
  background: rgba(8, 32, 37, 0.94) !important;
}
body .gradio-container .tabs [role="tab"] {
  background: var(--lecta-surface-2) !important;
  border: 1px solid var(--lecta-border) !important;
  border-radius: var(--lecta-radius-sm) !important;
  color: var(--lecta-text-dim) !important;
  box-shadow: none !important;
}
body .gradio-container .tabs [role="tab"] button {
  background: transparent !important;
  border: none !important;
  color: inherit !important;
  padding: 8px 14px !important;
  font-weight: 700 !important;
  font-size: 13.5px !important;
}
body .gradio-container .tabs [role="tab"]:hover {
  border-color: var(--lecta-border-strong) !important;
  color: var(--lecta-text) !important;
  background: var(--lecta-surface-3) !important;
}
body .gradio-container .tabs [role="tab"].selected,
body .gradio-container .tabs [role="tab"].selected button {
  background: linear-gradient(180deg, #e6a93c 0%, var(--lecta-accent-2) 100%) !important;
  border-color: var(--lecta-accent-2) !important;
  color: var(--lecta-on-accent) !important;
  box-shadow: 0 2px 12px rgba(217, 153, 47, 0.3) !important;
}
body .gradio-container .tabs [role="tab"].selected button {
  background: transparent !important;
  border: none !important;
}

/* --- Help overlay --- */
.lecta-help-backdrop {
  position: fixed;
  inset: 0;
  z-index: 5990;
  display: none;
  padding: 56px 18px 40px;
  background:
    radial-gradient(960px 520px at 50% -12%, rgba(23, 70, 80, 0.55), transparent 62%),
    rgba(2, 12, 14, 0.84);
  -webkit-backdrop-filter: blur(7px) saturate(115%);
  backdrop-filter: blur(7px) saturate(115%);
  overflow: auto;
}
body.lecta-help-open .lecta-help-backdrop { display: block; }
.lecta-help-panel {
  position: relative;
  max-width: 880px;
  margin: 0 auto;
  border-radius: 18px;
  border: 1px solid var(--lecta-border-strong);
  background: linear-gradient(180deg, #0b262b 0%, #071d22 42%, #05171b 100%);
  box-shadow:
    0 34px 80px rgba(0, 0, 0, 0.62),
    inset 0 0 0 1px rgba(76, 201, 193, 0.07);
  color: var(--lecta-text);
  font-size: 13.5px;
  line-height: 1.6;
  overflow: hidden;
}
/* thin accent ribbon across the top edge */
.lecta-help-panel::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--lecta-accent) 0%, #edc06a 45%, var(--lecta-info) 100%);
}
.lecta-help-head {
  position: sticky;
  top: 0;
  z-index: 4;
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 18px 24px 14px;
  background: linear-gradient(180deg, rgba(11, 38, 43, 0.985) 0%, rgba(9, 30, 35, 0.92) 100%);
  border-bottom: 1px solid var(--lecta-border);
  -webkit-backdrop-filter: blur(8px);
  backdrop-filter: blur(8px);
}
.lecta-help-head-text { flex: 1 1 auto; min-width: 0; }
.lecta-help-head h2 {
  margin: 0;
  font-size: 19px;
  line-height: 1.2;
  letter-spacing: 0.01em;
  color: var(--lecta-text);
}
.lecta-help-sub {
  margin: 3px 0 0;
  font-size: 12px;
  color: var(--lecta-text-dim);
}
.lecta-help-body {
  padding: 6px 24px 26px;
}
.lecta-help-panel h2 { color: var(--lecta-text); }
.lecta-help-panel h3 {
  margin: 22px 0 10px;
  padding: 0 0 6px;
  font-size: 12.5px;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--lecta-info);
  border-bottom: 1px solid var(--lecta-border);
}
.lecta-help-panel h3:first-child { margin-top: 10px; }
.lecta-help-lead {
  margin: 12px 0 4px;
  padding: 12px 14px;
  border-radius: var(--lecta-radius-sm);
  border: 1px solid var(--lecta-border);
  background: rgba(76, 201, 193, 0.055);
  color: #d3e7e7;
}
.lecta-help-panel p { margin: 7px 0; color: #cfe4e4; }
.lecta-help-panel ul,
.lecta-help-panel ol { margin: 8px 0 8px 20px; color: #cfe4e4; }
.lecta-help-panel li { margin: 5px 0; }
.lecta-help-panel li::marker { color: var(--lecta-accent); }
.lecta-help-panel b { color: var(--lecta-text); }
.lecta-help-panel code,
.lecta-help-panel kbd {
  padding: 1px 6px;
  border-radius: 5px;
  background: rgba(76, 201, 193, 0.12);
  border: 1px solid var(--lecta-border);
  color: #8fe3dc;
  font-size: 12px;
  font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
}
.lecta-help-panel table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  margin: 10px 0 14px;
  font-size: 12.5px;
  border: 1px solid var(--lecta-border);
  border-radius: var(--lecta-radius-sm);
  overflow: hidden;
}
.lecta-help-panel th {
  padding: 8px 12px;
  background: var(--lecta-surface-3);
  color: var(--lecta-text-dim);
  text-align: left;
  text-transform: uppercase;
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.07em;
  border-bottom: 1px solid var(--lecta-border-strong);
}
.lecta-help-panel td {
  padding: 8px 12px;
  border-bottom: 1px solid var(--lecta-border);
  vertical-align: top;
  color: #d7eaea;
}
.lecta-help-panel tr:last-child td { border-bottom: none; }
.lecta-help-panel tbody tr:nth-child(even) td { background: rgba(76, 201, 193, 0.035); }
.lecta-help-panel tr:hover td { background: rgba(217, 153, 47, 0.075); }
.lecta-help-close {
  flex: 0 0 auto;
  width: 34px; height: 34px;
  border-radius: 50%;
  cursor: pointer;
  color: var(--lecta-text);
  background: var(--lecta-surface-3);
  border: 1px solid var(--lecta-border-strong);
  font-size: 17px;
  line-height: 1;
  transition: background 0.15s ease, transform 0.15s ease;
}
.lecta-help-close:hover { background: var(--lecta-err); border-color: var(--lecta-err); transform: rotate(90deg); }
.lecta-help-note {
  margin: 12px 0;
  padding: 11px 14px;
  border-left: 3px solid var(--lecta-accent);
  border-radius: 0 var(--lecta-radius-sm) var(--lecta-radius-sm) 0;
  background: rgba(217, 153, 47, 0.09);
  color: #e2f1f1;
  font-size: 12.5px;
}
.lecta-help-panel::-webkit-scrollbar { width: 10px; }
.lecta-help-panel::-webkit-scrollbar-thumb { background: var(--lecta-surface-4); border-radius: 8px; }
@media (max-width: 720px) {
  #lecta-help-btn { top: 8px; right: 10px; }
  .lecta-help-backdrop { padding: 16px 8px 24px; }
  .lecta-help-panel { border-radius: 13px; font-size: 13px; }
  .lecta-help-head { padding: 14px 14px 10px; gap: 8px; }
  .lecta-help-head h2 { font-size: 16px; }
  .lecta-help-body { padding: 4px 14px 18px; }
  .lecta-help-panel th, .lecta-help-panel td { padding: 6px 8px; }
}

/* =======================================================================
   Field affordance. Kept at the very END of the stylesheet on purpose:
   an earlier generic rule sets `background: var(--lecta-surface)` on
   .options/.wrap-inner with !important, and with equal specificity the later
   declaration wins. Putting this block first made it look like nothing changed.
   ======================================================================= */

/* The select itself: Gradio renders a bare <input> inside .wrap-inner, which
   has no border and no fill, so it read as plain text. */
body .gradio-container .block .wrap-inner {
  background: #08242a !important;
  border: 2px solid var(--lecta-border-strong) !important;
  border-radius: var(--lecta-radius-sm) !important;
  min-height: 40px !important;
  cursor: pointer !important;
  transition: border-color 0.13s ease, background 0.13s ease, box-shadow 0.13s ease;
}
body .gradio-container .block .wrap-inner:hover {
  border-color: var(--lecta-accent) !important;
  background: #0b2c33 !important;
}
body .gradio-container .block .wrap-inner:focus-within {
  border-color: var(--lecta-accent) !important;
  box-shadow: var(--lecta-focus) !important;
}
body .gradio-container .block .wrap-inner input,
body .gradio-container .block .secondary-wrap input {
  color: var(--lecta-text) !important;
  font-weight: 600 !important;
  cursor: pointer !important;
  padding-left: 4px !important;
}
/* the chevron: bright, unmistakably a menu */
body .gradio-container .block .icon-wrap,
body .gradio-container .block .icon-wrap svg {
  color: var(--lecta-info) !important;
  opacity: 1 !important;
}

/* The open list of choices: Gradio paints it with --background-fill-primary,
   which reads as a flat light panel over the petrol UI. */
body .gradio-container ul.options,
body .gradio-container .options {
  background: #08242a !important;
  border: 2px solid var(--lecta-border-strong) !important;
  border-radius: var(--lecta-radius-sm) !important;
  box-shadow: 0 14px 34px rgba(0, 0, 0, 0.55) !important;
  padding: 4px !important;
}
body .gradio-container ul.options li.item,
body .gradio-container .options .item {
  color: var(--lecta-text) !important;
  font-size: 13px !important;
  padding: 7px 10px !important;
  border-radius: 6px !important;
  cursor: pointer !important;
}
body .gradio-container ul.options li.item:hover,
body .gradio-container ul.options li.item.selected,
body .gradio-container .options .item:hover,
body .gradio-container .options .item.selected {
  background: rgba(217, 153, 47, 0.18) !important;
  color: var(--lecta-text) !important;
  font-weight: 600 !important;
}

/* Read-only status boxes ("Model loaded successfully!") used to render as a
   45 px box with its own scrollbar and clipped text. */
body .gradio-container .sidebar textarea,
body .gradio-container .block textarea {
  background: var(--lecta-surface-3) !important;
  border: 1px solid var(--lecta-border) !important;
  border-radius: var(--lecta-radius-sm) !important;
  color: var(--lecta-text) !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  line-height: 1.45 !important;
  min-height: 32px !important;
  max-height: 76px !important;
  padding: 7px 9px !important;
  resize: none !important;
  overflow-y: auto !important;
}
body .gradio-container .sidebar textarea[disabled] {
  color: var(--lecta-ok) !important;
  border-color: rgba(63, 185, 141, 0.35) !important;
  background: rgba(63, 185, 141, 0.08) !important;
  -webkit-text-fill-color: var(--lecta-ok) !important;
  opacity: 1 !important;
}
"""

# Dictionary of known Gradio Russian strings → English.
# Used by the MutationObserver in custom_head to fix Gradio's built-in
# i18n when the browser's locale is set to Russian and navigator.language
# cannot be overridden in time (Gradio 5+ loads its bundle before our
# <head> script runs).
GRADIO_RU_EN_MAP = {
    "Перетащите файл сюда": "Drop file here",
    "- или -": "- or -",
    "Нажмите для загрузки": "Click to upload",
    "Очистить": "Clear",
    "Отправить": "Submit",
    "Ошибка": "Error",
    "Использовать через API": "Use via API",
    "Создано с помощью Gradio": "Built with Gradio",
    "Настройки": "Settings",
}

# Build the JS dictionary literal for injection into the page
_ru_en_js = "{" + ", ".join(
    f"{json.dumps(k)}: {json.dumps(v)}" for k, v in GRADIO_RU_EN_MAP.items()
) + "}"

# NOTE: _head_i18n IS an f-string (it interpolates _ru_en_js), therefore every
# literal JS brace inside it MUST be doubled: {{ }}. Keep it that way.
_head_i18n = f"""
<script>
// Clear any stored Gradio i18n preference so the MutationObserver
// fallback below can take full effect.
try {{
  for (var i = localStorage.length - 1; i >= 0; i--) {{
    var k = localStorage.key(i);
    if (/lang|locale|i18n/i.test(k)) localStorage.removeItem(k);
  }}
}} catch (e) {{}}
</script>
<script>
// Replace known Gradio Russian strings with English equivalents.
// This runs after the Gradio bundle has already rendered its i18n.
// It works by walking all text nodes and replacing exact matches
// from a dictionary, using a single MutationObserver to catch
// dynamically-added elements.
(function() {{
  var MAP = {_ru_en_js};

  function isEditable(el) {{
    var tag = el.tagName;
    return tag === 'TEXTAREA' || tag === 'INPUT' || tag === 'SCRIPT' || tag === 'STYLE';
  }}

  function fixNode(node) {{
    var raw = node.nodeValue;
    if (!raw || !raw.trim()) return;
    var trimmed = raw.trim();
    if (MAP.hasOwnProperty(trimmed)) {{
      node.nodeValue = raw.replace(trimmed, MAP[trimmed]);
    }}
  }}

  function walkAndFix(root) {{
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null, false);
    var node;
    while ((node = walker.nextNode())) {{
      if (!isEditable(node.parentElement)) fixNode(node);
    }}
  }}

  // Initial sweep
  walkAndFix(document.body);

  // Watch for changes
  var observer = new MutationObserver(function() {{
    observer.disconnect();
    walkAndFix(document.body);
    observer.observe(document.body, {{ subtree: true, characterData: true, childList: true }});
  }});
  observer.observe(document.body, {{ subtree: true, characterData: true, childList: true }});
}})();
</script>
"""

# NOTE: _head_hotkeys is a PLAIN string (no f prefix) because this JS is full of
# single braces. Never add an f prefix here — doing so raises
# "SyntaxError: f-string: expecting a valid expression after '{'".
_head_hotkeys = """
<script>
document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.key === 'Enter') {
        const btns = ['#tts_btn', '#fb2_gen_btn', '#parse_btn', '#demo_tts_btn'];
        for (let id of btns) {
            let el = document.querySelector(id);
            if (el && el.offsetParent !== null) {
                let b = el.tagName.toLowerCase() === 'button' ? el : el.querySelector('button');
                if (b) { b.click(); break; } 
            }
        }
    }
    if (e.ctrlKey && (e.key === 's' || e.key === 'S' || e.key === 'ы' || e.key === 'Ы')) {
        e.preventDefault(); 
        let el = document.querySelector('#save_xml_btn');
        if (el && el.offsetParent !== null) {
            let b = el.tagName.toLowerCase() === 'button' ? el : el.querySelector('button');
            if (b) b.click();
        }
    }
    if (e.key === 'Escape') {
        let btns = document.querySelectorAll('button');
        for (let b of btns) {
            if (b.innerText.includes('Stop') && b.offsetParent !== null) { b.click(); }
        }
    }
    if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey) {
        let activeTag = document.activeElement.tagName.toLowerCase();
        if (activeTag === 'input') {
            let btns = document.querySelectorAll('button');
            for (let b of btns) {
                if (b.textContent.includes('Rename') && b.offsetParent !== null) {
                    b.click();
                    e.preventDefault();
                    e.stopPropagation();
                    return;
                }
            }
        }
    }
});
</script>
"""

_head_steps = """
<script>
// Clickable workflow strip. It follows the pipeline (upload → analyze → voice
// → generate → done) and lets the user jump to any stage by clicking it.
(function () {
  function steps() { return document.querySelectorAll('.lecta-step'); }

  function setStep(n) {
    var items = steps();
    for (var i = 0; i < items.length; i++) {
      var s = parseInt(items[i].getAttribute('data-step'), 10);
      items[i].classList.toggle('active', s === n);
      items[i].classList.toggle('done', s < n);
      if (s === n) { items[i].setAttribute('aria-current', 'step'); }
      else { items[i].removeAttribute('aria-current'); }
    }
  }
  window.lectaSetStep = setStep;

  // Gradio 6 drops the `id` we give gr.Tab, so tabs are matched by their
  // visible caption. Inner tabs are rendered deeper in the DOM than the
  // top-level ones, so the deepest visible match wins.
  var TAB_TEXT = {
    'fb2tts': 'project manager',
    'analyze': 'analyze',
    'tts': 'create audiobook',
    'cover': 'cover'
  };

  function depth(el) {
    var d = 0;
    while (el) { d += 1; el = el.parentElement; }
    return d;
  }

  function clickTab(name) {
    var tab = document.querySelector('[role="tab"][id="' + name + '"]');
    if (!tab) {
      var needle = TAB_TEXT[name] || name;
      var best = null, bestDepth = -1;
      var all = document.querySelectorAll('[role="tab"]');
      for (var i = 0; i < all.length; i++) {
        var t = all[i];
        if (!t.getClientRects().length) { continue; }  // hidden
        var txt = (t.textContent || '').toLowerCase();
        if (txt.indexOf(needle) === -1) { continue; }
        var d = depth(t);
        if (d > bestDepth) { best = t; bestDepth = d; }
      }
      tab = best;
    }
    if (!tab) { return false; }
    if (!isSelected(tab)) {
      var btn = tab.querySelector('button') || tab;
      btn.click();
    }
    return true;
  }

  function isSelected(tab) {
    return tab.classList.contains('selected') || tab.getAttribute('aria-selected') === 'true';
  }

  function tabKind(tab) {
    var txt = (tab.textContent || '').toLowerCase();
    if (txt.indexOf('analyze') !== -1) { return 'analyze'; }
    if (txt.indexOf('create audiobook') !== -1) { return 'tts'; }
    if (txt.indexOf('cover') !== -1) { return 'cover'; }
    return '';
  }

  function flash(el) {
    if (!el) { return; }
    try { el.scrollIntoView({ block: 'center', behavior: 'smooth' }); } catch (e) { el.scrollIntoView(); }
    el.classList.remove('lecta-flash');
    void el.offsetWidth;  // restart the animation
    el.classList.add('lecta-flash');
    setTimeout(function () { el.classList.remove('lecta-flash'); }, 1200);
  }

  function openSidebar() {
    var sb = document.querySelector('.sidebar');
    if (sb && sb.getBoundingClientRect().x < 0) {
      var btn = document.querySelector('button[aria-label="Toggle Sidebar"]');
      if (btn) { btn.click(); }
    }
  }

  function findVoiceSelect() {
    // The TTS tab owns the voice list; matching it by caption avoids relying
    // on the sidebar drawer (which Gradio keeps closed until clicked).
    var nodes = document.querySelectorAll('[data-testid="block-info"], label, span');
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      if (!el.getClientRects().length) { continue; }
      if ((el.textContent || '').trim().toLowerCase() === 'select main voice') {
        var block = el.closest('.block') || el.parentElement;
        if (block) { return block; }
      }
    }
    return null;
  }

  function goto(target) {
    switch (target) {
      case 'upload':
        clickTab('fb2tts');
        flash(document.getElementById('file_queue_table'));
        return true;
      case 'analyze':
        if (clickTab('analyze')) { return true; }
        clickTab('fb2tts');
        flash(document.getElementById('file_queue_table'));
        return false;
      case 'voice':
        openSidebar();
        if (clickTab('tts')) {
          setTimeout(function () { flash(findVoiceSelect()); }, 300);
          return true;
        }
        flash(document.querySelector('.sidebar'));
        return true;
      case 'tts':
        return clickTab('tts');
      case 'files':
        clickTab('tts');
        setTimeout(function () { flash(document.getElementById('file_checkboxes')); }, 300);
        return true;
      default:
        return false;
    }
  }

  document.addEventListener('click', function (e) {
    var t = e.target;
    if (!t || !t.closest) { return; }

    var step = t.closest('.lecta-step');
    if (step) {
      e.preventDefault();
      var n = parseInt(step.getAttribute('data-step'), 10);
      var ok = goto(step.getAttribute('data-goto'));
      if (ok !== false) { setStep(n); }
      return;
    }

    // keep the strip in sync when the user switches tabs by hand
    var tab = t.closest('[role="tab"]');
    if (!tab) { return; }
    var kind = tabKind(tab);
    if (kind === 'analyze') { setStep(2); }
    else if (kind === 'tts') { setStep(4); }
  }, true);

  document.addEventListener('keydown', function (e) {
    var t = e.target;
    if (!t || !t.classList || !t.classList.contains('lecta-step')) { return; }
    if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') {
      e.preventDefault();
      t.click();
    }
  }, true);

  // Gradio selects tabs programmatically after an upload/analysis, which fires
  // no click event here — watch the tab container so the strip keeps up.
  function syncFromTabs() {
    var tabs = document.querySelectorAll('[role="tab"]');
    for (var i = 0; i < tabs.length; i++) {
      if (!isSelected(tabs[i])) { continue; }
      var kind = tabKind(tabs[i]);
      if (kind === 'analyze') { setStep(2); return; }
      if (kind === 'tts') { setStep(4); return; }
    }
  }

  var observer = new MutationObserver(syncFromTabs);
  function observeTabs() {
    var lists = document.querySelectorAll('[role="tablist"]');
    for (var i = 0; i < lists.length; i++) {
      if (lists[i].getAttribute('data-lecta-observed') === '1') { continue; }
      lists[i].setAttribute('data-lecta-observed', '1');
      observer.observe(lists[i], { subtree: true, attributes: true, attributeFilter: ['class'] });
    }
    // Gradio may replace the whole tab container instead of flipping a class,
    // in which case no mutation ever fires on the new nodes: sync on every
    // pass as well, so the strip is at most one tick behind.
    syncFromTabs();
  }
  observeTabs();
  setInterval(observeTabs, 600);  // inner tabs are rendered later, on demand
})();
</script>
"""

_head_dark = """
<script>
// LECTA is a dark-only product: Gradio would otherwise follow the OS and
// render white dropdowns and cream info badges. Force the dark palette and
// keep it forced even if something toggles the class later.
(function () {
  function apply() {
    var html = document.documentElement;
    var body = document.body;
    if (html && !html.classList.contains('dark')) { html.classList.add('dark'); }
    if (body && !body.classList.contains('dark')) { body.classList.add('dark'); }
    if (html) { html.style.colorScheme = 'dark'; }
  }
  apply();
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', apply);
  }
  function watch(el) {
    if (!el) { return; }
    new MutationObserver(apply).observe(el, { attributes: true, attributeFilter: ['class'] });
  }
  watch(document.documentElement);
  if (document.body) {
    watch(document.body);
  } else {
    document.addEventListener('DOMContentLoaded', function () { watch(document.body); });
  }
})();
</script>
"""

_head_help = """
<script>
// Help overlay (? button, top right). Delegated listeners keep working after
// Gradio re-renders the gr.HTML component.
(function () {
  function backdrop() { return document.querySelector('.lecta-help-backdrop'); }
  function isOpen() { return document.body.classList.contains('lecta-help-open'); }
  function setOpen(open) {
    open = !!open;
    document.body.classList.toggle('lecta-help-open', open);
    var back = backdrop();
    // Gradio scopes custom CSS, so its scoped `display:none` outranks the
    // `.lecta-help-open` rule. An inline style always wins.
    if (back) {
      back.style.display = open ? 'block' : '';
      if (open) { back.scrollTop = 0; }
    }
  }

  document.addEventListener('click', function (e) {
    var target = e.target;
    if (!target || !target.closest) { return; }
    if (target.closest('#lecta-help-btn')) {
      e.preventDefault();
      setOpen(!isOpen());
      return;
    }
    if (target.closest('[data-lecta-help-close]')) {
      e.preventDefault();
      setOpen(false);
      return;
    }
    if (target.classList && target.classList.contains('lecta-help-backdrop')) {
      setOpen(false);
    }
  }, true);

  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && isOpen()) { setOpen(false); }
    if (e.key === 'F1') { e.preventDefault(); setOpen(!isOpen()); }
  }, true);
})();
</script>
<script>
// Sidebar ergonomics — DISABLED ON PURPOSE.
// This script used to append an (i) badge into Gradio's label and hide the
// original info text. Gradio re-renders those labels on every state update,
// which fed the MutationObserver below, so each re-render added another badge
// and the page kept growing until the main thread stalled.
// The hover card is now pure CSS (see `.sidebar label .md` in custom_css).
(function () {
  return;
  var canHover = window.matchMedia ? window.matchMedia('(hover: hover)').matches : true;
  // Gradio re-creates sidebar blocks on every state update. Without these
  // guards each re-render would append another badge and the page would keep
  // growing until the main thread stalls.
  var done = (typeof WeakSet === 'function') ? new WeakSet() : null;
  var budget = 200;

  function enhance() {
    if (!canHover || budget <= 0) { return; }
    var infos = document.querySelectorAll('.sidebar label .md');
    for (var i = 0; i < infos.length; i++) {
      var info = infos[i];
      if (info.dataset.lectaTip === '1') { continue; }
      if (done && done.has(info)) { continue; }
      var label = info.closest('label') || info.parentElement;
      var holder = label ? (label.querySelector('div') || label) : null;
      var text = (info.textContent || '').trim();
      if (!holder || !text || text.length > 320) { continue; }
      if (holder.querySelector('.lecta-tip')) { continue; }  // badge already there
      info.dataset.lectaTip = '1';
      if (done) { done.add(info); }
      var badge = document.createElement('span');
      badge.className = 'lecta-tip';
      badge.textContent = 'i';
      badge.setAttribute('role', 'note');
      badge.setAttribute('aria-label', text);
      badge.setAttribute('data-tip', text);
      holder.appendChild(badge);
      info.style.display = 'none';
      budget -= 1;
      if (budget <= 0) { return; }
    }
  }

  var pending = null;
  function schedule() {
    if (pending) { return; }
    pending = setTimeout(function () { pending = null; enhance(); }, 250);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', schedule);
  } else {
    schedule();
  }
  if (document.body) {
    new MutationObserver(schedule).observe(document.body, { childList: true, subtree: true });
  }
})();
</script>
"""

custom_head = _head_dark + _head_i18n + _head_hotkeys + _head_steps + _head_help


# ═══ TIME FORMATTERS ═══

def format_time_hms(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0: return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def format_audio_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m} m. {s} s."


# ═══ HTML PROGRESS BAR GENERATORS ═══

def get_upload_progress_html(pct, current, total, label):
    """Upload progress bar (app.py)"""
    return f"""<div style="background:#1e293b;padding:12px;border-radius:8px;border:1px solid #334155;margin-top:8px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;color:#f8fafc;font-size:14px;">
            <span>🔄 Auto-parsing: {label}</span><span style="color:#e2a33c;">{pct}%</span>
        </div>
        <div style="width:100%;background:#0f172a;border-radius:8px;overflow:hidden;height:16px;box-shadow:inset 0 2px 4px rgba(0,0,0,0.5);">
            <div class="lecta-pb-fill" style="--pb-pct:{pct}%;height:100%;background:linear-gradient(90deg,#c8862a,#e2a33c);transition:width 0.3s ease;"></div>
        </div>
        <div style="color:#8fb0b3;font-size:12px;margin-top:4px;">Project {current} of {total}</div>
    </div>"""

def get_metrics_html(percent, elapsed, remaining, speed):
    """Single TTS progress bar (tts_tab.py)"""
    return f"""
    <div style="background: var(--lecta-surface-2); padding: 15px; border-radius: 8px; border: 1px solid var(--lecta-border); margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px; font-weight: bold; font-size: 18px; color: var(--lecta-text);">
            <span>TTS progress</span><span style="color: #e2a33c;">{percent}%</span>
        </div>
        <div style="width: 100%; background-color: #061a1e; border-radius: 10px; overflow: hidden; height: 28px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);">
            <div class="lecta-pb-fill" style="--pb-pct:{percent}%;height:100%;background:linear-gradient(90deg,#c8862a,#e2a33c);transition:width 0.3s ease;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 10px; color: #94a3b8; font-family: monospace; font-size: 15px;">
            <span>⏱ Elapsed: <span style="color:#4cc9c1">{elapsed}</span></span>
            <span>⏳ Remaining: <span style="color:#e05a6f">{remaining}</span></span>
            <span>⚡ Speed: <span style="color:#3fb98d">{speed}</span></span>
        </div>
    </div>
    """

def get_batch_metrics_html(project_name, project_pct, project_idx, total_projects, 
                           batch_pct, elapsed, remaining, speed):
    """Two-level batch TTS progress bar (tts_tab.py)"""
    return f"""
    <div style="background: var(--lecta-surface-2); padding: 15px; border-radius: 8px; border: 1px solid var(--lecta-border); margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 3px; font-weight: bold; font-size: 14px; color: #94a3b8;">
            <span>📦 Batch TTS: project {project_idx} of {total_projects}</span>
            <span style="color: #7fd6cf;">{batch_pct}%</span>
        </div>
        <div style="width: 100%; background-color: #061a1e; border-radius: 8px; overflow: hidden; height: 12px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.5); margin-bottom: 14px;">
            <div class="lecta-pb-fill" style="--pb-pct:{batch_pct}%;height:100%;background:linear-gradient(90deg,#1b6d73,#4cc9c1);transition:width 0.3s ease;"></div>
        </div>
        
        <div style="display: flex; justify-content: space-between; margin-bottom: 3px; font-weight: bold; font-size: 14px; color: var(--lecta-text);">
            <span>📁 {project_name}</span>
            <span style="color: #e2a33c;">{project_pct}%</span>
        </div>
        <div style="width: 100%; background-color: #061a1e; border-radius: 8px; overflow: hidden; height: 18px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.5); margin-bottom: 10px;">
            <div class="lecta-pb-fill" style="--pb-pct:{project_pct}%;height:100%;background:linear-gradient(90deg,#c8862a,#e2a33c);transition:width 0.3s ease;"></div>
        </div>
        
        <div style="display: flex; justify-content: space-between; margin-top: 8px; color: #94a3b8; font-family: monospace; font-size: 13px;">
            <span>⏱ Elapsed: <span style="color:#4cc9c1">{elapsed}</span></span>
            <span>⏳ Remaining: <span style="color:#e05a6f">{remaining}</span></span>
            <span>⚡ Speed: <span style="color:#3fb98d">{speed}</span></span>
        </div>
    </div>
    """

def get_batch_summary_html(stats):
    """Сводная таблица после пакетной озвучки (tts_tab.py)"""
    if not stats:
        return ""
    
    total_dur = sum(s[1] for s in stats)
    total_size = sum(s[2] for s in stats)
    total_proc = sum(s[3] for s in stats)
    
    rows_html = ""
    for i, (name, dur, size, proc) in enumerate(stats, 1):
        rows_html += f"""
        <tr>
            <td style="padding:6px 12px; border-bottom:1px solid var(--lecta-border);">{i}</td>
            <td style="padding:6px 12px; border-bottom:1px solid var(--lecta-border); font-weight:bold; color:var(--lecta-text);">{name}</td>
            <td style="padding:6px 12px; border-bottom:1px solid var(--lecta-border); color:#4cc9c1;">{format_audio_time(dur)}</td>
            <td style="padding:6px 12px; border-bottom:1px solid var(--lecta-border); color:#e2a33c;">{format_audio_time(proc)}</td>
            <td style="padding:6px 12px; border-bottom:1px solid var(--lecta-border); color:#3fb98d;">{size:.1f} MB</td>
        </tr>"""
    
    return f"""
    <div style="background: var(--lecta-surface-2); padding: 18px; border-radius: 8px; border: 1px solid var(--lecta-border); margin-bottom: 10px;">
        <div style="font-weight: bold; font-size: 18px; color: var(--lecta-text); margin-bottom: 12px;">
            📊 Batch TTS summary
        </div>
        <table style="width:100%; border-collapse:collapse; font-size:14px;">
            <thead>
                <tr style="color:#8fb0b3; text-align:left; border-bottom:2px solid var(--lecta-border-strong);">
                    <th style="padding:6px 12px;">#</th>
                    <th style="padding:6px 12px;">Project</th>
                    <th style="padding:6px 12px;">🎧 Duration</th>
                    <th style="padding:6px 12px;">⏱ Processing</th>
                    <th style="padding:6px 12px;">💾 Size</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
                <tr style="font-weight:bold; border-top:2px solid var(--lecta-border-strong);">
                    <td style="padding:8px 12px;" colspan="2">📦 TOTAL ({len(stats)} projects)</td>
                    <td style="padding:8px 12px; color:#4cc9c1;">{format_audio_time(total_dur)}</td>
                    <td style="padding:8px 12px; color:#e2a33c;">{format_audio_time(total_proc)}</td>
                    <td style="padding:8px 12px; color:#3fb98d;">{total_size:.1f} MB</td>
                </tr>
            </tbody>
        </table>
    </div>
    """

def get_parse_metrics_html(percent, action="Waiting..."):
    """Parse progress bar (parse_tab.py)"""
    return f"""
    <div style="background: var(--lecta-surface-2); padding: 15px; border-radius: 8px; border: 1px solid var(--lecta-border); margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px; font-weight: bold; font-size: 16px; color: var(--lecta-text);">
            <span>Status: {action}</span><span style="color: #e2a33c;">{percent}%</span>
        </div>
        <div style="width: 100%; background-color: #061a1e; border-radius: 10px; overflow: hidden; height: 20px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);">
            <div class="lecta-pb-fill" style="--pb-pct:{percent}%;height:100%;background:linear-gradient(90deg,#c8862a,#e2a33c);transition:width 0.3s ease;"></div>
        </div>
    </div>
    """

def get_vocab_metrics_html(percent, elapsed, remaining, speed, action="Waiting..."):
    """Vocabulary parser progress bar (vocab_tab.py)"""
    return f"""
    <div style="background: var(--lecta-surface-2); padding: 15px; border-radius: 8px; border: 1px solid var(--lecta-border); margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px; font-weight: bold; font-size: 16px; color: var(--lecta-text);">
            <span>Status: {action}</span><span style="color: #e2a33c;">{percent}%</span>
        </div>
        <div style="width: 100%; background-color: #061a1e; border-radius: 10px; overflow: hidden; height: 20px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);">
            <div class="lecta-pb-fill" style="--pb-pct:{percent}%;height:100%;background:linear-gradient(90deg,#c8862a,#e2a33c);transition:width 0.3s ease;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 10px; color: #94a3b8; font-family: monospace; font-size: 14px;">
            <span>⏱ Elapsed: <span style="color:#4cc9c1">{elapsed}</span></span>
            <span>⏳ Remaining: <span style="color:#e05a6f">{remaining}</span></span>
            <span>⚡ Speed: <span style="color:#3fb98d">{speed}</span></span>
        </div>
    </div>
    """

# ═══ HELP OVERLAY + WORKFLOW GUIDE ═══

def _help_steps_rows():
    """Rows for the "inference steps → time" table in the help overlay."""
    try:
        from libs.estimates import steps_table_rows
        return steps_table_rows()
    except Exception:
        return []


def help_overlay_html() -> str:
    """Floating "?" button (top right) plus the full product help overlay."""
    rows = "".join(
        f"<tr><td><code>{steps}</code></td><td>{per_k}</td><td>{per_book}</td></tr>"
        for steps, per_k, per_book in _help_steps_rows()
    )
    return f"""
<button id="lecta-help-btn" type="button" title="Help — every setting explained (F1)" aria-label="Help">?</button>
<div class="lecta-help-backdrop" role="dialog" aria-modal="true" aria-label="LECTA help">
  <div class="lecta-help-panel">
    <div class="lecta-help-head">
      <div class="lecta-help-head-text">
        <h2>Help &amp; reference</h2>
        <p class="lecta-help-sub">Every voice, slider and trade-off — <kbd>Esc</kbd> closes this window</p>
      </div>
      <button class="lecta-help-close" type="button" data-lecta-help-close title="Close (Esc)">&times;</button>
    </div>
    <div class="lecta-help-body">
    <p class="lecta-help-lead">LECTA turns a book (FB2, PDF, DOCX, EPUB, RTF, HTML or a web article) into an MP3 audiobook
    with Russian, English and Hebrew voices. Everything you need is on one screen: pick a project,
    analyse it, choose a voice and pauses, generate.</p>

    <h3>1. Quick start (5 steps)</h3>
    <ol>
      <li><b>📁 Project Manager → Upload file(s)</b> — drag a book into the queue and press
          <code>⬇️ Upload file(s)</code>. The book is parsed automatically and appears in
          <b>Select project</b>. A whole article can be fetched by URL.</li>
      <li><b>🔍 Analyze</b> — splits the book into chapters and writes the XML the voice engine
          reads. Rerun it after editing the text.</li>
      <li><b>🎧 TTS tab</b> — choose the voice in <b>Select main voice</b>, then set the
          <b>speed</b>, <b>bitrate</b> and <b>F5-TTS inference steps</b>.</li>
      <li>Press <b>TTS (Ctrl+Enter)</b> for one project or <b>Batch TTS</b> for several.
          <b>Stop</b> keeps the audio that is already finished.</li>
      <li>The MP3 appears in the table on the right with a player; the folder opens automatically
          (toggle <b>📂 Auto-open folder</b>).</li>
    </ol>
    <div class="lecta-help-note">Long jobs: tick <b>⚡ Background mode</b> before uploading — the work
    continues in a separate process even if you close this tab.</div>

    <h3>2. Which voice model to pick</h3>
    <table>
      <tr><th>Model</th><th>Language</th><th>Character</th><th>When to use</th></tr>
      <tr><td><b>ESpeech-TTS (F5-TTS)</b></td><td>RU / EN</td><td>Expressive, natural prosody</td>
          <td>Default for Russian fiction. 2.7 GB, needs the GPU.</td></tr>
      <tr><td><b>Misha24-10 (F5-TTS)</b></td><td>RU</td><td>Lighter, slightly flatter</td>
          <td>Faster on weak GPUs, long non-fiction.</td></tr>
      <tr><td><b>Silero v5_6 / v5_cis</b></td><td>RU</td><td>Fast, steady, "radio" reading</td>
          <td>Huge books, hot GPU, many hours of text.</td></tr>
      <tr><td><b>Vosk 0.10</b></td><td>RU</td><td>Robotic but instant</td>
          <td>Drafts and checking the text flow.</td></tr>
      <tr><td><b>Silero English v3</b></td><td>EN</td><td>Neutral English</td>
          <td>English-only books without the cloud.</td></tr>
      <tr><td><b>⚡ Edge TTS (cloud)</b></td><td>RU / EN / HE</td><td>Neural Microsoft voices</td>
          <td>Hebrew always uses Edge. Also the fastest way for RU/EN — needs internet.</td></tr>
    </table>
    <p>The <b>Stress placement</b> model writes Russian stress marks (<code>+</code>) so the voice
    reads "замо́к/за́мок" correctly. RuAccent is the better default; Silero stress is lighter.</p>

    <h3>3. Speed, quality and time</h3>
    <p><b>F5-TTS inference steps</b> is the main quality/speed knob: 8 = draft, 10–14 = normal,
    20+ = maximum detail. Generation time grows linearly with the step count. Measured on this
    machine (RTX 3060 Laptop 6 GB, ESpeech, speed 1, stress marks on):</p>
    <table>
      <tr><th>Steps</th><th>Time per 1000 characters</th><th>300 000-character book</th></tr>
      {rows}
    </table>
    <p><b>Speed</b> changes how fast the voice speaks (0.9 = slower and calmer, 1.1 = quicker), not
    the generation time. <b>Audio bitrate</b>: 96–128 kbps is transparent for speech, 192 is
    archival; higher values only make files bigger. <b>Add stress marks (RU)</b> costs ~2 s per 1000
    characters and noticeably improves Russian pronunciation.</p>

    <h3>4. Pauses (✂️ Audio pauses)</h3>
    <p>A TTS engine adds its own air at commas and full stops, so LECTA controls both the gaps
    <i>between</i> fragments and the silence <i>inside</i> them. Foreground and background
    generation always use these same values.</p>
    <table>
      <tr><th>Setting</th><th>What it does</th><th>Recommended</th></tr>
      <tr><td><b>Pause profile</b></td><td>Preset for all values below</td>
          <td><code>compact</code> for narration, <code>natural</code> for fiction,
          <code>learning</code> for study material</td></tr>
      <tr><td><b>Max pause inside a phrase</b></td><td>Caps the engine's own pauses at commas</td>
          <td>0.15–0.25 s for tight reading</td></tr>
      <tr><td><b>Language-switch gap</b></td><td>Pause between RU / EN / HE fragments</td>
          <td>0.16 s</td></tr>
      <tr><td><b>Between sentences / paragraphs</b><br>(Settings → Audio &amp; pauses)</td>
          <td>Gaps LECTA inserts itself</td><td>0.28 / 0.45 s</td></tr>
      <tr><td><b>Edge trim pad</b></td><td>Silence kept around each fragment; too small clips
          quiet consonants</td><td>0.03 s</td></tr>
      <tr><td><b>Anti-click fade</b></td><td>Micro fade at cut points only, never over speech</td>
          <td>0.008 s</td></tr>
      <tr><td><b>📚 Dictionary mode</b></td><td>Longer breaths between languages, for vocabulary
          lists</td><td>off for books</td></tr>
    </table>
    <div class="lecta-help-note">Already have an MP3 with big pauses? Open
    <b>✂️ Remove pauses</b>, choose the file and the <code>compact</code> profile — the original is
    never overwritten.</div>

    <h3>5. GPU, ECO and temperature</h3>
    <ul>
      <li><b>Compute device</b> <code>auto</code> uses the GPU (fast, ~6 GB VRAM);
          <code>cpu</code> keeps the GPU free but is 10–50× slower.</li>
      <li><b>♻ ECO</b> limits the work to 2 threads, adds a 15 s cooldown and lowers the thermal
          limit to 78 °C — use it on laptops with a hot case.</li>
      <li>The GPU temperature badge refreshes every 5 s. Above the limit synthesis pauses until the
          card cools down (~80 °C by default); the log then shows <code>❄️ Cooled …</code>.</li>
      <li><b>🛠️ System &amp; Cleanup → 🔍 Check for updates</b> compares your model files with the
          registry and downloads only what is missing.</li>
    </ul>

    <h3>6. Hotkeys</h3>
    <table>
      <tr><th>Key</th><th>Action</th></tr>
      <tr><td><code>Ctrl+Enter</code></td><td>Run the main action (Analyze / TTS / Save)</td></tr>
      <tr><td><code>Ctrl+S</code></td><td>Save the XML editor</td></tr>
      <tr><td><code>Esc</code></td><td>Stop the running job / close this help</td></tr>
      <tr><td><code>Delete</code></td><td>Delete the selected file</td></tr>
      <tr><td><code>F2</code></td><td>Rename the selected project</td></tr>
      <tr><td><code>F1</code> or the <b>?</b> button</td><td>Open this help</td></tr>
    </table>

    <h3>7. If something goes wrong</h3>
    <ul>
      <li><b>"Local TTS returned silent audio"</b> — the model produced nothing for one fragment
          (usually a broken leftover such as <code>для ,</code> from parsing). The job stops on
          purpose instead of writing silence: rerun <b>Analyze</b> and start TTS again.</li>
      <li><b>"Cloud speech synthesis failed"</b> — no internet, or Microsoft throttled the requests.
          Retry, or switch that language to local synthesis in the sidebar.</li>
      <li><b>The reading sounds choppy</b> — lower <b>Max pause inside a phrase</b> to 0.15 s and
          keep the <code>compact</code> profile.</li>
      <li><b>Generation is very slow</b> — lower the steps to 10, or use Silero / Edge TTS.</li>
      <li><b>GPU 85 °C+</b> — press ♻ ECO, close other GPU apps, or switch to
          <b>Silero v5_6</b>.</li>
    </ul>
    </div>
  </div>
</div>
"""


def workflow_steps_html(active: int = 1) -> str:
    """Clickable 5-step strip: it follows the pipeline and navigates on click."""
    steps = [
        ("Add a book", "Upload it or paste a URL — the book is parsed automatically", "upload"),
        ("Analyze", "Split into chapters and build the XML for the voice", "analyze"),
        ("Pick a voice", "Sidebar: model, language flags, pauses", "voice"),
        ("Generate", "TTS (one project) or Batch TTS (many)", "tts"),
        ("Done", "MP3 + player; the package opens in Explorer", "files"),
    ]
    cells = []
    for i, (title, hint, target) in enumerate(steps, 1):
        cls = "lecta-step"
        if i == active:
            cls += " active"
        elif i < active:
            cls += " done"
        cells.append(
            f"<div class='{cls}' data-step='{i}' data-goto='{target}' "
            f"role='button' tabindex='0' title='{title} — click to jump here'>"
            f"<span class='lecta-step-num'>{i}</span>"
            f"<span class='lecta-step-body'><b>{title}</b>{hint}</span>"
            "</div>"
        )
    return "<div class='lecta-steps' id='lecta-steps'>" + "".join(cells) + "</div>"
