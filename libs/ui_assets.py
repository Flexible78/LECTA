# libs/ui_assets.py — CSS, JavaScript and HTML progress-bar generators
import json

custom_css = """
/* ====================================================================
   LECTA design system - production UI theme
   Surfaces: deep slate | Accent: amber | Data: cyan / emerald / rose
   ==================================================================== */
:root {
  --lecta-bg: #0b1120;
  --lecta-surface: #111827;
  --lecta-surface-2: #1e293b;
  --lecta-surface-3: #24334a;
  --lecta-border: #334155;
  --lecta-border-strong: #475569;
  --lecta-text: #f8fafc;
  --lecta-text-dim: #94a3b8;
  --lecta-accent: #dc762d;
  --lecta-accent-2: #c65e1e;
  --lecta-info: #52acde;
  --lecta-ok: #30ac80;
  --lecta-err: #da4e68;
  --lecta-radius: 12px;
  --lecta-radius-sm: 8px;
  --lecta-shadow: 0 8px 24px rgba(2, 6, 23, 0.45);
  --lecta-focus: 0 0 0 3px rgba(220, 118, 45, 0.28);
}

/* --- App shell --- */
gradio-app, .gradio-container {
  background: radial-gradient(1200px 600px at 15% -10%, #121a29 0%, var(--lecta-bg) 55%) !important;
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
  color: #64748b !important;
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
  background: rgba(17, 24, 39, 0.92) !important;
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
  color: #0b1120 !important;
  background: linear-gradient(180deg, var(--lecta-accent), var(--lecta-accent-2)) !important;
  box-shadow: 0 2px 10px rgba(220, 118, 45, 0.25) !important;
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
label > span, .gr-form label, span[data-testid=block-info] {
  color: var(--lecta-text-dim) !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  letter-spacing: 0.03em !important;
}

/* --- Inputs --- */
input[type=text], input[type=number], input[type=password], textarea, .gr-input, select {
  background: #0f172a !important;
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
  color: #0b1120 !important;
  border: none !important;
  font-weight: 700 !important;
  border-radius: var(--lecta-radius-sm) !important;
  box-shadow: 0 4px 14px rgba(220, 118, 45, 0.22) !important;
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
.gr-dataframe tbody tr:hover { background: rgba(220, 118, 45, 0.08) !important; }

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
.progress-level { background-color: #0f172a !important; border-radius: 6px !important; }
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

custom_head = _head_i18n + _head_hotkeys


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
            <span>🔄 Auto-parsing: {label}</span><span style="color:#f97316;">{pct}%</span>
        </div>
        <div style="width:100%;background:#0f172a;border-radius:8px;overflow:hidden;height:16px;box-shadow:inset 0 2px 4px rgba(0,0,0,0.5);">
            <div class="lecta-pb-fill" style="--pb-pct:{pct}%;height:100%;background:linear-gradient(90deg,#ea580c,#f97316);transition:width 0.3s ease;"></div>
        </div>
        <div style="color:#94a3b8;font-size:12px;margin-top:4px;">Project {current} of {total}</div>
    </div>"""

def get_metrics_html(percent, elapsed, remaining, speed):
    """Single TTS progress bar (tts_tab.py)"""
    return f"""
    <div style="background: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px; font-weight: bold; font-size: 18px; color: #f8fafc;">
            <span>TTS progress</span><span style="color: #f97316;">{percent}%</span>
        </div>
        <div style="width: 100%; background-color: #0f172a; border-radius: 10px; overflow: hidden; height: 28px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);">
            <div class="lecta-pb-fill" style="--pb-pct:{percent}%;height:100%;background:linear-gradient(90deg,#ea580c,#f97316);transition:width 0.3s ease;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 10px; color: #94a3b8; font-family: monospace; font-size: 15px;">
            <span>⏱ Elapsed: <span style="color:#38bdf8">{elapsed}</span></span>
            <span>⏳ Remaining: <span style="color:#f43f5e">{remaining}</span></span>
            <span>⚡ Speed: <span style="color:#10b981">{speed}</span></span>
        </div>
    </div>
    """

def get_batch_metrics_html(project_name, project_pct, project_idx, total_projects, 
                           batch_pct, elapsed, remaining, speed):
    """Two-level batch TTS progress bar (tts_tab.py)"""
    return f"""
    <div style="background: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 3px; font-weight: bold; font-size: 14px; color: #94a3b8;">
            <span>📦 Batch TTS: project {project_idx} of {total_projects}</span>
            <span style="color: #c084fc;">{batch_pct}%</span>
        </div>
        <div style="width: 100%; background-color: #0f172a; border-radius: 8px; overflow: hidden; height: 12px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.5); margin-bottom: 14px;">
            <div class="lecta-pb-fill" style="--pb-pct:{batch_pct}%;height:100%;background:linear-gradient(90deg,#7c3aed,#a78bfa);transition:width 0.3s ease;"></div>
        </div>
        
        <div style="display: flex; justify-content: space-between; margin-bottom: 3px; font-weight: bold; font-size: 14px; color: #f8fafc;">
            <span>📁 {project_name}</span>
            <span style="color: #f97316;">{project_pct}%</span>
        </div>
        <div style="width: 100%; background-color: #0f172a; border-radius: 8px; overflow: hidden; height: 18px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.5); margin-bottom: 10px;">
            <div class="lecta-pb-fill" style="--pb-pct:{project_pct}%;height:100%;background:linear-gradient(90deg,#ea580c,#f97316);transition:width 0.3s ease;"></div>
        </div>
        
        <div style="display: flex; justify-content: space-between; margin-top: 8px; color: #94a3b8; font-family: monospace; font-size: 13px;">
            <span>⏱ Elapsed: <span style="color:#38bdf8">{elapsed}</span></span>
            <span>⏳ Remaining: <span style="color:#f43f5e">{remaining}</span></span>
            <span>⚡ Speed: <span style="color:#10b981">{speed}</span></span>
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
            <td style="padding:6px 12px; border-bottom:1px solid #334155;">{i}</td>
            <td style="padding:6px 12px; border-bottom:1px solid #334155; font-weight:bold; color:#f1f5f9;">{name}</td>
            <td style="padding:6px 12px; border-bottom:1px solid #334155; color:#38bdf8;">{format_audio_time(dur)}</td>
            <td style="padding:6px 12px; border-bottom:1px solid #334155; color:#f97316;">{format_audio_time(proc)}</td>
            <td style="padding:6px 12px; border-bottom:1px solid #334155; color:#10b981;">{size:.1f} MB</td>
        </tr>"""
    
    return f"""
    <div style="background: #1e293b; padding: 18px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 10px;">
        <div style="font-weight: bold; font-size: 18px; color: #f8fafc; margin-bottom: 12px;">
            📊 Batch TTS summary
        </div>
        <table style="width:100%; border-collapse:collapse; font-size:14px;">
            <thead>
                <tr style="color:#94a3b8; text-align:left; border-bottom:2px solid #475569;">
                    <th style="padding:6px 12px;">#</th>
                    <th style="padding:6px 12px;">Project</th>
                    <th style="padding:6px 12px;">🎧 Duration</th>
                    <th style="padding:6px 12px;">⏱ Processing</th>
                    <th style="padding:6px 12px;">💾 Size</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
                <tr style="font-weight:bold; border-top:2px solid #475569;">
                    <td style="padding:8px 12px;" colspan="2">📦 TOTAL ({len(stats)} projects)</td>
                    <td style="padding:8px 12px; color:#38bdf8;">{format_audio_time(total_dur)}</td>
                    <td style="padding:8px 12px; color:#f97316;">{format_audio_time(total_proc)}</td>
                    <td style="padding:8px 12px; color:#10b981;">{total_size:.1f} MB</td>
                </tr>
            </tbody>
        </table>
    </div>
    """

def get_parse_metrics_html(percent, action="Waiting..."):
    """Parse progress bar (parse_tab.py)"""
    return f"""
    <div style="background: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px; font-weight: bold; font-size: 16px; color: #f8fafc;">
            <span>Status: {action}</span><span style="color: #f97316;">{percent}%</span>
        </div>
        <div style="width: 100%; background-color: #0f172a; border-radius: 10px; overflow: hidden; height: 20px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);">
            <div class="lecta-pb-fill" style="--pb-pct:{percent}%;height:100%;background:linear-gradient(90deg,#ea580c,#f97316);transition:width 0.3s ease;"></div>
        </div>
    </div>
    """

def get_vocab_metrics_html(percent, elapsed, remaining, speed, action="Waiting..."):
    """Vocabulary parser progress bar (vocab_tab.py)"""
    return f"""
    <div style="background: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px; font-weight: bold; font-size: 16px; color: #f8fafc;">
            <span>Status: {action}</span><span style="color: #f97316;">{percent}%</span>
        </div>
        <div style="width: 100%; background-color: #0f172a; border-radius: 10px; overflow: hidden; height: 20px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);">
            <div class="lecta-pb-fill" style="--pb-pct:{percent}%;height:100%;background:linear-gradient(90deg,#ea580c,#f97316);transition:width 0.3s ease;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 10px; color: #94a3b8; font-family: monospace; font-size: 14px;">
            <span>⏱ Elapsed: <span style="color:#38bdf8">{elapsed}</span></span>
            <span>⏳ Remaining: <span style="color:#f43f5e">{remaining}</span></span>
            <span>⚡ Speed: <span style="color:#10b981">{speed}</span></span>
        </div>
    </div>
    """