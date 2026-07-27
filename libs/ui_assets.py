# libs/ui_assets.py — CSS, JavaScript и HTML-генераторы прогресс-баров
import json

custom_css = """
/* Force custom HTML progress bar fill widths via CSS custom properties.
   Gradio 6 may override inline width on divs inside gr.HTML, so we
   use !important with a variable set via inline style. */
.lecta-pb-fill { width: var(--pb-pct, 0%) !important; }

/* Style Gradio's built-in progress bar (used during model loading) */
.progress-level { background-color: #1a1a1a !important; border-radius: 4px !important; }
.progress-level > div[style*="width"], .progress-bar {
    background-color: #FF8C00 !important; 
    background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent) !important;
    background-size: 1rem 1rem !important;
    animation: progress-stripes 1s linear infinite !important;
}
@keyframes progress-stripes { from { background-position: 1rem 0; } to { background-position: 0 0; } }
.progress-text, .eta-text, .progress-info { color: #87CEEB !important; font-weight: bold !important; text-shadow: 1px 1px 2px #000000 !important; opacity: 1 !important; display: inline-block !important; }
.meta-text, .eta-level { text-align: right !important; display: block !important; width: 100% !important; }

/* Hide the Gradio footer — not needed in a local tool */
footer { display: none !important; }



/* Block card visual language */
.block-card {
  background: #1e293b !important;
  border-radius: 8px !important;
  padding: 12px !important;
  border: 1px solid #334155 !important;
}
.block-card h3 {
  color: #38bdf8 !important;
  font-size: 13px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.05em !important;
  margin-bottom: 8px !important;
}

/* Ensure status/readonly textboxes have enough height instead of one-line scrollers */
textarea[readonly] {
  min-height: 40px !important;
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

custom_head = f"""
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
