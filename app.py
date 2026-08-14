import os
os.environ["GRADIO_LANGUAGE"] = "en"

import gradio as gr
from pathlib import Path
import shutil
import re
import sys
import logging
import inspect
import numpy as np
import threading
import webbrowser
import subprocess
import json
import tkinter as tk

from libs.utils import get_data_list, data_path, change_pitch
from libs.accent import accentizer
from libs.tts import synth, set_tts_device
from libs.tts_preprocessor import TextParse
from config import AppConfig
import config as config_module
from gr_tabs.parse_tab import parse_tab, get_all_projects_xml
from gr_tabs.tts_tab import tts_tab
from gr_tabs.settings_tab import settings_tab
from gr_tabs.cover_tab import cover_tab
from gr_tabs.vocab_tab import vocab_tab
from libs.russian import normalize_russian

from libs.document_parser import parse_and_save_document
from libs.web_scraper import scrape_and_save_article
from libs.ui_assets import custom_css, custom_head, get_upload_progress_html
from libs.project_manager import (
    refresh_data, remove_dataset, remove_all_datasets,
    create_fb2_file, update_existing_fb2, delete_created_file
)
from libs.system_tools import (
    clean_tmp_folder, get_installed_models, get_models_disk_info,
    delete_selected_model,
    get_voice_models_choices, update_voice_model, update_all_voice_models,
    check_all_voice_models, stop_model_update, quick_check_models_local
)
from libs.voice_preview import preview_voice, PREVIEW_PHRASES
from libs.multilingual_router import process_multilingual_text
import libs.multilingual_router as router
from libs.fb2_processor import FB2Processor 

os.system("color")

# ═══ ЗАГРУЗКА СОХРАНЁННЫХ ФЛАГОВ EDGE TTS ═══
try:
    saved = AppConfig.load_user_settings()
    router.USE_EDGE_FOR_ENGLISH = getattr(saved, 'use_edge_english', False)
    router.USE_EDGE_FOR_RUSSIAN = getattr(saved, 'use_edge_russian', False)
    router.USE_EDGE_FOR_HEBREW = getattr(saved, 'use_edge_hebrew', True)
    router.DICTIONARY_MODE = getattr(saved, 'dict_mode', False)
except Exception:
    pass  # fallback to defaults 

class FullColorFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        if record.levelno >= logging.ERROR: return f"\033[91m{msg}\033[0m" 
        if record.levelno == logging.INFO: return f"\033[92m{msg}\033[0m" 
        return msg

CURRENT_DIR = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if logger.hasHandlers(): logger.handlers.clear()
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(FullColorFormatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(sh)
logger.propagate = False

class RedStderr:
    def __init__(self, original_stderr): self.original_stderr = original_stderr
    def write(self, msg): self.original_stderr.write(f"\033[91m{msg}\033[0m")
    def flush(self): self.original_stderr.flush()
sys.stderr = RedStderr(sys.stderr)

def global_exception_handler(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("CRITICAL ERROR:", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = global_exception_handler

# ==============================================================================
# ЧТЕНИЕ ИЗ БУФЕРА ОБМЕНА WINDOWS
# ==============================================================================
def read_clipboard_path() -> str:
    """Извлекает пути к файлам из буфера обмена Windows.
    Поддерживает: копирование файлов в Проводнике (CF_HDROP) и копирование текстовых путей."""
    CREATE_NO_WINDOW = 0x08000000
    
    # Способ 1: PowerShell Get-Clipboard -Format FileDropList (для файлов из Проводника)
    try:
        process = subprocess.Popen(
            ["powershell", "-NoProfile", "-command",
             "Get-Clipboard -Format FileDropList | Select-Object -ExpandProperty FullName"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=CREATE_NO_WINDOW)
        out, err = process.communicate(timeout=5)
        if out.strip():
            paths = [p.strip() for p in out.strip().split('\n') if p.strip()]
            if paths:
                return "\n".join(paths)
    except Exception:
        pass
    
    # Способ 2: PowerShell Get-Clipboard -Format Text (для текстовых путей)
    try:
        process = subprocess.Popen(
            ["powershell", "-NoProfile", "-command",
             "Get-Clipboard -Format Text"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=CREATE_NO_WINDOW)
        out, err = process.communicate(timeout=5)
        if out.strip():
            valid_paths = []
            for line in out.strip().split('\n'):
                p = line.strip().strip('"').strip("'")
                if p and os.path.exists(p):
                    valid_paths.append(p)
            if valid_paths:
                return "\n".join(valid_paths)
    except Exception:
        pass
    
    # Способ 3: tkinter (запасной вариант)
    try:
        root = tk.Tk(); root.withdraw()
        clip_text = root.clipboard_get()
        root.destroy()
        if clip_text:
            valid_paths = []
            for line in clip_text.strip().split('\n'):
                p = line.strip().strip('"').strip("'")
                if p and os.path.exists(p):
                    valid_paths.append(p)
            if valid_paths:
                return "\n".join(valid_paths)
    except Exception:
        pass
    
    return ""

def read_clipboard_text() -> str:
    try:
        root = tk.Tk(); root.withdraw()
        clip_text = root.clipboard_get()
        root.destroy()
        if clip_text: return clip_text.strip()
    except Exception: pass
    return ""

gr.set_static_paths(paths=[str(data_path),])
app_config = AppConfig.parse_args()
txt_parser = TextParse(False)

def process_url_wrapper(url, remove_ru):
    if not url or not url.strip(): return gr.update(), gr.update()
    gr.Info("Fetching the webpage and extracting text...", duration=3)
    ab_name, error_msg = scrape_and_save_article(url, remove_ru)
    if error_msg:
        gr.Warning(error_msg)
        return gr.update(), gr.update()
    gr.Info("Article downloaded successfully!", duration=4)
    return refresh_data(ab_name), refresh_data(ab_name)

def _extract_file_path(file_obj):
    """Extract path from a Gradio file object (string or object with .name)"""
    if isinstance(file_obj, str):
        return file_obj
    if hasattr(file_obj, 'name'):
        return file_obj.name
    return str(file_obj)

def process_file_wrapper(manual_path, drop_files, remove_ru):
    """ГЕНЕРАТОР: загружает файлы → создаёт FB2 → парсит с прогресс-баром.
    Каждый yield обновляет: dropdown проектов + прогресс + статус."""
    # ── Шаг 1: Сбор путей ──
    file_paths = []
    if manual_path and manual_path.strip():
        raw_paths = re.split(r'[\n|]+', manual_path)
        for p in raw_paths:
            p = p.strip().strip('"').strip("'")
            if p and os.path.exists(p):
                file_paths.append(p)
    if drop_files is not None:
        if isinstance(drop_files, list):
            for f in drop_files:
                fp = _extract_file_path(f)
                if fp and os.path.exists(fp) and fp not in file_paths:
                    file_paths.append(fp)
        else:
            fp = _extract_file_path(drop_files)
            if fp and os.path.exists(fp) and fp not in file_paths:
                file_paths.append(fp)
    if not file_paths:
        yield gr.update(), gr.update(), "", ""
        return
    
    # ── Шаг 2: Конвертация в FB2 (быстро, без парсинга) ──
    processed = []
    errors = []
    for i, fp in enumerate(file_paths, 1):
        fname = Path(fp).name
        gr.Info(f"[{i}/{len(file_paths)}] Loading: {fname}...", duration=2)
        ab_name, error_msg = parse_and_save_document(fp, remove_ru)
        if error_msg:
            errors.append(f"{fname}: {error_msg}")
        else:
            processed.append(ab_name)
    if errors:
        for err in errors:
            gr.Warning(err)
    if not processed:
        yield gr.update(), gr.update(), "", ""
        return
    
    first_ab = processed[0]
    total = len(processed)
    drop_update = refresh_data(first_ab)
    
    # ── Первый yield: обновляем dropdown + показываем начало прогресса ──
    yield drop_update, drop_update, get_upload_progress_html(0, 0, total, "Starting..."), f"📦 Loaded {total} projects. Starting parse..."
    
    # ── Шаг 3: Парсинг каждого проекта с живым прогрессом ──
    # Читаем сохранённые настройки парсинга
    try:
        fresh_cfg = AppConfig.load_user_settings()
        ps_ch_size = getattr(fresh_cfg, 'ch_size', 200)
        ps_punctuation = getattr(fresh_cfg, 'punctuation', False)
        ps_translit = getattr(fresh_cfg, 'translit', True)
        ps_sound_effect = getattr(fresh_cfg, 'sound_effect', False)
    except Exception:
        ps_ch_size, ps_punctuation, ps_translit, ps_sound_effect = 200, False, True, False
    
    for i, ab_name in enumerate(processed):
        try:
            proc = FB2Processor()
            for pct, msg in proc.process_book(
                ab_path=ab_name, replace=True,
                sound_effect=ps_sound_effect,
                punctuation=ps_punctuation,
                translit=ps_translit,
                ch_size=ps_ch_size
            ):
                overall_pct = int((i + pct / 100) / total * 100)
                yield (
                    gr.update(), gr.update(),
                    get_upload_progress_html(overall_pct, i + 1, total, f"{ab_name}: {msg}"),
                    f"📄 [{i+1}/{total}] Parsing: {ab_name} — {msg}"
                )
        except Exception as e:
            logger.warning(f"⚠️ Auto-parse error {ab_name}: {e}")
            yield (
                gr.update(), gr.update(),
                get_upload_progress_html(int((i+1)/total*100), i+1, total, f"⚠️ Error: {ab_name}"),
                f"⚠️ [{i+1}/{total}] Parse error {ab_name}: {e}"
            )
    
    # ── Финальный yield: скрываем прогресс-бар ──
    gr.Info(f"✅ Loaded and parsed: {total} projects", duration=5)
    yield (
        gr.update(), gr.update(),
        "",  # сбрасываем прогресс-бар
        f"🎉 All {total} projects parsed! Ready for TTS."
    )

def clean_srt_timings(text):
    """Clean SRT timecodes from text (for dictionary preparation)"""
    if not text: return text
    # Удаляем таймкоды 00:00:19,560 --> 00:00:21,570
    text = re.sub(r'\d{2}:\d{2}:\d{2}[,\.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,\.]\d{3}', '', text)
    # Удаляем одиночные номера строк SRT
    text = re.sub(r'(?m)^\d+$\n?', '', text)
    # Удаляем JSON структуру, если вставили JSON массив
    text = re.sub(r'[{}[\]",:]|id|time|ORIGINAL|EN|RU', '', text)
    # Чистим лишние переносы строк
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def toggle_tab(ab_path): return gr.Tabs(visible=True, selected=0), ab_path
def toggle_tab_parse(ab_path): return gr.Tabs(visible=True, selected=1), ab_path
def put_accents(string): return accentizer.process_accent(string, r'\+\w+|\w+\+\w+')

def text_to_audio(string, spk, rate=1, noise=None, pitch=None, ref_audio=None, ref_text=''):
    if not synth: return gr.update(label = "Model not loaded!")
    def safe_synth(t, disable_norm=False):
        txt = t if disable_norm else txt_parser.garbage(normalize_russian(t))
        res = synth.synth_audio(txt, spk, rate, noise, ref_audio, ref_text)
        if res is None: return None, None
        a, b = res
        if isinstance(a, (int, float)): return int(a), b
        elif isinstance(b, (int, float)): return int(b), a
        else: return 24000, a 
    try:
        sr, np_audio = process_multilingual_text(
            string, safe_synth,
            use_edge_en=router.USE_EDGE_FOR_ENGLISH,
            use_edge_he=router.USE_EDGE_FOR_HEBREW,
            use_edge_ru=router.USE_EDGE_FOR_RUSSIAN,
            dict_mode=router.DICTIONARY_MODE
        )
        if np_audio is None: return None
        if pitch != 50: np_audio = change_pitch(np_audio, sr, pitch)
        # Конвертируем float32 в int16 для Gradio (чтобы избежать предупреждения)
        if np_audio.dtype == np.float32 or np_audio.dtype == np.float64:
            np_audio = np.clip(np_audio, -1.0, 1.0)
            np_audio = (np_audio * 32767).astype(np.int16)
        return (sr, np_audio)
    except Exception as e:
        logger.error(f"Router error: {e}", exc_info=True)
        return None

def rename_project_folder(old_name, new_name):
    if not old_name or not new_name: return gr.update(), gr.update(), gr.update(visible=False)
    safe_new_name = "".join([c for c in new_name if c.isalnum() or c in (' ', '_', '-')]).rstrip()
    if not safe_new_name: 
        gr.Warning("Invalid project name!")
        return gr.update(), gr.update(), gr.update(visible=False)
    old_dir = data_path / old_name
    new_dir = data_path / safe_new_name
    if not old_dir.exists():
        gr.Warning(f"Project {old_name} not found!")
        return gr.update(), gr.update(), gr.update(visible=False)
    if new_dir.exists():
        gr.Warning(f"Project named {safe_new_name} already exists!")
        return gr.update(), gr.update(), gr.update()
    try:
        old_dir.rename(new_dir)
        gr.Info(f"Project renamed to {safe_new_name}")
        new_choices = sorted(get_data_list())
        return gr.update(choices=new_choices, value=safe_new_name), gr.update(choices=new_choices, value=safe_new_name), gr.update(visible=False)
    except Exception as e:
        logger.error(f"Rename error: {e}", exc_info=True)
        gr.Warning(f"Error: {e}")
        return gr.update(), gr.update(), gr.update(visible=False)

tts_models_list = [
    ('Vosk 0.10 (dev. 56 voices)', 2),
    ('Silero v5_5 (5 voices)', 3), ('Silero v5_cis (60 voices)', 4),
    ('Misha24-10 (F5-TTS)', 5), ('ESpeech-TTS (F5-TTS)', 6),
    ('Silero English v3 (English)', 7),
]
accent_models_list = [('RuAccent', 1), ('Silero stress', 2)]

def tts_model_load(ver, progress=gr.Progress()): return synth.load(ver)
def acc_model_load(ver, progress=gr.Progress()): return accentizer.load(ver)

def change_tts_model(mver):
    sp_list = synth.speakers_list()
    speaker = sp_list[0][1] if isinstance(sp_list[0], tuple) and sp_list else (sp_list[0] if sp_list else "")
    ref_group = gr.update(visible=True) if mver in [5, 6] else gr.update(visible=False)
    return gr.update(value=speaker, choices=sp_list), gr.update(visible=True), gr.update(interactive=True), ref_group

def change_acc_model(mver): return gr.update(interactive=True)

def load_existing_fb2(ab_name):
    if not ab_name: return "", ""
    fb2_path = data_path / ab_name / f"{ab_name}.fb2"
    if not fb2_path.exists(): return ab_name, ""
    try:
        content = fb2_path.read_text(encoding="utf-8")
        paras = re.findall(r'<p[^>]*>(.*?)</p>', content, flags=re.DOTALL | re.IGNORECASE)
        if not paras: raw_text = re.sub(r'<[^>]+>', '', content).strip()
        else: raw_text = "\n".join(paras).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        return ab_name, raw_text
    except Exception as e: return ab_name, f"Error: {e}"

global_shortcuts = """
<style>
.gradio-container { max-width: 98% !important; padding: 10px !important; }
.form { border-radius: 8px !important; box-shadow: none !important; }
div[data-testid="file-name"], span[data-testid="file-name"], .file-name, .file-preview {
    white-space: pre-wrap !important; word-break: break-all !important; overflow-wrap: break-word !important; overflow: visible !important; font-size: 13px !important; line-height: 1.2 !important;
}
.small-column { min-width: 260px !important; }
.small-column .upload-container { padding: 5px !important; }
.uniform-row { align-items: flex-end !important; }
.fixed-height-btn { height: 42px !important; display: flex; align-items: center; justify-content: center; font-weight: bold !important; }

/* УБИВАЕМ УРОДЛИВЫЕ СКРОЛЛБАРЫ ИЗ ТЕКСТОВЫХ СТАТУСОВ */
textarea[readonly] { overflow-y: hidden !important; resize: none !important; }

#tts_btn, #batch_tts_btn, #demo_tts_btn, #fb2_gen_btn { background: linear-gradient(135deg, #00aff0 0%, #005a9e 100%) !important; color: white !important; border: none !important; box-shadow: inset 0px 1px 2px rgba(255, 255, 255, 0.5), 0 4px 10px rgba(0, 175, 240, 0.4) !important; border-radius: 6px !important; }
#tts_btn:hover, #batch_tts_btn:hover, #demo_tts_btn:hover, #fb2_gen_btn:hover { background: linear-gradient(135deg, #00c3ff 0%, #0078d7 100%) !important; }
#batch_tts_btn { background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important; box-shadow: inset 0px 1px 2px rgba(255, 255, 255, 0.5), 0 4px 10px rgba(249, 115, 22, 0.4) !important; }
#batch_tts_btn:hover { background: linear-gradient(135deg, #fb923c 0%, #f97316 100%) !important; }
</style>
<script>
document.addEventListener('keydown', function(e) {
    function clickVisibleButton(selector) {
        let btns = document.querySelectorAll(selector);
        for (let b of btns) { if (b.offsetParent !== null) { b.click(); return; } }
    }
    if (e.key === 'Delete') clickVisibleButton('#del_file_btn');
    if (e.key === 'F2') clickVisibleButton('#rename_file_btn, #show_rename_btn');
    if (e.ctrlKey && e.key === 'Enter') clickVisibleButton('#parse_btn, #tts_btn, #fb2_gen_btn, #demo_tts_btn');
});
</script>
"""

with gr.Blocks(title="LECTA — Text-to-Speech for Russian, English and Hebrew") as App:
    gr.HTML(global_shortcuts, visible=False)
    gr.Markdown(
        "<div style='text-align:center; margin-bottom:8px;'>"
        "<h1 style='margin:0; color:#38bdf8;'>LECTA</h1>"
        "<p style='margin:2px 0; color:#94a3b8; font-size:14px;'>"
        "Text-to-Speech for Russian, English and Hebrew"
        "</p>"
        "<p style='margin:2px 0; color:#64748b; font-size:12px;'>"
        "KATAV writes, LECTA reads."
        "</p>"
        "<p style='margin:2px 0; color:#64748b; font-size:12px;'>"
        "<a href='https://github.com/Flexible78/KATAV' target='_blank' rel='noopener' style='color:#38bdf8;'>KATAV on GitHub</a>"
        "&nbsp;&nbsp;|&nbsp;&nbsp;"
        "<a href='" + config_module.LECTA_REPO_URL + "' target='_blank' rel='noopener' style='color:#38bdf8;'>LECTA on GitHub</a>"
        "&nbsp;&nbsp;|&nbsp;&nbsp;"
        "Built on <a href='https://gitverse.ru/diger/fb2tts' target='_blank' rel='noopener' style='color:#64748b;'>fb2tts</a> by diger"
        "</p>"
        "</div>"
    )
    
    tts_state = gr.State()
    acc_state = gr.State()
    ab_state = gr.State()
    
    with gr.Sidebar():
        tts_sel = gr.Dropdown(value='', allow_custom_value=True, label='Select TTS model', choices=tts_models_list, interactive=True)
        tts_status = gr.Textbox(show_label=False, visible=True, lines=1)
        acc_sel = gr.Dropdown(value='', allow_custom_value=True, label='Stress placement', choices=accent_models_list, interactive=True)
        acc_status = gr.Textbox(show_label=False, visible=True, lines=1)

        gr.Markdown("---")
        with gr.Row():
            device_radio = gr.Radio(
                choices=["auto", "cpu"],
                value="auto",
                label="🖥 Compute device",
                info="auto = GPU (fast, 6GB VRAM) | cpu = RAM (slower, but does not load the GPU)",
                interactive=True
            )
        
        with gr.Row():
            eco_btn = gr.Button("♻ ECO (quiet / cool)", size="sm", elem_id="eco_btn",
                variant="secondary")
        
        gr.Markdown("---")
        with gr.Row():
            preview_voice_btn = gr.Button("▶ Preview voice", size="sm", elem_id="preview_voice_btn")
        preview_audio = gr.Audio(label="Voice preview", type="filepath", interactive=False, autoplay=True, visible=False)
        preview_status = gr.Markdown("")
        
        gr.Markdown("---")
        fast_ru_cb = gr.Checkbox(label="⚡ Russian via cloud (Edge TTS)", value=router.USE_EDGE_FOR_RUSSIAN, info="Instant speed! Requires internet. Voice: Dmitry")
        fast_eng_cb = gr.Checkbox(label="⚡ English via cloud", value=router.USE_EDGE_FOR_ENGLISH, info="Fast generation via Microsoft Edge")
        
        # ДОБАВЛЕН ИВРИТ ЧЕРЕЗ ОБЛАКО
        fast_heb_cb = gr.Checkbox(label="⚡ Hebrew via cloud", value=router.USE_EDGE_FOR_HEBREW, info="Microsoft Edge TTS (recommended — local TTS does not support Hebrew)")
        
        dict_mode_cb = gr.Checkbox(label="📚 Dictionary mode (long pauses)", value=router.DICTIONARY_MODE, info="1 sec breath between languages")
        
        gr.Markdown("---")
        with gr.Row():
            restart_btn = gr.Button("🔄 Restart", variant="primary")
            quit_btn = gr.Button("🚪 Quit", variant="stop")
        
        def restart_app(): os._exit(42)
        def stop_app(): os._exit(0)
        
        js_restart = "function(){ document.body.innerHTML = '<h1 style=\"color:#38bdf8; text-align:center; margin-top:20%; font-family:sans-serif;\">🔄 Restarting server...<br>Please wait.</h1>'; setTimeout(() => location.reload(), 2000); }"
        js_exit = "function(){ document.body.innerHTML = '<h1 style=\"color:#f8fafc; text-align:center; margin-top:20%; font-family:sans-serif;\">Server stopped 🛑<br><br>You can close this tab.</h1>'; setTimeout(() => { window.open('', '_self', ''); window.close(); }, 500); }"

    with gr.Tabs() as project_tabs:
        with gr.Tab("📁 Project Manager", id="fb2tts") as fb2tts_tab:
            with gr.Row(visible=False): remove_ru_cb = gr.Checkbox(label="Remove Russian text", value=False)
            
            with gr.Row(elem_classes=["uniform-row"]):
                with gr.Column(scale=5):
                    ab_path = gr.Dropdown(value='', label="Select project", allow_custom_value=True, choices=get_data_list(), interactive=True)
                with gr.Column(scale=3):
                    with gr.Row():
                        show_rename_btn = gr.Button("✏️ Rename", elem_id="show_rename_btn", elem_classes=["fixed-height-btn"])
                        rm_dataset = gr.Button("❌ Delete", variant="secondary", elem_classes=["fixed-height-btn"])
                        rm_all_dataset = gr.Button("💣 Delete ALL", variant="stop", elem_classes=["fixed-height-btn"])
            
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes="block-card"):
                        with gr.Row(elem_classes=["uniform-row"]):
                            paste_file_btn = gr.Button("📋 From clipboard", variant="primary", elem_classes=["fixed-height-btn"], scale=1)
                            manual_file_input = gr.Textbox(label="File paths (one per line)", lines=2, scale=4, placeholder="C:\\books\\book1.fb2\nC:\\books\\book2.pdf")
                        upload_text_file = gr.File(label="Or drag files here: PDF, DOCX, EPUB, RTF, HTML, FB2...", file_count="multiple", height=100)
                        process_file_btn = gr.Button("⬇️ Upload file(s)", variant="primary")
                
                with gr.Column(scale=1):
                    with gr.Group(elem_classes="block-card"):
                        with gr.Row(elem_classes=["uniform-row"]):
                            paste_url_btn = gr.Button("📋 From clipboard", variant="primary", elem_classes=["fixed-height-btn"], scale=1)
                            url_input = gr.Textbox(label="🌐 Or article URL", lines=1, scale=4)
                        url_btn = gr.Button("⬇️ Download article", variant="primary")
            
            with gr.Row():
                upload_progress_html = gr.HTML(value="", visible=True)
                upload_status_text = gr.Textbox(label="Upload status", lines=1, interactive=False, visible=True)
            
            with gr.Row(visible=False) as rename_proj_panel:
                new_proj_name = gr.Textbox(label="Enter new project name", scale=4)
                confirm_rename_btn = gr.Button("💾 Save new name", scale=1, variant="primary")
                
            with gr.Tabs(visible=False) as inner_tabs:
                cover_tab(ab_path, ab_state)
                file_content_box, parse_df_output = parse_tab(ab_path, acc_state, tts_state)
                vocab_tab(ab_path, file_content_box)
                tts_tab(ab_path, tts_state)

        with gr.Tab("📝 Create FB2 / Editor"):
            gr.Markdown("### ✨ Create a new project or edit an existing one")
            
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Group():
                        gr.Markdown("#### 📥 Load existing")
                        load_project_dropdown = gr.Dropdown(
                            label="Select project",
                            choices=get_data_list(),
                            interactive=True,
                            info="Will load the project's FB2 file into the editor below"
                        )
                
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("#### 🗑 Delete")
                        fb2_delete_btn = gr.Button(
                            "🗑 Delete project (Delete)",
                            variant="stop",
                            elem_id="del_file_btn",
                            elem_classes=["fixed-height-btn"]
                        )
            
            gr.Markdown("---")
            
            with gr.Row():
                with gr.Column(scale=4):
                    fb2_title_input = gr.Textbox(
                        label="📛 Project name",
                        value="MyAudiobook",
                        placeholder="Enter a name (used as folder and file name)"
                    )
                    fb2_text_input = gr.Textbox(
                        label="📝 Text (Enter-separated lines = new paragraphs)",
                        lines=14,
                        placeholder="Paste text here...\nEach line becomes a separate paragraph in FB2.\n\nYou can also paste SRT subtitles and clean them with the button below."
                    )
                
                with gr.Column(scale=1, min_width=200):
                    with gr.Group():
                        gr.Markdown("#### ⚡ Actions")
                        fb2_generate_btn = gr.Button(
                            "✨ Save (Ctrl+Enter)",
                            variant="primary",
                            elem_id="fb2_gen_btn",
                            elem_classes=["fixed-height-btn"]
                        )
                        fb2_update_btn = gr.Button(
                            "🔄 Overwrite",
                            variant="secondary",
                            elem_classes=["fixed-height-btn"]
                        )
                        clean_srt_btn = gr.Button(
                            "✂️ Clean SRT (timecodes)",
                            variant="secondary",
                            elem_classes=["fixed-height-btn"]
                        )
                    
                    with gr.Group():
                        gr.Markdown("#### 📦 Result")
                        fb2_file_output = gr.File(label="Download FB2", height=80)
                        editor_status = gr.Textbox(
                            label="Status",
                            lines=2,
                            interactive=False,
                            placeholder="The operation result will appear here..."
                        )

        with gr.Tab(label="🎙️ Demo TTS"):
            with gr.Group(visible=False) as f5_gr:
                with gr.Row(): 
                    ref_audio = gr.Audio(label='Your voice sample', elem_classes="small-audio")
                with gr.Row(): 
                    ref_text = gr.Textbox(label='Text in sample', lines=1, placeholder="Enter the text spoken in the sample", interactive=True)
            with gr.Row():
                spk_sel = gr.Dropdown(value='', label='Select voice', choices=[''], interactive=True)
                speech_rate = gr.Slider(0, 3, 1, step=0.1, label="Set speed", interactive=True)
                noise_lvl = gr.Slider(0, 64, 16, step=1, label="Noise level", interactive=True)
                pitch_sel = gr.Slider(0, 100, 50, step=1, label="Pitch", interactive=True)
            with gr.Row():
                text_input = gr.Textbox(label='Text', lines=2, placeholder="English | עברית | Russian", interactive=True, max_length=220)
                audio_output = gr.Audio(interactive=False, buttons=[])
            with gr.Row():
                accent_button = gr.Button("Add stress marks", interactive=False, elem_classes=["fixed-height-btn"])
                tts_button = gr.Button("Convert to speech (Ctrl+Enter)", interactive=False, elem_id="demo_tts_btn", elem_classes=["fixed-height-btn"])

        with gr.Tab("🛠️ System & Cleanup") as system_tab:
            with gr.Row():
                with gr.Column(scale=1):
                    clean_tmp_btn = gr.Button("🧹 Clean tmp folder", variant="primary")
                    tmp_status = gr.Textbox(label="Result", interactive=False, lines=1)
                with gr.Column(scale=1):
                    model_to_del = gr.Dropdown(choices=get_installed_models(), label="Delete model")
                    with gr.Row():
                        confirm_del_cb = gr.Checkbox(label="I understand this deletes files from disk", value=False, scale=2)
                        del_model_btn = gr.Button("🗑 Delete model", variant="stop", scale=1)
                    model_del_status = gr.Textbox(label="Result", interactive=False, lines=1)
                    disk_space_display = gr.Textbox(label="Disk space", interactive=False, lines=1, value=get_models_disk_info()[0])
            gr.Markdown("---")
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 🔄 Update / download voice models")
                    voice_model_sel = gr.Dropdown(
                        choices=get_voice_models_choices(),
                        label="Select model to update",
                        allow_custom_value=True,
                        interactive=True
                    )
                    with gr.Row():
                        update_one_model_btn = gr.Button("⬇️ Update selected", variant="primary")
                        update_all_models_btn = gr.Button("⬇️⬇️ Update ALL models", variant="secondary")
                    with gr.Row():
                        check_models_btn = gr.Button("🔍 Check for updates", variant="secondary")
                        stop_model_btn = gr.Button("🛑 Abort", variant="stop")
                    with gr.Row():
                        update_del_model = gr.Dropdown(choices=get_installed_models(), label="Delete model", scale=2)
                        update_del_confirm_cb = gr.Checkbox(label="I understand this deletes files from disk", value=False, scale=2)
                        update_del_btn = gr.Button("🗑 Delete", variant="stop", scale=1)
                    update_del_status = gr.Textbox(label="Result", interactive=False, lines=1)
                with gr.Column(scale=3):
                    voice_model_status = gr.Textbox(
                        label="Model update status",
                        interactive=False, lines=10,
                        placeholder="Select a model and click 'Update'..."
                    )

        with gr.Tab("⚙️ Settings"):
            settings_tab(tts_state)

    # ==============================================================================
    # БЛОК ОБРАБОТЧИКОВ СОБЫТИЙ 
    # ==============================================================================
    def set_fast_russian(val):
        router.USE_EDGE_FOR_RUSSIAN = val
        AppConfig.save_user_settings({'use_edge_russian': val})
    def set_fast_english(val):
        router.USE_EDGE_FOR_ENGLISH = val
        AppConfig.save_user_settings({'use_edge_english': val})
    def set_fast_hebrew(val):
        try: router.USE_EDGE_FOR_HEBREW = val
        except: pass
        AppConfig.save_user_settings({'use_edge_hebrew': val})
    def set_dict_mode(val):
        router.DICTIONARY_MODE = val
        AppConfig.save_user_settings({'dict_mode': val})
    
    def eco_preset():
        """Apply ECO mode: reduce workers, add cooldown, lower temp limit."""
        import config as cfg
        import gr_tabs.tts_tab as ttab
        cfg.TTS_COOLDOWN_SEC = 15
        cfg.TTS_GPU_TEMP_LIMIT_C = 78
        cfg.TTS_GPU_TEMP_RESUME_C = 70
        # Recreate semaphore with 2 permits
        try:
            new_sema = threading.Semaphore(2)
            ttab._tts_sema = new_sema
        except Exception:
            pass
        return "♻ ECO mode: 2 workers, cooldown 15 s, temp limit 78°C"
    
    def preview_voice_handler(tts_model_val, spk):
        """Handler for preview voice button."""
        if not tts_model_val:
            return "⚠️ Select a TTS model first", gr.update(visible=False)
        return preview_voice(tts_model_val, spk)
        
    device_radio.change(fn=set_tts_device, inputs=device_radio, outputs=tts_status)
    fast_ru_cb.change(set_fast_russian, inputs=fast_ru_cb, outputs=[])
    fast_eng_cb.change(set_fast_english, inputs=fast_eng_cb, outputs=[])
    fast_heb_cb.change(set_fast_hebrew, inputs=fast_heb_cb, outputs=[])
    dict_mode_cb.change(set_dict_mode, inputs=dict_mode_cb, outputs=[])

    eco_btn.click(fn=eco_preset, outputs=tts_status)
    preview_voice_btn.click(
        fn=preview_voice_handler,
        inputs=[tts_sel, spk_sel],
        outputs=[preview_status, preview_audio]
    )

    tts_sel.select(tts_model_load, inputs=tts_sel, outputs=[tts_state, tts_status], show_progress_on=tts_status)
    acc_sel.select(acc_model_load, inputs=acc_sel, outputs=[acc_state, acc_status], show_progress_on=acc_status)
    restart_btn.click(fn=restart_app, js=js_restart)
    quit_btn.click(fn=stop_app, js=js_exit)

    paste_file_btn.click(fn=read_clipboard_path, outputs=manual_file_input)
    paste_url_btn.click(fn=read_clipboard_text, outputs=url_input)

    process_file_btn.click(
        fn=process_file_wrapper, 
        inputs=[manual_file_input, upload_text_file, remove_ru_cb], 
        outputs=[ab_path, load_project_dropdown, upload_progress_html, upload_status_text]
    ).then(toggle_tab_parse, inputs=ab_path, outputs=[inner_tabs, ab_state]
    ).then(get_all_projects_xml, inputs=ab_path, outputs=parse_df_output)
    
    url_btn.click(
        fn=process_url_wrapper, 
        inputs=[url_input, remove_ru_cb], 
        outputs=[ab_path, load_project_dropdown]
    ).then(toggle_tab, inputs=ab_path, outputs=[inner_tabs, ab_state])
    
    rm_dataset.click(remove_dataset, inputs=ab_path, outputs=[ab_path, inner_tabs])
    rm_all_dataset.click(remove_all_datasets, outputs=[ab_path, inner_tabs])
    ab_path.change(toggle_tab, inputs=ab_path, outputs=[inner_tabs, ab_state])
    
    show_rename_btn.click(lambda: gr.update(visible=True), outputs=rename_proj_panel)
    confirm_rename_btn.click(
        rename_project_folder,
        inputs=[ab_path, new_proj_name],
        outputs=[ab_path, load_project_dropdown, rename_proj_panel]
    )

    load_project_dropdown.change(
        fn=lambda name: (*load_existing_fb2(name), f"📂 Loaded: {name}" if name else ""),
        inputs=[load_project_dropdown],
        outputs=[fb2_title_input, fb2_text_input, editor_status]
    )
    clean_srt_btn.click(
        fn=lambda t: (clean_srt_timings(t), "✂️ SRT timecodes removed"),
        inputs=fb2_text_input,
        outputs=[fb2_text_input, editor_status]
    )
    fb2_generate_btn.click(
        fn=lambda r, b: (*create_fb2_file(r, b), gr.update(choices=sorted(get_data_list())), f"✨ Project '{b}' created and saved!"),
        inputs=[fb2_text_input, fb2_title_input],
        outputs=[fb2_file_output, ab_path, load_project_dropdown, editor_status]
    )
    fb2_update_btn.click(
        fn=lambda r, b: (*update_existing_fb2(r, b), f"🔄 Project '{b}' overwritten!"),
        inputs=[fb2_text_input, fb2_title_input],
        outputs=[fb2_file_output, ab_path, load_project_dropdown, editor_status]
    )
    fb2_delete_btn.click(
        fn=lambda b: (*delete_created_file(b), f"🗑 Project deleted" if b else "⚠️ Nothing to delete"),
        inputs=[fb2_title_input],
        outputs=[fb2_file_output, ab_path, editor_status]
    )

    accent_button.click(put_accents, inputs=text_input, outputs=text_input)
    tts_button.click(text_to_audio, inputs=[text_input, spk_sel, speech_rate, noise_lvl, pitch_sel, ref_audio, ref_text], outputs=audio_output)
    tts_state.change(change_tts_model, inputs=tts_state, outputs=[spk_sel, fb2tts_tab, tts_button, f5_gr])
    acc_state.change(change_acc_model, inputs=acc_state, outputs=accent_button)

    clean_tmp_btn.click(fn=clean_tmp_folder, outputs=tmp_status)
    
    # Delete model handlers — reuse the same function with confirmation
    del_model_btn.click(
        fn=delete_selected_model,
        inputs=[model_to_del, confirm_del_cb, tts_state, acc_state],
        outputs=[model_to_del, model_del_status, disk_space_display, voice_model_sel, update_del_model]
    )
    
    # Duplicate delete button inside the Update section (reuses same handler)
    update_del_btn.click(
        fn=delete_selected_model,
        inputs=[update_del_model, update_del_confirm_cb, tts_state, acc_state],
        outputs=[update_del_model, update_del_status, disk_space_display, voice_model_sel, model_to_del]
    )
    update_one_model_btn.click(fn=lambda mid: update_voice_model(mid)[0], inputs=voice_model_sel, outputs=voice_model_status)
    update_all_models_btn.click(fn=update_all_voice_models, outputs=voice_model_status)
    check_models_btn.click(fn=check_all_voice_models, outputs=voice_model_status)
    stop_model_btn.click(fn=stop_model_update, outputs=voice_model_status, queue=False)
    # Refresh disk info and model lists when opening the System tab
    def _refresh_system_tab():
        models = get_installed_models()
        voices = get_voice_models_choices()
        return (quick_check_models_local(), get_models_disk_info()[0], models, models, voices)
    system_tab.select(
        fn=_refresh_system_tab,
        outputs=[voice_model_status, disk_space_display, model_to_del, update_del_model, voice_model_sel]
    )

def _check_port_lecta(port: int) -> bool:
    """Return True if *port* is free; otherwise log the owner and decide."""
    if os.name != "nt":
        return True
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue).OwningProcess"],
            capture_output=True, text=True, timeout=5,
            creationflags=0x08000000,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return True
        for line in result.stdout.strip().splitlines():
            pid_str = line.strip()
            if pid_str and pid_str.isdigit():
                pid = int(pid_str)
                try:
                    info = subprocess.run(
                        ["powershell", "-NoProfile", "-Command",
                         f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue | Select-Object ProcessName -ExpandProperty ProcessName)"],
                        capture_output=True, text=True, timeout=5,
                        creationflags=0x08000000,
                    )
                    image = info.stdout.strip() or "unknown"
                except Exception:
                    image = "unknown"
                logger.info("[PORT] %d busy, owner PID %d (%s)", port, pid, image)
                # If it's an old LECTA instance, kill it
                if image.lower() in ("python.exe", "pythonw.exe"):
                    try:
                        cmdline_result = subprocess.run(
                            ["powershell", "-NoProfile", "-Command",
                             f"(Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\").CommandLine"],
                            capture_output=True, text=True, timeout=5,
                            creationflags=0x08000000,
                        )
                        cmdline = cmdline_result.stdout.strip()
                        if "app.py" in cmdline or "fb2tts" in cmdline:
                            logger.info("[PORT] Killing old LECTA instance (PID %d)", pid)
                            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                                           capture_output=True, timeout=10,
                                           creationflags=0x08000000)
                    except Exception:
                        pass
                else:
                    logger.error("[PORT] %d held by foreign process %s (PID %d). Stop it or use LECTA_PORT env var.",
                                  port, image, pid)
                    return False
    except Exception as e:
        logger.debug("[PORT] check skipped: %s", e)
    return True

if __name__ == "__main__":
    sound_dir = CURRENT_DIR / "sound"
    if not data_path.exists(): data_path.mkdir(parents=True, exist_ok=True)
    port = app_config.port if app_config.port else 7860

    if not _check_port_lecta(port):
        print(f"[PORT] {port} is held by another application. Set LECTA_PORT env var to use a different port.")
        sys.exit(1)

    threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()

    launch_kwargs = {
        "server_name": "127.0.0.1",
        "server_port": port,
        "share": app_config.share,
        "debug": app_config.debug,
        "inbrowser": False,
        "show_api": False,
        "allowed_paths": [str(sound_dir), str(data_path), str(CURRENT_DIR / "libs")],
        "css": custom_css,
        "head": custom_head,
        "theme": gr.themes.Soft(
            primary_hue="sky",
            neutral_hue="slate",
            radius_size=gr.themes.sizes.radius_lg,
            spacing_size=gr.themes.sizes.spacing_md,
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        ),
    }

    favicon = CURRENT_DIR / "libs" / "favicon" / "lecta.ico"
    if favicon.is_file():
        launch_kwargs["favicon_path"] = str(favicon)

    # Gradio 6 removed/moved several launch() parameters (show_api, favicon_path,
    # analytics_enabled...). Keep only what the installed version really accepts
    # instead of guessing from the docs.
    allowed = set(inspect.signature(App.launch).parameters)
    dropped = sorted(k for k in launch_kwargs if k not in allowed)
    if dropped:
        logger.info("[LAUNCH] Gradio %s ignores: %s", gr.__version__, ", ".join(dropped))
    launch_kwargs = {k: v for k, v in launch_kwargs.items() if k in allowed}

    logger.info("[LAUNCH] Starting on http://127.0.0.1:%d", port)
    try:
        App.queue().launch(**launch_kwargs)
    except OSError as e:
        logger.error("[PORT] Could not bind %d: %s", port, e)
        print(f"[PORT] {port} is not available. Free it, or set LECTA_PORT env var to use a different port.")
        sys.exit(1)