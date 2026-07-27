import logging
import gradio as gr
from pathlib import Path
from typing import Tuple
import re
import time
import sys
from libs.fb2_processor import FB2Processor
from libs.utils import data_path, get_data_list, now_dir
from libs.ui_assets import get_parse_metrics_html
from config import AppConfig, config

processor: FB2Processor = None
logger = logging.getLogger(__name__)

def _play_done_sound():
    """Проигрывает звуковой сигнал по завершению парсинга."""
    try:
        if sys.platform == 'win32':
            import winsound
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        else:
            print('\a', end='', flush=True)
    except Exception:
        pass

def get_xml_files(ab_path):
    if not ab_path or str(ab_path).startswith('<gradio'):
        return []
    xml_dir = data_path / str(ab_path) / "xml"
    if not xml_dir.exists():
        return []
    # Безопасная сортировка: сначала числовые имена, потом строковые
    def safe_sort(x):
        try:
            return (0, float(x))
        except ValueError:
            return (1, x.lower())
    return [[f] for f in get_data_list(xml_dir, "*.xml", sort=safe_sort)]

def get_all_projects_xml(ab_path=None):
    """Собирает XML-файлы из ВСЕХ проектов в data/. 
    Возвращает одну колонку в формате: 'Проект / файл.xml'.
    Используется после пакетной загрузки."""
    rows = []
    if not data_path.exists():
        return rows
    def safe_sort(x):
        try: return (0, float(x))
        except ValueError: return (1, x.lower())
    for project_dir in sorted(data_path.iterdir()):
        if not project_dir.is_dir():
            continue
        xml_dir = project_dir / "xml"
        if not xml_dir.exists():
            continue
        for f in get_data_list(xml_dir, "*.xml", sort=safe_sort):
            rows.append([f"{project_dir.name} / {f}.xml"])
    return rows if rows else [["⚠️ No XML files in any project"]]

def show_file_content(data: gr.SelectData, ab_path: str):
    if not ab_path or str(ab_path).startswith('<gradio'): 
        return "", gr.update(interactive=False), ""
    value = str(data.value)
    # Поддержка формата "проект / файл.xml" (из get_all_projects_xml)
    if " / " in value:
        proj_name, file_name = value.split(" / ", 1)
        xml_dir = data_path / proj_name / "xml"
    else:
        xml_dir = data_path / str(ab_path) / "xml"
        file_name = value
    file_path = xml_dir / f"{file_name}"
    if not file_path.suffix:
        file_path = xml_dir / f"{file_name}.xml"
    try:
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            return content, gr.update(interactive=True), str(file_path)
        return f"File not found: {file_name}", gr.update(interactive=False), ""
    except Exception as e:
        return f"Error: {str(e)}", gr.update(interactive=False), ""

def del_file(filename: str, ab_path: str):
    if not filename or not ab_path: 
        return get_all_projects_xml(), ""
    file_path = Path(filename)
    if file_path.exists():
        file_path.unlink()
        logger.info(f"🗑 Deleted XML file: {file_path.name}")
        gr.Info(f"Deleted: {file_path.name}", duration=2)
    return get_all_projects_xml(), ""

def magic_clean_xml(content, ab_path, cur_file):
    if not content or not cur_file: 
        return content, "⚠️ Select a file in the table on the left first!"
        
    clean_album = re.sub(r'[\W_]*\d{6,}[\W_]*\d*$', '', ab_path)
    if len(clean_album) > 35: 
        clean_album = clean_album[:35] + "..."
    if not clean_album: 
        clean_album = ab_path[:20]

    content = re.sub(r'<\?xml[^>]*\?>\s*', '', content)
    content = re.sub(r'(<(?:speak|root)[^>]*album=")[^"]*("?[^>]*>)', rf'\g<1>{clean_album}\g<2>', content)
    content = re.sub(r'(<(?:speak|root)[^>]*title=")[^"]*("?[^>]*>)', rf'\g<1>{clean_album}\g<2>', content)
    
    def clean_head(match):
        speak_tag = match.group(1)
        p1_text = match.group(2)
        if len(p1_text) < 150:
            return f'{speak_tag}\n  <break time="30"/>\n'
        else:
            return f'{speak_tag}\n  <break time="30"/>\n  <p>{p1_text}</p>\n'
            
    content = re.sub(r'(<(?:speak|root)[^>]*>)[\s\S]*?<p>(.*?)</p>\s*', clean_head, content, count=1)
    content = re.sub(r'\s*<p>\s*[a-zA-Z0-9_]+_csv_[^<]*</p>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\s*<p>[^<]*(ru en he|ru en|en ru|he en)[^<]*</p>', '', content, flags=re.IGNORECASE)
    
    try:
        Path(cur_file).write_text(content, encoding='utf-8')
        gr.Info("✨ Cleanup: no junk, no empty lines!", duration=3)
        return content, "✨ Cleaned and SAVED!"
    except Exception as e:
        return content, f"❌ Save error: {e}"

def parse_fb2_wrapper(
    ab_path: str, replace: bool, ch_size: int, gender: bool, sound_effect: bool,
    accent: bool, single_vowel: bool, punctuation: bool, translit: bool, profanity: bool,
    remove_ru: bool, is_english: bool, translate: bool, auto_clean_xml: bool
):
    if not ab_path or str(ab_path).startswith('<gradio'):
        gr.Warning("Please select a project from the list above!")
        yield [], get_parse_metrics_html(0, "Error"), "❌ Error: No project selected"
        return

    AppConfig.save_user_settings({
        "gender": gender, "punctuation": punctuation, "translit": translit,
        "sound_effect": sound_effect, "ch_size": ch_size, "profanity": profanity, "single_vowel": single_vowel
    })
    
    global processor
    processor = FB2Processor(accent=accent, single_vowel=single_vowel)
    
    xml_dir = data_path / str(ab_path) / "xml"
    xml_dir.mkdir(parents=True, exist_ok=True)

    yield get_xml_files(ab_path), get_parse_metrics_html(0, "Initializing..."), "Initializing..."

    try:
        # Процессор теперь возвращает процент и сообщение!
        for percent, status_msg in processor.process_book(
            ab_path=ab_path, replace=replace, sound_effect=sound_effect, punctuation=punctuation,
            translit=translit, ch_size=ch_size, remove_ru=remove_ru, is_english=is_english, translate=translate
        ):
            yield get_xml_files(ab_path), get_parse_metrics_html(percent, status_msg), status_msg
            
        if auto_clean_xml:
            yield get_xml_files(ab_path), get_parse_metrics_html(95, "Cleaning XML..."), "✨ Auto-cleanup: removing junk..."
            clean_album = re.sub(r'[\W_]*\d{6,}[\W_]*\d*$', '', ab_path)
            if len(clean_album) > 35: clean_album = clean_album[:35] + "..."
            if not clean_album: clean_album = ab_path[:20]

            for xml_file in xml_dir.glob("*.xml"):
                content = xml_file.read_text(encoding="utf-8")
                content = re.sub(r'<\?xml[^>]*\?>\s*', '', content)
                content = re.sub(r'(<(?:speak|root)[^>]*album=")[^"]*("?[^>]*>)', rf'\g<1>{clean_album}\g<2>', content)
                content = re.sub(r'(<(?:speak|root)[^>]*title=")[^"]*("?[^>]*>)', rf'\g<1>{clean_album}\g<2>', content)
                
                def clean_head(match):
                    speak_tag = match.group(1); p1_text = match.group(2)
                    if len(p1_text) < 150: return f'{speak_tag}\n  <break time="30"/>\n'
                    else: return f'{speak_tag}\n  <break time="30"/>\n  <p>{p1_text}</p>\n'
                        
                content = re.sub(r'(<(?:speak|root)[^>]*>)[\s\S]*?<p>(.*?)</p>\s*', clean_head, content, count=1)
                content = re.sub(r'\s*<p>\s*[a-zA-Z0-9_]+_csv_[^<]*</p>', '', content, flags=re.IGNORECASE)
                content = re.sub(r'\s*<p>[^<]*(ru en he|ru en|en ru|he en)[^<]*</p>', '', content, flags=re.IGNORECASE)
                
                xml_file.write_text(content, encoding='utf-8')

        msg = "🛑 Stopped by user" if processor.stop_parsing else "✅ Completed"
        yield get_xml_files(ab_path), get_parse_metrics_html(100, msg), msg
        if not processor.stop_parsing:
            _play_done_sound()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        yield get_xml_files(ab_path), get_parse_metrics_html(0, "Error"), f"❌ Error: {e}"

def stop_parse():
    if processor: 
        processor.stop_parse()
    return get_parse_metrics_html(100, "Interrupting..."), "🛑 Stopping..."


# ═══════════════════════════════════════════════════════════════════
# BF11 — Multi-replace pattern engine for the ANALYZE editor
# ═══════════════════════════════════════════════════════════════════

def apply_patterns(content: str, patterns_text: str, whole_line: bool, use_regex: bool):
    """Apply find-and-remove/replace patterns to text.
    
    Patterns are one per line. Syntax:
      - "find"          → delete all occurrences
      - "find => replace" → replace all occurrences
    
    Args:
        content: The text to process.
        patterns_text: Newline-separated list of patterns.
        whole_line: If True, remove entire lines containing the pattern.
        use_regex: If True, treat patterns as regex; otherwise plain text.
    
    Returns:
        (cleaned_content, report_string)
    """
    if not content:
        return content, "⚠️ No content to process"
    if not patterns_text or not patterns_text.strip():
        return content, "⚠️ No patterns provided"
    
    lines = [l for l in patterns_text.strip().split('\n') if l.strip()]
    report_lines = []
    total_fragments = 0
    total_chars = 0
    pattern_count = 0
    
    for line in lines:
        line = line.strip()
        
        # Parse: "find => replace" or just "find" (delete)
        if '=>' in line:
            parts = line.split('=>', 1)
            find_str = parts[0].strip()
            replace_str = parts[1].strip()
        else:
            find_str = line
            replace_str = ''
        
        if not find_str:
            continue
        
        pattern_count += 1
        
        try:
            if use_regex:
                if whole_line:
                    pattern = re.compile(r'^.*' + find_str + r'.*\n?', re.MULTILINE)
                else:
                    pattern = re.compile(find_str)
                count = len(pattern.findall(content))
                new_content = pattern.sub(replace_str, content)
            else:
                if whole_line:
                    escaped = re.escape(find_str)
                    pattern = re.compile(r'^.*' + escaped + r'.*\n?', re.MULTILINE)
                    count = len(pattern.findall(content))
                    new_content = pattern.sub(replace_str + ('\n' if replace_str else ''), content)
                else:
                    count = content.count(find_str)
                    new_content = content.replace(find_str, replace_str)
            
            chars_removed = len(content) - len(new_content)
            content = new_content
            
            display = find_str if len(find_str) <= 60 else find_str[:57] + '...'
            if replace_str:
                r_display = replace_str if len(replace_str) <= 30 else replace_str[:27] + '...'
                report_lines.append(f'"{display}" => "{r_display}" — {count} replaced')
            else:
                report_lines.append(f'"{display}" — {count} removed')
            
            total_fragments += count
            total_chars += chars_removed
        except re.error as e:
            report_lines.append(f'❌ Bad regex: "{find_str[:60]}" — {e}')
    
    if report_lines:
        report_lines.append('')
    report_lines.append(f'Total: {pattern_count} patterns, {total_fragments} fragments, {total_chars:,} characters removed')
    report_lines.append('Not saved yet — press Manual save (Ctrl+S)')
    
    return content, '\n'.join(report_lines)


def _add_pattern(patterns_text: str, selection: str):
    """Append a selected fragment to the patterns list."""
    if not selection or not selection.strip():
        return patterns_text, "⚠️ Nothing selected"
    sel = selection.strip()
    if patterns_text and patterns_text.strip():
        new_text = patterns_text.strip() + '\n' + sel
    else:
        new_text = sel
    return new_text, f"📋 Added pattern: \"{sel[:60]}{'...' if len(sel) > 60 else ''}\""


def _count_patterns(content: str, patterns_text: str, whole_line: bool, use_regex: bool):
    """Count matches only, without modifying content."""
    if not content:
        return content, "⚠️ No content to process"
    if not patterns_text or not patterns_text.strip():
        return content, "⚠️ No patterns provided"
    
    lines = [l for l in patterns_text.strip().split('\n') if l.strip()]
    report_lines = []
    total_fragments = 0
    
    for line in lines:
        line = line.strip()
        if '=>' in line:
            find_str = line.split('=>', 1)[0].strip()
        else:
            find_str = line
        if not find_str:
            continue
        
        try:
            if use_regex:
                if whole_line:
                    pattern = re.compile(r'^.*' + find_str + r'.*$', re.MULTILINE)
                else:
                    pattern = re.compile(find_str)
                count = len(pattern.findall(content))
            else:
                if whole_line:
                    escaped = re.escape(find_str)
                    pattern = re.compile(r'^.*' + escaped + r'.*$', re.MULTILINE)
                    count = len(pattern.findall(content))
                else:
                    count = content.count(find_str)
            
            display = find_str if len(find_str) <= 60 else find_str[:57] + '...'
            report_lines.append(f'"{display}" — {count} matches')
            total_fragments += count
        except re.error as e:
            report_lines.append(f'❌ Bad regex: "{find_str[:60]}" — {e}')
    
    if report_lines:
        report_lines.append('')
    report_lines.append(f'Total: {total_fragments} matches across {len(lines)} patterns (no changes made)')
    
    return content, '\n'.join(report_lines)

def parse_tab(ab_path, acc_state, tts_state):
    with gr.Tab("🔍 ANALYZE") as pr_tab:
        gr.Markdown("After changing the TTS model, you **must** re-process the fb2.")
        
        with gr.Row():
            with gr.Column(scale=2):
                sound_effect = gr.Checkbox(label="Voice events", value=False)
                single_vowel = gr.Checkbox(label="Moskvich", value=False)
                remove_ru = gr.Checkbox(label="🚫 Remove Russian text", value=False)
                is_english = gr.Checkbox(label="🇬🇧 English text (no number russification)", value=False)
                gender = gr.Checkbox(label="Gender detection", interactive=False, value=False, visible=False)
                profanity = gr.Checkbox(label="Bleep profanity", interactive=False, value=False, visible=False)
            
            with gr.Column(scale=2):
                translit = gr.Checkbox(label="Translit", value=False)
                translate = gr.Checkbox(label="🌐 Translate to Russian", value=False)
                accent = gr.Checkbox(label="Add stress marks", interactive=False, value=False)
                repl = gr.Checkbox(label="Overwrite old files", value=True)
                auto_clean = gr.Checkbox(label="✨ Auto-clean XML (Junk + Name + 3s intro)", value=True)
                bilingual = gr.Checkbox(label="Multilingual", interactive=False, value=False, visible=False)
        
        with gr.Row():
            with gr.Column(scale=4):
                punctuation = gr.Checkbox(label="Remove punctuation", value=False)
                ch_size = gr.Slider(50, 400, 400, step=10, label="Line length", interactive=True)
            
            with gr.Column(scale=1):
                parse_btn = gr.Button("▶ Process text (Ctrl+Enter)", variant="primary", elem_id="parse_btn")
                stop_btn = gr.Button("🚫 Stop (Esc)")
        
        with gr.Row():
            metrics_panel = gr.HTML(value=get_parse_metrics_html(0, "Waiting..."))
            
        with gr.Group(elem_id="log_group"):
            status = gr.Textbox(label="Live log", show_label=True, lines=1, interactive=False)
        
        with gr.Row():
            with gr.Column(scale=1, min_width=150):
                refresh_btn = gr.Button("🔄 Refresh XML list")
                df_output = gr.DataFrame(headers=['File name'], value=[], interactive=False, max_height=720, type='array')
                del_btn = gr.Button("❌ Delete file (Delete)", interactive=False, elem_id="del_file_btn")
            
            with gr.Column(scale=5, min_width=500):
                file_content = gr.Textbox(label="File content (with word wrap)", interactive=True, lines=25, max_lines=40, elem_id="file_content")
                
                with gr.Row():
                    save_btn = gr.Button("📝 Manual save (Ctrl+S)", elem_id="save_xml_btn")
                    magic_clean_btn = gr.Button("✨ Manual Auto-Clean", variant="secondary")
                
                # ── BF11: Find & remove repeated fragments ──
                undo_stack = gr.State([])
                with gr.Accordion("🧹 Find & remove repeated fragments", open=False):
                    with gr.Row():
                        copy_selection_btn = gr.Button("📋 Copy selection", variant="secondary", scale=1)
                    patterns_input = gr.Textbox(
                        label="Patterns — one per line",
                        lines=6,
                        placeholder="<p>garbage</p>\nbad word => good word\nPEREVOD.",
                        info="Syntax: find => replace. Without => the pattern is deleted.",
                        elem_id="patterns_input"
                    )
                    with gr.Row():
                        whole_line_cb = gr.Checkbox(label="Whole line", value=False)
                        regex_cb = gr.Checkbox(label="Regex", value=False)
                    with gr.Row():
                        count_btn = gr.Button("🔎 Count", scale=1)
                        remove_all_btn = gr.Button("🧹 Remove all", variant="stop", scale=1)
                        undo_btn = gr.Button("↩ Undo", scale=1)
                    cleanup_report = gr.Textbox(label="Cleanup report", lines=4, interactive=False)
                
                cur_file = gr.State()

    parse_btn.click(
        fn=lambda: (gr.update(value="⏳ Initializing..."), get_parse_metrics_html(0, "Starting...")), 
        outputs=[status, metrics_panel]
    ).then(
        fn=parse_fb2_wrapper,
        inputs=[ab_path, repl, ch_size, gender, sound_effect, accent, single_vowel, punctuation, translit, profanity, remove_ru, is_english, translate, auto_clean],
        outputs=[df_output, metrics_panel, status]
    ).then(
        fn=get_all_projects_xml, outputs=df_output
    )
    refresh_btn.click(fn=get_all_projects_xml, outputs=df_output)
    
    # VIP QUEUE: Мгновенное прерывание с сохранением!
    stop_btn.click(stop_parse, outputs=[metrics_panel, status], queue=False)
    
    df_output.select(fn=show_file_content, inputs=ab_path, outputs=[file_content, del_btn, cur_file])
    save_btn.click(fn=lambda f_c, c_f: FB2Processor.save_xml(f_c, c_f), inputs=[file_content, cur_file], outputs=status)
    magic_clean_btn.click(fn=magic_clean_xml, inputs=[file_content, ab_path, cur_file], outputs=[file_content, status])
    del_btn.click(fn=del_file, inputs=[cur_file, ab_path], outputs=[df_output, file_content])

    # ── BF11: Find & remove repeated fragments handlers ──
    _copy_sel_js = """
    function() {
        var ta = document.querySelector('#file_content textarea');
        if (!ta) return;
        var sel = ta.value.substring(ta.selectionStart, ta.selectionEnd);
        if (!sel) return;
        // Copy to clipboard (multiple fallback strategies)
        try { navigator.clipboard.writeText(sel); } catch(e) { try { document.execCommand('copy'); } catch(e2) {} }
        // Append selection to the patterns textarea via DOM
        var pta = document.querySelector('#patterns_input textarea');
        if (pta) {
            pta.value = (pta.value ? pta.value + '\\n' : '') + sel;
            pta.dispatchEvent(new Event('input', { bubbles: true }));
        }
    }
    """
    copy_selection_btn.click(
        fn=None,
        inputs=[],
        outputs=[],
        js=_copy_sel_js
    )
    count_btn.click(
        fn=_count_patterns,
        inputs=[file_content, patterns_input, whole_line_cb, regex_cb],
        outputs=[file_content, cleanup_report]
    )
    
    def _remove_wrapper(content, patterns, whole_line, use_regex, undo_list):
        """Apply removals and push current content to undo stack."""
        undo_list = (undo_list or []).copy()
        undo_list.append(content)
        if len(undo_list) > 10:
            undo_list = undo_list[-10:]
        new_content, report = apply_patterns(content, patterns, whole_line, use_regex)
        return new_content, report, undo_list
    
    remove_all_btn.click(
        fn=_remove_wrapper,
        inputs=[file_content, patterns_input, whole_line_cb, regex_cb, undo_stack],
        outputs=[file_content, cleanup_report, undo_stack]
    )
    
    def _undo_wrapper(undo_list):
        if not undo_list:
            return gr.update(), "⚠️ Nothing to undo", undo_list
        prev = undo_list[-1]
        new_list = undo_list[:-1]
        return prev, f"↩ Undone — restored {len(prev):,} characters", new_list
    
    undo_btn.click(
        fn=_undo_wrapper,
        inputs=[undo_stack],
        outputs=[file_content, cleanup_report, undo_stack]
    )

    pr_tab.select(fn=get_all_projects_xml, outputs=df_output)
    acc_state.change(fn=lambda: gr.update(interactive=True, value=False), outputs=accent)
    return file_content, df_output