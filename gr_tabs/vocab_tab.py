import gradio as gr
import re
import logging
import time
from pathlib import Path
from deep_translator import GoogleTranslator
from libs.utils import data_path
from libs.ui_assets import format_time_hms, get_vocab_metrics_html
import csv
import json

logger = logging.getLogger(__name__)
stop_vocab = False

def stop_vocab_fn():
    global stop_vocab
    stop_vocab = True
    logger.info("🛑 Vocabulary parser stop signal...")

def clean_api_errors(text):
    """Очищает готовый словарь от строк с ошибками 'HTTPSConnectionPool'"""
    if not text: return text
    lines = text.split('\n')
    clean_lines = [line for line in lines if "Translation error: HTTPSConnectionPool" not in line and "[Translation error" not in line]
    return "\n".join(clean_lines)

def save_vocab_project(content, current_project):
    if not content or "⚠️" in content or "Starting" in content:
        return "⚠️ No data to save.", []
        
    try:
        if current_project and not str(current_project).startswith('<gradio'):
            proj_dir = data_path / str(current_project)
            xml_dir = proj_dir / "xml"
            project_name = current_project
        else:
            project_name = f"Vocab_{time.strftime('%Y%m%d_%H%M%S')}"
            proj_dir = data_path / project_name
            xml_dir = proj_dir / "xml"
            
        xml_dir.mkdir(parents=True, exist_ok=True)
        
        max_num = 0
        for f in xml_dir.glob("*.xml"):
            try:
                num = float(f.stem)
                if num > max_num: max_num = int(num)
            except ValueError: pass
        vocab_filename = str(max_num + 1)
        
        files_created = []
        
        # 1. Raw text (.txt)
        txt_file = proj_dir / f"Vocab_raw_{vocab_filename}.txt"
        txt_file.write_text(content, encoding='utf-8')
        files_created.append(str(txt_file))
        
        # 2. XML (for TTS)
        xml_content = f'<speak autor="Dictionary" album="{project_name}">\n  <break time="30"/>\n'
        for line in content.split('\n'):
            if line.strip() and not line.startswith('[🛑'):
                clean_line = line.replace('\t', ' , ')
                xml_content += f'  <p>{clean_line}</p>\n'
        xml_content += '</speak>'
        xml_file = xml_dir / f"{vocab_filename}.xml"
        xml_file.write_text(xml_content, encoding='utf-8')
        files_created.append(str(xml_file))
        
        # Prepare data for structured formats (JSON, CSV, MD)
        struct_data = []
        for line in content.split('\n'):
            if line.strip() and not line.startswith('[🛑'):
                parts = line.split('\t')
                if len(parts) >= 2:
                    struct_data.append({"Word": parts[0].strip(), "Translation": parts[1].strip()})
        
        if struct_data:
            # 3. JSON
            json_file = proj_dir / f"Vocab_{vocab_filename}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(struct_data, f, ensure_ascii=False, indent=4)
            files_created.append(str(json_file))
            
            # 4. CSV (UTF-8-SIG for Excel)
            csv_file = proj_dir / f"Vocab_{vocab_filename}.csv"
            with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["Word", "Translation"])
                writer.writeheader()
                writer.writerows(struct_data)
            files_created.append(str(csv_file))
            
            # 5. MD (Markdown)
            md_file = proj_dir / f"Vocab_{vocab_filename}.md"
            md_content = f"# Vocabulary project: {project_name}\n\n| Source | Translation |\n|---|---|\n"
            for row in struct_data: md_content += f"| {row['Word']} | {row['Translation']} |\n"
            md_file.write_text(md_content, encoding='utf-8')
            files_created.append(str(md_file))
        
        return f"✅ Saved files ({len(files_created)} items) to project '{project_name}'", files_created
    except Exception as e:
        return f"❌ Save error: {e}", []

def extract_and_translate(text, src_langs_ui, tgt_langs_ui, min_length, include_context, tts_format, current_project):
    global stop_vocab
    stop_vocab = False
    
    if not text.strip():
        yield "⚠️ Paste article text!", get_vocab_metrics_html(0, "00:00", "00:00", "0.0 w/s", "Error"), "⚠️ Empty text", []
        return
        
    if not src_langs_ui or not tgt_langs_ui:
        yield "⚠️ Select languages!", get_vocab_metrics_html(0, "00:00", "00:00", "0.0 w/s", "Error"), "⚠️ No languages selected", []
        return
        
    yield "⏳ Preparing...", get_vocab_metrics_html(0, "00:00", "00:00", "0.0 w/s", "Collecting words..."), "⏳ Collecting words...", []
    
    clean_text = re.sub(r'<[^>]+>', ' ', text)
    clean_sent_text = clean_text.replace('\n', ' ')
    sentences = re.split(r'(?<=[.!?])\s+', clean_sent_text)
    
    lang_codes = {"English (en)": "en", "Hebrew (he)": "iw", "Russian (ru)": "ru"}
    src_langs = [lang_codes[l] for l in src_langs_ui]
    tgt_langs = [lang_codes[l] for l in tgt_langs_ui]
        
    words = []
    if "en" in src_langs: words.extend(re.findall(r'\b[a-zA-Z\']+\b', clean_text.lower()))
    if "iw" in src_langs: words.extend(re.findall(r'\b[א-ת]+\b', clean_text))
    if "ru" in src_langs: words.extend(re.findall(r'\b[а-яА-ЯёЁ]+\b', clean_text.lower()))
        
    unique_words = set()
    for w in words:
        cw = w.strip("'")
        if len(cw) >= min_length: unique_words.add(cw)
            
    sorted_words = sorted(list(unique_words))
    total_words = len(sorted_words)
    
    if not sorted_words:
        yield "⚠️ No words found.", get_vocab_metrics_html(0, "00:00", "00:00", "0.0 w/s", "Error"), "⚠️ No words found", []
        return
        
    translators = {lang: GoogleTranslator(source='auto', target=lang) for lang in tgt_langs}
    
    results = []
    start_time = time.time()
    
    for i, word in enumerate(sorted_words):
        if stop_vocab:
            results.append("\n[🛑 Parsing interrupted by user]")
            break
            
        try:
            trans_words = []
            for t_lang in tgt_langs: trans_words.append(translators[t_lang].translate(word))
                
            example = ""
            trans_examples = []
            if include_context:
                for sent in sentences:
                    if re.search(rf'\b{re.escape(word)}\b', sent, re.IGNORECASE):
                        example = sent.strip()
                        break
                if example:
                    for t_lang in tgt_langs: trans_examples.append(translators[t_lang].translate(example))
            
            if tts_format:
                line = f"{word}\t" + "\t".join(trans_words)
                results.append(line)
                if example:
                    ex_line = f"{example}\t" + "\t".join(trans_examples)
                    results.append(ex_line)
            else:
                trans_str = " | ".join(trans_words)
                res_str = f"🔹 {word} — {trans_str}"
                if example:
                    ex_trans_str = " | ".join(trans_examples)
                    res_str += f"\n   📝 {example}\n   🔄 {ex_trans_str}\n"
                results.append(res_str)
                
        except Exception as e:
            logger.error(f"Translation error: {e}")
            results.append(f"{word} — [Translation error: {e}]")
            
        if i % 3 == 0 or i == total_words - 1:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            rem_sec = (total_words - (i + 1)) / speed if speed > 0 else 0
            pct = int(((i + 1) / total_words) * 100)
            
            html = get_vocab_metrics_html(pct, format_time_hms(elapsed), format_time_hms(rem_sec), f"{speed:.1f} w/s", "Translating...")
            yield "\n".join(results), html, f"⏳ Translated {i+1} of {total_words}", []
            
    # AUTOSAVE
    final_text = "\n".join(results)
    save_msg, saved_files = save_vocab_project(final_text, current_project)
    
    status_prefix = "🛑 Interrupted" if stop_vocab else "✅ Completed"
    pct_final = int(((i + 1) / total_words) * 100) if stop_vocab else 100
    final_html = get_vocab_metrics_html(pct_final, format_time_hms(time.time() - start_time), "00:00", f"{speed:.1f} w/s", status_prefix)
    
    yield final_text, final_html, f"{status_prefix}\n{save_msg}", saved_files

def vocab_tab(ab_path_box=None, file_content_box=None):
    if ab_path_box is None: ab_path_box = gr.State("")
        
    with gr.Tab("📚 Vocabulary Parser") as vocab_tab_elem:
        gr.Markdown("### 🧠 Multi-Extractor for vocabulary words")
        
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(label="Source text", lines=15, max_lines=20)
            with gr.Column(scale=1):
                src_langs = gr.CheckboxGroup(["English (en)", "Hebrew (he)", "Russian (ru)"], label="SOURCE languages", value=["English (en)"])
                tgt_langs = gr.CheckboxGroup(["Russian (ru)", "English (en)", "Hebrew (he)"], label="TARGET languages", value=["Russian (ru)", "Hebrew (he)"])
                min_len = gr.Slider(minimum=1, maximum=15, value=4, step=1, label="Minimum word length")
                
                with gr.Group():
                    include_context = gr.Checkbox(label="📖 Add context (sentences)", value=True)
                    tts_format = gr.Checkbox(label="🎧 TTS format", value=True)
                
                with gr.Row(elem_classes=["uniform-row"]):
                    parse_btn = gr.Button("▶ Start", variant="primary", elem_classes=["fixed-height-btn"])
                    stop_btn = gr.Button("🛑 Stop (Auto-save)", variant="stop", elem_classes=["fixed-height-btn"])
        
        with gr.Row():
            with gr.Column(scale=4):
                metrics_panel = gr.HTML(value=get_vocab_metrics_html(0, "00:00", "00:00", "0.0 w/s"))
                output_text = gr.Textbox(label="Ready dictionary", lines=10, max_lines=15, interactive=True)
            with gr.Column(scale=1):
                save_btn = gr.Button("💾 Manual save (Export)", variant="primary", elem_classes=["fixed-height-btn"])
                clean_errors_btn = gr.Button("🧹 Clean API junk", variant="secondary", elem_classes=["fixed-height-btn"])
                save_status = gr.Textbox(label="Execution status", interactive=False, lines=3)
                download_files = gr.File(label="Download exports", interactive=False)
            
        parse_btn.click(
            fn=lambda: (gr.update(value="⏳ Starting..."), get_vocab_metrics_html(0, "00:00", "00:00", "0.0 w/s", "Preparing")),
            outputs=[output_text, metrics_panel]
        ).then(
            fn=extract_and_translate,
            inputs=[input_text, src_langs, tgt_langs, min_len, include_context, tts_format, ab_path_box],
            outputs=[output_text, metrics_panel, save_status, download_files]
        )
        
        stop_btn.click(fn=stop_vocab_fn, queue=False)
        save_btn.click(fn=save_vocab_project, inputs=[output_text, ab_path_box], outputs=[save_status, download_files])
        clean_errors_btn.click(fn=clean_api_errors, inputs=output_text, outputs=output_text)
        
        if file_content_box is not None:
            vocab_tab_elem.select(fn=lambda x: x, inputs=file_content_box, outputs=input_text)