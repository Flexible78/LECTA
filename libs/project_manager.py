# libs/project_manager.py
import shutil
import logging
import gradio as gr
from libs.utils import data_path, get_data_list

logger = logging.getLogger(__name__)

def refresh_data(ab_name):
    if not data_path.exists(): data_path.mkdir(parents=True, exist_ok=True)
    return {"value": ab_name, "choices": sorted(get_data_list()), "__type__": "update"}

def remove_dataset(ab_name):
    if not ab_name: return {"value": '', "choices": sorted(get_data_list()), "__type__": "update"}, {"visible": False, "__type__": "update"}
    ab_path = data_path / ab_name
    try:
        if ab_path.exists(): shutil.rmtree(ab_path)
        gr.Info(f'Project {ab_name} deleted.')
        logger.info(f'🗑 Project fully removed: {ab_name}')
    except Exception as e: logger.error(f"Deletion error: {e}")
    if not data_path.exists(): data_path.mkdir(parents=True, exist_ok=True)
    return {"value": '', "choices": sorted(get_data_list()), "__type__": "update"}, {"visible": False, "__type__": "update"}

def remove_all_datasets():
    try:
        for item in data_path.iterdir():
            if item.is_dir(): shutil.rmtree(item)
        gr.Info('ALL projects removed from the data folder.')
        logger.info('💣 ALL projects removed from the data folder.')
    except Exception as e: logger.error(f"Deletion error: {e}")
    if not data_path.exists(): data_path.mkdir(parents=True, exist_ok=True)
    return {"value": '', "choices": sorted(get_data_list()), "__type__": "update"}, {"visible": False, "__type__": "update"}

def create_fb2_file(raw_text, book_title):
    if not raw_text or not raw_text.strip(): return None, gr.update()
    safe_title = book_title if book_title.strip() else "Audiobook"
    clean_text = raw_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    paragraphs = clean_text.split('\n')
    sections_xml = ""
    current_paras = []
    
    for p in paragraphs:
        if p.strip(): current_paras.append(f"<p>{p.strip()}</p>")
        if len(current_paras) >= 9999:
            sections_xml += "<section>\n" + "\n".join(current_paras) + "\n</section>\n"
            current_paras = []
            
    if current_paras: sections_xml += "<section>\n" + "\n".join(current_paras) + "\n</section>\n"
    
    safe_name = "".join([c for c in safe_title if c.isalnum() or c in (' ', '_')]).rstrip()
    fb2_template = f"<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<FictionBook xmlns=\"http://www.gribuser.ru/xml/fictionbook/2.0\">\n  <description><title-info><book-title>{safe_title}</book-title></title-info></description>\n  <body>{sections_xml}</body>\n</FictionBook>"

    ab_path_dir = data_path / safe_name
    ab_path_dir.mkdir(parents=True, exist_ok=True)
    cover_path = ab_path_dir / "cover.jpg"
    if not cover_path.exists():
        try:
            from PIL import Image
            Image.new('RGB', (300, 300), color=(50, 50, 50)).save(cover_path, format='JPEG')
        except ImportError: pass

    destination_path = ab_path_dir / f"{safe_name}.fb2"
    destination_path.write_text(fb2_template, encoding="utf-8")
    gr.Info(f"Project '{safe_name}' saved!", duration=4)
    return str(destination_path), refresh_data(safe_name)

def update_existing_fb2(raw_text, book_title):
    if not raw_text or not raw_text.strip(): return None, gr.update(), gr.update()
    safe_title = book_title if book_title.strip() else "Audiobook"
    safe_name = "".join([c for c in safe_title if c.isalnum() or c in (' ', '_')]).rstrip()
    ab_path_dir = data_path / safe_name
    if ab_path_dir.exists(): shutil.rmtree(ab_path_dir)
    file_path, upd = create_fb2_file(raw_text, book_title)
    gr.Info(f"Project '{safe_name}' overwritten!", duration=4)
    return file_path, upd, gr.update(choices=sorted(get_data_list()))

def delete_created_file(book_title):
    if not book_title: return None, refresh_data("")
    safe_name = "".join([c for c in book_title if c.isalnum() or c in (' ', '_')]).rstrip() or "Audiobook"
    ab_path_dir = data_path / safe_name
    if ab_path_dir.exists():
        shutil.rmtree(ab_path_dir)
        gr.Info(f"Project '{safe_name}' deleted!", duration=4)
        logger.info(f"🗑 Project deleted via Editor: {safe_name}")
    return None, refresh_data("")