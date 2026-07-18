import re
import shutil
import logging
from pathlib import Path
from libs.utils import data_path

logger = logging.getLogger(__name__)

def parse_and_save_document(file_path_str, remove_ru):
    dropbox_path = Path(file_path_str)
    ab_name_raw = dropbox_path.stem
    ext = dropbox_path.suffix.lower()
    ab_name = re.sub(r'[^a-zA-Z0-9_\-а-яА-Я]', '', ab_name_raw) or "MyBook"
        
    logger.info(f"📥 Начата обработка локального файла: {dropbox_path.name}")
    
    ab_path = data_path / ab_name
    ab_path.mkdir(parents=True, exist_ok=True)
    
    cover_path = ab_path / "cover.jpg"
    if not cover_path.exists():
        from PIL import Image
        Image.new('RGB', (300, 300), color=(50, 50, 50)).save(cover_path, format='JPEG')
        
    destination_path = ab_path / f"{ab_name}.fb2"
    
    if ext == '.fb2':
        shutil.copy(file_path_str, destination_path)
        logger.info(f"✅ Локальный файл '{ext}' успешно скопирован: {ab_name}")
        return ab_name, None

    raw_text = ""
    if ext in ['.html', '.htm']:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(dropbox_path.read_text(encoding='utf-8', errors='ignore'), 'html.parser')
            raw_text = soup.get_text(separator='\n')
        except ImportError:
            return None, "Установите библиотеку: pip install beautifulsoup4"
    elif ext == '.rtf':
        try:
            from striprtf.striprtf import rtf_to_text
            raw_text = rtf_to_text(dropbox_path.read_text(errors='ignore'))
        except ImportError:
            return None, "Установите библиотеку: pip install striprtf"
    elif ext in ['.txt', '.md']:
        for enc in ['utf-8', 'utf-16', 'cp1251', 'latin-1']:
            try:
                raw_text = dropbox_path.read_text(encoding=enc)
                break
            except UnicodeDecodeError: continue
    elif ext == '.json':
        import json
        try:
            data = json.loads(dropbox_path.read_text(encoding='utf-8'))
            def extract_strings(obj):
                strings = []
                if isinstance(obj, dict):
                    for v in obj.values(): strings.extend(extract_strings(v))
                elif isinstance(obj, list):
                    for item in obj: strings.extend(extract_strings(item))
                elif isinstance(obj, str): strings.append(obj)
                return strings
            raw_text = "\n".join(extract_strings(data))
        except Exception as e:
            logger.error(f"Ошибка чтения JSON: {e}")
            return None, f"Ошибка чтения JSON: {e}"
    elif ext == '.csv':
        import csv
        try:
            with open(file_path_str, 'r', encoding='utf-8', newline='') as f: raw_text = "\n".join([" ".join(row) for row in csv.reader(f)])
        except UnicodeDecodeError:
            with open(file_path_str, 'r', encoding='cp1251', newline='') as f: raw_text = "\n".join([" ".join(row) for row in csv.reader(f)])
    elif ext == '.pdf':
        try:
            import PyPDF2
            with open(file_path_str, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    t = page.extract_text()
                    if t: raw_text += t + "\n"
        except ImportError:
            return None, "Установите библиотеку: pip install PyPDF2"
    elif ext == '.epub':
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
            book = epub.read_epub(file_path_str)
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    raw_text += BeautifulSoup(item.get_content(), 'html.parser').get_text(separator='\n') + "\n"
        except ImportError:
            return None, "Установите библиотеки: pip install EbookLib beautifulsoup4"
    elif ext in ['.docx', '.doc']:
        if ext == '.docx':
            try:
                import docx
                raw_text = "\n".join([p.text for p in docx.Document(file_path_str).paragraphs])
            except ImportError:
                return None, "Установите библиотеку: pip install python-docx"
        else:
            matches = re.findall(r'[А-Яа-яЁё0-9\s.,!?\-a-zA-Z]{4,}', dropbox_path.read_bytes().decode('cp1251', errors='ignore'))
            raw_text = "\n".join(matches)

    if remove_ru:
        raw_text = re.sub(r'[А-Яа-яЁё]+', '', raw_text)
        raw_text = re.sub(r' +', ' ', raw_text)
        raw_text = "\n".join([s.strip() for s in raw_text.splitlines() if s.strip()])

    clean_text = raw_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    paragraphs = clean_text.split('\n')
    CHUNK_SIZE = 9999
    sections_xml = ""
    current_paras = []
    for p in paragraphs:
        if p.strip(): current_paras.append(f"<p>{p.strip()}</p>")
        if len(current_paras) >= CHUNK_SIZE:
            sections_xml += "<section>\n" + "\n".join(current_paras) + "\n</section>\n"
            current_paras = []
    if current_paras:
        sections_xml += "<section>\n" + "\n".join(current_paras) + "\n</section>\n"
    
    fb2_template = f"<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<FictionBook xmlns=\"http://www.gribuser.ru/xml/fictionbook/2.0\">\n  <description><title-info><book-title>{ab_name}</book-title></title-info></description>\n  <body>{sections_xml}</body>\n</FictionBook>"
    destination_path.write_text(fb2_template, encoding="utf-8")

    logger.info(f"✅ Локальный файл '{ext}' успешно сконвертирован в FB2: {ab_name}")
    return ab_name, None