import re
import logging
import requests
from bs4 import BeautifulSoup
from libs.utils import data_path

logger = logging.getLogger(__name__)

def scrape_and_save_article(url, remove_ru):
    if not url or not url.strip():
        return None, "Empty URL"
        
    try:
        logger.info(f"🌐 Started parsing URL: {url.strip()}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url.strip(), headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Вырезаем мусор
        for el in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]):
            el.extract()
            
        main_content = soup.find('article') or soup.find('main') or soup.find('body') or soup
        blocks = main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li'])
        
        if blocks:
            lines = [block.get_text(separator=' ', strip=True) for block in blocks]
        else:
            lines = [line.strip() for line in main_content.get_text(separator='\n').splitlines()]
            
        raw_text = "\n".join([line for line in lines if line])
        
        if remove_ru:
            raw_text = re.sub(r'[А-Яа-яЁё]+', '', raw_text)
            raw_text = re.sub(r' +', ' ', raw_text)
            raw_text = "\n".join([s.strip() for s in raw_text.splitlines() if s.strip()])
        
        if not raw_text.strip():
            return None, "No text found or it was filtered out!"
        
        title = soup.title.string if soup.title else url.split('/')[-1]
        ab_name = re.sub(r'[^a-zA-Z0-9_\-а-яА-Я]', '', title)[:35]
        if not ab_name:
            ab_name = "WebArticle"
        
        ab_path = data_path / ab_name
        ab_path.mkdir(parents=True, exist_ok=True)
        
        cover_path = ab_path / "cover.jpg"
        if not cover_path.exists():
            from PIL import Image
            Image.new('RGB', (300, 300), color=(50, 50, 50)).save(cover_path, format='JPEG')
            
        clean_text = raw_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        paragraphs = clean_text.split('\n')
        sections_xml = ""
        current_paras = []
        
        for p in paragraphs:
            if p.strip():
                current_paras.append(f"<p>{p.strip()}</p>")
            if len(current_paras) >= 9999:
                sections_xml += "<section>\n" + "\n".join(current_paras) + "\n</section>\n"
                current_paras = []
                
        if current_paras:
            sections_xml += "<section>\n" + "\n".join(current_paras) + "\n</section>\n"
        
        fb2_template = f"<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<FictionBook xmlns=\"http://www.gribuser.ru/xml/fictionbook/2.0\">\n  <description><title-info><book-title>{ab_name}</book-title></title-info></description>\n  <body>{sections_xml}</body>\n</FictionBook>"
        (ab_path / f"{ab_name}.fb2").write_text(fb2_template, encoding="utf-8")
        
        logger.info(f"✅ Article '{ab_name}' downloaded and saved successfully.")
        return ab_name, None
        
    except ImportError:
        return None, "Install the libraries: pip install requests beautifulsoup4"
    except Exception as e:
        logger.error(f"URL fetch error: {e}")
        return None, f"Failed to download the article: {e}"