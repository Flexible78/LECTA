import re
import logging
import gradio as gr
from pathlib import Path
from lxml import etree
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from config import config
from libs.utils import data_path
from libs.sql_db import sql_db
from libs.tts_preprocessor import TextParse
from libs.russian import normalize_russian

logger = logging.getLogger(__name__)

@dataclass
class BookMetadata:
    first_name: str
    last_name: str
    book_title: str
    annotation: str

class FB2Processor:
    def __init__(self, accent: bool = False, single_vowel: bool = False):
        self.parser = TextParse(accent, single_vowel)
        self.list_of_snd = {}
        self.sound_pattern = re.compile(r'^$')
        self.stop_parsing = False
        self.update_sound_pattern()

    def _compile_sound_pattern(self) -> re.Pattern:
        patterns = '|'.join(f"({patt})" for patt in self.list_of_snd.keys())
        return re.compile(rf'{patterns}')
    
    def update_sound_pattern(self):
        new_list = dict(sql_db.select('list_of_snd', {'pattern': False, 'sound_type': False}))
        if new_list != self.list_of_snd:
            self.list_of_snd = new_list
            self.sound_pattern = self._compile_sound_pattern()

    def stop_parse(self):
        self.stop_parsing = True

    # === ЗАЩИТА ОТ СТРОК (ФИКС КРАША) ===
    def remove_namespaces(self, file_path) -> etree._Element:
        # Конвертируем строку в объект Path, если нужно
        path_obj = Path(file_path) if isinstance(file_path, str) else file_path
        
        if not path_obj.exists():
            return etree.Element("empty")
            
        parser = etree.XMLParser(remove_blank_text=True, ns_clean=True, strip_cdata=True, recover=True)
        with open(path_obj, 'rb') as f:
            xml_content = f.read()
            
        root = etree.fromstring(xml_content, parser=parser)
        
        for elem in root.iter():
            if isinstance(elem.tag, str): 
                elem.tag = etree.QName(elem).localname
            for attr_name in list(elem.attrib.keys()):
                if isinstance(attr_name, str) and '}' in attr_name:
                    local_name = attr_name.split('}', 1)[1]
                    elem.attrib[local_name] = elem.attrib[attr_name]
                    del elem.attrib[attr_name]
        return root

    def extract_metadata(self, root: etree._Element) -> BookMetadata:
        first_name = root.xpath('string(//description/title-info/author/first-name)').strip()
        last_name = root.xpath('string(//description/title-info/author/last-name)').strip()
        book_title = root.xpath('string(//description/title-info/book-title)').strip()
        annotation_text = ""
        annotations = root.xpath('//description/title-info/annotation')
        if annotations and annotations[0].text:
            annotation_text = annotations[0].text.strip()
        return BookMetadata(first_name, last_name, book_title, annotation_text)

    def extract_notes(self, root: etree._Element) -> Dict[str, str]:
        notes = {}
        note_sections = root.xpath("//body[@name='notes']/section")
        for nt in note_sections:
            nts = " ".join(t for t in nt.itertext() if re.search(r'[а-яА-ЯёЁa-zA-Zא-ת]', t))
            if nt.get('id'):
                notes[nt.get('id')] = nts
        return notes

    def split_sections(self, in_xml: etree._Element) -> List[Dict[str, dict]]:
        sections = []
        for index, sect in enumerate(in_xml.xpath("./section"), start=1):
            if (sub_sects := sect.xpath("./section")):
                titles = sect.xpath("./title")
                if len(titles) >= 1:
                    title_text = ''
                    for title in titles:
                        for sel in title:
                            if sel.text is not None:
                                title_text = title_text + sel.text + '. '
                for sub_index, sub_sect in enumerate(sub_sects, start=1):
                    if sub_index == 1:
                        sections.append({f'{index}_{sub_index}': {'sect': sub_sect, 'title': title_text}})
                    else:
                        sections.append({f'{index}_{sub_index}': {'sect': sub_sect}})
            else:
                sections.append({f'{index}': {'sect': sect}})

        return sections

    def optimize_chunk(self, text: str, max_length: int = 200) -> List[str]:
        result = []
        buffer = ""
        sentences = re.findall(r'[^.!?]+[.!?]|[^.!?]+$', text)

        for sentence in sentences:
            sentence_clean = sentence.strip()
            if not sentence_clean: continue
            sentence_with_space = sentence_clean + " "
            if len(sentence_with_space) > max_length:
                if buffer:
                    result.append(buffer.rstrip()); buffer = ""
                sub_parts = [sp.strip() for sp in sentence_clean.split(',') if sp.strip()]
                sub_buffer = ""
                for sp in sub_parts:
                    if len(sp) > max_length:
                        for sp2 in re.split(r'\s+(?=и\s+)', sp):
                            sp2_clean = sp2.strip()
                            if sp2_clean: result.append(sp2_clean)
                    else:
                        trial = sub_buffer + sp + ", "
                        if len(trial) <= max_length: sub_buffer = trial
                        else:
                            if sub_buffer: result.append(sub_buffer.rstrip())
                            sub_buffer = sp + ", "
                if sub_buffer: result.append(sub_buffer.rstrip())
            else:
                trial = buffer + sentence_with_space
                if len(trial) <= max_length: buffer = trial
                else:
                    if buffer: result.append(buffer.rstrip())
                    buffer = sentence_with_space

        if buffer: result.append(buffer.rstrip())
        return result

    def sound_check(self, text: str) -> List[etree._Element]:
        if not text.strip(): return []
        parts = self.sound_pattern.split(text)
        result = []
        for part in parts:
            if part is None or not part.strip(): continue
            match = self.sound_pattern.match(part)
            if match:
                value = list(self.list_of_snd.values())[match.lastindex - 1]
                elem = etree.Element("sound", value=value)
                result.append(elem)
            else:
                p_elem = etree.Element("p")
                p_elem.text = part
                result.append(p_elem)
        return result

    def parse_lines(self, parent: etree._Element, max_length: int = 200, remove_ru: bool = False, is_english: bool = False, translate: bool = False) -> None:
        elements = list(parent)
        for elem in elements:
            if self.stop_parsing: break
            if elem.text:
                if remove_ru:
                    elem.text = re.sub(r'[А-Яа-яЁё]+', '', elem.text)
                    elem.text = re.sub(r' +', ' ', elem.text).strip()
                
                if translate:
                    try:
                        from deep_translator import GoogleTranslator
                        text_to_translate = elem.text.strip()
                        if text_to_translate:
                            translated = GoogleTranslator(source='auto', target='ru').translate(text_to_translate)
                            if translated and re.search(r'[a-zA-Z]', translated):
                                def force_en_ru(match):
                                    word = match.group(0)
                                    res = GoogleTranslator(source='en', target='ru').translate(word)
                                    return res if res else word
                                translated = re.sub(r'[a-zA-Z\']+', force_en_ru, translated)
                            if translated: elem.text = translated
                    except Exception as e:
                        elem.text = f"[TRANSLATION ERROR] " + elem.text

                # Определяем, есть ли в тексте некириллические/нелатинские символы (иврит, арабский, CJK)
                has_non_european = bool(re.search(r'[\u0590-\u05FF\u0600-\u06FF\u4E00-\u9FFF]', elem.text))
                
                if not is_english and not has_non_european:
                    # Чисто русский текст: полная обработка (нормализация + ударения)
                    elem.text = normalize_russian(elem.text)
                    elem.text = self.parser.preprocess(elem.text)
                elif has_non_european and not is_english:
                    # Смешанный текст (русский + иврит/арабский): только лёгкая чистка мусора,
                    # без normalize_russian (чтобы не конвертировать латиницу в кириллицу и не ломать иврит)
                    elem.text = self.parser.garbage(elem.text)
                # else: is_english=True — текст не обрабатываем (поведение не изменилось)

                if len(elem.text) > max_length:
                    for chunk in self.optimize_chunk(elem.text, max_length):
                        new_p = etree.Element("p")
                        new_p.text = chunk
                        parent.insert(parent.index(elem), new_p)
                    parent.remove(elem)
                elif not re.search(r'[а-яА-ЯёЁa-zA-Z0-9א-ת]', elem.text):
                    empty = etree.Element("empty-line")
                    parent.insert(parent.index(elem), empty)
                    parent.remove(elem)

    def check_cite(self, element: etree._Element, notes: Dict[str, str]) -> None:
        for cite_elem in element.xpath('./cite | ./poem | ./epigraph'):
            for idx, child in enumerate(cite_elem):
                p = etree.Element("cite")
                if idx == 0:
                    cite_elem.addprevious(etree.Element("break", time="2"))
                    p.set("position", "start")
                if child.tag == "text-author" and child.text: p.text = "Author " + child.text
                else: p.text = child.text or ""
                cite_elem.addprevious(p)
            cite_elem.getparent().remove(cite_elem)

        for p_with_note in element.xpath('./p[a[@type="note"]]'):
            a = p_with_note.xpath('./a')[0]
            href = a.get("href", "")
            note_id = href[1:]
            if note_id in notes:
                p_elem = etree.Element("p")
                p_elem.text = (p_with_note.text or "")
                p_with_note.addprevious(p_elem)
                cite = etree.Element("cite", position="start")
                cite.text = notes[note_id]
                p_with_note.addprevious(cite)
                l_elem = etree.Element("p")
                l_elem.text = (a.tail or "")
                p_with_note.addprevious(l_elem)
                p_with_note.getparent().remove(p_with_note)

    @staticmethod
    def save_xml(content: str, filename: str):
        if not content: return "Content is empty"
        file_path = Path(filename)
        try:
            parser = etree.XMLParser(resolve_entities=False, no_network=True)
            etree.fromstring(content.encode("utf-8"), parser=parser)
            file_path.write_text(content, encoding="utf-8")
            return f"Saved: {file_path.name}"
        except Exception as e: return f"Error: {str(e)}"

    def process_book(
        self, ab_path: str, replace: bool = False, sound_effect: bool = True,
        punctuation: bool = True, translit: bool = True, ch_size: int = 200,
        remove_ru: bool = False, is_english: bool = False, translate: bool = False
    ) -> Tuple[int, str]:
        self.stop_parsing = False
        config.punctuation = punctuation
        config.translit = translit
        
        if not ab_path:
            yield 0, "❌ Error: No project selected"
            return
            
        work_dir = data_path / ab_path
        fb2_file = work_dir / f"{ab_path}.fb2"
        xml_path = work_dir / "xml"
        xml_path.mkdir(parents=True, exist_ok=True)

        if not fb2_file.exists():
            yield 0, f"❌ File {fb2_file} not found"
            return

        try:
            root = self.remove_namespaces(fb2_file)
        except Exception as e:
            yield 0, f"❌ XML parsing error: {e}"
            return

        body = root.xpath("//body[not(@name)]")
        if not body:
            yield 0, "❌ No main body found in FB2"
            return

        metadata = self.extract_metadata(root)
        notes = self.extract_notes(root)
        sections = self.split_sections(body[0])
        total_sections = len(sections)

        for idx, section_dict in enumerate(sections, start=1):
            if self.stop_parsing:
                yield int((idx / total_sections) * 100), "🛑 Interrupted by user"
                return

            (f_name, sect_data), = section_dict.items()
            xml_file = xml_path / f"{f_name}.xml"

            if xml_file.exists() and not replace:
                yield int((idx / total_sections) * 100), f"🟡 Skipping: {f_name}.xml already exists"
                continue

            element = sect_data["sect"]
            etree.strip_elements(element, "image")
            etree.strip_tags(element, "strong", "emphasis", "sup", "stanza")

            titles = element.xpath(".//title")
            if len(titles) >= 1:
                n_title = etree.Element('p')
                n_title.text = ''
                if sect_data.get('title') is not None:
                    n_title.text = sect_data['title']
                for title in titles:
                    for sel in title:
                        if sel.text: n_title.text = n_title.text + sel.text + '. '
                etree.strip_elements(element, 'title')
                element.insert(0, etree.Element('break', time='5'))
                element.insert(0, n_title)
                element.insert(0, etree.Element('break', time='5'))

            if idx == 1:
                bt = etree.Element("p")
                bt.text = f"{metadata.first_name} {metadata.last_name}. {metadata.book_title}."
                element.insert(0, bt)
                element.insert(0, etree.Element("break", time="5"))
                if metadata.annotation:
                    ann = etree.Element("p")
                    ann.text = metadata.annotation
                    element.insert(0, ann)
                element.insert(0, etree.Element("break", time="5"))

            for sub in element.xpath("./subtitle"): sub.tag = "p"
            self.check_cite(element, notes)
            
            if sound_effect:
                for p in list(element.xpath("./p")):
                    if p.text:
                        sounds = self.sound_check(p.text)
                        for s in sounds: p.addprevious(s)
                        p.getparent().remove(p)

            self.parse_lines(element, max_length=ch_size, remove_ru=remove_ru, is_english=is_english, translate=translate)

            speak = etree.Element("speak", attrib={"autor": f"{metadata.first_name} {metadata.last_name}", "album": metadata.book_title})
            speak.extend(list(element))
            tree = etree.ElementTree(speak)
            
            try: tree.write(str(xml_file), encoding="utf-8", pretty_print=True)
            except FileNotFoundError:
                xml_path.mkdir(parents=True, exist_ok=True)
                tree.write(str(xml_file), encoding="utf-8", pretty_print=True)

            yield int((idx / total_sections) * 100), f"✅ Processed: {f_name}"
            
        yield 100, "🎉 Done"