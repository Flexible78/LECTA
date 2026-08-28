import json
import os
import gzip
from pathlib import Path
from zipfile import ZipFile, BadZipFile
import re
import itertools

# --- Razdel integration for bulletproof processing ---
try:
    from razdel import tokenize
    from razdel.segmenters.tokenize import TokenSegmenter, RULES
    from razdel.rule import FunctionRule, JOIN
    
    def smart_join_rule(split):
        """Правило для склеивания слов с дефисами и плюсами (В+ест-+Индии)"""
        L, R = split.left, split.right
        if L is None or R is None: return
        l_text = L.text if hasattr(L, 'text') else str(L)
        r_text = R.text if hasattr(R, 'text') else str(R)
        if not l_text.strip() or not r_text.strip(): return
        
        valid_chars = re.compile(r'[а-яА-ЯёЁa-zA-Z\+\-]')
        if valid_chars.match(l_text[-1]) and valid_chars.match(r_text[0]):
            return JOIN

    custom_tokenizer = TokenSegmenter()
    custom_tokenizer.rules = list(RULES) + [FunctionRule(smart_join_rule)]
    RAZDEL_AVAILABLE = True
except ImportError:
    custom_tokenizer = None
    RAZDEL_AVAILABLE = False

def safe_pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

from .omograph_model import OmographModel
from .accent_model import AccentModel
from .stress_usage_model import StressUsagePredictorModel
from .yo_homograph_model import YoHomographModel
from .text_preprocessor import TextPreprocessor
from .text_postprocessor import fix_capital
from libs.utils import download_model

class RUAccent:
    def __init__(self):
        self.omograph_model = OmographModel()
        self.accent_model = AccentModel()
        self.stress_usage_predictor = StressUsagePredictorModel()
        self.yo_homograph_model = YoHomographModel()
        self.normalize = re.compile(r"[^a-zA-Z0-9\sа-яА-ЯёЁ—.,!?:;\"\"''(){}\[\]«»„“”-]")
        self.omograph_models_paths = {'turbo3.1': '/nn/nn_omograph/turbo3.1'}
    
        self.accentuator_paths = ['/nn/nn_accent', '/nn/nn_stress_usage_predictor','/nn/nn_yo_homograph_resolver', '/dictionary']
        self.letters_accent = {'о': '+о', 'О': '+О'}
        self.koziev_paths = []
        self.tiny_mode = False
        
        # Variables for custom dictionaries
        self.wildcard_rules = []
        self.manual_accents_dict = {}
        self.master_pattern = None
        
    def load(
        self,
        omograph_model_size="turbo3.1",
        use_dictionary=False,
        custom_dict=None,
        custom_homographs=None,
        device="cpu",
        workdir=None
        ):
        models_path = Path.cwd() / "models"
        if workdir:
            self.workdir = Path(workdir)
        else:
            self.workdir = Path(__file__).resolve().parent
            
        self.custom_dict = custom_dict or {}
        self.accents = {}

        if not self.workdir.exists():
            print('Загрузка RuAccent')
            url = f"https://myfreenet.ru/models/RuAccent.zip"
            zip_path = models_path / "RuAccent.zip"
            m, status = download_model(url, zip_path)
            if m is not None:
                try:
                    with ZipFile(zip_path, "r") as model_ref:
                        model_ref.extractall(str(models_path))
                except Exception as e:
                    print(f"ОШИБКА: Архив RuAccent поврежден ({e}). Битый файл удален. Перезапустите приложение.")
                finally:
                    zip_path.unlink(missing_ok=True)
        
        dictionary_dir = self.workdir / "dictionary"
        
        self.omographs = json.load(gzip.open(dictionary_dir / "omographs.json.gz"))
        self.omographs.update({"коса": ["к+оса", "кос+а"]})
        self.omographs.update(custom_homographs or {})
        self.omograph_model.load(self.workdir / self.omograph_models_paths[omograph_model_size][1:], device=device)

        self.yo_words = json.load(gzip.open(dictionary_dir / "yo_words.json.gz")) 
        self.accent_model.load(self.workdir / "nn" / "nn_accent", device=device)
        self.yo_homographs = json.load(gzip.open(dictionary_dir / "yo_homographs.json.gz")) 
        self.yo_homograph_model.load(self.workdir / "nn" / "nn_yo_homograph_resolver", device=device)

        self.accents.update(json.load(gzip.open(dictionary_dir / "accents_nn.json.gz")))
        self.accents.update(self.letters_accent)
        
        # 🚀 DICTIONARY COMPILATION FROM THE PROVIDED JSON (word_dict.json)
        self._compile_custom_dict()
        
        self.accents.update(self.manual_accents_dict)
        self.stress_usage_predictor.load(self.workdir / "nn" / "nn_stress_usage_predictor", device=device)

    def _compile_custom_dict(self):
        """Компилирует правила (Wildcards + Регистры) напрямую из переданного JSON словаря"""
        self.wildcard_rules = []
        self.manual_accents_dict = {}
        
        if not self.custom_dict:
            return
            
        print(f"📖 RuAccent: Обработка пользовательского словаря ({len(self.custom_dict)} записей)")
        wildcards_count = 0
        
        for k_phrase, v_phrase in self.custom_dict.items():
            k_phrase = k_phrase.strip()
            v_phrase = v_phrase.strip()
            
            if '*' in k_phrase:
                has_start_star = k_phrase.startswith('*')
                has_end_star = k_phrase.endswith('*')
                parts = k_phrase.split('*')
                regex_parts = []
                replacement_template = v_phrase
                group_idx = 1
                
                for i, part in enumerate(parts):
                    if part:
                        regex_parts.append(re.escape(part))
                    else:
                        is_edge = (i == 0) or (i == len(parts) - 1)
                        if is_edge:
                            regex_parts.append(r'(\w*)')
                            if i == 0:
                                replacement_template = r'\g<1>' + replacement_template
                                group_idx += 1
                            elif i == len(parts) - 1:
                                g_num = 2 if has_start_star else 1
                                replacement_template = replacement_template + f'\\g<{g_num}>'
                        else:
                            regex_parts.append(r'\w*')
                
                pattern_str = "".join(regex_parts)
                if not has_start_star: pattern_str = r'\b' + pattern_str
                if not has_end_star: pattern_str = pattern_str + r'\b'
                
                try:
                    compiled = re.compile(pattern_str, re.IGNORECASE)
                    self.wildcard_rules.append((compiled, replacement_template, False))
                    wildcards_count += 1
                except Exception: pass
            else:
                # Generate case variants for regular words
                k_words = k_phrase.split()
                if len(k_words) > 4:
                    cases = [tuple(str.lower for _ in k_words)]
                else:
                    funcs = [str.lower, str.capitalize, str.upper]
                    cases = itertools.product(funcs, repeat=len(k_words))
                
                for combo in cases:
                    new_k = " ".join([f(w.lower()) for f, w in zip(combo, k_words)])
                    if new_k.isupper(): new_v = v_phrase.upper()
                    elif new_k.istitle(): new_v = v_phrase.capitalize()
                    else: new_v = v_phrase
                    self.manual_accents_dict[new_k] = new_v

        print(f"   - Точных совпадений сгенерировано: {len(self.manual_accents_dict)}")
        print(f"   - Wildcard правил (*): {wildcards_count}\n")

        # Master regex for exact matches
        if self.manual_accents_dict:
            sorted_keys = sorted(self.manual_accents_dict.keys(), key=len, reverse=True)
            escaped_keys = [re.escape(k) for k in sorted_keys]
            pattern_str = r'(?<![\w+])(?:' + '|'.join(escaped_keys) + r')(?![\w+])'
            try:
                self.master_pattern = re.compile(f'({pattern_str})')
            except re.error:
                self.master_pattern = None

    def count_vowels(self, text):
        return sum(1 for char in text if char in "аеёиоуыэюяАЕЁИОУЫЭЮЯ")

    def has_punctuation(self, text):
        for char in text:
            if char in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~": return True
        return False

    def delete_spaces_before_punc(self, text):
        punc = "!\"#$%&'()*,./:;<=>?@[\\]^_`{|}-"
        for char in punc:
            if char == '-': text = text.replace(" " + char, char).replace(char + " ", char)
            text = text.replace(" " + char, char)
        return text.replace('~', '-')

    def extract_entities(self, data):
        return [item['entity'] for item in data]

    def _process_yo(self, words, sentence):
        lower_sentence = sentence.lower()
        yo_predictions = None
        if 'е' in lower_sentence:
            yo_predictions = self.extract_entities(self.yo_homograph_model.predict_yo_homographs(lower_sentence))
        
        for i, word in enumerate(words):
            lower_word = word.lower()
            words[i] = fix_capital(word, self.yo_words.get(lower_word, word))
            if yo_predictions and yo_predictions[i] == "YO":
                words[i] = fix_capital(word, self.yo_homographs.get(lower_word, word))
        return words

    def _process_omographs(self, text):
        splitted_text = text
        founded_omographs = []
        texts = []
        hypotheses = []
    
        for i, word in enumerate(splitted_text):
            variants = self.omographs.get(word)
            if variants:
                founded_omographs.append({"word": word, "variants": variants, "position": i})
                texts.append(splitted_text)
                hypotheses.append(variants)
    
        if len(founded_omographs) > 0:
            texts_batch = []
            hypotheses_batch = [val for sublist in hypotheses for val in sublist]
            num_hypotheses = [len(i) for i in hypotheses]
            
            for o, t in zip(founded_omographs, texts):
                position = o["position"]
                t_back = t[position]
                t[position] = ' <w>' + t[position] + '</w> '
                for _ in range(len(o["variants"])):
                    texts_batch.append(self.delete_spaces_before_punc(" ".join(t.copy())))
                t[position] = t_back
            cls_batch = self.omograph_model.classify(texts_batch, hypotheses_batch, num_hypotheses)
            for cls_index, omograph in enumerate(founded_omographs):
                splitted_text[omograph["position"]] = cls_batch[cls_index]
        return splitted_text

    def _process_accent(self, text, stress_usages):
        splitted_text = text
        for i, word in enumerate(splitted_text):
            if '+' in word: continue
            if stress_usages[i] == "STRESS":
                lower_word = word.lower()
                stressed_word = self.accents.get(lower_word, lower_word)
                if stressed_word == lower_word and not self.has_punctuation(lower_word) and self.count_vowels(lower_word) > 1:
                    splitted_text[i] = self.accent_model.put_accent(word)
                else:
                    match = re.finditer(r'\+', stressed_word)
                    word_fixed = list(word)
                    for j, e in enumerate(list(match)):
                        word_fixed = word_fixed[:e.start() + j] + ["+"] + list(word)[e.end() - 1:]
                    splitted_text[i] = "".join(word_fixed)
        return splitted_text

    def process_yo(self, text):
        sentences = TextPreprocessor.split_by_sentences(text)
        outputs = []
        for sentence in sentences:
            words, remaining_text = TextPreprocessor.split_by_words(sentence)
            processed_words = self._process_yo(words, sentence)
            processed_text = "".join([l+r for l,r in zip(remaining_text, processed_words)])
            processed_text = self.delete_spaces_before_punc(processed_text)
            outputs.append(processed_text)
        return " ".join(outputs)
    
    def process_all_internal(self, text):
        text = re.sub(self.normalize, "", text)
        sentences = TextPreprocessor.split_by_sentences(text)
        outputs = []
        for sentence in sentences:
            words, remaining_text = TextPreprocessor.split_by_words(sentence)
            if len(words) == 0:
                outputs.append("".join(remaining_text))
                continue
            stress_usages = self.extract_entities(self.stress_usage_predictor.predict_stress_usage(sentence))
            processed_words = self._process_yo(words, sentence)
            processed_words = self._process_omographs(processed_words)
            processed_words = self._process_accent(processed_words, stress_usages)
            processed_sentence = "".join([l+r for l,r in zip(remaining_text, processed_words)] + [remaining_text[-1]])
            processed_sentence = self.delete_spaces_before_punc(processed_sentence)
            outputs.append(processed_sentence)
        return "".join(outputs)

    def process_all(self, text, skip_regex=None):
        """
        🚀 УМНАЯ И БЕЗОПАСНАЯ ОБРАБОТКА (Интеграция Razdel + Wildcards + OOM Защита)
        """
        if skip_regex:
            pattern = re.compile(skip_regex)
            matches = list(pattern.finditer(text))
            
            if not matches:
                return self._process_robust(text)

            indices = [(match.start(), match.end()) for match in matches]
            skipped = [text[l:r] for l,r in indices]
            
            elems = []
            for l,r in safe_pairwise(indices):
                start = l[1]
                end = r[0]
                elem = text[start:end]
                elems.extend([elem])

            first_elem = text[:indices[0][0]]
            last_elem = text[indices[-1][1]:]

            elems = [first_elem] + elems + [last_elem]

            results = []
            for e in elems:
                if len(e) == 0:
                    results.append(e)
                    continue
                results.append(self._process_robust(e))
                
            return "".join([results[0]] + [l+r for l,r in zip(skipped, results[1:])])
        else:
            return self._process_robust(text)

    def _process_robust(self, text):
        """Скрытая бронебойная логика, спасающая от падений памяти (OOM) и рваных пробелов"""
        if not RAZDEL_AVAILABLE:
            return self.process_all_internal(text)
            
        try:
            # Step 1: Wildcards
            text_with_wildcards = text
            for pattern, replacement_template, is_literal in self.wildcard_rules:
                def replacer(match):
                    result = match.expand(replacement_template)
                    original_text = match.group(0)
                    
                    if not is_literal:
                        orig_words = original_text.split()
                        repl_words = result.split()
                        
                        if len(orig_words) == len(repl_words):
                            final_words = []
                            for o_w, r_w in zip(orig_words, repl_words):
                                if len(o_w) >= 2:
                                    if o_w[0].isupper() and o_w[1].isupper(): res = r_w.upper()
                                    elif o_w[0].isupper() and o_w[1].islower(): res = r_w.capitalize()
                                    elif o_w[0].islower() and o_w[1].islower(): res = r_w.lower()
                                    else: res = r_w
                                elif len(o_w) == 1:
                                    if o_w[0].isupper(): res = r_w.upper()
                                    else: res = r_w.lower()
                                else:
                                    res = r_w
                                final_words.append(res)
                            result = " ".join(final_words)
                        else:
                            if original_text.isupper(): result = result.upper()
                            elif original_text.istitle(): result = result.capitalize()
                            elif original_text and original_text[0].isupper(): result = result[0].upper() + result[1:]

                    fin_words = result.split()
                    normalized_words = []
                    for w in fin_words:
                        if len(w) >= 2:
                            if w[0].isupper() and w[1].isupper(): n_w = w.upper()
                            elif w[0].isupper() and w[1].islower(): n_w = w.capitalize()
                            elif w[0].islower() and w[1].islower(): n_w = w.lower()
                            else: n_w = w
                        else: n_w = w
                        normalized_words.append(n_w)
                    return " ".join(normalized_words)
                
                text_with_wildcards = pattern.sub(replacer, text_with_wildcards)

            # Step 2: Exact dictionary
            text_with_dict = text_with_wildcards
            if self.master_pattern:
                parts = self.master_pattern.split(text_with_wildcards)
                processed_parts = []
                for part in parts:
                    if not part: continue
                    processed_parts.append(self.manual_accents_dict.get(part, part))
                text_with_dict = "".join(processed_parts)
            
            # Step 3: Token synchronizers
            src_tokens = list(custom_tokenizer(text_with_dict))
            if not src_tokens: return text
            
            src_ru_indices = [i for i, t in enumerate(src_tokens) if re.search(r'[а-яА-ЯёЁa-zA-Z]', t.text)]
            
            # Step 4: Safe 100-token chunks (OOM protection)
            text_before = text_with_dict.replace('+', '')
            text_before_toks = list(custom_tokenizer(text_before))
            
            text_after = ""
            current_idx = 0
            while current_idx < len(text_before_toks):
                chunk_toks = text_before_toks[current_idx:current_idx + 100]
                if current_idx + 100 >= len(text_before_toks):
                    cut_idx = len(chunk_toks)
                else:
                    cut_idx = len(chunk_toks)
                    for i in range(len(chunk_toks) - 1, -1, -1):
                        if chunk_toks[i].text in ['.', '!', '?', ';', ',', ':']:
                            cut_idx = i + 1
                            break
                            
                start_char = chunk_toks[0].start
                end_char = chunk_toks[cut_idx - 1].stop
                chunk_text = text_before[start_char:end_char]
                
                if current_idx == 0:
                    prefix_spaces = text_before[0:start_char]
                else:
                    prev_end_char = text_before_toks[current_idx - 1].stop
                    prefix_spaces = text_before[prev_end_char:start_char]
                    
                text_after += prefix_spaces
                
                m_space = re.match(r'^(.*?)([\s\n]*)$', chunk_text, re.DOTALL)
                if m_space:
                    core_text = m_space.group(1)
                    spaces = m_space.group(2)
                    if core_text:
                        text_after += self.process_all_internal(core_text)
                    text_after += spaces
                
                current_idx += cut_idx
                
            last_stop = text_before_toks[-1].stop
            if last_stop < len(text_before):
                text_after += text_before[last_stop:]
                
            # Step 5: Perfect gluing (keep original spaces and hyphens)
            trg_tokens = list(custom_tokenizer(text_after))
            trg_ru_tokens = [t for t in trg_tokens if re.search(r'[а-яА-ЯёЁa-zA-Z]', t.text)]
            
            if len(src_ru_indices) == len(trg_ru_tokens):
                class FakeToken:
                    def __init__(self, t): self.text = t
                    
                result_tokens = list(src_tokens)
                for idx_src, token_trg in zip(src_ru_indices, trg_ru_tokens):
                    token_src = result_tokens[idx_src]
                    if '+' in token_src.text:
                        continue 
                    result_tokens[idx_src] = FakeToken(token_trg.text)

                sep_sizes = [y.start - x.stop for x, y in safe_pairwise(src_tokens)]
                sep_sizes.append(0)
                with_sep = ((t.text, ' ' * sep) for t, sep in zip(result_tokens, sep_sizes))
                return ''.join(itertools.chain.from_iterable(with_sep))
            else:
                return text_with_dict

        except Exception as e:
            print(f"⚠️ Ошибка бронебойного парсинга razdel: {e}")
            return self.process_all_internal(text)