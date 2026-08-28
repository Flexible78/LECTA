# -*- coding: utf-8 -*-
import os
import sys
import threading
import queue
import tkinter as tk
from tkinter import filedialog, messagebox
import logging
import re
import itertools
import json

# --- SUPPRESS LOGS FOR CLEAN OUTPUT ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

# --- Set up import paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
    
# --- DEPENDENCIES ---
from ruaccent import RUAccent
import customtkinter as ctk

# Check onnxruntime
try:
    import onnxruntime as ort
except ImportError:
    ort = None

# Import razdel for tokenization
RAZDEL_AVAILABLE = False
custom_tokenizer = None

try:
    from razdel import tokenize
    from itertools import pairwise, chain
    
    # Attempt to import internal components for custom rules
    try:
        from razdel.segmenters.tokenize import TokenSegmenter, RULES, RU
        from razdel.rule import FunctionRule, JOIN
        
        # Configure razdel rules to handle the + character
        def smart_join_rule(split):
            """Умное правило для склеивания слов с дефисами и плюсами (например: В+ест-+Индии)"""
            L = split.left
            R = split.right
            if L is None or R is None:
                return

            # Get the token text
            l_text = L.text if hasattr(L, 'text') else str(L)
            r_text = R.text if hasattr(R, 'text') else str(R)

            # Check that it's not whitespace (the splitter emits whitespace separately)
            if not l_text.strip() or not r_text.strip():
                return

            # We only glue when the junction has: Cyrillic, plus, or hyphen.
            # If the left part ends with Cyrillic, a plus, or a hyphen
            # And the right part starts with Cyrillic, a plus, or a hyphen
            # THEN WE GLUE THEM!
            
            valid_chars = re.compile(r'[а-яА-ЯёЁa-zA-Z\+\-]')
            
            # Check the last char of the left part and the first char of the right
            if valid_chars.match(l_text[-1]) and valid_chars.match(r_text[0]):
                # Make sure we don't glue two hyphens in a row or something odd without letters,
                # although it's safe for the neural net.
                # The key thing is that the glued token ends up with letters.
                return JOIN

        # Create the tokenizer and add a custom rule
        custom_tokenizer = TokenSegmenter()
        custom_tokenizer.rules = list(RULES) + [FunctionRule(smart_join_rule)]
        
        RAZDEL_AVAILABLE = True
        print("✅ razdel импортирован с поддержкой custom правил")
    except (ImportError, AttributeError) as e:
        print(f"⚠️ Не удалось импортировать внутренние компоненты razdel: {e}")
        print("⚠️ Будет использован стандартный tokenize без custom правил")
        RAZDEL_AVAILABLE = True  # razdel exists but without custom rules
        
except ImportError:
    print("⚠️ Модуль razdel не найден. Установите: pip install razdel")

# --- CustomTkinter setup ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# --- Check Python libraries ---
try:
    import customtkinter
except ImportError:
    messagebox.showerror(
        "Ошибка импорта",
        f"Библиотека customtkinter не найдена.\n"
        f"Пожалуйста, установите её:\n"
        f"pip install customtkinter"
    )
    sys.exit(1)

try:
    import ruaccent
except ImportError as e:
    messagebox.showerror(
        "Ошибка импорта",
        f"Библиотека ruaccent не найдена.\n"
        f"Пожалуйста, установите необходимые компоненты:\n"
        f"pip install ruaccent onnxruntime-gpu\n\n"
        f"Исключение: {e}"
    )
    sys.exit(1)

if ort is None:
    messagebox.showwarning(
        "Отсутствует компонент",
        f"Библиотека onnxruntime не найдена.\n"
        f"Она необходима для работы модели.\n"
        f"Рекомендуемая установка (для GPU): pip install onnxruntime-gpu"
    )

class TextRedirector:
    def __init__(self, widget, tag="stdout"):
        self.widget, self.tag = widget, tag
        if tag == "stderr": self.widget.tag_config("stderr", foreground="#ff6b6b")
        else: self.widget.tag_config("stdout", foreground="#e0e0e0")
    def write(self, string):
        if string.strip() or string == '\n': self.widget.after(0, self._write, string)
    def _write(self, string): self.widget.insert(tk.END, string, self.tag); self.widget.see(tk.END)
    def flush(self): pass

class AccentizerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("RUAccent-GUI (v0.33)")
        self.geometry("700x750")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(9, weight=1)

        self.settings_file = os.path.join(SCRIPT_DIR, "settings_ruaccent.json")
        
        # --- CUDA CHECK ---
        self.cuda_is_available = False
        self.cuda_disabled_reason = "CUDA недоступна"
        
        if ort is not None:
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                self.cuda_is_available = True
            else:
                self.cuda_disabled_reason = "onnxruntime-gpu не найден (установлен CPU вариант)"
                print(f"⚠️ {self.cuda_disabled_reason}.")
                print("⚠️ Для использования GPU выполните: pip uninstall onnxruntime && pip install onnxruntime-gpu")
        else:
            self.cuda_disabled_reason = "onnxruntime не установлен"

        self.source_path = ctk.StringVar()
        self.custom_dict_path = ctk.StringVar()
        self.output_dir_path = ctk.StringVar()
        
        default_device = 'CUDA' if self.cuda_is_available else 'CPU'
        self.device_var = ctk.StringVar(value=default_device)
        
        self.load_settings()
        
        self.progress_queue = queue.Queue()
        self.processing_thread = None
        self.accentizer = None
        self.create_widgets()
        
        if not self.cuda_is_available:
            if hasattr(self, 'cuda_radio'):
                self.cuda_radio.configure(state="disabled")
                if hasattr(self, 'cuda_label'):
                    self.cuda_label.configure(text=self.cuda_disabled_reason)

        self.after(100, self.check_queue)

    def save_settings(self):
        settings = {
            "source_path": self.source_path.get(),
            "custom_dict_path": self.custom_dict_path.get(),
            "output_dir_path": self.output_dir_path.get(),
            "device": self.device_var.get(),
        }
        try:
            with open(self.settings_file, "w", encoding='utf-8') as f:
                json.dump(settings, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Не удалось сохранить настройки: {e}")

    def load_settings(self):
        if not os.path.exists(self.settings_file):
            return
        try:
            with open(self.settings_file, "r", encoding='utf-8') as f:
                settings = json.load(f)
            
            self.source_path.set(settings.get("source_path", ""))
            self.custom_dict_path.set(settings.get("custom_dict_path", ""))
            self.output_dir_path.set(settings.get("output_dir_path", os.path.join(SCRIPT_DIR, "output")))

            loaded_device = settings.get("device")
            if loaded_device:
                loaded_device = loaded_device.upper()
            
            if self.cuda_is_available:
                final_device = loaded_device if loaded_device == 'CPU' else 'CUDA'
            else:
                final_device = 'CPU'
            
            self.device_var.set(final_device)
            
        except Exception as e:
            print(f"⚠️ Не удалось загрузить настройки: {e}")

    def create_widgets(self):
        device_frame = ctk.CTkFrame(self); device_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew"); device_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(device_frame, text="Устройство:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        radio_frame = ctk.CTkFrame(device_frame); radio_frame.grid(row=0, column=1, sticky="w", padx=10)
        self.cpu_radio = ctk.CTkRadioButton(radio_frame, text="CPU", variable=self.device_var, value="CPU"); self.cpu_radio.pack(side="left", padx=5)
        self.cuda_radio = ctk.CTkRadioButton(radio_frame, text="GPU (CUDA)", variable=self.device_var, value="CUDA"); self.cuda_radio.pack(side="left")
        if not self.cuda_is_available:
            self.cuda_label = ctk.CTkLabel(device_frame, text=self.cuda_disabled_reason, text_color="gray")
            self.cuda_label.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="w")

        ctk.CTkLabel(self, text="Входной TXT файл:").grid(row=1, column=0, padx=10, pady=(10, 5), sticky="w")
        source_frame = ctk.CTkFrame(self); source_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew"); source_frame.grid_columnconfigure(0, weight=1)
        self.source_entry = ctk.CTkEntry(source_frame, textvariable=self.source_path, placeholder_text="Выберите .txt файл"); self.source_entry.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="ew")
        self.browse_button = ctk.CTkButton(source_frame, text="Обзор...", command=self.select_source_file); self.browse_button.grid(row=0, column=1, padx=(0, 10), pady=10)

        ctk.CTkLabel(self, text="Словарь ручных ударений:").grid(row=3, column=0, padx=10, pady=(10, 5), sticky="w")
        
        custom_dict_frame = ctk.CTkFrame(self); custom_dict_frame.grid(row=4, column=0, padx=10, pady=5, sticky="ew"); custom_dict_frame.grid_columnconfigure(0, weight=1)
        self.custom_dict_entry = ctk.CTkEntry(custom_dict_frame, textvariable=self.custom_dict_path, placeholder_text="Выберите или введите путь до файла словаря"); self.custom_dict_entry.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="ew")
        ctk.CTkButton(custom_dict_frame, text="Обзор...", command=self.select_custom_dict_file).grid(row=0, column=1, padx=(0, 10), pady=10)

        ctk.CTkLabel(self, text="Папка для сохранения:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
        output_frame = ctk.CTkFrame(self); output_frame.grid(row=6, column=0, padx=10, pady=5, sticky="ew"); output_frame.grid_columnconfigure(0, weight=1)
        self.output_entry = ctk.CTkEntry(output_frame, textvariable=self.output_dir_path); self.output_entry.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="ew")
        ctk.CTkButton(output_frame, text="Обзор...", command=self.select_output_dir).grid(row=0, column=1, padx=(0, 10), pady=10)

        self.start_button = ctk.CTkButton(self, text="Начать обработку", command=self.start_processing_thread)
        self.start_button.grid(row=7, column=0, padx=10, pady=10, sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(self); self.progress_bar.grid(row=8, column=0, padx=10, pady=(0, 5), sticky="ew"); self.progress_bar.set(0)
        
        log_label = ctk.CTkLabel(self, text="Лог выполнения:"); log_label.grid(row=9, column=0, padx=10, pady=(10, 0), sticky="w")
        self.log_textbox = ctk.CTkTextbox(self); self.log_textbox.grid(row=10, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.log_textbox.bind("<Key>", lambda event: "break")
        self.create_log_menu()
        
    def create_log_menu(self):
        self.log_menu = tk.Menu(self, tearoff=0); self.log_menu.add_command(label="Копировать все", command=self.copy_log_all)
        self.log_textbox.bind("<Button-3>", lambda e: self.show_log_menu(e))
    def show_log_menu(self, event): self.log_menu.post(event.x_root, event.y_root)
    def copy_log_all(self): all_text = self.log_textbox.get("1.0", tk.END); self.clipboard_clear(); self.clipboard_append(all_text)

    def select_source_file(self):
        initial_dir = os.path.dirname(self.source_path.get()) if self.source_path.get() else SCRIPT_DIR
        path = filedialog.askopenfilename(title="Выберите текстовый файл", initialdir=initial_dir, filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if path: self.source_path.set(path)
        
    def select_custom_dict_file(self):
        initial_dir = os.path.dirname(self.custom_dict_path.get()) if self.custom_dict_path.get() else SCRIPT_DIR
        path = filedialog.askopenfilename(title="Выберите файл словаря", initialdir=initial_dir, filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if path: self.custom_dict_path.set(path)

    def select_output_dir(self):
        initial_dir = self.output_dir_path.get()
        path = filedialog.askdirectory(title="Выберите папку для сохранения", initialdir=initial_dir)
        if path: self.output_dir_path.set(path)

    def start_processing_thread(self):
        if not self.source_path.get() or not self.output_dir_path.get():
            messagebox.showwarning("Внимание", "Пожалуйста, выберите исходный файл и выходную папку.")
            return
            
        self.save_settings()
        
        self.start_button.configure(state="disabled")
        self.progress_bar.set(0)
        self.log_textbox.delete("1.0", tk.END)
        self.processing_thread = threading.Thread(target=self.run_processing_worker, daemon=True)
        self.processing_thread.start()
        self.after(100, self.check_queue)

    def run_processing_worker(self):
        original_stdout, original_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = TextRedirector(self.log_textbox, "stdout"), TextRedirector(self.log_textbox, "stderr")
            source_file, output_folder, device = self.source_path.get(), self.output_dir_path.get(), self.device_var.get()
            print(f"🚀 Запуск обработки файла: {os.path.basename(source_file)}")
            print(f"📍 Устройство: {device}")
            
            if device == 'CUDA':
                if ort is None or 'CUDAExecutionProvider' not in ort.get_available_providers():
                    print("⚠️ ОШИБКА: Попытка запустить CUDA без поддержки onnxruntime-gpu.")
                    print("⚠️ Пожалуйста, установите onnxruntime-gpu (pip install onnxruntime-gpu).")
                    self.progress_queue.put(("error", "CUDA выбрана, но onnxruntime-gpu не найден.")); return
                else:
                    print("✅ onnxruntime GPU provider обнаружен.")

            if not RAZDEL_AVAILABLE:
                print("⚠️ ОШИБКА: Модуль razdel не найден.")
                print("⚠️ Для работы требуется установить: pip install razdel")
                self.progress_queue.put(("error", "Модуль razdel не установлен.")); return

            if custom_tokenizer is not None:
                print(f"🛡️ Режим: Токенизация с защитой через razdel (custom правила)\n")
            else:
                print(f"🛡️ Режим: Токенизация через razdel (стандартные правила)\n")

            manual_accents_dict = {}
            wildcard_rules = [] # List of tuples (compiled_regex, replacement, is_literal)
            
            custom_dict_file_str = self.custom_dict_path.get()
            
            # --- 1. LOAD THE DICTIONARY
            if custom_dict_file_str and os.path.exists(custom_dict_file_str):
                print(f"ℹ️ Найден словарь: {os.path.basename(custom_dict_file_str)}")
                print("ℹ️ Поддерживаются *, комментарии (#), флаги $ и ==.")
                lines_processed = 0
                wildcards_count = 0
                try:
                    with open(custom_dict_file_str, 'r', encoding='utf-8') as f:
                        for raw_line in f:
                            line = raw_line.strip()
                            if not line or line.startswith('#'): continue

                            is_strict = line.startswith('$')
                            content = line[1:].strip() if is_strict else line
                            
                            sep = '==' if '==' in content else '='
                            if sep not in content:
                                print(f"⚠️ Пропуск: '{line}'. Нет разделителя '{sep}'.")
                                continue
                            
                            key_raw, val_raw = content.split(sep, 1)
                            k_phrase, v_phrase = key_raw.strip(), val_raw.strip()
                            is_literal = (sep == '==')
                            
                            # Check for a Wildcard (*)
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
                                            # Always add groups (needed for capture)
                                            # Insert backreferences in the template to restore context
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
                                
                                flags = re.IGNORECASE if not is_strict else 0
                                
                                try:
                                    compiled = re.compile(pattern_str, flags)
                                    wildcard_rules.append((compiled, replacement_template, is_literal))
                                    wildcards_count += 1
                                    lines_processed += 1
                                except re.error as e:
                                    print(f"⚠️ Ошибка компиляции Wildcard '{k_phrase}': {e}")
                                continue

                            # Standard logic for simple words
                            if is_strict:
                                manual_accents_dict[k_phrase] = v_phrase
                                lines_processed += 1
                                continue

                            k_words = k_phrase.split()
                            if len(k_words) > 4:
                                cases = [tuple(str.lower for _ in k_words)]
                            else:
                                funcs = [str.lower, str.capitalize, str.upper]
                                cases = itertools.product(funcs, repeat=len(k_words))
                            
                            seen_keys = set()
                            for combo in cases:
                                new_k = " ".join([f(w.lower()) for f, w in zip(combo, k_words)])

                                if new_k in seen_keys:
                                    continue
                                seen_keys.add(new_k)

                                if is_literal:
                                    new_v = v_phrase
                                else:
                                    if new_k.isupper():
                                        new_v = v_phrase.upper()
                                    elif new_k.istitle():
                                        new_v = v_phrase.capitalize()
                                    else:
                                        new_v = v_phrase
                                
                                manual_accents_dict[new_k] = new_v
                            
                            lines_processed += 1

                    print(f"✅ Обработано строк: {lines_processed}")
                    print(f"   - Точных совпадений: {len(manual_accents_dict)}")
                    print(f"   - Wildcard правил (*): {wildcards_count}\n")
                    
                except Exception as e:
                    print(f"❌ Ошибка при чтении словаря: {e}\nБудет продолжено без словаря.\n")
            else:
                print("ℹ️ Словарь ручных ударений не найден. Будет использована только модель.\n")

            # --- 2. COMPILE THE MASTER REGEX ---
            master_pattern = None
            if manual_accents_dict:
                sorted_keys = sorted(manual_accents_dict.keys(), key=len, reverse=True)
                escaped_keys = [re.escape(k) for k in sorted_keys]
                # pattern_str = r'\b(?:' + '|'.join(escaped_keys) + r')\b'
                # Forbid a match if letters, digits, _ or + sit before or after the word
                pattern_str = r'(?<![\w+])(?:' + '|'.join(escaped_keys) + r')(?![\w+])'
                try:
                    master_pattern = re.compile(f'({pattern_str})')
                    print(f"🔍 Regex активен: точный словарь.")
                except re.error as e:
                    print(f"⚠️ Ошибка компиляции Regex: {e}. Точный словарь отключен.")
                    master_pattern = None

            # --- 3. LOAD RUACCENT ---
            self.accentizer = RUAccent()
            print(f"🔄 Загрузка модели RUAccent ('turbo3.1') на устройство '{device}'...")
            cache_dir = os.path.join(SCRIPT_DIR, "ruaccent_cache")
            os.makedirs(cache_dir, exist_ok=True)
            self.accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True, device=device, workdir=cache_dir)
            print("✅ Модель RUAccent успешно загружена. Начинаю обработку...\n")

            # --- 4. PROCESS THE FILE ---
            base_filename = os.path.splitext(os.path.basename(source_file))[0]
            output_filename = f"{base_filename}_processed.txt"
            output_file_path = os.path.join(output_folder, output_filename)

            os.makedirs(output_folder, exist_ok=True)
            with open(source_file, 'r', encoding='utf-8') as infile:
                lines = infile.readlines()
            
            total_lines = len(lines)
            if total_lines == 0: self.progress_queue.put(("error", "Входной файл пуст.")); return
            print(f"📄 Всего строк для обработки: {total_lines}\n")

            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                for i, line in enumerate(lines):
                    line = line.rstrip('\n')
                    if not line:
                        outfile.write('\n'); continue
                    
                    processed_line = self.process_line_with_razdel(line, manual_accents_dict, master_pattern, wildcard_rules, original_stderr)
                    
                    outfile.write(processed_line + '\n')
                    self.progress_queue.put(("progress", (i + 1) / total_lines))
            
            print(f"✅ Обработка завершена. Результат сохранен в: {output_file_path}")
            self.progress_queue.put(("success", output_file_path))
        except Exception as e:
            import traceback; print("\n--- !!! КРИТИЧЕСКАЯ ОШИБКА !!! ---"); traceback.print_exc()
            self.progress_queue.put(("error", f"Произошла критическая ошибка:\n{e}"))
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr; self.accentizer = None; print("--- Обработка завершена ---")

    def process_line_with_razdel(self, src_text, manual_dict, master_pattern, wildcard_rules, original_stderr):
        try:
            # Step 0: Wildcards
            text_with_wildcards = src_text
            for pattern, replacement_template, is_literal in wildcard_rules:
                
                def replacer(match):
                    result = match.expand(replacement_template)
                    original_text = match.group(0)
                    
                    # 1. Try to adapt the case (only for soft replacement =)
                    if not is_literal:
                        orig_words = original_text.split()
                        repl_words = result.split()
                        
                        if len(orig_words) == len(repl_words):
                            final_words = []
                            for o_w, r_w in zip(orig_words, repl_words):
                                if len(o_w) >= 2:
                                    if o_w[0].isupper() and o_w[1].isupper():
                                        res = r_w.upper()
                                    elif o_w[0].isupper() and o_w[1].islower():
                                        res = r_w.capitalize()
                                    elif o_w[0].islower() and o_w[1].islower():
                                        res = r_w.lower()
                                    else:
                                        res = r_w
                                elif len(o_w) == 1:
                                    if o_w[0].isupper():
                                        res = r_w.upper()
                                    else:
                                        res = r_w.lower()
                                else:
                                    res = r_w
                                final_words.append(res)
                            result = " ".join(final_words) # Update result
                        else:
                            # Fallback for words of different lengths
                            if original_text.isupper(): result = result.upper()
                            elif original_text.istitle(): result = result.capitalize()
                            elif original_text and original_text[0].isupper(): result = result[0].upper() + result[1:]

                    # 2. Final case normalization of the RESULT (for both modes)
                    # "If a word's case got mixed up by group substitution, fix it"
                    # АВТОобус -> АВТООБУС
                    # АвтоОБУС -> Автообус
                    
                    fin_words = result.split()
                    normalized_words = []
                    for w in fin_words:
                        if len(w) >= 2:
                            if w[0].isupper() and w[1].isupper():
                                n_w = w.upper()
                            elif w[0].isupper() and w[1].islower():
                                n_w = w.capitalize()
                            elif w[0].islower() and w[1].islower():
                                n_w = w.lower()
                            else:
                                n_w = w
                        else:
                            n_w = w
                        normalized_words.append(n_w)
                    
                    return " ".join(normalized_words)
                
                text_with_wildcards = pattern.sub(replacer, text_with_wildcards)

            # Step 1: Exact dictionary
            text_with_dict = text_with_wildcards
            if master_pattern:
                parts = master_pattern.split(text_with_wildcards)
                processed_parts = []
                for part in parts:
                    if not part: continue
                    if part in manual_dict:
                        processed_parts.append(manual_dict[part])
                    else:
                        processed_parts.append(part)
                text_with_dict = "".join(processed_parts)
            
            # Step 2: Tokenization
            tokenizer_func = custom_tokenizer if custom_tokenizer is not None else tokenize
            src_tokens = list(tokenizer_func(text_with_dict))
            if not src_tokens: return src_text
            
            # Step 3: Word indices
            src_ru_indices = []
            for i, token in enumerate(src_tokens):
                if re.search(r'[а-яА-ЯёЁa-zA-Z]', token.text):
                    src_ru_indices.append(i)
            
            # Step 4: Remove pluses
            text_before = text_with_dict.replace('+', '')
            
            # Step 5: RUAccent
            try:
                # Tokenize the text without pluses for an accurate word/punctuation count
                text_before_toks = list(tokenizer_func(text_before))
                
                # If tokens <= 250 (guaranteed to fit in 512 subwords), process it whole
                if len(text_before_toks) <= 120:
                    text_after = self.accentizer.process_all(text_before)
                else:
                    text_after = ""
                    current_idx = 0
                    while current_idx < len(text_before_toks):
                        # Take a chunk of up to 200 tokens (with headroom)
                        chunk_toks = text_before_toks[current_idx:current_idx + 100]
                        if current_idx + 100 >= len(text_before_toks):
                            cut_idx = len(chunk_toks)
                        else:
                            cut_idx = len(chunk_toks)
                            # Look for end-of-sentence marks
                            for i in range(len(chunk_toks) - 1, -1, -1):
                                if chunk_toks[i].text in ['.', '!', '?', ';']:
                                    cut_idx = i + 1
                                    break
                            # If none, look for commas
                            if cut_idx == len(chunk_toks):
                                for i in range(len(chunk_toks) - 1, -1, -1):
                                    if chunk_toks[i].text in [',', ':']:
                                        cut_idx = i + 1
                                        break
                                        
                        start_char = chunk_toks[0].start
                        end_char = chunk_toks[cut_idx - 1].stop
                        chunk_text = text_before[start_char:end_char]
                        
                        # Keep the spaces/characters between chunks
                        if current_idx == 0:
                            prefix_spaces = text_before[0:start_char]
                        else:
                            prev_end_char = text_before_toks[current_idx - 1].stop
                            prefix_spaces = text_before[prev_end_char:start_char]
                            
                        text_after += prefix_spaces
                        
                        # Detach trailing whitespace if it accidentally got included
                        m_space = re.match(r'^(.*?)([\s\n]*)$', chunk_text, re.DOTALL)
                        core_text = m_space.group(1)
                        spaces = m_space.group(2)
                        
                        if core_text:
                            text_after += self.accentizer.process_all(core_text)
                        text_after += spaces
                        
                        current_idx += cut_idx
                        
                    # Append the remaining whitespace at the end, if any
                    last_stop = text_before_toks[-1].stop
                    if last_stop < len(text_before):
                        text_after += text_before[last_stop:]

            except Exception as e:
                original_stderr.write(f"❌ Ошибка RUAccent в строке '{text_before[:50]}...': {e}\n")
                return text_with_dict
            
            # Step 6: Tokenize the result
            trg_tokens = list(tokenizer_func(text_after))
            trg_ru_tokens = []
            for token in trg_tokens:
                if re.search(r'[а-яА-ЯёЁa-zA-Z]', token.text):
                    trg_ru_tokens.append(token)
            
            # Step 8: Synchronization
            if len(src_ru_indices) == len(trg_ru_tokens):
                result_tokens = list(src_tokens)
                for idx_src, token_trg in zip(src_ru_indices, trg_ru_tokens):
                    token_src = result_tokens[idx_src]
                    
                    # If the dictionary already added a stress mark - keep it
                    if '+' in token_src.text:
                        continue 
                    
                    class FakeToken:
                        def __init__(self, t): self.text = t
                    result_tokens[idx_src] = FakeToken(token_trg.text)

                sep_sizes = [y.start - x.stop for x, y in pairwise(src_tokens)]
                sep_sizes.append(0)
                with_sep = ((t.text, ' ' * sep) for t, sep in zip(result_tokens, sep_sizes))
                return ''.join(chain.from_iterable(with_sep))
            
            else:
                return text_with_dict

        except Exception as e:
            import traceback
            print(f"⚠️ Ошибка при обработке строки через razdel: {e}")
            traceback.print_exc()
            return src_text

    def check_queue(self):
        try:
            while True:
                msg_type, *args = self.progress_queue.get_nowait()
                if msg_type == "progress": self.progress_bar.set(args[0])
                elif msg_type == "success": self.progress_bar.set(1.0); messagebox.showinfo("Успех", f"Обработка завершена!\nФайл сохранен в:\n{args[0]}")
                elif msg_type == "error": self.progress_bar.set(0); messagebox.showerror("Ошибка", args[0])
        except queue.Empty: pass
        finally:
            if self.processing_thread and not self.processing_thread.is_alive(): self.start_button.configure(state="normal")
            else: self.after(100, self.check_queue)

if __name__ == "__main__":
    app = AccentizerApp()
    app.mainloop()