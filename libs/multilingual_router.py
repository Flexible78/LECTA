import numpy as np
import logging
import sys
import io
import torch
import torchaudio
import asyncio
import concurrent.futures
import re
from pydub import AudioSegment

# =====================================================================
# ГЛОБАЛЬНЫЕ НАСТРОЙКИ (Управляются из app.py двумя чекбоксами)
# =====================================================================
USE_EDGE_FOR_ENGLISH = False
USE_EDGE_FOR_HEBREW = True   # Иврит через облако Microsoft Edge TTS (по умолчанию ВКЛ — локальный TTS не поддерживает иврит)
USE_EDGE_FOR_RUSSIAN = False  # Русский через облако Microsoft Edge TTS — ОЧЕНЬ быстро, нужен интернет
DICTIONARY_MODE = False # Тот самый отдельный рубильник для пауз!

# =====================================================================
# БЛОК 1: ЦВЕТНОЕ ЛОГИРОВАНИЕ
# =====================================================================
class ColorFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)

logger = logging.getLogger("Router")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(ColorFormatter())
    logger.addHandler(ch)

def create_silence(duration_sec, sample_rate):
    return np.zeros(int(sample_rate * duration_sec), dtype=np.float32)

def normalize_audio(audio_np):
    if audio_np is None or getattr(audio_np, 'size', 0) == 0: return None
    audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=0.0, neginf=0.0)
    
    if str(audio_np.dtype).startswith('int') or np.max(np.abs(audio_np)) > 10.0:
        audio_np = audio_np.astype(np.float32) / 32768.0
    else:
        audio_np = audio_np.astype(np.float32)

    audio_np = np.clip(audio_np, -1.0, 1.0)
    return audio_np

def resample_audio(audio_np, orig_sr, target_sr):
    audio_np = normalize_audio(audio_np)
    if audio_np is None: return None
    if orig_sr == target_sr: return audio_np
    try:
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
        resampled = torchaudio.functional.resample(audio_tensor, orig_sr, target_sr)
        return resampled.squeeze(0).numpy()
    except Exception as e:
        logger.error(f"Ошибка ресемплинга: {e}")
        return audio_np

def gentle_trim_and_fade(audio_np, sample_rate):
    if audio_np is None or len(audio_np) == 0: return audio_np
    abs_audio = np.abs(audio_np)
    max_amp = np.max(abs_audio)
    
    if max_amp < 0.01: return np.zeros(10, dtype=np.float32)
    
    threshold = 0.015
    active = np.where(abs_audio > threshold)[0]
    
    if len(active) == 0: return np.zeros(10, dtype=np.float32)
    
    pad = int(sample_rate * 0.1) 
    start = max(0, active[0] - pad)
    end = min(len(audio_np), active[-1] + pad)
    
    trimmed = audio_np[start:end]
    
    fade_samples = int(sample_rate * 0.05)
    if len(trimmed) > fade_samples * 2:
        fade_curve = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        trimmed[-fade_samples:] *= fade_curve
        
    return trimmed

def split_by_words(long_text, max_chars):
    words = long_text.split()
    word_chunks = []
    curr = ""
    for w in words:
        if len(curr) + len(w) + 1 <= max_chars:
            curr += w + " "
        else:
            if curr: word_chunks.append(curr.strip())
            curr = w + " "
    if curr: word_chunks.append(curr.strip())
    return word_chunks

def smart_split(text, max_chars):
    sentences = re.split(r'(?<=[.!?\n])\s+', text)
    chunks = []
    current_chunk = ""
    
    for s in sentences:
        s = s.strip()
        if not s: continue
        
        if len(s) > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            
            sub_sentences = re.split(r'(?<=[,;:—])\s+', s)
            sub_chunk = ""
            for sub in sub_sentences:
                sub = sub.strip()
                if not sub: continue
                
                if len(sub) > max_chars:
                    if sub_chunk: 
                        chunks.append(sub_chunk.strip())
                        sub_chunk = ""
                    chunks.extend(split_by_words(sub, max_chars))
                else:
                    if len(sub_chunk) + len(sub) + 1 <= max_chars:
                        sub_chunk += sub + " "
                    else:
                        if sub_chunk: chunks.append(sub_chunk.strip())
                        sub_chunk = sub + " "
            if sub_chunk: chunks.append(sub_chunk.strip())
        else:
            if len(current_chunk) + len(s) + 1 <= max_chars:
                current_chunk += s + " "
            else:
                if current_chunk: chunks.append(current_chunk.strip())
                current_chunk = s + " "
                
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def auto_split_mixed_languages(text):
    if '|' in text:
        return [p.strip() for p in text.split('|') if p.strip()]
        
    chunks = []
    current_chunk = ""
    current_type = None 
    
    for char in text:
        if '\u0590' <= char <= '\u05FF':
            char_type = 'heb'
        elif char.isalpha(): 
            if re.match(r'[а-яА-ЯёЁ]', char):
                char_type = 'ru'
            else:
                char_type = 'en'
        else:
            char_type = current_type if current_type else 'other'
            
        if current_type is None:
            current_type = char_type
            
        if char_type == current_type or char_type == 'other':
            current_chunk += char
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = char
            current_type = char_type
            
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

# ═══ Потокобезопасный executor для Edge TTS (создаётся один раз) ═══
_edge_executor = None

def _run_async_safely(coroutine_func, timeout=120):
    """Потокобезопасный запуск async-функции. Работает и внутри event loop, и без него."""
    global _edge_executor
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Нет работающего event loop — asyncio.run() безопасен
        return asyncio.run(coroutine_func())
    else:
        # Event loop уже работает (Gradio!) — запускаем в ОТДЕЛЬНОМ потоке
        if _edge_executor is None:
            _edge_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
            logger.debug(f"[EDGE TTS] Thread executor created (event loop running)")
        future = _edge_executor.submit(asyncio.run, coroutine_func())
        return future.result(timeout=timeout)

def get_edge_audio(text, voice="he-IL-AvriNeural", target_sr=24000):
    """Озвучка через Microsoft Edge TTS — потокобезопасная версия."""
    try:
        clean_text = text.replace('"', '').replace("'", '').strip()
        if not clean_text:
            return create_silence(0.5, target_sr)

        logger.info(f"[CLOUD] Voice: {voice} | Text: {clean_text[:50]}...")

        import edge_tts

        async def _synthesize():
            communicate = edge_tts.Communicate(clean_text, voice)
            audio_bytes = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]
            return audio_bytes

        # Потокобезопасный вызов: работает и внутри event loop Gradio, и в обычных потоках
        audio_bytes = _run_async_safely(_synthesize, timeout=120)

        if not audio_bytes:
            logger.warning("Edge TTS returned empty data")
            return create_silence(1.0, target_sr)

        audio_seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        audio_seg = audio_seg.set_frame_rate(target_sr).set_channels(1)

        samples = np.array(audio_seg.get_array_of_samples())
        if audio_seg.sample_width == 2:
            audio_np = samples.astype(np.float32) / 32768.0
        elif audio_seg.sample_width == 4:
            audio_np = samples.astype(np.float32) / 2147483648.0
        else:
            audio_np = samples.astype(np.float32)

        audio_np = gentle_trim_and_fade(audio_np, target_sr)
        logger.info(f"[CLOUD OK] {len(audio_np)/target_sr:.1f}s audio")
        return audio_np

    except Exception as e:
        logger.error(f"Cloud TTS error ({voice}): {e}")
        return create_silence(2.5, target_sr)

def process_multilingual_text(text, safe_synth_func,
                             use_edge_en=None, use_edge_he=None, use_edge_ru=None, dict_mode=None):
    """Мультиязычный TTS-роутер.
    
    Параметры флагов (если None — берутся из глобальных переменных модуля):
        use_edge_en: bool — использовать Edge TTS для английского
        use_edge_he: bool — использовать Edge TTS для иврита
        use_edge_ru: bool — использовать Edge TTS для русского
        dict_mode: bool — режим словаря (длинные паузы)
    """
    # ЯВНОЕ разрешение флагов: параметры функции > глобальные переменные
    _use_en = USE_EDGE_FOR_ENGLISH if use_edge_en is None else use_edge_en
    _use_he = USE_EDGE_FOR_HEBREW if use_edge_he is None else use_edge_he
    _use_ru = USE_EDGE_FOR_RUSSIAN if use_edge_ru is None else use_edge_ru
    _dict_mode = DICTIONARY_MODE if dict_mode is None else dict_mode
    
    # Отладочный лог: какие флаги используются            logger.debug(f"[CONFIG] [ROUTER] Flags: EN={_use_en} HE={_use_he} RU={_use_ru} DICT={_dict_mode}")
    
    parts = auto_split_mixed_languages(text)
    if not parts: return None, None
    
    audio_chunks = []
    target_sr = 24000 
    
    # === НАСТОЯЩАЯ УМНАЯ ПАУЗА ===
    lang_pause = 1.0 if _dict_mode else 0.4
    
    edge_count = 0
    local_count = 0
    
    for part_idx, part in enumerate(parts):
        try:
            # === СУПЕР ФИЛЬТР МУСОРА ===
            # Если в куске нет ни одной буквы или цифры, мы его игнорируем!
            if not re.search(r'[a-zA-Zа-яА-ЯёЁ0-9\u0590-\u05FF]', part):
                continue

            is_hebrew = bool(re.search(r'[\u0590-\u05FF]', part))
            is_english = bool(re.search(r'[a-zA-Z]', part) and not re.search(r'[а-яА-ЯёЁ]', part))

            if is_hebrew and _use_he:
                logger.info(f"[EDGE-HE] Hebrew: {part[:40]}...")
                heb_audio = get_edge_audio(part, voice="he-IL-AvriNeural", target_sr=target_sr)
                audio_chunks.append(heb_audio)
                edge_count += 1
                if part_idx < len(parts) - 1:
                    audio_chunks.append(create_silence(lang_pause, target_sr))
                    
            elif is_english and _use_en:
                logger.info(f"[EDGE-EN] English: {part[:40]}...")
                eng_audio = get_edge_audio(part, voice="en-US-ChristopherNeural", target_sr=target_sr)
                audio_chunks.append(eng_audio)
                edge_count += 1
                if part_idx < len(parts) - 1:
                    audio_chunks.append(create_silence(lang_pause, target_sr))
                    
            elif not is_english and not is_hebrew and _use_ru:
                # Русский через облако Microsoft Edge TTS — мгновенная скорость!
                logger.info(f"[EDGE-RU] Russian: {part[:40]}...")
                ru_audio = get_edge_audio(part, voice="ru-RU-DmitryNeural", target_sr=target_sr)
                audio_chunks.append(ru_audio)
                edge_count += 1
                if part_idx < len(parts) - 1:
                    audio_chunks.append(create_silence(lang_pause, target_sr))
                    
            else:
                disable_n = is_english
                lang_tag = "EN-LOCAL" if disable_n else "RU-LOCAL"
                char_limit = 145 if disable_n else 110
                
                chunks = smart_split(part, max_chars=char_limit)
                if not chunks: chunks = [part]
                
                logger.info(f"[LOCAL] [{lang_tag}] TTS: {part[:40]}...")
                local_count += 1
                
                for i, chunk in enumerate(chunks):
                    # Двойная проверка перед отправкой в нейросеть!
                    if not re.search(r'[a-zA-Zа-яА-ЯёЁ0-9\u0590-\u05FF]', chunk):
                        continue
                        
                    sr, aud = safe_synth_func(chunk, disable_norm=disable_n)
                    
                    if aud is not None:
                        aud_resampled = resample_audio(aud, sr, target_sr)
                        aud_trimmed = gentle_trim_and_fade(aud_resampled, target_sr)
                        audio_chunks.append(aud_trimmed)
                        
                        if i < len(chunks) - 1:
                            audio_chunks.append(create_silence(0.12, target_sr))
                    else:
                        logger.warning(f"[{lang_tag}] Empty audio chunk {i+1}")
                            
                if len(parts) > 1 and part_idx != len(parts) - 1:
                    audio_chunks.append(create_silence(lang_pause, target_sr))
                    
        except Exception as e:
            logger.exception(f"Processing error: '{part[:50]}...'")

    valid = [c for c in audio_chunks if c is not None and getattr(c, 'size', 0) > 0]
    if not valid: 
        logger.warning("[ROUTER] No valid audio chunks")
        return None, None
        
    logger.info(f"[ROUTER OK] Edge:{edge_count} | Local:{local_count} | Chunks:{len(valid)}")
    return target_sr, np.concatenate(valid)