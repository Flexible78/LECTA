import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from collections import OrderedDict, deque
import zipfile
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path

import gradio as gr
import libs.multilingual_router as router  # для доступа к флагам USE_EDGE_*
import numpy as np
from config import AppConfig, config, TTS_WORKERS, TTS_COOLDOWN_SEC, TTS_GPU_TEMP_LIMIT_C, TTS_GPU_TEMP_RESUME_C
from libs.accent import accentizer
from libs.estimates import steps_hint_html
from libs.fb2_processor import FB2Processor
from libs.thermal import get_gpu_temp
from libs.multilingual_router import process_multilingual_text
from libs.russian import normalize_russian
from libs.audio_tools import verify_audio_file
from libs.tts import get_device, get_model_short_name, synth
from libs.tts_preprocessor import TextParse
from libs.ui_assets import (
    format_audio_time,
    format_time_hms,
    get_batch_metrics_html,
    get_batch_summary_html,
    get_metrics_html,
)
from libs.utils import convert, data_path, get_data_list, now_dir
from lxml import etree
from pydub import AudioSegment, effects
from pydub.utils import mediainfo

logger = logging.getLogger(__name__)

sound_dir = now_dir / "sound"
stop_text_to_sp = False
_synthesis_completed = False
#: Filled by batch_tts_all_projects so the background worker can refuse to
#: package a project whose audio failed.
LAST_BATCH_FAILURES: list[str] = []
LAST_BATCH_ERRORS: list[str] = []
txt_parser = TextParse(False)

# ═══ ПЕРФ: глобальный executor + кэш сегментов (разделяется между проектами) ═══
_tts_executor = ThreadPoolExecutor(max_workers=TTS_WORKERS)
_tts_sema = threading.Semaphore(TTS_WORKERS)  # limits in-flight synthesis tasks
_local_segment_cache = OrderedDict()  # LRU: keeps memory bounded on long books
_LOCAL_CACHE_MAX = 2000
_local_cache_lock = threading.Lock()
_local_tts_lock = threading.Lock()  # защита от GPU-конкуренции Silero/F5

#: Duration probes (ffprobe) are slow and the file table is rebuilt often.
_duration_cache: dict[str, tuple[int, int, float]] = {}
_duration_cache_lock = threading.Lock()
_PURE_RU_HINT = re.compile(r"[а-яА-ЯёЁ]")
_NON_RU_SCRIPT = re.compile(r"[A-Za-z\u0590-\u05FF]")


def _is_pure_russian(text: str) -> bool:
    return bool(_PURE_RU_HINT.search(text or "")) and not _NON_RU_SCRIPT.search(text)

# ═══ THERMAL AUTO-THROTTLE ═══
_last_temp_poll = 0.0
_TEMP_POLL_INTERVAL = 3.0  # секунд между проверками
_thermal_throttle_active = False


# ═══ КАРТА МОДЕЛЕЙ ДЛЯ ОТСЛЕЖИВАНИЯ ═══
# Сохраняет какое короткое имя модели использовалось для каждого MP3 файла.
# Файл: data/<project>/tts_model_map.json  →  {"filename_stem": "Silero5_5", ...}
_MODEL_MAP_FILE = "tts_model_map.json"


def _poll_gpu_temp_thermal():
    """Periodic GPU temperature check with auto-throttle.
    Returns (is_hot_bool, temp_str_suffix).
    
    Every _TEMP_POLL_INTERVAL seconds, polls GPU temp via nvidia-smi.
    If temp >= TTS_GPU_TEMP_LIMIT_C, sets _thermal_throttle_active and
    blocks for 1-second intervals until cooldown or stop.
    """
    global _last_temp_poll, _thermal_throttle_active
    now = time.monotonic()
    if now - _last_temp_poll < _TEMP_POLL_INTERVAL:
        if _thermal_throttle_active:
            return True, f"\n🌡 GPU: throttling (>{TTS_GPU_TEMP_LIMIT_C}°C)"
        t = get_gpu_temp()
        if t is not None:
            return False, f"\n🌡 GPU: {t}°C"
        return False, ""
    _last_temp_poll = now
    temp = get_gpu_temp()
    if temp is None:
        _thermal_throttle_active = False
        return False, ""
    if temp >= TTS_GPU_TEMP_LIMIT_C:
        _thermal_throttle_active = True
        waited = 0
        while temp is not None and temp >= TTS_GPU_TEMP_RESUME_C:
            if stop_text_to_sp:
                _thermal_throttle_active = False
                return True, "\n🛑 Stop requested during thermal cooldown"
            if waited >= 300:
                _thermal_throttle_active = False
                return True, f"\n⚠️ Cooldown timeout — continuing anyway (GPU: {temp}°C)"
            time.sleep(1)
            waited += 1
            temp = get_gpu_temp()
        _thermal_throttle_active = False
        return False, f"\n❄️ Cooled to {temp}°C after {waited}s"
    _thermal_throttle_active = False
    return False, f"\n🌡 GPU: {temp}°C"


def _load_model_map(ab_path):
    """Загружает карту модель→файл из JSON. Возвращает dict."""
    map_path = data_path / ab_path / _MODEL_MAP_FILE
    if map_path.exists():
        try:
            with open(map_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_model_map_entry(ab_path, file_stem, model_short):
    """Добавляет/обновляет запись о модели для файла."""
    model_map = _load_model_map(ab_path)
    model_map[str(file_stem)] = model_short
    map_path = data_path / ab_path / _MODEL_MAP_FILE
    try:
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(model_map, f, ensure_ascii=False)
    except Exception:
        pass


def _build_file_index(ab_path, rows):
    """Build {display_name: absolute_mp3_path} from DataFrame rows."""
    index = {}
    if not ab_path:
        return index
    mp3_dir = data_path / ab_path / "mp3"
    for row in rows:
        if not row:
            continue
        name = row[0]
        if name and isinstance(name, str) and name.endswith(".mp3"):
            index[name] = str((mp3_dir / name).resolve())
    return index


def _project_from_path(p):
    """Извлекает имя проекта из абсолютного пути к mp3 (data/<project>/mp3/...)."""
    try:
        rel = Path(p).resolve().relative_to(data_path.resolve())
        return rel.parts[0]
    except Exception:
        return None


def _resolve_path(filename, ab_path, file_index):
    """Возвращает (project_name, absolute_path) по отображаемому имени файла.
    Приоритет — индекс; fallback — data_path/ab_path/mp3/filename."""
    if file_index and filename in file_index:
        p = Path(file_index[filename])
        project = _project_from_path(p)
        return project if project else str(ab_path), p
    fallback = data_path / str(ab_path) / "mp3" / str(filename)
    return str(ab_path), fallback


def _short_name(name, max_len=24):
    """Умно обрезает имя файла до max_len символов, избегая разрыва слов.
    Не добавляет спецсимволы (…) в имя файла.
    Сохраняет ведущие цифры/номера для корректной сортировки."""
    if not name:
        return name
    name = str(name).strip()
    if len(name) <= max_len:
        return name
    # Пробуем обрезать по последнему разделителю (пробел, _, -, .)
    truncated = name[:max_len]
    cut = max(truncated.rfind(" "), truncated.rfind("_"), truncated.rfind("-"))
    if cut > max_len // 2:
        return truncated[:cut]
    # Если нет удачного разделителя — просто обрезаем
    return truncated.rstrip(" _-")


def _concat_audio_segments(segments):
    """O(n) конкатенация AudioSegment через raw bytes (вместо O(n²) через оператор +).
    Все сегменты приводятся к единому формату (24kHz / 16-bit / mono).
    Пустой список — это ошибка вызывающего, а не повод писать тишину."""
    if not segments:
        return AudioSegment.empty()
    ref = segments[0]
    frame_rate, sample_width, channels = ref.frame_rate, ref.sample_width, ref.channels
    parts = []
    for seg in segments:
        if seg.frame_rate != frame_rate:
            seg = seg.set_frame_rate(frame_rate)
        if seg.sample_width != sample_width:
            seg = seg.set_sample_width(sample_width)
        if seg.channels != channels:
            seg = seg.set_channels(channels)
        parts.append(seg.raw_data)
    return AudioSegment(
        data=b"".join(parts),
        frame_rate=frame_rate,
        sample_width=sample_width,
        channels=channels,
    )


def _trim_and_normalize_audio(audio, *, fade_ms: int = 8):
    """Trim edge silence once per chapter and apply one global peak gain.

    Per-chunk ``effects.normalize`` was removed: it pumped loudness between
    batches and raised the noise floor.  This function works on the finished
    chapter only, and never clips.
    """
    import math

    if audio is None or len(audio) == 0:
        raise RuntimeError("Empty audio result")
    samples = np.frombuffer(audio.raw_data, dtype=np.int16)
    if samples.size == 0:
        raise RuntimeError("Empty audio result")
    peak_sample = int(np.max(np.abs(samples)))
    if peak_sample <= 0:
        raise RuntimeError("Audio result is fully silent")
    threshold = max(1, int(peak_sample * 0.002))
    active = np.where(np.abs(samples) > threshold)[0]
    if active.size == 0:
        raise RuntimeError("Audio result is fully silent")
    sr = audio.frame_rate
    start = max(0, int(active[0]) - int(sr * 0.02))
    end = min(len(samples), int(active[-1]) + int(sr * 0.05))
    trimmed = samples[start:end]
    n = min(int(sr * fade_ms / 1000), len(trimmed) // 4)
    if n > 1:
        ramp = np.linspace(0.0, 1.0, n)
        trimmed = trimmed.copy()
        trimmed[:n] = (trimmed[:n].astype(np.float32) * ramp).astype(np.int16)
        trimmed[-n:] = (trimmed[-n:].astype(np.float32) * ramp[::-1]).astype(np.int16)
    audio = AudioSegment(
        trimmed.tobytes(), frame_rate=sr, sample_width=2, channels=1
    )
    # One global gain: only boost quiet material, never push above -1 dBFS.
    peak_dbfs = audio.max_dBFS
    if peak_dbfs < -1.0:
        gain = min(12.0, -1.0 - peak_dbfs)
        if gain > 0.1:
            audio = audio.apply_gain(gain)
    return audio


def _export_mp3_atomic(out_audio, target_path, *, bitrate, cover_file=None, tags=None):
    """Write, verify and atomically replace one MP3.

    A previous valid file is never destroyed until the new file has passed
    ffprobe + full-decode + audible-signal checks.
    """
    import os as _os

    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.stem}.{uuid.uuid4().hex[:8]}.tmp.mp3")
    try:
        kwargs = {"format": "mp3", "bitrate": f"{int(bitrate)}"}
        if cover_file and Path(cover_file).is_file():
            kwargs["cover"] = str(cover_file)
        if tags:
            kwargs["tags"] = tags
        exported = out_audio.export(str(tmp), **kwargs)
        try:
            exported.close()
        except Exception:
            pass
        check = verify_audio_file(
            str(tmp),
            expected_duration=out_audio.duration_seconds,
            min_duration=0.2,
        )
        if not check.get("ok"):
            raise RuntimeError(
                "Audio verification failed: " + "; ".join(check.get("errors") or ["unknown"])
            )
        _os.replace(tmp, target)
        return check
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _save_partial_audio(segments, target_path, *, bitrate):
    """Publish only verified audio; preserve any output from an earlier run."""
    audio = _trim_and_normalize_audio(_concat_audio_segments(segments))
    target = Path(target_path)
    if target.exists():
        target = target.with_name(f"{target.stem}_{uuid.uuid4().hex[:8]}_PARTIAL.mp3")
    _export_mp3_atomic(audio, target, bitrate=bitrate)
    return str(target)


def parse_percent_from_html(html):
    """Извлекает процент из HTML прогресс-бара get_metrics_html"""
    m = re.search(r"(\d+)%", html)
    return int(m.group(1)) if m else 0


def _detect_lang(text):
    """Определяет доминирующий язык строки: 'ru', 'en', 'he' или 'mixed'."""
    if not text:
        return "other"
    has_he = bool(re.search(r"[\u0590-\u05FF]", text))
    has_ru = bool(re.search(r"[а-яА-ЯёЁ]", text))
    has_en = bool(re.search(r"[a-zA-Z]", text))
    count = sum([has_he, has_ru, has_en])
    if count > 1:
        return "mixed"
    if has_he:
        return "he"
    if has_ru:
        return "ru"
    if has_en:
        return "en"
    return "other"


def get_project_stats(project_name):
    """Собирает статистику по MP3-файлам проекта:
    возвращает (общая_длительность_сек, общий_размер_MB, время_обработки_сек)"""
    mp3_dir = data_path / project_name / "mp3"
    times_file = data_path / project_name / "parse_times.json"

    total_duration = 0.0
    total_size_mb = 0.0

    if mp3_dir.exists():
        for mp3 in mp3_dir.glob("*.mp3"):
            if "_PARTIAL" in mp3.name:
                continue
            try:
                total_duration += _probe_duration(mp3)
                total_size_mb += mp3.stat().st_size / (1024 * 1024)
            except Exception:
                pass

    processing_time = 0.0
    if times_file.exists():
        try:
            with open(times_file, "r", encoding="utf-8") as f:
                pt = json.load(f)
            processing_time = sum(pt.values())
        except Exception:
            pass

    return total_duration, total_size_mb, processing_time


def add_reverb(audio, room_size=0.3, wet_gain=-15, decay_rate=0.4):
    num_reflections = int(4 + room_size * 12)
    output = audio
    wet = audio + wet_gain
    for _ in range(num_reflections):
        delay = random.uniform(20, 300)
        additional_decay = min(1, delay / 1000) * 10
        echo = wet - additional_decay
        if delay > 100:
            echo = echo.low_pass_filter(3000).high_pass_filter(100)
        output = output.overlay(echo, position=int(delay))
    return output


# ═══ ПЕРФ: статические хелперы для параллельного синтеза ═══

def _apply_accents_gpu_safe(text: str) -> str:
    """Accent marks must not compete with F5 for the same CUDA context."""
    if getattr(accentizer, "accentizer", None) is None:
        return text
    try:
        if get_device() == "cpu":
            return accentizer.process_accent(text, r"\+\w+|\w+\+\w+")
        with _local_tts_lock:
            return accentizer.process_accent(text, r"\+\w+|\w+\+\w+")
    except Exception:
        return text


def _safe_synth_static(text, speaker_id, speed, noise, use_accents, disable_norm=False):
    """Module-level safe_synth с кэшированием. Используется в потоках executor'а."""
    txt = text if disable_norm else txt_parser.garbage(normalize_russian(text))
    if use_accents and not disable_norm and getattr(accentizer, "accentizer", None) is not None:
        txt = _apply_accents_gpu_safe(txt)
    # ПЕРФ (CPU): F5-TTS inference steps напрямую определяют скорость на CPU.
    # Принудительно ограничиваем nfe_step до 8 на CPU — выше почти неразличимо по качеству, но в 2-4 раза медленнее.
    _is_cpu = get_device() == "cpu"
    if _is_cpu and synth.ver in (5, 6) and noise:
        try:
            noise = min(int(noise), 8)
        except Exception:
            pass

    cache_key = f"{synth.ver}|{txt}|{speaker_id}|{speed}|{noise}"
    with _local_cache_lock:
        if cache_key in _local_segment_cache:
            _local_segment_cache.move_to_end(cache_key)
            return _local_segment_cache[cache_key]
    # ПЕРФ: блокировка нужна только на GPU (конкуренция за CUDA-контекст).
    # На CPU убираем её — иначе 4 worker'а сериализуются и скорость падает в N раз.
    def _try_synth():
        if _is_cpu:
            return synth.synth_audio(txt, speaker_id=speaker_id, speed=speed, noise=noise)
        with _local_tts_lock:
            return synth.synth_audio(txt, speaker_id=speaker_id, speed=speed, noise=noise)

    # ── АНТИ-КРАШ: при сбое синтеза (OOM/температура) чистим кэш CUDA,
    # ждём 2 сек и пробуем ещё раз. Если не вышло — возвращаем None; router
    # преобразует это в явную ошибку и не экспортирует файл с тишиной.
    res = None
    for attempt in range(2):
        try:
            res = _try_synth()
            break
        except Exception as e:
            logger.warning(f"⚠️ Synth error (attempt {attempt+1}/2): {e}")
            if attempt == 0:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                time.sleep(2)
    if res is None:
        return None, None
    a, b = res
    if isinstance(a, (int, float)):
        sr, aud_raw = int(a), b
    elif isinstance(b, (int, float)):
        sr, aud_raw = int(b), a
    else:
        sr, aud_raw = 24000, a
    with _local_cache_lock:
        _local_segment_cache[cache_key] = (sr, aud_raw)
        if len(_local_segment_cache) > _LOCAL_CACHE_MAX:
            _local_segment_cache.popitem(last=False)
    return sr, aud_raw


def _execute_text_chunk(text, speaker_id, speed, noise, use_accents,
                        use_edge_en, use_edge_he, use_edge_ru, dict_mode):
    """Выполняет синтез одного текстового чанка через process_multilingual_text.
    Возвращает (sr, np_audio, elapsed_sec)."""
    t0 = time.time()

    # ПЕРФ: для чисто русских чанков нормализация и ударения выполняются один
    # раз на весь чанк, а не на каждый 110-символьный кусок внутри роутера.
    prepared = None
    if _is_pure_russian(text):
        try:
            prepared = txt_parser.garbage(normalize_russian(text))
            if use_accents and getattr(accentizer, "accentizer", None) is not None:
                prepared = _apply_accents_gpu_safe(prepared)
        except Exception:
            logger.warning("Bulk preprocessing failed, using per-chunk path", exc_info=True)
            prepared = None

    if prepared is not None:
        text = prepared

        def synth_fn(t, disable_norm=False):
            return _safe_synth_static(t, speaker_id, speed, noise, False, disable_norm=True)
    else:
        def synth_fn(t, disable_norm=False):
            return _safe_synth_static(t, speaker_id, speed, noise, use_accents, disable_norm)

    sr, np_audio = process_multilingual_text(
        text, synth_fn,
        use_edge_en=use_edge_en,
        use_edge_he=use_edge_he,
        use_edge_ru=use_edge_ru,
        dict_mode=dict_mode,
    )
    elapsed = time.time() - t0
    return sr, np_audio, elapsed


# === ОСНОВНОЙ ГЕНЕРАТОР ОЗВУЧКИ ===
def tts(
    ab_path,
    repl,
    spk_sel,
    sp_rate,
    back_sound_sel,
    bitrate,
    noise_lvl,
    use_sound_effect,
    use_accents,
    auto_parse=True,
):
    work_dir = data_path / ab_path
    xml_path = work_dir / "xml"
    mp3_path = work_dir / "mp3"
    mp3_path.mkdir(parents=True, exist_ok=True)

    # Авто-парсинг: если XML отсутствует или исходник новее — парсим автоматически
    if auto_parse:
        fb2_file = work_dir / f"{ab_path}.fb2"
        need_parse = False
        if not xml_path.exists() or not list(xml_path.glob("*.xml")):
            need_parse = True
        elif fb2_file.exists():
            fb2_mtime = fb2_file.stat().st_mtime
            xml_files = list(xml_path.glob("*.xml"))
            if not xml_files or fb2_mtime > max(f.stat().st_mtime for f in xml_files):
                need_parse = True
        # "Overwrite existing MP3" only overwrites audio: re-parse happens when
        # XML is missing or older than the FB2 source, not on every run.
        if need_parse and fb2_file.exists():
            try:
                fresh_config = AppConfig.load_user_settings()
                parse_ch_size = getattr(fresh_config, "ch_size", 200)
                parse_punctuation = getattr(fresh_config, "punctuation", False)
                parse_translit = getattr(fresh_config, "translit", True)
                parse_sound_effect = getattr(fresh_config, "sound_effect", False)
                proc = FB2Processor()
                for _ in proc.process_book(
                    ab_path=ab_path,
                    replace=repl,
                    sound_effect=parse_sound_effect,
                    punctuation=parse_punctuation,
                    translit=parse_translit,
                    ch_size=parse_ch_size,
                ):
                    pass
            except Exception as e:
                print(f"Auto-parse error: {e}")

    try:
        Path("tmp/.config/TTS-Server").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    AppConfig.save_user_settings(
        {
            "spk_sel": spk_sel,
            "sp_rate": sp_rate,
            "back_sound_sel": back_sound_sel,
            "bitrate": bitrate,
            "noise_lvl": noise_lvl,
            "use_sound_effect": use_sound_effect,
            "use_accents": use_accents,
        }
    )

    global stop_text_to_sp, _synthesis_completed
    stop_text_to_sp = False
    _synthesis_completed = False
    last_final_mp3_path = ""
    failed_files: list[str] = []

    files = [x.stem for x in xml_path.glob("*.xml")]

    # Overwrite mode: удаляем MP3 со старыми именами (например, усечёнными),
    # которых больше нет в текущем наборе — чтобы не оставались «старые файлы».
    if repl:
        expected_names = {
            f"{ab_path}.mp3" if len(files) == 1 else f"{_short_name(f)}.mp3"
            for f in files
        }
        expected_stems = {Path(n).stem for n in expected_names}
        for old_mp3 in mp3_path.glob("*.mp3"):
            if "_PARTIAL" in old_mp3.stem:
                continue  # Recovery audio belongs to the user, not stale-output cleanup.
            if old_mp3.name not in expected_names:
                try:
                    old_mp3.unlink()
                    logger.info(f"🗑 Removed stale MP3: {old_mp3.name}")
                except Exception:
                    pass
        # Чистим устаревшие записи в карте моделей и таймингах
        model_map = _load_model_map(ab_path)
        if model_map:
            removed_keys = [k for k in model_map if k not in expected_stems and "_PARTIAL" not in k]
            if removed_keys:
                for k in removed_keys:
                    model_map.pop(k, None)
                try:
                    map_path_j = data_path / ab_path / _MODEL_MAP_FILE
                    with open(map_path_j, "w", encoding="utf-8") as f:
                        json.dump(model_map, f, ensure_ascii=False)
                except Exception:
                    pass
        pt_file = work_dir / "parse_times.json"
        if pt_file.exists():
            try:
                with open(pt_file, "r", encoding="utf-8") as f:
                    pt = json.load(f)
                if any(k not in expected_stems for k in pt):
                    with open(pt_file, "w", encoding="utf-8") as f:
                        json.dump({k: v for k, v in pt.items() if k in expected_stems or "_PARTIAL" in k}, f, ensure_ascii=False)
            except Exception:
                pass

    # --- Считаем общее количество строк для точного прогресс-бара ---
    total_lines = 0
    for file in files:
        x_file = xml_path / f"{file}.xml"
        if x_file.exists():
            try:
                rt = etree.parse(str(x_file)).getroot()
                total_lines += len(rt)
            except Exception:
                pass

    if total_lines == 0:
        yield (
            get_files_list(ab_path),
            "⚠️ No lines to synthesize!",
            get_metrics_html(0, "00:00", "00:00", "0.0"),
            gr.update(),
            {},
        )
        return

    # Кэшируем список файлов в начале — не обновляем каждые 3 строки (экономит десятки ffprobe-вызовов)
    cached_files_list = get_files_list(ab_path)

    times_file = work_dir / "parse_times.json"
    parse_times = {}
    if times_file.exists():
        try:
            with open(times_file, "r", encoding="utf-8") as f:
                parse_times = json.load(f)
        except:
            pass

    is_single_file = len(files) == 1

    def safe_sort(x):
        try:
            return (0, float(x))
        except ValueError:
            return (1, x.lower())

    global_start_time = time.monotonic()
    current_line = 0
    recent_line_times = deque(maxlen=20)
    last_progress_update = 0.0

    _device = get_device()
    init_msg = "⏳ Initializing engine..."
    if _device == "cpu":
        init_msg = ("⚠️ CPU mode — F5-TTS is 10–50× slower than GPU! "
                    "Switch Compute device to 'auto' for GPU. "
                    "(Use ECO button + thermal throttle to keep GPU cool.)")

    yield (
        cached_files_list,
        init_msg,
        get_metrics_html(0, "00:00", "Estimating...", "0.0"),
        gr.update(),
        _build_file_index(ab_path, cached_files_list),
    )

    def _tts_to_audio(np_audio, sr):
        """Конвертирует numpy-аудио в AudioSegment (без per-chunk normalize)."""
        if np_audio is None or getattr(np_audio, "size", 0) <= 0:
            raise RuntimeError("Synthesis returned empty audio")
        np_audio = np.nan_to_num(np_audio, nan=0.0, posinf=0.0, neginf=0.0)
        np_audio = np.clip(np_audio, -1.0, 1.0)
        if int(np.max(np.abs(np_audio)) * 32767) <= 0:
            raise RuntimeError("Synthesis returned silent audio")
        final_audio_int16 = (np_audio * 32767).astype(np.int16)
        return AudioSegment(
            final_audio_int16.tobytes(), frame_rate=int(sr), sample_width=2, channels=1
        ).set_frame_rate(24000).set_channels(1)

    for file in sorted(files, key=safe_sort):
        if is_single_file:
            final_name = ab_path
            # Имя проекта НЕ усекаем — MP3 должен называться ровно как проект
            short_final = final_name
        else:
            final_name = file
            # Короткое имя файла (до ~20 символов) для аккуратного листинга
            short_final = _short_name(final_name)

        mp3_file = mp3_path / f"{short_final}.mp3"
        partial_mp3_file = mp3_path / f"{short_final}_PARTIAL.mp3"

        if mp3_file.exists() and not repl:
            x_file = xml_path / f"{file}.xml"
            try:
                current_line += len(etree.parse(str(x_file)).getroot())
            except:
                pass
            elapsed = time.monotonic() - global_start_time
            pct = min(99, int(current_line / total_lines * 100))
            yield (
                cached_files_list,
                f"⏭ Skip: {short_final}.mp3 already exists",
                get_metrics_html(
                    pct,
                    format_time_hms(elapsed),
                    "00:00",
                    "-",
                ),
                gr.update(),
                _build_file_index(ab_path, cached_files_list),
            )
            continue

        file_start_time = time.monotonic()
        xml_file = xml_path / f"{file}.xml"
        root = etree.parse(str(xml_file)).getroot()
        autor = root.get("autor")
        album = root.get("album")

        was_interrupted = False
        try:
            fresh_cfg = AppConfig.load_user_settings()
            lead_ms = int(getattr(fresh_cfg, "lead_silence_ms", 0) or 0)
        except Exception:
            lead_ms = 0
        lead_ms = max(0, min(lead_ms, 500))
        audio_segments = [AudioSegment.silent(duration=lead_ms, frame_rate=24000)] if lead_ms else []

        # ── БУФЕР ДЛЯ ЯЗЫКОВО-ОСНОВАННОГО БАТЧИНГА + DEFERRED SYNTH ──
        text_buffer = []
        buffer_lang = None  # текущий язык буфера
        buffer_chars = 0  # суммарная длина символов в буфере
        # ПЕРФ: F5-TTS держит GPU-лок, поэтому крупные задачи дают многоминутные
        # «зависания» прогресса. Мелкие задачи — частые обновления и меньше VRAM.
        _BATCH_MAX_CHARS = 600 if synth.ver in (5, 6) else 2000
        pending_tasks = []  # ПЕРФ: задачи синтеза + аудио-эффекты (в порядке)
        chunk_timings = []  # ПЕРФ: замеры времени на каждый чанк
        last_task_kind = None
        last_task_lang = None

        def _take_gap(kind, lang):
            """Return the gap (ms) that belongs *before* this task, then remember it."""
            nonlocal last_task_kind, last_task_lang
            if last_task_kind is None or last_task_kind in ("sound", "break") or kind in ("sound", "break"):
                gap = 0
            elif lang != last_task_lang:
                gap = int(router.PAUSE_LANGUAGE_MS)
            else:
                gap = int(router.PAUSE_PARAGRAPH_MS)
            last_task_kind, last_task_lang = kind, lang
            return max(0, gap)

        def _submit_text_task(txt):
            """Submit a text chunk to the executor, throttled by semaphore.
            Blocks until a worker slot is free."""
            _tts_sema.acquire()
            try:
                f = _tts_executor.submit(
                    _execute_text_chunk, txt, spk_sel, sp_rate, noise_lvl, use_accents,
                    router.USE_EDGE_FOR_ENGLISH, router.USE_EDGE_FOR_HEBREW,
                    router.USE_EDGE_FOR_RUSSIAN, router.DICTIONARY_MODE,
                )
            except Exception:
                _tts_sema.release()
                raise
            f.add_done_callback(lambda _: _tts_sema.release())
            return f

        def _check_gpu_temp():
            """Return GPU temp string for log, or empty string if unavailable.
            Delegates to _poll_gpu_temp_thermal for auto-throttle."""
            is_hot, suffix = _poll_gpu_temp_thermal()
            return suffix

        def _wait_if_hot():
            """Block with 1-second polls until GPU cools down, respecting stop flag.
            Returns a status message suffix."""
            temp = get_gpu_temp()
            if temp is None or temp < TTS_GPU_TEMP_LIMIT_C:
                return ""
            waited = 0
            while temp is not None and temp >= TTS_GPU_TEMP_LIMIT_C:
                if stop_text_to_sp:
                    return "\n🛑 Stop requested during cooldown"
                if waited >= 300:  # 5 minutes max
                    return f"\n⚠️ Cooldown timeout — continuing anyway (GPU: {temp}°C)"
                time.sleep(1)
                waited += 1
                temp = get_gpu_temp()
            return f"\n❄️ Cooled to {temp}°C after {waited}s"

        def flush_text_buffer():
            """ПЕРФ: собирает текст в pending_tasks вместо немедленного синтеза."""
            nonlocal text_buffer, buffer_lang, buffer_chars
            if not text_buffer:
                return
            joined = "\n".join(text_buffer)
            lang = buffer_lang
            text_buffer.clear()
            buffer_lang = None
            buffer_chars = 0
            gap = _take_gap("text", lang)
            f = _submit_text_task(joined)
            pending_tasks.append({
                "type": "text", "future": f, "chars": len(joined),
                "gap_ms": gap, "lang": lang, "text": joined,
            })

        for i, line in enumerate(root):
            if stop_text_to_sp:
                # НЕ сбрасываем stop_text_to_sp здесь! Батч-функция должна видеть, что была остановка.
                was_interrupted = True
                break  # Выходим из цикла строк, но идем дальше по коду для сохранения файла!

            current_line += 1

            # Track line timing for rolling average calculations
            now = time.monotonic()
            recent_line_times.append((current_line, now))

            # Throttle: update at most 2x/sec, at least once per 2 sec, always on last line
            should_update = (current_line == total_lines)
            if not should_update and current_line % 3 == 0:
                should_update = (now - last_progress_update >= 0.5)
            if now - last_progress_update >= 2.0:
                should_update = True

            if should_update:
                last_progress_update = now
                elapsed = now - global_start_time

                # Compute speed from recent 20 lines (or fewer) for accurate remaining estimate
                if len(recent_line_times) >= 2:
                    first = recent_line_times[0]
                    latest = recent_line_times[-1]
                    recent_lines = latest[0] - first[0]
                    recent_time = latest[1] - first[1]
                    recent_speed = recent_lines / recent_time if recent_time > 0 else 0
                else:
                    recent_speed = 0

                # Overall speed for display
                speed = current_line / elapsed if elapsed > 0 else 0

                # Remaining: use recent speed if available
                remaining_lines = total_lines - current_line
                if len(recent_line_times) < 3:
                    rem_str = "Calculating..."
                elif recent_speed > 0:
                    rem_sec = remaining_lines / recent_speed
                    rem_str = format_time_hms(rem_sec)
                else:
                    rem_str = "Calculating..."

                # Cap at 5% during Phase A (line parsing), actual synthesis is Phase B
                pct = min(50, int(current_line / total_lines * 50))

                html = get_metrics_html(
                    pct,
                    format_time_hms(elapsed),
                    rem_str,
                    (f"{speed:.1f} lines/s" if speed > 0 else "—"),
                )
                log_txt = f"📄 Buffering for synthesis: {file}.xml — line {current_line} of {total_lines} (GPU is working)"
                # НЕ вызываем get_files_list каждые 3 строки — это запускает ffprobe на всех MP3!
                # Возвращаем кэшированный список вместо [] чтобы таблица не мигала.
                yield cached_files_list, log_txt, html, gr.update(), _build_file_index(ab_path, cached_files_list)

            # --- ЛОГИКА ГЕНЕРАЦИИ ЗВУКА (ПЕРФ: сбор задач вместо немедленного синтеза) ---
            audio = AudioSegment.empty()
            if line.tag == "sound" and use_sound_effect:
                flush_text_buffer()
                _take_gap("sound", None)
                pending_tasks.append({"type": "sound", "value": line.get("value"), "gap_ms": 0})
            elif line.tag == "break":
                flush_text_buffer()
                _take_gap("break", None)
                pending_tasks.append({
                    "type": "break",
                    "ms": router.explicit_break_ms(line.get("time")),
                    "gap_ms": 0,
                })
            elif (
                line.tag in ("cite", "empty-line")
                and use_sound_effect
                and not line.text
            ):
                # cite/empty-line БЕЗ текста: просто сброс буфера
                flush_text_buffer()
            elif line.tag in ("cite", "empty-line") and use_sound_effect and line.text:
                # cite/empty-line С текстом: defer to executor
                flush_text_buffer()
                task_meta = {"type": "text_inline"}
                if line.tag == "cite":
                    task_meta["cite"] = True
                    task_meta["cite_position"] = line.get("position")
                if line.tag == "empty-line":
                    task_meta["empty_line"] = True
                f = _submit_text_task(line.text)
                task_meta["future"] = f
                task_meta["text"] = line.text
                task_meta["chars"] = len(line.text or "")
                task_meta["gap_ms"] = _take_gap("text", _detect_lang(line.text))
                pending_tasks.append(task_meta)
            elif line.text:
                # Обычный текст: language-aware batching
                stripped = line.text.strip()
                line_lang = _detect_lang(stripped)

                # mixed-строки: сброс буфера + defer to executor
                if line_lang == "mixed":
                    flush_text_buffer()
                    f = _submit_text_task(stripped)
                    pending_tasks.append({
                        "type": "text_inline", "future": f, "chars": len(stripped),
                        "gap_ms": _take_gap("text", "mixed"), "text": stripped,
                    })
                else:
                    # Проверяем: можно ли добавить в текущий буфер?
                    can_batch = (
                        buffer_lang is not None
                        and buffer_lang == line_lang
                        and buffer_chars + len(stripped) + 1 <= _BATCH_MAX_CHARS
                    )
                    if not can_batch and text_buffer:
                        flush_text_buffer()
                    text_buffer.append(stripped)
                    buffer_lang = line_lang
                    buffer_chars += len(stripped) + 1  # +1 за '\n'

            # Для cite/empty-line с текстом аудио будет создано позже из future
            if line.tag in ("cite", "empty-line") and use_sound_effect and line.text:
                audio = None  # не добавляем сейчас — будет собрано из future

            # Добавляем аудио если оно не пустое (пустое = текст ушёл в буфер)
            # ПЕРФ: не добавляем audio_segments здесь — всё идёт через pending_tasks

        # Сбрасываем остатки буфера после цикла
        flush_text_buffer()

        # ── PHASE B: разрешаем все отложенные задачи синтеза (futures) ──
        text_task_count = sum(1 for t in pending_tasks if "future" in t)
        total_tasks = text_task_count if text_task_count > 0 else len(pending_tasks)
        done_tasks = 0
        total_chars = sum(t.get("chars", 0) for t in pending_tasks) or 1
        done_chars = 0
        phase_b_start = time.monotonic()
        last_phase_b_update = 0.0
        # Rolling average for ETA (last 10 completed tasks)
        chunk_times_deque = deque(maxlen=10)
        synthesis_failed = False
        failure_reason = ""
        failure_traceback = ""
        task_index = 0
        
        # GPU temperature check before synthesis
        cool_msg = _wait_if_hot()
        
        # Yield phase B start
        yield (
            cached_files_list,
            f"🎙 Synthesizing chunk 0 of {total_tasks}...{_check_gpu_temp()}{cool_msg}",
            get_metrics_html(50, format_time_hms(time.monotonic() - global_start_time), "Estimating...", "—"),
            gr.update(),
            _build_file_index(ab_path, cached_files_list),
        )
        
        try:
            for task_index, task in enumerate(pending_tasks, 1):
                if stop_text_to_sp:
                    was_interrupted = True
                    # Phase A already synthesized futures. Recover the ready,
                    # ordered prefix without waiting for more work after Stop.
                    future = task.get("future")
                    if future is not None and (not future.done() or future.cancelled()):
                        break
                    
                if task["type"] == "text":
                    sr, np_audio, elapsed = task["future"].result()
                    chunk_timings.append(elapsed)
                    chunk_times_deque.append(elapsed)
                    audio = _tts_to_audio(np_audio, sr)
                    gap = int(task.get("gap_ms", 0) or 0)
                    if gap > 0 and audio_segments:
                        audio_segments.append(AudioSegment.silent(duration=gap, frame_rate=24000))
                    audio_segments.append(audio)
                    done_tasks += 1
                    done_chars += task.get("chars", 0)
                elif task["type"] == "text_inline":
                    sr, np_audio, elapsed = task["future"].result()
                    chunk_timings.append(elapsed)
                    chunk_times_deque.append(elapsed)
                    audio = _tts_to_audio(np_audio, sr)
                    if task.get("cite"):
                        audio = add_reverb(audio)
                        if task.get("cite_position") == "start":
                            pr = AudioSegment.from_wav(sound_dir / "pause" / "min_cite.wav")
                            pr = effects.normalize(pr)
                            audio = pr + audio
                    if task.get("empty_line"):
                        pr = AudioSegment.from_wav(sound_dir / "pause" / "empty.wav")
                        pr = effects.normalize(pr)
                        audio = pr + audio
                    gap = int(task.get("gap_ms", 0) or 0)
                    if gap > 0 and audio_segments:
                        audio_segments.append(AudioSegment.silent(duration=gap, frame_rate=24000))
                    audio_segments.append(audio)
                    done_tasks += 1
                    done_chars += task.get("chars", 0)
                elif task["type"] == "sound":
                    sf = sound_dir / "events" / f"{task['value']}.wav"
                    if not sf.is_file():
                        raise RuntimeError(f"Sound effect not found: {sf.name}")
                    audio_segments.append(effects.normalize(AudioSegment.from_wav(str(sf))))
                elif task["type"] == "break":
                    ms = int(task.get("ms", 0) or 0)
                    if ms > 0:
                        audio_segments.append(AudioSegment.silent(duration=ms, frame_rate=24000))
                
                # Throttled phase B progress yield
                now_b = time.monotonic()
                should_yield = (
                    done_tasks == total_tasks
                    or (now_b - last_phase_b_update >= 0.5)
                )
                if now_b - last_phase_b_update >= 2.0:
                    should_yield = True
                
                if should_yield and done_tasks > 0:
                    last_phase_b_update = now_b
                    frac = min(1.0, done_chars / total_chars)
                    pct = min(99, 50 + int(frac * 49))
                    elapsed_total = now_b - global_start_time
                    
                    # Real remaining time, measured from elapsed synthesis time
                    phase_b_elapsed = max(0.001, now_b - phase_b_start)
                    if done_chars > 0 and phase_b_elapsed > 1.0:
                        chars_per_sec = done_chars / phase_b_elapsed
                        rem_sec = (total_chars - done_chars) / chars_per_sec
                        rem_str = format_time_hms(rem_sec)
                        speed_str = f"{chars_per_sec:.0f} chars/s"
                    else:
                        rem_str = "Estimating..."
                        speed_str = "Estimating..."
                    
                    yield (
                        cached_files_list,
                        f"🎙 Synthesizing chunk {done_tasks} of {total_tasks}...{_check_gpu_temp()}",
                        get_metrics_html(pct, format_time_hms(elapsed_total), rem_str, speed_str),
                        gr.update(),
                        _build_file_index(ab_path, cached_files_list),
                    )
        except Exception as exc:
            failure_reason = f"{type(exc).__name__}: {exc}"
            failure_traceback = traceback.format_exc()
            logger.exception("Synthesis failed after %s completed speech chunks", done_tasks)
            synthesis_failed = True

        if synthesis_failed or was_interrupted or stop_text_to_sp:
            # Never fall through to the complete-file exporter after a failure.
            # Do not shut down the shared executor or cancel another job's work.
            for pending in pending_tasks:
                future = pending.get("future")
                if future is not None:
                    future.cancel()
            if synthesis_failed:
                if task.get("future") is not None:
                    detail = f"Failed at chunk {done_tasks + 1} of {total_tasks}"
                else:
                    detail = f"Failed at audio task {task_index} after {done_tasks} of {total_tasks} speech chunks"
                detail += f" in {short_final}.xml: {failure_reason}"
                diagnostic = {
                    "file": str(xml_file), "completed_chunks": done_tasks,
                    "total_chunks": total_tasks, "task_index": task_index,
                    "task_type": task.get("type"), "text": task.get("text", ""),
                    "error": failure_reason, "traceback": failure_traceback,
                }
                try:
                    diagnostic_path = work_dir / f"tts_failure_{uuid.uuid4().hex[:8]}.json"
                    diagnostic_path.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
                    detail += f"\nDetails: {diagnostic_path.name}"
                except Exception:
                    logger.exception("Could not save synthesis failure details")
            else:
                detail = f"Stopped after {done_tasks} of {total_tasks} speech chunks"
            LAST_BATCH_FAILURES.append(f"{ab_path}::{short_final}::{detail}")
            recovered_path = ""
            if done_tasks:
                try:
                    # Omit optional music/cover/effects here: a decoration must
                    # never prevent recovery of already synthesized speech.
                    recovered_path = _save_partial_audio(audio_segments, partial_mp3_file, bitrate=bitrate)
                    _save_model_map_entry(ab_path, Path(recovered_path).stem, get_model_short_name(synth.ver))
                    detail += f"\nPartial file saved as {Path(recovered_path).name}"
                except Exception as exc:
                    logger.exception("Partial MP3 recovery failed for %s", short_final)
                    detail += f"\nPartial MP3 could not be saved: {type(exc).__name__}: {exc}"
            else:
                detail += "\nNo completed speech; no new MP3 was saved."
            rows = get_files_list(ab_path)
            yield (
                rows,
                ("❌ " if synthesis_failed else "🛑 ") + detail,
                get_metrics_html(
                    min(99, 50 + int(min(1.0, done_chars / total_chars) * 49)),
                    format_time_hms(time.monotonic() - global_start_time),
                    "-",
                    f"incomplete: {done_tasks} of {total_tasks}",
                ),
                _safe_audio(recovered_path) if recovered_path else gr.update(),
                _build_file_index(ab_path, rows),
            )
            return

        # ПЕРФ: лог параллельного синтеза
        if chunk_timings:
            total_chunk_time = sum(chunk_timings)
            avg = total_chunk_time / len(chunk_timings)
            max_t = max(chunk_timings)
            print(f"[PERF] {len(chunk_timings)} chunks synthesized (total text tasks: {text_task_count})")
            print(f"[PERF] Avg chunk time: {avg:.2f}s | Max: {max_t:.2f}s | Total: {total_chunk_time:.1f}s")

        # --- O(n) КОНКАТЕНАЦИЯ ВСЕХ СЕГМЕНТОВ (вместо O(n²) через оператор +) ---
        out_audio = _concat_audio_segments(audio_segments)
        if len(out_audio) == 0:
            logger.error("No audio segments produced for %s", short_final)
            failed_files.append(short_final)
            LAST_BATCH_FAILURES.append(f"{ab_path}::{short_final}::no audio produced")
            yield (
                get_files_list(ab_path),
                f"❌ No audio produced for {short_final}.mp3 — nothing was saved",
                get_metrics_html(99, format_time_hms(time.monotonic() - global_start_time), "-", "failed"),
                gr.update(),
                _build_file_index(ab_path, get_files_list(ab_path)),
            )
            continue

        # --- ФИНАЛЬНАЯ ОБРАБОТКА (Сработает даже при прерывании!) ---
        if back_sound_sel:
            back_sound_file = sound_dir / "back" / back_sound_sel
            if not back_sound_file.is_file():
                logger.warning("Background music missing: %s", back_sound_file)
            else:
                pr_audio = AudioSegment.from_wav(str(back_sound_file))
                duration = pr_audio.duration_seconds
                back_duration_ms = int(duration * 1000)
                clip_length_ms = 16000

                if back_duration_ms <= clip_length_ms:
                    st_poz = 0
                else:
                    max_start = back_duration_ms - clip_length_ms
                    st_poz = random.randint(0, max_start)

                pr_audio = pr_audio[st_poz : st_poz + clip_length_ms]
                pr_audio = pr_audio.fade_out(8000).fade_in(3000) - 10
                out_audio = out_audio.overlay(pr_audio, position=0)

        try:
            out_audio = _trim_and_normalize_audio(out_audio)
        except Exception as e:
            logger.error("Final audio processing failed for %s: %s", short_final, e)
            failed_files.append(short_final)
            LAST_BATCH_FAILURES.append(f"{ab_path}::{short_final}::processing: {e}")
            yield (
                get_files_list(ab_path),
                f"❌ Audio rejected for {short_final}.mp3: {e}",
                get_metrics_html(99, format_time_hms(time.monotonic() - global_start_time), "-", "failed"),
                gr.update(),
                _build_file_index(ab_path, get_files_list(ab_path)),
            )
            continue

        cover_file = work_dir / "cover.jpg"
        tags = {
            "artist": autor,
            "title": f"{short_final}",
            "track": f"{file}",
            "album": album,
        }

        # Определяем имя файла (полное или частичное)
        save_path = str(partial_mp3_file) if was_interrupted else str(mp3_file)

        try:
            _export_mp3_atomic(
                out_audio,
                save_path,
                bitrate=bitrate,
                cover_file=cover_file if cover_file.exists() else None,
                tags=tags,
            )
        except Exception as e:
            logger.exception("MP3 export/verify failed for %s", short_final)
            failed_files.append(short_final)
            LAST_BATCH_FAILURES.append(f"{ab_path}::{short_final}::export: {e}")
            yield (
                get_files_list(ab_path),
                f"❌ Not saved: {short_final}.mp3 — {e}",
                get_metrics_html(99, format_time_hms(time.monotonic() - global_start_time), "-", "export failed"),
                gr.update(),
                _build_file_index(ab_path, get_files_list(ab_path)),
            )
            continue

        # Запоминаем модель, использованную для этого файла (читаем synth.ver непосредственно перед сохранением)
        model_short = get_model_short_name(synth.ver)
        # При прерывании файл сохраняется с суффиксом _PARTIAL — сохраняем метаданные под его реальным именем
        saved_stem = partial_mp3_file.stem if was_interrupted else mp3_file.stem
        _save_model_map_entry(ab_path, saved_stem, model_short)

        if not was_interrupted:
            last_final_mp3_path = str(mp3_file)

        elapsed_file = time.monotonic() - file_start_time
        parse_times[str(saved_stem)] = elapsed_file
        try:
            tmp_times = times_file.with_suffix(".json.tmp")
            with open(tmp_times, "w", encoding="utf-8") as f:
                json.dump(parse_times, f)
            tmp_times.replace(times_file)
        except Exception:
            pass

        if was_interrupted:
            yield (
                get_files_list(ab_path),
                f"🛑 Stopped! Partial file saved as {short_final}_PARTIAL.mp3",
                get_metrics_html(
                    99,
                    format_time_hms(time.monotonic() - global_start_time),
                    "-",
                    f"stopped at chunk {done_tasks} of {total_tasks}",
                ),
                _safe_audio(last_final_mp3_path),
                _build_file_index(ab_path, get_files_list(ab_path)),
            )
            return  # Выходим из функции полностью, чтобы не начинать следующий файл

        # ── PHASE C: Writing MP3, then 100% only when file is on disk ──
        # Update cache after saving new file
        cached_files_list = get_files_list(ab_path)
        elapsed_now = time.monotonic() - global_start_time
        # Phase C: yield "Writing MP3..."
        yield (
            cached_files_list,
            f"💾 Writing {short_final}.mp3...",
            get_metrics_html(99, format_time_hms(elapsed_now), "00:00", "Writing MP3..."),
            _safe_audio(last_final_mp3_path),
            _build_file_index(ab_path, cached_files_list),
        )
        # Confirm file is on disk, then show 100%
        if Path(save_path).is_file():
            yield (
                cached_files_list,
                f"✅ Saved: {short_final}.mp3",
                get_metrics_html(100, format_time_hms(elapsed_now), "00:00", "—"),
                _safe_audio(last_final_mp3_path),
                _build_file_index(ab_path, cached_files_list),
            )

    # ФИНАЛ — only now can we show 100%
    _synthesis_completed = not failed_files
    if failed_files:
        gr.Warning(f"Failed: {', '.join(failed_files[:5])}")
        final_msg = f"⚠️ Finished with {len(failed_files)} failed file(s): " + ", ".join(failed_files[:5])
        final_pct = 99
    else:
        gr.Info("Done")
        final_msg = "🎉 ALL FILES SUCCESSFULLY SYNTHESIZED!"
        final_pct = 100
    yield (
        cached_files_list,
        final_msg,
        get_metrics_html(
            final_pct, format_time_hms(time.monotonic() - global_start_time), "00:00", "-"
        ),
        _safe_audio(last_final_mp3_path),
        _build_file_index(ab_path, cached_files_list),
    )


def _probe_duration(path: Path) -> float:
    """ffprobe duration with a per-file cache (table refresh must stay instant)."""
    try:
        st = path.stat()
        signature = (st.st_mtime_ns, st.st_size)
    except OSError:
        return 0.0
    key = str(path)
    with _duration_cache_lock:
        cached = _duration_cache.get(key)
    if cached is not None and cached[:2] == signature:
        return cached[2]
    try:
        duration = float(mediainfo(str(path)).get("duration") or 0.0)
    except Exception:
        duration = 0.0
    with _duration_cache_lock:
        _duration_cache[key] = (*signature, duration)
    return duration


def get_files_list(ab_name):
    d_path = data_path / ab_name / "mp3"
    rows = []
    times_file = data_path / ab_name / "parse_times.json"
    parse_times = {}
    if times_file.exists():
        try:
            with open(times_file, "r", encoding="utf-8") as f:
                parse_times = json.load(f)
        except:
            pass

    # Карта: короткое имя модели для каждого файла
    model_map = _load_model_map(ab_name)

    if d_path.exists():
        files = [x for x in d_path.glob("*.mp3")]

        def safe_sort(file_path):
            stem = file_path.stem
            try:
                return (0, float(stem))
            except ValueError:
                return (1, stem.lower())

        sorted_files = sorted(files, key=safe_sort)
        # Probing hundreds of chapters serially froze the tab; run it in parallel
        # and reuse cached results on every later refresh.
        durations = {}
        if sorted_files:
            with ThreadPoolExecutor(max_workers=min(8, len(sorted_files))) as probe_pool:
                durations = dict(zip(sorted_files, probe_pool.map(_probe_duration, sorted_files)))

        for file_path in sorted_files:
            size_in_mb = round(file_path.stat().st_size / (1024 * 1024), 2)
            file_key = file_path.stem
            elapsed = parse_times.get(file_key, 0)
            ptime_str = format_audio_time(elapsed) if elapsed > 0 else "N/A"
            # Короткое имя модели (из карты) или '?'
            model_label = model_map.get(file_key, "?")
            # Частичные файлы помечаем в колонке обработки
            if "_PARTIAL" in file_path.name:
                ptime_str = "interrupted"
            duration = durations.get(file_path, 0.0)
            # Повреждённый/недописанный MP3: показываем строку, а не падаем
            duration_str = format_audio_time(duration) if duration > 0 else "—"
            rows.append(
                [
                    file_path.name,
                    model_label,
                    f"{size_in_mb}M",
                    duration_str,
                    ptime_str,
                ]
            )
    return rows


def create_zip_archive(ab_path, file_index):
    """Упаковывает ВСЕ mp3 из индекса/таблицы в zip.
    Если индекс пуст — fallback на сканирование папки текущего проекта."""
    # ⚠️ Предупреждение: синтез не завершён — архив может быть неполным
    if not _synthesis_completed:
        gr.Warning("⚠️ Synthesis not finished yet! The archive may be incomplete.")

    mp3_files = []
    if file_index:
        for p in file_index.values():
            path = Path(p)
            if path.exists():
                mp3_files.append(path)
    else:
        mp3_dir = data_path / ab_path / "mp3"
        if not mp3_dir.exists():
            raise gr.Error(f"Files folder not found!")
        mp3_files = sorted(mp3_dir.glob("*.mp3"))
    if not mp3_files:
        raise gr.Error("No MP3 files in the folder!")

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_filename = tmp.name
    try:
        added = 0
        seen_names = {}
        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in mp3_files:
                arcname = file_path.name
                # Дедуп: если имя уже встречалось — добавляем индекс, НЕ пропускаем
                if arcname in seen_names:
                    seen_names[arcname] += 1
                    stem, ext = file_path.stem, file_path.suffix
                    arcname = f"{stem}_{seen_names[arcname]}{ext}"
                else:
                    seen_names[arcname] = 1
                zipf.write(str(file_path), arcname)
                added += 1

        total_found = len(mp3_files)
        print(f"Added {added} files to archive out of {total_found} found")
        if added != total_found:
            gr.Warning(f"⚠️ Mismatch: {added} in archive vs {total_found} on disk!")

        return gr.update(visible=True, value=zip_filename)
    except Exception as e:
        if Path(zip_filename).exists():
            Path(zip_filename).unlink()
        raise gr.Error(f"Error: {str(e)}")


def create_selected_zip_archive(ab_path, selected_filenames, file_index):
    """Упаковывает ТОЛЬКО выбранные файлы в zip, используя индекс для резолва путей."""
    # ⚠️ Предупреждение: синтез не завершён — архив может быть неполным
    if not _synthesis_completed:
        gr.Warning("⚠️ Synthesis not finished yet! The archive may be incomplete.")

    if not selected_filenames:
        raise gr.Error("No files selected!")

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_filename = tmp.name
    try:
        added = 0
        seen_names = {}
        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            for fname in selected_filenames:
                _, file_path = _resolve_path(fname, ab_path, file_index)
                if not file_path.exists():
                    gr.Warning(f"File not found on disk: {fname} — skipped")
                    continue
                arcname = fname
                if arcname in seen_names:
                    seen_names[arcname] += 1
                    stem, ext = file_path.stem, file_path.suffix
                    arcname = f"{stem}_{seen_names[arcname]}{ext}"
                else:
                    seen_names[arcname] = 1
                zipf.write(str(file_path), arcname)
                added += 1

        print(f"Added {added} files to archive out of {len(selected_filenames)} selected")
        return gr.update(visible=True, value=zip_filename)
    except Exception as e:
        if Path(zip_filename).exists():
            Path(zip_filename).unlink()
        raise gr.Error(f"Error: {str(e)}")


# ── ХЕЛПЕРЫ ДЛЯ ФИКСА B: чекбоксы, массовые операции ──

def get_file_checkbox_choices(ab_name, df_output=None):
    """Возвращает список имён mp3-файлов для CheckboxGroup (со сбросом выбора).
    Подпись всегда показывает, файлы КАКОГО проекта перечислены — чтобы не
    путать со списком проектов batch TTS слева."""
    label = f"MP3 files of project: {ab_name or '(no project selected)'}"
    # Безопасно резолвим имена из df_output: файлы из ДРУГИХ проектов (batch TTS)
    # показываем с префиксом проекта, но в чекбоксы берём только файлы текущего проекта.
    choices = []
    if df_output is not None:
        mp3_dir = data_path / str(ab_name) / "mp3" if ab_name else None
        for row in df_output:
            if not row or not row[0]:
                continue
            name = str(row[0])
            if " | " in name:
                # вид "project | file.mp3" из batch TTS
                if name.startswith(f"{ab_name} | "):
                    choices.append(name.split(" | ", 1)[1])
            elif mp3_dir is not None and (mp3_dir / name).exists():
                choices.append(name)
    else:
        files = get_files_list(ab_name)
        choices = [row[0] for row in files]
    # Убираем дубли, сохраняя порядок
    seen = set()
    uniq = []
    for c in choices:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return gr.update(choices=uniq, value=[], label=label)


def delete_all_project_files(ab_name, confirm_state, df_output):
    """Удаляет ВСЕ mp3-файлы текущего проекта с диска.
    Двухкликовое подтверждение: первый клик — предупреждение, второй — удаление."""
    if not ab_name:
        gr.Warning("No project selected!")
        return df_output, gr.update(), "idle", gr.update(), {}

    if confirm_state != "confirm":
        files = get_files_list(ab_name)
        gr.Warning(f"⚠️ Will delete ALL {len(files)} MP3 files of project '{ab_name}'. Click again to confirm.")
        return (
            df_output,
            gr.update(value=f"⚠️ Confirm: delete ALL {len(files)} files"),
            "confirm",
            gr.update(),
            {},
        )

    # Второй клик — удаление
    mp3_dir = data_path / ab_name / "mp3"
    deleted = 0
    if mp3_dir.exists():
        for file_path in list(mp3_dir.glob("*.mp3")):
            try:
                file_path.unlink()
                deleted += 1
            except Exception as e:
                gr.Warning(f"Delete error {file_path.name}: {e}")
    # Чистим карту моделей проекта
    try:
        map_path_j = data_path / ab_name / _MODEL_MAP_FILE
        if map_path_j.exists():
            map_path_j.unlink()
    except Exception:
        pass

    gr.Info(f"🗑 Deleted all {deleted} MP3 files of project '{ab_name}'")
    return (
        [],
        gr.update(value="🗑 Delete ALL project files"),
        "idle",
        gr.update(choices=[], value=[]),
        {},
    )


def select_all_files(ab_name, df_output=None):
    """Выбрать все mp3-файлы проекта/таблицы."""
    if df_output is not None:
        return gr.update(value=[row[0] for row in df_output])
    files = get_files_list(ab_name)
    return gr.update(value=[row[0] for row in files])


def deselect_all_files():
    """Снять выделение со всех файлов."""
    return gr.update(value=[])


def delete_selected_files(selected_filenames, ab_name, confirm_state, file_index, df_output):
    """Удаляет выбранные файлы с диска и возвращает обновлённую таблицу.
    Двухкликовое подтверждение: первый клик — предупреждение, второй — удаление."""
    if not selected_filenames:
        gr.Warning("No files selected!")
        return df_output, gr.update(), "idle", gr.update(), file_index

    if confirm_state != "confirm":
        files_str = ", ".join(selected_filenames[:5])
        if len(selected_filenames) > 5:
            files_str += f" ... and {len(selected_filenames) - 5} more"
        gr.Warning(f"⚠️ Will be deleted: {files_str}. Click again to confirm.")
        return (
            gr.update(),
            gr.update(value="⚠️ Confirm deletion"),
            "confirm",
            gr.update(),
            file_index,
        )

    # Второй клик — удаление
    deleted = 0
    errors = 0
    for fname in selected_filenames:
        proj, file_path = _resolve_path(fname, ab_name, file_index)
        if file_path.exists():
            try:
                file_path.unlink()
                deleted += 1
                # Чистим запись о модели в РЕАЛЬНОМ проекте файла
                model_map = _load_model_map(proj)
                if file_path.stem in model_map:
                    model_map.pop(file_path.stem, None)
                    map_path_j = data_path / proj / _MODEL_MAP_FILE
                    try:
                        with open(map_path_j, "w", encoding="utf-8") as f:
                            json.dump(model_map, f, ensure_ascii=False)
                    except Exception:
                        pass
            except Exception as e:
                gr.Warning(f"Delete error {fname}: {e}")
                errors += 1

    # Удаляем записи из индекса
    for fname in selected_filenames:
        if file_index and fname in file_index:
            del file_index[fname]

    new_df = [row for row in (df_output or []) if row[0] not in selected_filenames]
    new_choices = gr.update(choices=[row[0] for row in new_df])

    gr.Info(f"🗑 Deleted {deleted} files" + (f", errors: {errors}" if errors else ""))
    return (
        new_df,
        gr.update(value="🗑 Delete selected files"),
        "idle",
        new_choices,
        file_index,
    )


def download_selected_files_zip(ab_name, selected_filenames, file_index):
    """Создаёт zip из выбранных файлов (через create_selected_zip_archive)."""
    if not selected_filenames:
        raise gr.Error("No files selected!")
    return create_selected_zip_archive(ab_name, selected_filenames, file_index)


def download_current_file(file_path):
    """Скачать выбранный mp3-файл."""
    if not file_path or not Path(file_path).is_file():
        raise gr.Error("File not found!")
    return gr.update(value=file_path)


# ── УДАЛЕНИЕ ПРОЕКТОВ ИЗ ПАКЕТНОГО СПИСКА TTS ──
# Внутренние каталоги LECTA никогда не считаются проектами.
_PROTECTED_PROJECT_DIRS = {"tts_cache", "__pycache__", "_audio_tools", "_archive"}


def _list_project_dirs():
    """Реальные проекты в data_path (без служебных и скрытых каталогов).
    Специально читает tts_tab.data_path, чтобы функцию можно было тестировать
    на временном каталоге."""
    base = Path(data_path)
    if not base.is_dir():
        return []
    return sorted(
        item.name
        for item in base.iterdir()
        if item.is_dir()
        and item.name not in _PROTECTED_PROJECT_DIRS
        and not item.name.startswith(("_", "."))
    )


def _bg_job_running() -> str:
    """Имя проекта, который сейчас обрабатывает фоновый воркер, или ''.
    Статус старше 15 минут считаем протухшим, чтобы упавший воркер не
    блокировал удаление навсегда."""
    try:
        status_file = _bg_status_path()
        if not status_file.exists():
            return ""
        if time.time() - status_file.stat().st_mtime > 15 * 60:
            return ""
        data = json.loads(status_file.read_text(encoding="utf-8"))
        if str(data.get("state", "")).lower() in {"running", "starting"}:
            return str(data.get("project") or "").strip() or "background job"
    except Exception:
        return ""
    return ""


def _busy_reason() -> str:
    """Причина, по которой удалять проекты сейчас нельзя ('' = можно)."""
    busy = _bg_job_running()
    if busy:
        return (
            f"Background worker is busy with '{busy}'. "
            "Wait for it to finish (or close the job) before deleting projects."
        )
    return ""


def _real_project_names(names):
    """Оставляет только существующие реальные проекты — защита от удаления
    служебных каталогов и путей вида ../."""
    available = set(_list_project_dirs())
    result = []
    for raw in names or []:
        name = str(raw).strip()
        if name in available and name not in result:
            result.append(name)
    return result


def _delete_project_dirs(names):
    """Удаляет каталоги проектов. Возвращает (deleted, errors)."""
    deleted, errors = [], []
    for name in names:
        target = Path(data_path) / name
        try:
            if target.is_dir():
                shutil.rmtree(target)
                deleted.append(name)
                logger.info("Project folder deleted: %s", target)
        except Exception as exc:  # noqa: BLE001 - show the real OS error
            errors.append(f"{name}: {exc}")
            logger.error("Project delete failed %s: %s", target, exc)
    return deleted, errors


def _project_delete_view(keep_project, button_label):
    """Обновления всех зависимых компонентов после удаления проектов."""
    remaining = _list_project_dirs()
    keep = keep_project if keep_project in remaining else ""
    return (
        gr.update(choices=remaining, value=[]),   # batch_project_sel: снимаем отметки
        gr.update(choices=remaining, value=keep),  # ab_path: текущий проект (если жив)
        [],                                        # df_output
        {},                                        # batch_file_index_state
        gr.update(choices=[], value=[]),           # file_checkboxes
        gr.update(value=button_label),
    )


def delete_selected_projects(selected_projects, confirm_state, current_project, df_output, file_index):
    """Удаляет отмеченные в списке проекты (двухкликовое подтверждение)."""
    busy = _busy_reason()
    if busy:
        gr.Warning(busy)
        return (gr.update(), gr.update(), df_output, file_index, gr.update(),
                gr.update(value="🗑 Delete selected"), "idle")

    names = _real_project_names(selected_projects)
    if not names:
        gr.Warning("Tick at least one project in the list first.")
        return (gr.update(), gr.update(), df_output, file_index, gr.update(),
                gr.update(value="🗑 Delete selected"), "idle")

    if confirm_state != "confirm":
        preview = ", ".join(names[:5]) + (f" … +{len(names) - 5}" if len(names) > 5 else "")
        gr.Warning(
            f"⚠️ Will permanently delete {len(names)} project(s) with all their files: "
            f"{preview}. Click the button again to confirm."
        )
        return (
            gr.update(),
            gr.update(),
            df_output,
            file_index,
            gr.update(),
            gr.update(value=f"⚠️ Confirm: delete {len(names)} project(s)"),
            "confirm",
        )

    deleted, errors = _delete_project_dirs(names)
    for name in deleted:
        logger.info("Deleted project: %s", name)
    if errors:
        gr.Warning(f"Deleted {len(deleted)} project(s), {len(errors)} error(s): " + "; ".join(errors[:3]))
    else:
        gr.Info(f"🗑 Deleted {len(deleted)} project(s): " + ", ".join(deleted[:5])
                + (" …" if len(deleted) > 5 else ""))
    return (*_project_delete_view(current_project, "🗑 Delete selected"), "idle")


def delete_all_projects(confirm_state, current_project, df_output, file_index):
    """Удаляет ВСЕ проекты из data_path (двухкликовое подтверждение)."""
    busy = _busy_reason()
    if busy:
        gr.Warning(busy)
        return (gr.update(), gr.update(), df_output, file_index, gr.update(),
                gr.update(value="💣 Delete ALL projects"), "idle")

    names = _list_project_dirs()
    if not names:
        gr.Warning("There are no projects to delete.")
        return (gr.update(), gr.update(), df_output, file_index, gr.update(),
                gr.update(value="💣 Delete ALL projects"), "idle")

    if confirm_state != "confirm":
        gr.Warning(
            f"⚠️ This will permanently delete ALL {len(names)} projects with their audio, "
            "text, covers and archives. Click again to confirm."
        )
        return (
            gr.update(),
            gr.update(),
            df_output,
            file_index,
            gr.update(),
            gr.update(value=f"⚠️ Confirm: delete ALL {len(names)} projects"),
            "confirm",
        )

    deleted, errors = _delete_project_dirs(names)
    if errors:
        gr.Warning(f"Deleted {len(deleted)} project(s), {len(errors)} error(s): " + "; ".join(errors[:3]))
    else:
        gr.Info(f"💣 Deleted ALL {len(deleted)} project(s)")
    return (*_project_delete_view("", "💣 Delete ALL projects"), "idle")


def snd_list():
    snd_path = sound_dir / "back"
    if snd_path.exists():
        return gr.update(value="", choices=sorted([x.name for x in snd_path.iterdir()]))
    return gr.update(value="", choices=[])


def del_file(filename, ab_name, file_index, df_output):
    if not filename:
        gr.Warning("Select a file in the table first, then click Delete")
        return df_output, file_index
    file_path = Path(filename)
    # Ищем отображаемое имя по реальному пути
    display_name = None
    if file_index:
        for disp, p in file_index.items():
            if Path(p).resolve() == file_path.resolve():
                display_name = disp
                break
    if file_path.exists():
        try:
            file_path.unlink()
            gr.Info(f"Deleted file {file_path.name}", duration=2)
            # Чистим запись о модели в РЕАЛЬНОМ проекте файла
            proj = _project_from_path(file_path) or str(ab_name)
            model_map = _load_model_map(proj)
            if file_path.stem in model_map:
                model_map.pop(file_path.stem, None)
                map_path = data_path / proj / _MODEL_MAP_FILE
                try:
                    with open(map_path, "w", encoding="utf-8") as f:
                        json.dump(model_map, f, ensure_ascii=False)
                except Exception:
                    pass
        except Exception as e:
            gr.Warning(f"Delete error: {e}")
    # Удаляем запись из индекса
    if file_index and display_name and display_name in file_index:
        del file_index[display_name]
    new_df = [row for row in (df_output or []) if row[0] != display_name]
    return new_df, file_index


def sel_file(data: gr.SelectData, ab_path, file_index):
    # Имя файла теперь чистый текст (колонка 0)
    filename_raw = data.row_value[0]
    if not filename_raw or not str(filename_raw).endswith(".mp3"):
        gr.Warning("Could not determine file name from table row")
        return gr.update(), "", gr.update(), gr.update(), gr.update(), gr.update()
    _, selected_path = _resolve_path(filename_raw, ab_path, file_index)
    stem_name = selected_path.stem
    return (
        gr.update(),
        str(selected_path),
        gr.update(visible=True),
        gr.update(value=stem_name),
        str(selected_path),
        gr.update(),
    )


def rename_selected_file(current_path, new_name, ab_path, file_index, df_output):
    if not current_path or not new_name:
        return df_output, gr.update(visible=False), file_index
    old_path = Path(current_path)
    if not old_path.exists():
        gr.Warning("File not found!")
        return df_output, gr.update(visible=False), file_index

    safe_name = "".join(
        [c for c in new_name if c.isalnum() or c in (" ", "_", "-")]
    ).rstrip()
    if not safe_name:
        safe_name = "renamed_audio"
    new_path = old_path.parent / f"{safe_name}.mp3"

    if new_path.exists() and new_path != old_path:
        gr.Warning("A file with this name already exists!")
        return df_output, gr.update(), file_index

    # Определяем отображаемое имя в индексе
    display_name = None
    if file_index:
        for disp, p in file_index.items():
            if p == str(old_path):
                display_name = disp
                break

    try:
        old_path.rename(new_path)
        gr.Info(f"File successfully renamed to {safe_name}.mp3")
        proj = _project_from_path(old_path) or str(ab_path)
        times_file = data_path / proj / "parse_times.json"
        if times_file.exists():
            with open(times_file, "r", encoding="utf-8") as f:
                parse_times = json.load(f)
            old_stem = old_path.stem
            if old_stem in parse_times:
                parse_times[safe_name] = parse_times.pop(old_stem)
                with open(times_file, "w", encoding="utf-8") as f:
                    json.dump(parse_times, f)
        # Переносим запись о модели в tts_model_map.json РЕАЛЬНОГО проекта
        model_map = _load_model_map(proj)
        if old_path.stem in model_map:
            model_map[safe_name] = model_map.pop(old_path.stem)
            map_path = data_path / proj / _MODEL_MAP_FILE
            try:
                with open(map_path, "w", encoding="utf-8") as f:
                    json.dump(model_map, f, ensure_ascii=False)
            except Exception:
                pass
        # Обновляем индекс
        if file_index and display_name:
            del file_index[display_name]
            new_display_name = display_name
            if " | " in new_display_name:
                parts = new_display_name.split(" | ")
                new_display_name = f"{parts[0]} | {new_path.name}"
            else:
                new_display_name = new_path.name
            file_index[new_display_name] = str(new_path.resolve())
        # Обновляем таблицу
        if df_output is not None:
            new_df = []
            for row in df_output:
                if row[0] == display_name:
                    new_row = list(row)
                    new_row[0] = new_display_name
                    new_df.append(new_row)
                else:
                    new_df.append(row)
            df_output = new_df
    except Exception as e:
        gr.Warning(f"Rename error: {e}")
    return df_output, gr.update(visible=False), file_index


def stop_tts():
    global stop_text_to_sp
    stop_text_to_sp = True
    return "🛑 Stopping after current line (saving file)..."


# ═══ ФОНОВЫЙ ВОРКЕР (переживает закрытие GUI/браузера) ═══
# Запускает полный цикл «парсинг + озвучка + пакет из 3 файлов» в отдельном
# детачнутом процессе. GUI может упасть или быть закрытым — работа идёт.

def _bg_status_path() -> Path:
    return Path(__file__).resolve().parent.parent / "tmp" / "bg_worker_status.json"


def _bg_job_path() -> Path:
    """Distinct from the status path — writing the job over the status file
    used to make the worker read an empty job (and process every project)."""
    return Path(__file__).resolve().parent.parent / "tmp" / "bg_worker_job.json"


def _bg_worker_script() -> Path:
    return Path(__file__).resolve().parent.parent / "worker.py"


def _bg_log_path() -> Path:
    return Path(__file__).resolve().parent.parent / "tmp" / "bg_worker.log"


def launch_background_job(
    ab_path,
    spk_sel=None,
    sp_rate=None,
    back_sound_sel=None,
    bitrate=None,
    noise_lvl=None,
    use_sound_effect=None,
    use_accents=None,
    repl=None,
    selected_projects=None,
    auto_open=None,
    sound_done=None,
):
    """Запускает фоновый воркер (полный цикл: парсинг + TTS + пакет).
    Возвращает строку-статус для лога. Процесс отвязан от GUI — его можно закрыть.
    Параметры None подтягиваются из сохранённых настроек."""
    projects = []
    if selected_projects and isinstance(selected_projects, list) and len(selected_projects) > 0:
        projects = sorted(selected_projects)
    elif ab_path and str(ab_path).strip() and not str(ab_path).startswith("<gradio"):
        projects = [str(ab_path)]
    else:
        projects = sorted(get_data_list())

    if not projects:
        raise gr.Error("No projects selected!")

    # Читаем сохранённые настройки (парсинг + TTS), как в остальном приложении
    try:
        fresh_cfg = AppConfig.load_user_settings()
        ps_ch_size = getattr(fresh_cfg, "ch_size", 200)
        ps_punctuation = getattr(fresh_cfg, "punctuation", False)
        ps_translit = getattr(fresh_cfg, "translit", True)
        ps_sound_effect = getattr(fresh_cfg, "sound_effect", False)
        saved_spk = getattr(fresh_cfg, "spk_sel", "")
        saved_rate = getattr(fresh_cfg, "sp_rate", 1.0)
        saved_back = getattr(fresh_cfg, "back_sound_sel", "")
        saved_bitrate = getattr(fresh_cfg, "bitrate", 96)
        saved_noise = getattr(fresh_cfg, "noise_lvl", 10)
        saved_accents = bool(getattr(fresh_cfg, "use_accents", True))
        saved_lead = int(getattr(fresh_cfg, "lead_silence_ms", 0) or 0)
    except Exception:
        ps_ch_size, ps_punctuation, ps_translit, ps_sound_effect = 200, False, True, False
        saved_spk, saved_rate, saved_back, saved_bitrate, saved_noise = "", 1.0, "", 96, 10
        saved_accents, saved_lead = True, 0

    # Текущий выбор модели и облачные флаги
    model_ver = getattr(synth, "ver", None) or 5
    device = get_device()

    job = {
        "projects": projects,
        "parse": {
            "sound_effect": bool(ps_sound_effect),
            "punctuation": bool(ps_punctuation),
            "translit": bool(ps_translit),
            "ch_size": int(ps_ch_size),
        },
        "tts": {
            "model_ver": int(model_ver),
            "device": device,
            "spk_sel": (spk_sel if spk_sel else saved_spk) or "",
            "sp_rate": float(sp_rate if sp_rate is not None else saved_rate),
            "back_sound_sel": (back_sound_sel if back_sound_sel else saved_back) or "",
            "bitrate": int(bitrate if bitrate is not None else saved_bitrate),
            "noise_lvl": int(noise_lvl if noise_lvl is not None else saved_noise),
            "use_sound_effect": bool(use_sound_effect) if use_sound_effect is not None else bool(ps_sound_effect),
            "use_accents": bool(use_accents) if use_accents is not None else saved_accents,
            "repl": bool(repl) if repl is not None else True,
            "use_edge_en": router.USE_EDGE_FOR_ENGLISH,
            "use_edge_he": router.USE_EDGE_FOR_HEBREW,
            "use_edge_ru": router.USE_EDGE_FOR_RUSSIAN,
            "dict_mode": router.DICTIONARY_MODE,
            "auto_open": True if auto_open is None else bool(auto_open),
            "audio": router.generation_settings(),
            "lead_silence_ms": saved_lead,
        },
        "sound_done": True if sound_done is None else bool(sound_done),
        "status_file": str(_bg_status_path()),
        "log_file": str(_bg_log_path()),
    }

    job_file = _bg_job_path()
    try:
        job_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_job = job_file.with_name(f".{job_file.name}.{uuid.uuid4().hex[:8]}.tmp")
        tmp_job.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_job, job_file)
    except Exception as e:
        raise gr.Error(f"Cannot write job file: {e}")

    # Сбрасываем статус, чтобы GUI не показывал старый
    try:
        _bg_status_path().write_text(
            json.dumps({"state": "starting", "stage": "init", "project": "", "pct": 0,
                        "message": "Launching...", "error": None, "package_dir": None},
                       ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass

    worker_py = _bg_worker_script()
    if not worker_py.exists():
        raise gr.Error(f"Worker script not found: {worker_py}")

    flags = 0
    if os.name == "nt":
        # Отвязываем процесс от консоли: переживёт закрытие GUI/браузера
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    try:
        logf = open(str(_bg_log_path()), "w", encoding="utf-8")
        subprocess.Popen(
            [sys.executable, str(worker_py), str(job_file)],
            stdout=logf,
            stderr=subprocess.STDOUT,
            creationflags=flags,
            cwd=str(Path(__file__).resolve().parent.parent),
            close_fds=True,
        )
    except Exception as e:
        raise gr.Error(f"Background launch failed: {e}")

    will_open = " and opened in Explorer." if (auto_open is None or auto_open) else "."
    return (
        f"⚡ Background job started for {len(projects)} project(s): "
        + ", ".join(projects)
        + ". You can close the GUI and the browser — processing continues. "
        "The 3-file package will be saved next to the source"
        + will_open
    )


def read_bg_status():
    """Читает status-файл фонового воркера и возвращает HTML для панели.
    Полностью crash-proof: никогда не бросает исключение."""
    status_file = _bg_status_path()
    if not status_file.exists():
        return "<div style='color:#94a3b8;font-size:13px;'>Background: no job</div>"
    try:
        st = json.loads(status_file.read_text(encoding="utf-8"))
    except Exception:
        return "<div style='color:#eab308;font-size:13px;'>Background: reading...</div>"
    state = st.get("state", "running")
    pct = st.get("pct", 0)
    msg = st.get("message", "") or ""
    project = st.get("project", "") or ""
    pkg = st.get("package_dir", "") or ""
    stage = st.get("stage", "") or ""

    if state == "done":
        color, label = "#10b981", "✅ Background done"
    elif state == "error":
        color, label = "#f43f5e", "❌ Background error"
    else:
        color, label = "#38bdf8", f"⏳ Background: {stage}"
    html = f"<div style='color:{color};font-size:13px;'><b>{label}</b> — {pct}%"
    if project:
        html += f" | {project}"
    if msg:
        html += f"<br><span style='color:#94a3b8;'>{msg[:160]}</span>"
    if pkg:
        html += f"<br>📦 {pkg}"
    html += "</div>"
    return html


# === ПАКЕТНАЯ ОЗВУЧКА ВСЕХ ПРОЕКТОВ ===
def batch_tts_all_projects(
    spk_sel,
    sp_rate,
    back_sound_sel,
    bitrate,
    noise_lvl,
    use_sound_effect,
    use_accents,
    repl,
    selected_projects=None,
    auto_open=True,
):
    """Пакетная озвучка выбранных проектов.
    Если selected_projects не указан или пуст — берутся все проекты."""
    global LAST_BATCH_FAILURES, LAST_BATCH_ERRORS
    LAST_BATCH_FAILURES = []
    LAST_BATCH_ERRORS = []
    if (
        selected_projects
        and isinstance(selected_projects, list)
        and len(selected_projects) > 0
    ):
        all_projects = sorted(selected_projects)
    else:
        all_projects = sorted(get_data_list())

    if not all_projects:
        yield (
            [],
            "⚠️ No projects for batch TTS!",
            get_batch_metrics_html("", 0, 0, 0, 0, "00:00", "00:00", "0.0"),
            gr.update(),
            batch_file_index,
        )
        return

    global stop_text_to_sp, _synthesis_completed
    stop_text_to_sp = False
    _synthesis_completed = False

    total = len(all_projects)

    # --- Pre-scan total lines across all projects for accurate batch progress ---
    total_lines_batch = 0
    for project in all_projects:
        xml_dir = data_path / project / "xml"
        if xml_dir.exists():
            for xf in xml_dir.glob("*.xml"):
                try:
                    total_lines_batch += len(etree.parse(str(xf)).getroot())
                except Exception:
                    pass

    global_start = time.monotonic()
    batch_recent_line_times = deque(maxlen=20)
    batch_last_progress_update = 0.0
    processed_count = 0
    batch_current_line = 0
    all_files = []  # НАКАПЛИВАЕМ MP3 ВСЕХ ПРОЕКТОВ
    batch_stats = []  # для финальной сводной таблицы
    batch_file_index = {}  # display_name -> absolute_mp3_path
    seen_display_names = set()  # для отслеживания коллизий имён

    # Загружаем настройки парсинга из сохранённой конфигурации
    try:
        fresh_config = AppConfig.load_user_settings()
        parse_ch_size = getattr(fresh_config, "ch_size", 200)
        parse_punctuation = getattr(fresh_config, "punctuation", False)
        parse_translit = getattr(fresh_config, "translit", True)
        parse_sound_effect = getattr(fresh_config, "sound_effect", False)
    except Exception:
        parse_ch_size = 200
        parse_punctuation = False
        parse_translit = True
        parse_sound_effect = False

    prep_msg = "Preparing..." if total_lines_batch == 0 else f"📦 Preparing {total} projects ({total_lines_batch} lines)..."
    yield (
        [],
        prep_msg,
        get_batch_metrics_html("Preparing...", 0, 0, total, 0, "00:00", "...", "..."),
        gr.update(),
        batch_file_index,
    )

    last_final_mp3_path = ""

    for idx, project in enumerate(all_projects, 1):
        if stop_text_to_sp:
            stop_text_to_sp = False
            elapsed = time.monotonic() - global_start
            pct = min(99, int(processed_count / total * 100)) if total > 0 else 0

            # Показываем частичную сводку по завершённым проектам
            partial_html = get_batch_metrics_html(
                f"stopped at {processed_count} of {total}",
                0,
                idx,
                total,
                pct,
                format_time_hms(elapsed),
                "-",
                "Stopped",
            )
            if batch_stats:
                partial_html = get_batch_summary_html(batch_stats) + "\n" + partial_html

            yield (
                all_files,
                f"🛑 Stopped! Synthesized {processed_count} of {total}",
                partial_html,
                _safe_audio(last_final_mp3_path),
                batch_file_index,
            )
            return

        work_dir = data_path / project
        xml_dir = work_dir / "xml"
        fb2_file = work_dir / f"{project}.fb2"

        # ── THERMAL: check GPU temp before starting each project ──
        _poll_gpu_temp_thermal()

        has_xml = xml_dir.exists() and bool(list(xml_dir.glob("*.xml")))

        # txt-проект без fb2 и без xml → создаём fb2 из первого *.txt
        if not has_xml and not fb2_file.exists():
            txt_files = sorted(work_dir.glob("*.txt"))
            if txt_files:
                try:
                    raw_text = txt_files[0].read_text(encoding="utf-8", errors="ignore")
                    raw_text = raw_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    body = "".join(f"<p>{p.strip()}</p>\n" for p in raw_text.split("\n") if p.strip())
                    fb2_xml = (
                        '<?xml version="1.0" encoding="utf-8"?>\n'
                        '<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">\n'
                        f'  <description><title-info><book-title>{project}</book-title></title-info></description>\n'
                        f'  <body><section>\n{body}</section></body>\n'
                        '</FictionBook>'
                    )
                    fb2_file.write_text(fb2_xml, encoding="utf-8")
                except Exception as e:
                    yield (all_files, f"⚠️ [{idx}/{total}] {project}: failed to create fb2 from txt: {e}", gr.update(), gr.update(), batch_file_index)
                    continue

        # нет ни xml, ни fb2 (и txt не нашёлся) → пропуск с логом
        if not has_xml and not fb2_file.exists():
            yield (all_files, f"⚠️ [{idx}/{total}] {project}: skipped — no XML/FB2/TXT", gr.update(), gr.update(), batch_file_index)
            continue

        # Авто-парсинг FB2 если XML ещё нет, либо включён Overwrite (repl) и есть FB2
        need_parse = (not has_xml) or (repl and fb2_file.exists())
        if need_parse:
            batch_pre_pct = int((idx - 1) / total * 100)
            elapsed = time.monotonic() - global_start
            speed = idx / elapsed if elapsed > 0 else 0
            rem = max(0, (total - idx) / speed) if speed > 0 else 0
            sec_per_proj = max(0, 1 / speed) if speed > 0 else 0
            yield (
                all_files,
                f"📁 [{idx}/{total}] {project}: Parsing FB2 → XML...",
                get_batch_metrics_html(
                        f"project {idx} of {total} — {project}",
                        0,
                        idx,
                        total,
                        batch_pre_pct,
                        format_time_hms(elapsed),
                        format_time_hms(rem),
                        f"~{format_time_hms(sec_per_proj)}/proj",
                    ),
                    gr.update(),
                    batch_file_index,
                )
            try:
                # Используем сохранённые настройки парсинга из конфигурации
                proc = FB2Processor()
                for _ in proc.process_book(
                    ab_path=project,
                    replace=repl,
                    sound_effect=parse_sound_effect,
                    punctuation=parse_punctuation,
                    translit=parse_translit,
                    ch_size=parse_ch_size,
                ):
                    if stop_text_to_sp:
                        break

                # Авто-очистка XML после парсинга (как в parse_tab)
                clean_album = re.sub(r"[\W_]*\d{6,}[\W_]*\d*$", "", project)
                if len(clean_album) > 35:
                    clean_album = clean_album[:35] + "..."
                if not clean_album:
                    clean_album = project[:20]

                def _clean_head(match):
                    speak_tag = match.group(1)
                    p1_text = match.group(2)
                    if len(p1_text) < 150:
                        return f'{speak_tag}\n  <break time="30"/>\n'
                    else:
                        return (
                            f'{speak_tag}\n  <break time="30"/>\n  <p>{p1_text}</p>\n'
                        )

                for xml_file in xml_dir.glob("*.xml"):
                    content = xml_file.read_text(encoding="utf-8")
                    content = re.sub(r"<\?xml[^>]*\?>\s*", "", content)
                    content = re.sub(
                        r'(<(?:speak|root)[^>]*album=")[^"]*("?[^>]*>)',
                        rf"\g<1>{clean_album}\g<2>",
                        content,
                    )
                    content = re.sub(
                        r'(<(?:speak|root)[^>]*title=")[^"]*("?[^>]*>)',
                        rf"\g<1>{clean_album}\g<2>",
                        content,
                    )
                    content = re.sub(
                        r"(<(?:speak|root)[^>]*>)[\s\S]*?<p>(.*?)</p>\s*",
                        _clean_head,
                        content,
                        count=1,
                    )
                    content = re.sub(
                        r"\s*<p>\s*[a-zA-Z0-9_]+_csv_[^<]*</p>",
                        "",
                        content,
                        flags=re.IGNORECASE,
                    )
                    content = re.sub(
                        r"\s*<p>[^<]*(ru en he|ru en|en ru|he en)[^<]*</p>",
                        "",
                        content,
                        flags=re.IGNORECASE,
                    )
                    xml_file.write_text(content, encoding="utf-8")
            except Exception as e:
                yield (
                    all_files,
                    f"⚠️ [{idx}/{total}] {project}: Parse error: {e}",
                    get_batch_metrics_html(
                        f"project {idx} of {total} — {project}",
                        0,
                        idx,
                        total,
                        batch_pre_pct,
                        format_time_hms(elapsed),
                        "...",
                        "...",
                    ),
                    gr.update(),
                    batch_file_index,
                )
                continue

        # Запуск TTS для проекта
        proj_start = time.monotonic()
        for tts_result in tts(
            project,
            repl,
            spk_sel,
            sp_rate,
            back_sound_sel,
            bitrate,
            noise_lvl,
            use_sound_effect,
            use_accents,
            auto_parse=False,
        ):
            if stop_text_to_sp:
                break
            files_list, log_msg, tts_html, audio_update, _ = tts_result

            # Извлекаем процент текущего проекта из HTML tts()
            project_pct = parse_percent_from_html(tts_html)

            # Track batch-level line timing for rolling average
            now_batch = time.monotonic()

            # Estimate lines done so far from percentages
            if total_lines_batch > 0:
                # Completed projects' lines + estimated fraction of current
                batch_current_line = int(((idx - 1) + project_pct / 100) / total * total_lines_batch)
                batch_recent_line_times.append((batch_current_line, now_batch))

            # Throttle batch progress updates
            should_update_batch = (
                (idx == total and project_pct >= 100)
                or (now_batch - batch_last_progress_update >= 0.5)
            )
            if now_batch - batch_last_progress_update >= 2.0:
                should_update_batch = True

            if not should_update_batch:
                continue

            batch_last_progress_update = now_batch

            # Общий прогресс: (завершённые проекты + доля текущего) / всего
            batch_pct = min(99, int(((idx - 1) + project_pct / 100) / total * 100))

            elapsed = now_batch - global_start

            # Compute batch-level speed from recent lines for accurate remaining
            if total_lines_batch > 0 and len(batch_recent_line_times) >= 3:
                first_bt = batch_recent_line_times[0]
                latest_bt = batch_recent_line_times[-1]
                batch_recent_lines = latest_bt[0] - first_bt[0]
                batch_recent_time = latest_bt[1] - first_bt[1]
                batch_recent_speed = batch_recent_lines / batch_recent_time if batch_recent_time > 0 else 0
                rem_sec = ((total_lines_batch - batch_current_line) / batch_recent_speed) if batch_recent_speed > 0 else 0
            else:
                # Fallback: proportional to projects
                projects_done = (idx - 1) + project_pct / 100
                speed_proj = projects_done / elapsed if elapsed > 0 else 0
                rem_sec = max(0, (total - projects_done) / speed_proj) if speed_proj > 0 else 0

            # Формат скорости: lines/sec if we have line data, else proj/s
            if total_lines_batch > 0 and len(batch_recent_line_times) >= 3:
                speed_str = f"{batch_recent_speed:.1f} lines/s"
            else:
                projects_done = (idx - 1) + project_pct / 100
                speed_proj = projects_done / elapsed if elapsed > 0 else 0
                if speed_proj > 0.01:
                    speed_str = f"{speed_proj:.2f} proj/s"
                else:
                    sec_per_proj = max(0, 1 / speed_proj) if speed_proj > 0 else 0
                    speed_str = f"~{format_time_hms(sec_per_proj)}/proj"

            rem_str = format_time_hms(rem_sec) if rem_sec > 0 else "Calculating..."

            batch_html = get_batch_metrics_html(
                f"project {idx} of {total} — {project}",
                project_pct,
                idx,
                total,
                batch_pct,
                format_time_hms(elapsed),
                rem_str,
                speed_str,
            )
            # НАКАПЛИВАЕМ: текущие файлы проекта + все предыдущие
            combined_display = all_files + files_list
            yield combined_display, f"[{idx}/{total}] {project}: {log_msg}", batch_html, audio_update, batch_file_index

        # Собираем статистику по завершённому проекту
        dur, size_mb, proc_time = get_project_stats(project)
        batch_stats.append(
            (
                project,
                dur,
                size_mb,
                proc_time if proc_time > 0 else time.monotonic() - proj_start,
            )
        )
        processed_count += 1

        # ── BF14: Cooldown between files in batch mode ──
        if TTS_COOLDOWN_SEC > 0 and idx < total:
            cool_msg = f"\n❄️ Cooling down {TTS_COOLDOWN_SEC}s..."
            elapsed_c = time.monotonic() - global_start
            batch_html_c = get_batch_metrics_html(
                f"project {idx} of {total} — {project}",
                100,
                idx,
                total,
                int(idx / total * 100),
                format_time_hms(elapsed_c),
                format_time_hms(TTS_COOLDOWN_SEC),
                "Cooling down...",
            )
            yield all_files, f"[{idx}/{total}] {project}: ✅ done{cool_msg}", batch_html_c, gr.update(), batch_file_index
            # Interruptible sleep
            slept = 0
            while slept < TTS_COOLDOWN_SEC:
                if stop_text_to_sp:
                    break
                time.sleep(1)
                slept += 1

        # НАКАПЛИВАЕМ: добавляем MP3 завершённого проекта в общий список
        project_files = get_files_list(project)
        for r in project_files:
            rc = list(r)
            filename = rc[0]
            display_name = filename
            if display_name in seen_display_names:
                display_name = f"{project} | {filename}"
            seen_display_names.add(display_name)
            batch_file_index[display_name] = str((data_path / project / "mp3" / filename).resolve())
            rc[0] = display_name
            all_files.append(rc)
        project_failures = [x for x in LAST_BATCH_FAILURES if x.startswith(f"{project}::")]
        if project_failures:
            LAST_BATCH_ERRORS.extend(project_failures)
        if project_files:
            last_file_name = project_files[-1][0]
            if "_PARTIAL" not in last_file_name:
                last_final_mp3_path = str(data_path / project / "mp3" / last_file_name)

    elapsed = time.monotonic() - global_start
    speed_final = processed_count / elapsed if elapsed > 0 else 0
    sec_per_proj_final = max(0, 1 / speed_final) if speed_final > 0 else 0

    # Финальный HTML: двойной бар + сводная таблица
    final_bars = get_batch_metrics_html(
        "✅ Completed",
        100,
        total,
        total,
        100,
        format_time_hms(elapsed),
        "00:00",
        f"~{format_time_hms(sec_per_proj_final)}/proj",
    )
    summary_table = get_batch_summary_html(batch_stats)
    final_html = summary_table + "\n" + final_bars if summary_table else final_bars

    _synthesis_completed = not LAST_BATCH_FAILURES
    # 📂 Автооткрытие папки с готовыми MP3 (по чекбоксу)
    if auto_open and all_projects:
        opened = open_folder_locally(data_path / all_projects[-1] / "mp3")
        if opened:
            logger.info("Opened MP3 folder: %s", data_path / all_projects[-1] / "mp3")
    if LAST_BATCH_FAILURES:
        gr.Warning(f"⚠️ {len(LAST_BATCH_FAILURES)} file(s) failed")
        final_msg = (
            f"⚠️ DONE with {len(LAST_BATCH_FAILURES)} failure(s): "
            + ", ".join(x.split('::')[1] for x in LAST_BATCH_FAILURES[:5])
        )
    else:
        gr.Info("Done")
        final_msg = f"🎉 DONE! Synthesized {processed_count} of {total} projects"
    yield (
        all_files,
        final_msg,
        final_html,
        _safe_audio(last_final_mp3_path),
        batch_file_index,
    )


def open_folder_locally(path) -> bool:
    """Открывает папку в системном файл-менеджере (локальный запуск GUI).
    Возвращает True при успехе."""
    try:
        p = Path(path)
        if not p.is_dir():
            return False
        if sys.platform == "win32":
            os.startfile(str(p))  # type: ignore[attr-defined]
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(p)])
        return True
    except Exception:
        return False


def _safe_audio(p):
    """gr.update для плеера: только если это реальный ФАЙЛ, иначе без изменений (защита от PermissionError на пустом/папочном пути)."""
    try:
        if p and Path(p).is_file():
            return gr.update(value=p)
    except Exception:
        pass
    return gr.update()


def _play_completion_sound():
    """Гарантированно проигрывает ЛЮБОЙ доступный звук завершения (base64 data-URI).
    🔇 Отключается чекбоксом (настройка sound_done)."""
    import base64, time
    try:
        _cfg = AppConfig.load_user_settings()
        if not getattr(_cfg, "sound_done", True):
            return gr.update(value="")
    except Exception:
        pass
    events_dir = sound_dir / "events"
    wav = None
    try:
        cfg = AppConfig.load_user_settings()
        chosen = events_dir / cfg.completion_sound
        if chosen.exists():
            wav = chosen
    except Exception:
        pass
    if wav is None:
        wavs = sorted(events_dir.glob("*.wav")) + sorted(events_dir.glob("*.mp3"))
        wav = wavs[0] if wavs else None
    if wav is None:
        return gr.update(value="")
    try:
        b64 = base64.b64encode(wav.read_bytes()).decode("ascii")
    except Exception:
        return gr.update(value="")
    mime = "audio/mpeg" if wav.suffix.lower() == ".mp3" else "audio/wav"
    key = int(time.time() * 1000)
    html = (
        f'<audio autoplay src="data:{mime};base64,{b64}"></audio>'
        f'<span style="display:none">{key}</span>'
    )
    return gr.update(value=html)


def change_tts_model(mver):
    sp_list = synth.speakers_list()
    speaker = sp_list[0]
    if isinstance(speaker, tuple):
        speaker = sp_list[0][1]

    try:
        fresh_config = AppConfig.load_user_settings()
        saved_spk = getattr(fresh_config, "spk_sel", "")

        if saved_spk and any(
            saved_spk == s or (isinstance(s, tuple) and saved_spk == s[1])
            for s in sp_list
        ):
            speaker = saved_spk
    except Exception:
        pass

    return gr.update(value=speaker, choices=sp_list)


# === ОТРИСОВКА ВКЛАДКИ ===
def tts_tab(ab_path, tts_state):
    with gr.Tab(label="🎧 Create Audiobook", id="tts") as tts_tab_ui:
        with gr.Row():
            # ── LEFT COLUMN: batch TTS block on top, then TTS controls ──
            with gr.Column(scale=5, min_width=320):
                # ── Batch TTS block (pinned to top) ──
                with gr.Row():
                    batch_project_sel = gr.CheckboxGroup(
                        choices=sorted(get_data_list()),
                        label="Select projects for batch TTS (empty = all)",
                        interactive=True, elem_id="batch_project_sel",
                    )
                with gr.Row():
                    select_all_btn = gr.Button("Select all", size="sm", scale=1)
                    deselect_all_btn = gr.Button("Deselect all", size="sm", scale=1)

                # ── Удаление проектов (выбранных / всех) ──
                with gr.Row():
                    del_projects_btn = gr.Button(
                        "🗑 Delete selected", size="sm", variant="stop",
                        scale=1, elem_id="del_projects_btn",
                    )
                    del_all_projects_btn = gr.Button(
                        "💣 Delete ALL projects", size="sm", variant="stop",
                        scale=1, elem_id="del_all_projects_btn",
                    )
                gr.Markdown(
                    "<div style='font-size:12px;color:#94a3b8;margin-top:-6px;'>"
                    "Deletes the ticked project folders (or all of them) after a second "
                    "confirmation. Internal folders (<code>_archive</code>, "
                    "<code>tts_cache</code>) are never touched."
                    "</div>"
                )
                project_del_confirm = gr.State("idle")
                project_del_all_confirm = gr.State("idle")

                with gr.Row():
                    bg_run_btn = gr.Button(
                        "⚡ Run in background (close GUI OK)",
                        variant="secondary",
                        size="sm",
                        elem_id="bg_run_btn",
                    )
                bg_status_html = gr.HTML(
                    value="<div style='color:#94a3b8;font-size:13px;'>Background: no job</div>"
                )
                bg_timer = gr.Timer(2, active=True)

                metrics_panel = gr.HTML(
                    value=get_metrics_html(0, "00:00", "00:00", "0.0 lines/s")
                )
                with gr.Group(elem_id="log_group"):
                    output_log = gr.Textbox(
                        label="Live TTS log", lines=3, max_lines=3,
                        interactive=False, value="Waiting to start...",
                    )

                with gr.Row():
                    try:
                        fresh_config = AppConfig.load_user_settings()
                        saved_spk = getattr(fresh_config, "spk_sel", "")
                    except:
                        saved_spk = ""
                    spk_sel = gr.Dropdown(
                        value=saved_spk,
                        label="Select main voice",
                        choices=[saved_spk] if saved_spk else [""],
                        interactive=True,
                    )
                    sp_rate = gr.Slider(
                        0, 3, config.sp_rate, step=0.1,
                        label="Set speed", interactive=True,
                        info="1.0 = normal · 0.9 = calmer and slower · 1.1 = quicker. "
                             "Changes the speaking rate, not the generation time",
                    )
                with gr.Row():
                    back_sound_sel = gr.Dropdown(
                        value=config.back_sound_sel,
                        allow_custom_value=True,
                        label="Select music for table of contents",
                        choices=[""], interactive=True,
                    )
                    bitrate = gr.Slider(
                        24, 256, config.bitrate, step=2,
                        label="Set audio bitrate", interactive=True,
                        info="kbps. 96–128 is transparent for speech, 192 is archival — "
                             "higher only makes files bigger",
                    )
                    noise_lvl = gr.Slider(
                        4, 32, config.noise_lvl, step=2,
                        label="F5-TTS inference steps",
                        info="Main quality/speed knob: 8 = draft, 10–14 = normal, "
                             "20+ = maximum detail. Time grows linearly — see the hint below",
                        interactive=True,
                    )

                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            use_sound_effect = gr.Checkbox(label="Audio effects", value=False)
                            repl = gr.Checkbox(label="Overwrite existing MP3", value=True)
                            use_accents = gr.Checkbox(label="Add stress marks (RU)", value=True)
                        auto_open_cb = gr.Checkbox(label="📂 Auto-open folder when done (MP3 in Explorer)", value=True)
                        try:
                            _sd_cfg = AppConfig.load_user_settings()
                            _sound_done_default = bool(getattr(_sd_cfg, "sound_done", True))
                        except Exception:
                            _sound_done_default = True
                        sound_done_cb = gr.Checkbox(label="🔊 Sound signal when done", value=_sound_done_default)
                    with gr.Column(scale=1, min_width=150):
                        tts_button = gr.Button("TTS (Ctrl+Enter)", elem_id="tts_btn", variant="primary", size="sm")
                        batch_tts_btn = gr.Button("Batch TTS (selected)", variant="primary", size="sm", elem_id="batch_tts_btn")
                        stop_btn = gr.Button("Stop", variant="stop", size="sm")

                # Live cost estimate: steps + stress marks -> time (measured on this GPU).
                steps_hint = gr.HTML(
                    steps_hint_html(int(config.noise_lvl or 12), accents=bool(config.use_accents))
                )

                def _update_steps_hint(steps, accents):
                    return steps_hint_html(int(steps or 12), accents=bool(accents))

                noise_lvl.change(
                    _update_steps_hint, inputs=[noise_lvl, use_accents], outputs=steps_hint
                )
                use_accents.change(
                    _update_steps_hint, inputs=[noise_lvl, use_accents], outputs=steps_hint
                )

            # ── RIGHT COLUMN: file table, player, buttons ──
            with gr.Column(scale=4, min_width=360):
                cur_file = gr.State()
                batch_file_index_state = gr.State({})

                df_output = gr.DataFrame(
                    headers=["File name", "Model", "Size", "Dur.", "Proc."],
                    label="MP3 files of the selected project",
                    interactive=False,
                    datatype=["str", "str", "str", "str", "str"],
                    column_widths=["200px", "80px", "60px", "60px", "60px"],
                    type="array", wrap=True,
                )
                audio_player = gr.Audio(label="Player", type="filepath", interactive=False, autoplay=False)
                completion_sound_html = gr.HTML(visible=False)

                # ── File action buttons ──
                with gr.Row():
                    del_btn = gr.Button("Delete file (Delete)", elem_id="del_file_btn", variant="stop")
                    row_download_btn = gr.DownloadButton("Download file", value=None, variant="secondary", visible=True, size="sm")
                    create_arh_btn = gr.Button("Create archive", size="sm")

                with gr.Row(visible=False) as rename_panel:
                    new_name_input = gr.Textbox(label="New file name (without .mp3)", scale=3)
                    rename_btn = gr.Button("Rename", scale=1, variant="secondary", elem_id="rename_file_btn")

                download_btn = gr.DownloadButton("Download archive", value=None, variant="primary", visible=False, size="sm")

                # ── Bulk operations ──
                file_checkboxes = gr.CheckboxGroup(
                    choices=[], label="Select files for bulk operations",
                    interactive=True, elem_id="file_checkboxes",
                )
                with gr.Row():
                    sel_all_files_btn = gr.Button("Select all", size="sm", scale=1)
                    desel_all_files_btn = gr.Button("Deselect all", size="sm", scale=1)

                with gr.Row():
                    dl_selected_btn = gr.Button("Download selected (zip)", size="sm", variant="primary")
                    del_selected_btn = gr.Button("🗑 Delete selected files", size="sm", variant="stop", elem_id="del_selected_files_btn")
                dl_selected_output = gr.DownloadButton(visible=False, size="sm")

                del_all_files_btn = gr.Button(
                    "🗑 Delete ALL project files", variant="stop", size="sm",
                    elem_id="del_all_files_btn",
                )

                del_confirm_state = gr.State("idle")
                del_all_confirm_state = gr.State("idle")

    download_btn.click(
        fn=lambda: gr.DownloadButton(visible=False), outputs=download_btn
    )
    create_arh_btn.click(create_zip_archive, inputs=[ab_path, batch_file_index_state], outputs=download_btn)

    # ── ФИКС B: скачивание выбранной строки ──
    row_download_btn.click(
        fn=download_current_file,
        inputs=[cur_file],
        outputs=[row_download_btn],
    )

    df_output.select(
        sel_file,
        inputs=[ab_path, batch_file_index_state],
        outputs=[del_btn, cur_file, rename_panel, new_name_input, audio_player, row_download_btn],
    )
    rename_btn.click(
        rename_selected_file,
        inputs=[cur_file, new_name_input, ab_path, batch_file_index_state, df_output],
        outputs=[df_output, rename_panel, batch_file_index_state],
    )

    # ── ФИКС B: выбрать все / снять всё для файлов ──
    sel_all_files_btn.click(
        fn=select_all_files, inputs=[ab_path, df_output], outputs=[file_checkboxes]
    )
    desel_all_files_btn.click(
        fn=deselect_all_files, outputs=[file_checkboxes]
    )

    # ── ФИКС B: скачать выбранные (zip) ──
    dl_selected_btn.click(
        fn=download_selected_files_zip,
        inputs=[ab_path, file_checkboxes, batch_file_index_state],
        outputs=[dl_selected_output],
    )
    dl_selected_output.click(
        fn=lambda: gr.DownloadButton(visible=False),
        outputs=[dl_selected_output],
    )

    # ── ФИКС B: удалить выбранные (с подтверждением) ──
    # Сброс подтверждения при изменении выбора файлов
    file_checkboxes.change(
        fn=lambda: (gr.update(value="🗑 Delete selected files"), "idle"),
        outputs=[del_selected_btn, del_confirm_state],
    )
    del_selected_btn.click(
        fn=delete_selected_files,
        inputs=[file_checkboxes, ab_path, del_confirm_state, batch_file_index_state, df_output],
        outputs=[df_output, del_selected_btn, del_confirm_state, file_checkboxes, batch_file_index_state],
    )

    # ── Удаление ВСЕХ файлов проекта (двухкликовое подтверждение) ──
    del_all_files_btn.click(
        fn=delete_all_project_files,
        inputs=[ab_path, del_all_confirm_state, df_output],
        outputs=[df_output, del_all_files_btn, del_all_confirm_state, file_checkboxes, batch_file_index_state],
    )

    tts_button.click(
        fn=lambda: (
            gr.update(value="⏳ Preparing files..."),
            get_metrics_html(0, "00:00", "00:00", "0.0 lines/s"),
        ),
        outputs=[output_log, metrics_panel],
    ).then(
        fn=tts,
        inputs=[
            ab_path,
            repl,
            spk_sel,
            sp_rate,
            back_sound_sel,
            bitrate,
            noise_lvl,
            use_sound_effect,
            use_accents,
        ],
        outputs=[df_output, output_log, metrics_panel, audio_player, batch_file_index_state],
    ).then(
        fn=_play_completion_sound,
        outputs=completion_sound_html,
    ).then(
        fn=get_file_checkbox_choices,
        inputs=[ab_path, df_output],
        outputs=[file_checkboxes],
    )

    def select_all_projects():
        return gr.update(value=sorted(get_data_list()))

    def deselect_all_projects():
        return gr.update(value=[])

    def refresh_project_list():
        return gr.update(choices=sorted(get_data_list()))

    select_all_btn.click(fn=select_all_projects, outputs=batch_project_sel)
    deselect_all_btn.click(fn=deselect_all_projects, outputs=batch_project_sel)

    # ── Удаление выбранных / всех проектов (двухкликовое подтверждение) ──
    def _reset_project_delete_confirm():
        return (
            gr.update(value="🗑 Delete selected"),
            gr.update(value="💣 Delete ALL projects"),
            "idle",
            "idle",
        )

    batch_project_sel.change(
        fn=_reset_project_delete_confirm,
        outputs=[del_projects_btn, del_all_projects_btn, project_del_confirm, project_del_all_confirm],
    )
    del_projects_btn.click(
        fn=delete_selected_projects,
        inputs=[batch_project_sel, project_del_confirm, ab_path, df_output, batch_file_index_state],
        outputs=[batch_project_sel, ab_path, df_output, batch_file_index_state,
                 file_checkboxes, del_projects_btn, project_del_confirm],
    )
    del_all_projects_btn.click(
        fn=delete_all_projects,
        inputs=[project_del_all_confirm, ab_path, df_output, batch_file_index_state],
        outputs=[batch_project_sel, ab_path, df_output, batch_file_index_state,
                 file_checkboxes, del_all_projects_btn, project_del_all_confirm],
    )

    tts_tab_ui.select(fn=refresh_project_list, outputs=batch_project_sel).then(
        snd_list, outputs=back_sound_sel
    )

    batch_tts_btn.click(
        fn=lambda: (
            gr.update(value="📦 Starting batch TTS..."),
            get_batch_metrics_html("Preparing...", 0, 0, 0, 0, "00:00", "...", "0.0"),
        ),
        outputs=[output_log, metrics_panel],
    ).then(
        fn=batch_tts_all_projects,
        inputs=[
            spk_sel,
            sp_rate,
            back_sound_sel,
            bitrate,
            noise_lvl,
            use_sound_effect,
            use_accents,
            repl,
            batch_project_sel,
            auto_open_cb,
        ],
        outputs=[df_output, output_log, metrics_panel, audio_player, batch_file_index_state],
    ).then(
        fn=_play_completion_sound,
        outputs=completion_sound_html,
    ).then(
        fn=get_file_checkbox_choices,
        inputs=[ab_path, df_output],
        outputs=[file_checkboxes],
    )

    stop_btn.click(stop_tts, outputs=output_log, queue=False)

    # ── Фоновый воркер: запуск + живой статус ──
    bg_run_btn.click(
        fn=launch_background_job,
        inputs=[
            ab_path,
            spk_sel,
            sp_rate,
            back_sound_sel,
            bitrate,
            noise_lvl,
            use_sound_effect,
            use_accents,
            repl,
            batch_project_sel,
            auto_open_cb,
            sound_done_cb,
        ],
        outputs=output_log,
    )
    sound_done_cb.change(lambda v: AppConfig.save_user_settings({"sound_done": bool(v)}), inputs=sound_done_cb, outputs=[]).then(
        fn=read_bg_status,
        outputs=bg_status_html,
    )
    bg_timer.tick(fn=read_bg_status, outputs=bg_status_html)

    del_btn.click(del_file, inputs=[cur_file, ab_path, batch_file_index_state, df_output], outputs=[df_output, batch_file_index_state]).then(
        fn=lambda: gr.update(visible=False), outputs=[rename_panel]
    ).then(
        fn=get_file_checkbox_choices, inputs=[ab_path, df_output], outputs=[file_checkboxes]
    )

    def _reset_table_on_tab(ab_name):
        rows = get_files_list(ab_name)
        index = _build_file_index(ab_name, rows)
        return rows, index

    tts_tab_ui.select(fn=_reset_table_on_tab, inputs=ab_path, outputs=[df_output, batch_file_index_state]).then(
        fn=get_file_checkbox_choices, inputs=[ab_path, df_output], outputs=[file_checkboxes]
    )
    # При смене проекта сразу показываем ТОЛЬКО файлы выбранного проекта,
    # а не накопленные файлы из предыдущих (batch) проектов.
    ab_path.change(fn=_reset_table_on_tab, inputs=ab_path, outputs=[df_output, batch_file_index_state]).then(
        fn=get_file_checkbox_choices, inputs=[ab_path, df_output], outputs=[file_checkboxes]
    )
    tts_state.change(change_tts_model, inputs=tts_state, outputs=spk_sel)
