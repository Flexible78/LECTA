import json
import random
import re
import tempfile
import time
import zipfile
from pathlib import Path

import gradio as gr
import libs.multilingual_router as router  # для доступа к флагам USE_EDGE_*
import numpy as np
from config import AppConfig, config
from libs.accent import accentizer
from libs.fb2_processor import FB2Processor
from libs.multilingual_router import process_multilingual_text
from libs.russian import normalize_russian
from libs.tts import get_model_short_name, synth
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

sound_dir = now_dir / "sound"
stop_text_to_sp = False
_synthesis_completed = False
txt_parser = TextParse(False)

# ═══ КАРТА МОДЕЛЕЙ ДЛЯ ОТСЛЕЖИВАНИЯ ═══
# Сохраняет какое короткое имя модели использовалось для каждого MP3 файла.
# Файл: data/<project>/tts_model_map.json  →  {"filename_stem": "Silero5_5", ...}
_MODEL_MAP_FILE = "tts_model_map.json"


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
    Все сегменты приводятся к единому формату (24kHz / 16-bit / mono)."""
    if not segments:
        return AudioSegment.silent(duration=1000, frame_rate=24000)
    if len(segments) == 1:
        return segments[0]
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
                info = mediainfo(str(mp3))
                total_duration += float(info.get("duration", 0))
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
):
    work_dir = data_path / ab_path
    xml_path = work_dir / "xml"
    mp3_path = work_dir / "mp3"
    mp3_path.mkdir(parents=True, exist_ok=True)

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
        }
    )

    global stop_text_to_sp, _synthesis_completed
    stop_text_to_sp = False
    _synthesis_completed = False

    files = [x.stem for x in xml_path.glob("*.xml")]

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
            "⚠️ Нет строк для озвучки!",
            get_metrics_html(0, "00:00", "00:00", "0.0"),
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

    global_start_time = time.time()
    current_line = 0

    yield (
        cached_files_list,
        "⏳ Инициализация движка...",
        get_metrics_html(0, "00:00", "Оценка...", "0.0"),
    )

    # ── Общий safe_synth (используется flush_text_buffer и cite/empty-line обработкой) ──
    def safe_synth(t, disable_norm=False):
        txt = t if disable_norm else txt_parser.garbage(normalize_russian(t))
        if use_accents and not disable_norm:
            try:
                txt = accentizer.process_accent(txt, r"\+\w+|\w+\+\w+")
            except Exception:
                pass
        res = synth.synth_audio(txt, speaker_id=spk_sel, speed=sp_rate, noise=noise_lvl)
        if res is None:
            return None, None
        a, b = res
        if isinstance(a, (int, float)):
            sr, aud_raw = int(a), b
        elif isinstance(b, (int, float)):
            sr, aud_raw = int(b), a
        else:
            sr, aud_raw = 24000, a
        return sr, aud_raw

    def _tts_to_audio(np_audio, sr):
        """Конвертирует numpy-аудио в AudioSegment."""
        if np_audio is not None and getattr(np_audio, "size", 0) > 0:
            np_audio = np.nan_to_num(np_audio, nan=0.0, posinf=0.0, neginf=0.0)
            np_audio = np.clip(np_audio, -1.0, 1.0)
            final_audio_int16 = (np_audio * 32767).astype(np.int16)
            audio = AudioSegment(
                final_audio_int16.tobytes(), frame_rate=sr, sample_width=2, channels=1
            )
            return effects.normalize(audio)
        return AudioSegment.silent(duration=500)

    for file in sorted(files, key=safe_sort):
        if is_single_file:
            final_name = ab_path
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
            yield (
                cached_files_list,
                f"⏭ Пропуск: {short_final}.mp3 уже существует",
                get_metrics_html(
                    int((current_line / total_lines) * 100),
                    format_time_hms(time.time() - global_start_time),
                    "00:00",
                    "-",
                ),
            )
            continue

        file_start_time = time.time()
        xml_file = xml_path / f"{file}.xml"
        root = etree.parse(str(xml_file)).getroot()
        autor = root.get("autor")
        album = root.get("album")

        was_interrupted = False
        audio_segments = [
            AudioSegment.silent(duration=1000, frame_rate=24000)
        ]  # Начальная пауза 1 сек + накопление сегментов

        # ── БУФЕР ДЛЯ ЯЗЫКОВО-ОСНОВАННОГО БАТЧИНГА ──
        text_buffer = []
        buffer_lang = None  # текущий язык буфера
        buffer_chars = 0  # суммарная длина символов в буфере
        _BATCH_MAX_CHARS = 1000

        def flush_text_buffer():
            """Сбрасывает накопленные строки в TTS одним батчем."""
            nonlocal text_buffer, buffer_lang, buffer_chars
            if not text_buffer:
                return
            joined = "\n".join(text_buffer)
            text_buffer.clear()
            buffer_lang = None
            buffer_chars = 0
            try:
                sr, np_audio = process_multilingual_text(
                    joined,
                    safe_synth,
                    use_edge_en=router.USE_EDGE_FOR_ENGLISH,
                    use_edge_he=router.USE_EDGE_FOR_HEBREW,
                    use_edge_ru=router.USE_EDGE_FOR_RUSSIAN,
                    dict_mode=router.DICTIONARY_MODE,
                )
            except Exception as e:
                print(f"Router Error: {e}")
                np_audio, sr = None, None
            audio_segments.append(_tts_to_audio(np_audio, sr))

        for i, line in enumerate(root):
            if stop_text_to_sp:
                # НЕ сбрасываем stop_text_to_sp здесь! Батч-функция должна видеть, что была остановка.
                was_interrupted = True
                break  # Выходим из цикла строк, но идем дальше по коду для сохранения файла!

            current_line += 1

            if current_line % 3 == 0 or current_line == total_lines:
                elapsed = time.time() - global_start_time
                speed = current_line / elapsed if elapsed > 0 else 0
                rem_sec = (total_lines - current_line) / speed if speed > 0 else 0
                pct = int((current_line / total_lines) * 100)

                html = get_metrics_html(
                    pct,
                    format_time_hms(elapsed),
                    format_time_hms(rem_sec),
                    f"{speed:.1f} стр/с",
                )
                log_txt = f"▶ В работе: {file}.xml\n🎙 Озвучка строки: {current_line} из {total_lines}..."
                # НЕ вызываем get_files_list каждые 3 строки — это запускает ffprobe на всех MP3!
                # Возвращаем кэшированный список вместо [] чтобы таблица не мигала.
                yield cached_files_list, log_txt, html

            # --- ЛОГИКА ГЕНЕРАЦИИ ЗВУКА (с буферизацией для Edge TTS) ---
            audio = AudioSegment.empty()
            if line.tag == "sound" and use_sound_effect:
                flush_text_buffer()
                sound_file = sound_dir / "events" / f"{line.get('value')}.wav"
                audio = AudioSegment.from_wav(str(sound_file))
                audio = effects.normalize(audio)
            elif line.tag == "break":
                flush_text_buffer()
                slt = int(line.get("time")) * 100
                audio = AudioSegment.silent(duration=slt)
            elif (
                line.tag in ("cite", "empty-line")
                and use_sound_effect
                and not line.text
            ):
                # cite/empty-line БЕЗ текста: просто сброс буфера
                flush_text_buffer()
            elif line.tag in ("cite", "empty-line") and use_sound_effect and line.text:
                # cite/empty-line С текстом: обрабатываем инлайн, чтобы применить реверб/префикс
                flush_text_buffer()
                try:
                    sr, np_audio = process_multilingual_text(
                        line.text,
                        safe_synth,
                        use_edge_en=router.USE_EDGE_FOR_ENGLISH,
                        use_edge_he=router.USE_EDGE_FOR_HEBREW,
                        use_edge_ru=router.USE_EDGE_FOR_RUSSIAN,
                        dict_mode=router.DICTIONARY_MODE,
                    )
                except Exception as e:
                    print(f"Router Error: {e}")
                    np_audio, sr = None, None
                audio = _tts_to_audio(np_audio, sr)
            elif line.text:
                # Обычный текст: language-aware batching
                stripped = line.text.strip()
                line_lang = _detect_lang(stripped)

                # mixed-строки: сброс буфера и обработка как есть (роутер разобьёт)
                if line_lang == "mixed":
                    flush_text_buffer()
                    try:
                        sr, np_audio = process_multilingual_text(
                            stripped,
                            safe_synth,
                            use_edge_en=router.USE_EDGE_FOR_ENGLISH,
                            use_edge_he=router.USE_EDGE_FOR_HEBREW,
                            use_edge_ru=router.USE_EDGE_FOR_RUSSIAN,
                            dict_mode=router.DICTIONARY_MODE,
                        )
                    except Exception as e:
                        print(f"Router Error: {e}")
                        np_audio, sr = None, None
                    audio_segments.append(_tts_to_audio(np_audio, sr))
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

            if line.tag == "cite" and use_sound_effect:
                audio = add_reverb(audio)
                if line.get("position") == "start":
                    pr_audio = AudioSegment.from_wav(
                        sound_dir / "pause" / "min_cite.wav"
                    )
                    pr_audio = effects.normalize(pr_audio)
                    audio = pr_audio + audio
            if line.tag == "empty-line" and use_sound_effect:
                pr_audio = AudioSegment.from_wav(sound_dir / "pause" / "empty.wav")
                pr_audio = effects.normalize(pr_audio)
                audio = pr_audio + audio

            # Добавляем аудио если оно не пустое (пустое = текст ушёл в буфер)
            if len(audio) > 0 or (line.tag != "" and not line.text):
                audio_segments.append(audio)

        # Сбрасываем остатки буфера после цикла
        flush_text_buffer()

        # --- O(n) КОНКАТЕНАЦИЯ ВСЕХ СЕГМЕНТОВ (вместо O(n²) через оператор +) ---
        out_audio = _concat_audio_segments(audio_segments)

        # --- ФИНАЛЬНАЯ ОБРАБОТКА (Сработает даже при прерывании!) ---
        if back_sound_sel:
            back_sound_file = sound_dir / "back" / back_sound_sel
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

        cover_file = work_dir / "cover.jpg"
        tags = {
            "artist": autor,
            "title": f"{short_final}",
            "track": f"{file}",
            "album": album,
        }

        # Определяем имя файла (полное или частичное)
        save_path = str(partial_mp3_file) if was_interrupted else str(mp3_file)

        if cover_file.exists():
            out_audio.export(
                save_path,
                format="mp3",
                bitrate=f"{bitrate}",
                cover=str(cover_file),
                tags=tags,
            )
        else:
            out_audio.export(save_path, format="mp3", bitrate=f"{bitrate}", tags=tags)

        # Запоминаем модель, использованную для этого файла (читаем synth.ver непосредственно перед сохранением)
        model_short = get_model_short_name(synth.ver) if synth.ver else "Модель?"
        _save_model_map_entry(ab_path, short_final, model_short)

        elapsed_file = time.time() - file_start_time
        parse_times[str(short_final)] = elapsed_file
        with open(times_file, "w", encoding="utf-8") as f:
            json.dump(parse_times, f)

        if was_interrupted:
            pct = int((current_line / total_lines) * 100) if total_lines > 0 else 0
            yield (
                get_files_list(ab_path),
                f"🛑 Остановлено! Частичный файл сохранен как {short_final}_PARTIAL.mp3",
                get_metrics_html(
                    pct,
                    format_time_hms(time.time() - global_start_time),
                    "-",
                    "Остановлено",
                ),
            )
            return  # Выходим из функции полностью, чтобы не начинать следующий файл

        # Обновляем кэш после сохранения нового файла
        cached_files_list = get_files_list(ab_path)
        # Вывод об обновлении файла (если не прервано)
        yield (
            cached_files_list,
            f"✅ Сохранен: {short_final}.mp3",
            get_metrics_html(
                int((current_line / total_lines) * 100),
                format_time_hms(time.time() - global_start_time),
                format_time_hms(
                    (total_lines - current_line)
                    / (current_line / (time.time() - global_start_time))
                ),
                "-",
            ),
        )

    # ФИНАЛ
    _synthesis_completed = True
    gr.Info("Готово")
    yield (
        cached_files_list,
        "🎉 ВСЕ ФАЙЛЫ УСПЕШНО ОЗВУЧЕНЫ!",
        get_metrics_html(
            100, format_time_hms(time.time() - global_start_time), "00:00", "-"
        ),
    )


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

        for file_path in sorted_files:
            info = mediainfo(str(file_path))
            size_in_mb = round(file_path.stat().st_size / (1024 * 1024), 2)
            file_key = file_path.stem
            elapsed = parse_times.get(file_key, 0)
            ptime_str = format_audio_time(elapsed) if elapsed > 0 else "Н/Д"
            # Короткое имя модели (из карты) или '?'
            model_label = model_map.get(file_key, "?")
            rows.append(
                [
                    file_path.name,
                    model_label,
                    f"{size_in_mb}M",
                    format_audio_time(float(info["duration"])),
                    ptime_str,
                ]
            )
    return rows


def create_zip_archive(ab_path):
    mp3_dir = data_path / ab_path / "mp3"
    if not mp3_dir.exists():
        raise gr.Error(f"Папка с файлами не найдена!")

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_filename = tmp.name
    try:
        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in mp3_dir.rglob("*"):
                if file_path.is_file():
                    arcname = f"{ab_path}/{file_path.relative_to(mp3_dir.parent)}"
                    zipf.write(str(file_path), str(arcname))
        return gr.update(visible=True, value=zip_filename)
    except Exception as e:
        if Path(zip_filename).exists():
            Path(zip_filename).unlink()
        raise gr.Error(f"Ошибка: {str(e)}")


def snd_list():
    snd_path = sound_dir / "back"
    if snd_path.exists():
        return gr.update(value="", choices=sorted([x.name for x in snd_path.iterdir()]))
    return gr.update(value="", choices=[])


def del_file(filename, ab_name):
    if not filename:
        return get_files_list(ab_name)
    file_path = Path(filename)
    if file_path.exists():
        try:
            file_path.unlink()
            gr.Info(f"Удален файл {file_path.name}", duration=2)
            # Чистим запись о модели в tts_model_map.json
            model_map = _load_model_map(ab_name)
            if file_path.stem in model_map:
                model_map.pop(file_path.stem, None)
                map_path = data_path / ab_name / _MODEL_MAP_FILE
                try:
                    with open(map_path, "w", encoding="utf-8") as f:
                        json.dump(model_map, f, ensure_ascii=False)
                except Exception:
                    pass
        except Exception as e:
            gr.Warning(f"Ошибка удаления: {e}")
    return get_files_list(ab_name)


def sel_file(data: gr.SelectData, ab_path):
    mp3_dir = data_path / ab_path / "mp3"
    # Имя файла теперь чистый текст (колонка 0)
    filename_raw = data.row_value[0]
    if not filename_raw or not str(filename_raw).endswith(".mp3"):
        gr.Warning("Не удалось определить имя файла из строки таблицы")
        return gr.update(), "", gr.update(), gr.update(), gr.update()
    selected_path = mp3_dir / filename_raw
    stem_name = selected_path.stem
    return (
        gr.update(interactive=True),
        str(selected_path),
        gr.update(visible=True),
        gr.update(value=stem_name),
        str(selected_path),
    )


def rename_selected_file(current_path, new_name, ab_path):
    if not current_path or not new_name:
        return get_files_list(ab_path), gr.update(visible=False)
    old_path = Path(current_path)
    if not old_path.exists():
        gr.Warning("Файл не найден!")
        return get_files_list(ab_path), gr.update(visible=False)

    safe_name = "".join(
        [c for c in new_name if c.isalnum() or c in (" ", "_", "-")]
    ).rstrip()
    if not safe_name:
        safe_name = "renamed_audio"
    new_path = old_path.parent / f"{safe_name}.mp3"

    if new_path.exists() and new_path != old_path:
        gr.Warning("Файл с таким именем уже существует!")
        return get_files_list(ab_path), gr.update()

    try:
        old_path.rename(new_path)
        gr.Info(f"Файл успешно переименован в {safe_name}.mp3")
        times_file = data_path / ab_path / "parse_times.json"
        if times_file.exists():
            with open(times_file, "r", encoding="utf-8") as f:
                parse_times = json.load(f)
            old_stem = old_path.stem
            if old_stem in parse_times:
                parse_times[safe_name] = parse_times.pop(old_stem)
                with open(times_file, "w", encoding="utf-8") as f:
                    json.dump(parse_times, f)
        # Переносим запись о модели в tts_model_map.json
        model_map = _load_model_map(ab_path)
        if old_path.stem in model_map:
            model_map[safe_name] = model_map.pop(old_path.stem)
            map_path = data_path / ab_path / _MODEL_MAP_FILE
            try:
                with open(map_path, "w", encoding="utf-8") as f:
                    json.dump(model_map, f, ensure_ascii=False)
            except Exception:
                pass
    except Exception as e:
        gr.Warning(f"Ошибка переименования: {e}")
    return get_files_list(ab_path), gr.update(visible=False)


def stop_tts():
    global stop_text_to_sp
    stop_text_to_sp = True
    return "🛑 Остановка после текущей строки (Сохраняем файл)..."


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
):
    """Пакетная озвучка выбранных проектов.
    Если selected_projects не указан или пуст — берутся все проекты."""
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
            "⚠️ Нет проектов для пакетной озвучки!",
            get_batch_metrics_html("", 0, 0, 0, 0, "00:00", "00:00", "0.0"),
        )
        return

    global stop_text_to_sp, _synthesis_completed
    stop_text_to_sp = False
    _synthesis_completed = False

    total = len(all_projects)
    global_start = time.time()
    processed_count = 0
    all_files = []  # НАКАПЛИВАЕМ MP3 ВСЕХ ПРОЕКТОВ
    batch_stats = []  # для финальной сводной таблицы

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

    yield (
        [],
        f"📦 Подготовка {total} проектов...",
        get_batch_metrics_html("Подготовка...", 0, 0, total, 0, "00:00", "...", "..."),
    )

    for idx, project in enumerate(all_projects, 1):
        if stop_text_to_sp:
            stop_text_to_sp = False
            elapsed = time.time() - global_start
            pct = int(processed_count / total * 100) if total > 0 else 0

            # Показываем частичную сводку по завершённым проектам
            partial_html = get_batch_metrics_html(
                project,
                0,
                idx,
                total,
                pct,
                format_time_hms(elapsed),
                "-",
                "Остановлено",
            )
            if batch_stats:
                partial_html = get_batch_summary_html(batch_stats) + "\n" + partial_html

            yield (
                all_files,
                f"🛑 Остановлено! Озвучено {processed_count} из {total}",
                partial_html,
            )
            return

        work_dir = data_path / project
        xml_dir = work_dir / "xml"
        fb2_file = work_dir / f"{project}.fb2"

        if not fb2_file.exists():
            continue

        # Авто-парсинг FB2 если XML ещё нет
        if not xml_dir.exists() or not list(xml_dir.glob("*.xml")):
            batch_pre_pct = int((idx - 1) / total * 100)
            elapsed = time.time() - global_start
            speed = idx / elapsed if elapsed > 0 else 0
            rem = (total - idx) / speed if speed > 0 else 0
            sec_per_proj = 1 / speed if speed > 0 else 0
            yield (
                all_files,
                f"📁 [{idx}/{total}] {project}: Парсинг FB2 → XML...",
                get_batch_metrics_html(
                    project,
                    0,
                    idx,
                    total,
                    batch_pre_pct,
                    format_time_hms(elapsed),
                    format_time_hms(rem),
                    f"~{format_time_hms(sec_per_proj)}/пр",
                ),
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
                    f"⚠️ [{idx}/{total}] {project}: Ошибка парсинга: {e}",
                    get_batch_metrics_html(
                        project,
                        0,
                        idx,
                        total,
                        batch_pre_pct,
                        format_time_hms(elapsed),
                        "...",
                        "...",
                    ),
                )
                continue

        # Запуск TTS для проекта
        proj_start = time.time()
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
        ):
            if stop_text_to_sp:
                break
            files_list, log_msg, tts_html = tts_result

            # Извлекаем процент текущего проекта из HTML tts()
            project_pct = parse_percent_from_html(tts_html)

            # Общий прогресс: (завершённые проекты + доля текущего) / всего
            batch_pct = int(((idx - 1) + project_pct / 100) / total * 100)

            elapsed = time.time() - global_start
            # Оценка оставшегося времени: пропорционально оставшимся проектам
            projects_done = (idx - 1) + project_pct / 100
            speed_proj = projects_done / elapsed if elapsed > 0 else 0
            rem_sec = (total - projects_done) / speed_proj if speed_proj > 0 else 0

            # Формат скорости: если проектов в секунду < 0.01, показываем мин/проект
            if speed_proj > 0.01:
                speed_str = f"{speed_proj:.2f} пр/с"
            else:
                sec_per_proj = 1 / speed_proj if speed_proj > 0 else 0
                speed_str = f"~{format_time_hms(sec_per_proj)}/пр"

            batch_html = get_batch_metrics_html(
                project,
                project_pct,
                idx,
                total,
                batch_pct,
                format_time_hms(elapsed),
                format_time_hms(rem_sec),
                speed_str,
            )
            # НАКАПЛИВАЕМ: текущие файлы проекта + все предыдущие
            combined_display = all_files + files_list
            yield combined_display, f"[{idx}/{total}] {project}: {log_msg}", batch_html

        # Собираем статистику по завершённому проекту
        dur, size_mb, proc_time = get_project_stats(project)
        batch_stats.append(
            (
                project,
                dur,
                size_mb,
                proc_time if proc_time > 0 else time.time() - proj_start,
            )
        )
        processed_count += 1

        # НАКАПЛИВАЕМ: добавляем MP3 завершённого проекта в общий список
        project_files = get_files_list(project)
        all_files.extend(project_files)

    elapsed = time.time() - global_start
    speed_final = processed_count / elapsed if elapsed > 0 else 0
    sec_per_proj_final = 1 / speed_final if speed_final > 0 else 0

    # Финальный HTML: двойной бар + сводная таблица
    final_bars = get_batch_metrics_html(
        "✅ Завершено",
        100,
        total,
        total,
        100,
        format_time_hms(elapsed),
        "00:00",
        f"~{format_time_hms(sec_per_proj_final)}/пр",
    )
    summary_table = get_batch_summary_html(batch_stats)
    final_html = summary_table + "\n" + final_bars if summary_table else final_bars

    _synthesis_completed = True
    gr.Info("Готово")
    yield (
        all_files,
        f"🎉 ГОТОВО! Озвучено {processed_count} из {total} проектов",
        final_html,
    )


def _play_completion_sound():
    """Проигрывает звук завершения, только если синтез завершился успешно."""
    if _synthesis_completed:
        cfg = AppConfig.load_user_settings()
        return str(sound_dir / "events" / cfg.completion_sound)
    return gr.update()


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
    with gr.Tab(label="🎧 ОЗВУЧКА", id=2) as tts_tab_ui:
        with gr.Row():
            try:
                fresh_config = AppConfig.load_user_settings()
                saved_spk = getattr(fresh_config, "spk_sel", "")
            except:
                saved_spk = ""
            spk_sel = gr.Dropdown(
                value=saved_spk,
                label="Выбрать основной голос",
                choices=[saved_spk] if saved_spk else [""],
                interactive=True,
            )
            sp_rate = gr.Slider(
                0,
                3,
                config.sp_rate,
                step=0.1,
                label="Задать скорость",
                interactive=True,
            )
        with gr.Row():
            back_sound_sel = gr.Dropdown(
                value=config.back_sound_sel,
                allow_custom_value=True,
                label="Выбрать музыку для оглавлений",
                choices=[""],
                interactive=True,
            )
            bitrate = gr.Slider(
                24,
                256,
                config.bitrate,
                step=2,
                label="Задать битрейт аудио",
                interactive=True,
            )
            noise_lvl = gr.Slider(
                0,
                64,
                config.noise_lvl,
                step=1,
                label="Уровень шума(выше-лучше)",
                interactive=True,
            )

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    use_sound_effect = gr.Checkbox(label="Аудио эффекты", value=False)
                    repl = gr.Checkbox(
                        label="Перезаписать существующие MP3", value=True
                    )
                    use_accents = gr.Checkbox(
                        label="Проставить ударения (РУ)", value=True
                    )
            with gr.Column(scale=1, min_width=150):
                tts_button = gr.Button(
                    "🟢 TTS (Ctrl+Enter)",
                    elem_id="tts_btn",
                    variant="primary",
                    size="sm",
                )
                batch_tts_btn = gr.Button(
                    "📦 Пакетная озвучка выбранных",
                    variant="primary",
                    size="sm",
                    elem_id="batch_tts_btn",
                )
                stop_btn = gr.Button("🚫 Прервать", variant="stop", size="sm")

        with gr.Row():
            with gr.Column(scale=1):
                batch_project_sel = gr.CheckboxGroup(
                    choices=sorted(get_data_list()),
                    label="🎯 Выберите проекты для пакетной озвучки (пусто = все)",
                    interactive=True,
                    elem_id="batch_project_sel",
                )
                with gr.Row():
                    select_all_btn = gr.Button("✅ Выбрать все", size="sm", scale=1)
                    deselect_all_btn = gr.Button("⬜ Снять всё", size="sm", scale=1)

        metrics_panel = gr.HTML(
            value=get_metrics_html(0, "00:00", "00:00", "0.0 стр/с")
        )
        with gr.Group(elem_id="log_group"):
            output_log = gr.Textbox(
                label="Живой лог озвучки",
                lines=3,
                max_lines=3,
                interactive=False,
                value="В ожидании старта...",
            )

        with gr.Row():
            cur_file = gr.State()

        # Таблица аудио: узкие колонки + колонка «Модель» (короткое имя)
        df_output = gr.DataFrame(
            headers=["Имя файла", "Модель", "Размер", "Длит.", "Обраб."],
            interactive=False,
            datatype=["str", "str", "str", "str", "str"],
            column_widths=["220px", "100px", "70px", "80px", "80px"],
            type="array",
            wrap=True,
        )
        audio_player = gr.Audio(label="Плеер", type="filepath", interactive=False)

        # ── Панель управления файлами (объединены логически) ──
        with gr.Group():
            with gr.Row(visible=False) as rename_panel:
                new_name_input = gr.Textbox(label="Новое имя файла (без .mp3)", scale=3)
                rename_btn = gr.Button(
                    "✏️ Переименовать",
                    scale=1,
                    variant="secondary",
                    elem_id="rename_file_btn",
                )

            with gr.Row():
                del_btn = gr.Button(
                    "❌ Удалить файл (Delete)",
                    interactive=False,
                    elem_id="del_file_btn",
                    size="sm",
                )
                create_arh_btn = gr.Button("📦 Создать архив", size="sm")
                download_btn = gr.DownloadButton(
                    "📥 Скачать архив",
                    value=None,
                    variant="primary",
                    visible=False,
                    size="sm",
                )

    download_btn.click(
        fn=lambda: gr.DownloadButton(visible=False), outputs=download_btn
    )
    create_arh_btn.click(create_zip_archive, inputs=ab_path, outputs=download_btn)
    df_output.select(
        sel_file,
        inputs=ab_path,
        outputs=[del_btn, cur_file, rename_panel, new_name_input, audio_player],
    )
    rename_btn.click(
        rename_selected_file,
        inputs=[cur_file, new_name_input, ab_path],
        outputs=[df_output, rename_panel],
    )

    tts_button.click(
        fn=lambda: (
            gr.update(value="⏳ Подготовка файлов..."),
            get_metrics_html(0, "00:00", "00:00", "0.0 стр/с"),
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
        outputs=[df_output, output_log, metrics_panel],
    ).then(
        fn=_play_completion_sound,
        outputs=audio_player,
    )

    def select_all_projects():
        return gr.update(value=sorted(get_data_list()))

    def deselect_all_projects():
        return gr.update(value=[])

    def refresh_project_list():
        return gr.update(choices=sorted(get_data_list()))

    select_all_btn.click(fn=select_all_projects, outputs=batch_project_sel)
    deselect_all_btn.click(fn=deselect_all_projects, outputs=batch_project_sel)
    tts_tab_ui.select(fn=refresh_project_list, outputs=batch_project_sel).then(
        snd_list, outputs=back_sound_sel
    )

    batch_tts_btn.click(
        fn=lambda: (
            gr.update(value="📦 Запуск пакетной озвучки..."),
            get_batch_metrics_html("Подготовка...", 0, 0, 0, 0, "00:00", "...", "0.0"),
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
        ],
        outputs=[df_output, output_log, metrics_panel],
    ).then(
        fn=_play_completion_sound,
        outputs=audio_player,
    )

    stop_btn.click(stop_tts, outputs=output_log, queue=False)
    del_btn.click(del_file, inputs=[cur_file, ab_path], outputs=[df_output]).then(
        fn=lambda: gr.update(visible=False), outputs=[rename_panel]
    )
    tts_tab_ui.select(fn=get_files_list, inputs=ab_path, outputs=df_output)
    tts_state.change(change_tts_model, inputs=tts_state, outputs=spk_sel)
