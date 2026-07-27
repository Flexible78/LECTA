import shutil
import logging
import gradio as gr
import requests
from pathlib import Path

from libs.utils import download_model

logger = logging.getLogger(__name__)

# Глобальный флаг для остановки скачивания моделей
_stop_model_update = False

def stop_model_update():
    """Устанавливает флаг остановки — update_all_voice_models() проверяет его после каждой модели."""
    global _stop_model_update
    _stop_model_update = True
    return "🛑 Stopping after the current model..."

# Высчитываем корневую директорию (fb2tts/) относительно папки libs/
CURRENT_DIR = Path(__file__).resolve().parent.parent

# =============================================================================
# РЕЕСТР ГОЛОСОВЫХ МОДЕЛЕЙ (для кнопки "Обновить модели")
# =============================================================================
# Каждый пункт: (id, название, список файлов для скачивания)
# Файл: (url, локальный_путь_относительно_models/)
VOICE_MODELS_REGISTRY = [
    ("vosk_010", "Vosk 0.10 (56 голосов)", [
        ("https://myfreenet.ru/models/vosk-model-tts-ru-0.10-multi.zip",
         "vosk-model-tts-ru-0.10-multi.zip"),
    ]),
    ("silero_ru", "Silero v5_5 (Russian, 5 voices)", [
        ("https://models.silero.ai/models/tts/ru/v5_5_ru.pt", "silero/v5_5_ru.pt"),
    ]),
    ("silero_cis", "Silero v5_cis (60 голосов)", [
        ("https://models.silero.ai/models/tts/ru/v5_cis_base_nostress.pt",
         "silero/v5_cis_base_nostress.pt"),
    ]),
    ("silero_en", "Silero English v3", [
        ("https://models.silero.ai/models/tts/en/v3_en.pt", "silero/v3_en.pt"),
    ]),
    ("f5_misha", "Misha24-10 (F5-TTS)", [
        ("https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN/resolve/main/F5TTS_v1_Base_v4_winter/model_212000.safetensors",
         "F5TTS_v1_Base_v4_winter/model_212000.safetensors"),
    ]),
    ("f5_espeech", "ESpeech-TTS (F5-TTS)", [
        ("https://huggingface.co/ESpeech/ESpeech-TTS-1_RL-V2/resolve/main/espeech_tts_rlv2.pt",
         "ESpeech-TTS-1_RL-V2/espeech_tts_rlv2.pt"),
    ]),
    ("vocos", "Vocoder Vocos-mel-24khz (for F5-TTS)", [
        ("https://huggingface.co/charactr/vocos-mel-24khz/resolve/main/pytorch_model.bin",
         "vocos-mel-24khz/pytorch_model.bin"),
        ("https://huggingface.co/charactr/vocos-mel-24khz/resolve/main/config.yaml",
         "vocos-mel-24khz/config.yaml"),
    ]),
    ("silero_stress", "Silero Stress (stress marks)", [
        ("https://github.com/snakers4/silero-stress/raw/refs/heads/master/src/silero_stress/data/accentor.pt",
         "silero_stress/accentor.pt"),
    ]),
]

# =============================================================================
# СТАРЫЕ ВЕРСИИ ФАЙЛОВ ДЛЯ АВТО-УДАЛЕНИЯ ПРИ ОБНОВЛЕНИИ
# =============================================================================
# Ключ — model_id, значение — список старых путей (относительно models/) которые
# нужно удалить после успешного обновления модели на новую версию.
OLD_FILES_CLEANUP = {
    "silero_ru": ["silero/v5_ru.pt"],          # заменён на v5_5_ru.pt
    # При будущих обновлениях добавляйте старые версии сюда:
    # "silero_cis": ["silero/v5_cis_base_v1.pt"],
    # "f5_misha": ["F5TTS_v1_Base_v4_winter/model_old.pt"],
}

def clean_tmp_folder():
    tmp_dir = CURRENT_DIR / "tmp"
    if not tmp_dir.exists(): return "The tmp folder is already clean."
    
    deleted_size, deleted_count = 0, 0
    for item in tmp_dir.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                size = item.stat().st_size
                item.unlink()
                deleted_size += size
                deleted_count += 1
            elif item.is_dir():
                for subitem in item.rglob('*'):
                    if subitem.is_file(): deleted_size += subitem.stat().st_size
                shutil.rmtree(item)
                deleted_count += 1
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            
    mb_freed = deleted_size / (1024 * 1024)
    return f"✅ Cleaned {deleted_count} items. Freed {mb_freed:.2f} MB."

def get_installed_models():
    """Returns list of (label_with_size, folder_name) for the Delete model dropdown."""
    models_dir = CURRENT_DIR / "models"
    if not models_dir.exists():
        return []
    result = []
    for item in sorted(models_dir.iterdir()):
        if item.is_dir():
            try:
                size_bytes = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            except Exception:
                size_bytes = 0
        else:
            try:
                size_bytes = item.stat().st_size
            except Exception:
                size_bytes = 0
        label = f"{item.name} — {_format_size(size_bytes)}"
        result.append((label, item.name))
    return result

def get_models_disk_info():
    """Returns a human-readable summary: total models size and free disk space."""
    models_dir = CURRENT_DIR / "models"
    total_bytes = 0
    count = 0
    if models_dir.exists():
        for item in models_dir.iterdir():
            count += 1
            if item.is_dir():
                try:
                    total_bytes += sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                except Exception:
                    pass
            else:
                try:
                    total_bytes += item.stat().st_size
                except Exception:
                    pass
    try:
        usage = shutil.disk_usage(models_dir if models_dir.exists() else CURRENT_DIR)
        free_str = _format_size(usage.free)
    except Exception:
        free_str = "?"
    return f"{count} models ({_format_size(total_bytes)} on disk) · Free: {free_str}", count

def delete_selected_model(model_name, confirmed, tts_model_ver=None, acc_model_ver=None):
    """Delete a model folder. Requires confirmation, protects active models.
    Returns: (model_del_dropdown_update, model_del_status, disk_info_update, update_section_dropdown_update, update_section_model_del_dropdown_update)
    """
    if not model_name:
        return (
            gr.update(), "⚠️ No model selected",
            get_models_disk_info()[0],
            gr.update(), gr.update()
        )
    if not confirmed:
        return (
            gr.update(), "⚠️ Confirm deletion with the checkbox first",
            get_models_disk_info()[0],
            gr.update(), gr.update()
        )
    
    # Map model folder names to the internal model IDs used by tts_sel / acc_sel
    folder_to_tts_id = {
        "vosk-model-tts-ru-0.10-multi": 2,
        "silero": (3, 4, 7),      # Silero folder contains multiple models (v5_5, cis, en)
        "v5_5_ru.pt": 3,
        "v5_cis_base_nostress.pt": 4,
        "F5TTS_v1_Base_v4_winter": 5,
        "ESpeech-TTS-1_RL-V2": 6,
        "v3_en.pt": 7,
    }
    folder_to_acc_id = {
        "silero_stress": 2,
    }
    
    in_use_tts = folder_to_tts_id.get(model_name)
    in_use_acc = folder_to_acc_id.get(model_name)
    
    if in_use_tts is not None and tts_model_ver is not None:
        try:
            tts_ver = int(tts_model_ver)
        except (TypeError, ValueError):
            tts_ver = None
        if tts_ver is not None:
            if isinstance(in_use_tts, tuple):
                if tts_ver in in_use_tts:
                    return (
                        gr.update(), f"⚠️ Model is in use (Silero). Select another TTS model first.",
                        get_models_disk_info()[0],
                        gr.update(), gr.update()
                    )
            elif tts_ver == in_use_tts:
                return (
                    gr.update(), f"⚠️ Model is in use. Select another TTS model first.",
                    get_models_disk_info()[0],
                    gr.update(), gr.update()
                )
    if in_use_acc is not None and acc_model_ver is not None:
        try:
            acc_ver = int(acc_model_ver)
        except (TypeError, ValueError):
            acc_ver = None
        if acc_ver is not None and acc_ver == in_use_acc:
            return (
                gr.update(), f"⚠️ Model is in use. Select another stress model first.",
                get_models_disk_info()[0],
                gr.update(), gr.update()
            )
    
    target_path = CURRENT_DIR / "models" / model_name
    if not target_path.exists():
        return (
            gr.update(choices=get_installed_models(), value=""),
            "⚠️ Path not found!",
            get_models_disk_info()[0],
            gr.update(choices=get_voice_models_choices()),
            gr.update(choices=get_installed_models(), value="")
        )
    
    # Calculate size before deletion
    try:
        if target_path.is_dir():
            freed_bytes = sum(f.stat().st_size for f in target_path.rglob("*") if f.is_file())
        else:
            freed_bytes = target_path.stat().st_size
    except Exception:
        freed_bytes = 0
    freed_str = _format_size(freed_bytes)
    
    try:
        if target_path.is_dir():
            shutil.rmtree(target_path)
        else:
            target_path.unlink()
        new_choices = get_installed_models()
        voice_choices = get_voice_models_choices()
        disk_info, _ = get_models_disk_info()
        return (
            gr.update(choices=new_choices, value=""),
            f"✅ Deleted: {model_name} — freed {freed_str}",
            disk_info,
            gr.update(choices=voice_choices),
            gr.update(choices=new_choices, value="")
        )
    except Exception as e:
        return (
            gr.update(choices=get_installed_models()),
            f"❌ Error: {e}",
            get_models_disk_info()[0],
            gr.update(), gr.update()
        )

# =============================================================================
# ОБНОВЛЕНИЕ / СКАЧИВАНИЕ ГОЛОСОВЫХ МОДЕЛЕЙ
# =============================================================================
def _format_size(size_bytes):
    """Форматирует размер байтов в человекочитаемый вид (КБ/МБ/ГБ)."""
    if size_bytes <= 0:
        return "?"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.0f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def _get_file_size(url, local_path):
    """Возвращает размер файла в байтах: с диска если установлен, иначе из HTTP content-length.
    Возвращает 0 если размер не удалось определить (ошибка сети/файла)."""
    # Сначала проверяем локальный файл
    if local_path.exists():
        try:
            return local_path.stat().st_size
        except Exception:
            pass
    # Для Vosk zip: проверяем распакованную папку
    if str(local_path).endswith('.zip'):
        extracted_dir = local_path.parent / local_path.stem
        if extracted_dir.exists():
            try:
                return sum(f.stat().st_size for f in extracted_dir.rglob('*') if f.is_file())
            except Exception:
                pass
    # Запрашиваем content-length с сервера (HEAD запрос)
    try:
        resp = requests.head(url, allow_redirects=True, timeout=8)
        cl = resp.headers.get('content-length')
        if cl:
            return int(cl)
    except Exception:
        pass
    return 0

def get_voice_models_choices():
    """Возвращает список (id, название_с_размером) для Dropdown обновления моделей.
    Показывает размер установленной модели на диске, или ❌ если отсутствует.
    БЕЗ сети — только проверка локальных файлов."""
    choices = []
    for m_id, m_name, files in VOICE_MODELS_REGISTRY:
        installed_bytes = 0
        all_installed = True
        for url, rel_path in files:
            local_path = CURRENT_DIR / "models" / rel_path
            is_installed = local_path.exists()
            # Vosk: zip удаляется после распаковки — проверяем папку
            if not is_installed and rel_path.endswith('.zip'):
                extracted_dir = CURRENT_DIR / "models" / Path(rel_path).stem
                if extracted_dir.exists():
                    is_installed = True
                    installed_bytes += sum(f.stat().st_size for f in extracted_dir.rglob('*') if f.is_file())
            if is_installed and local_path.exists():
                installed_bytes += local_path.stat().st_size
            if not is_installed:
                all_installed = False
        if all_installed:
            label = f"{m_name} ✅ ({_format_size(installed_bytes)})"
        else:
            label = f"{m_name} ❌"
        choices.append((m_id, label))
    return choices

def quick_check_models_local():
    """БЫСТРАЯ локальная проверка (БЕЗ сети) — какие модели установлены / отсутствуют.
    Возвращает строку-статус для UI."""
    installed_list = []
    missing_list = []
    for m_id, m_name, files in VOICE_MODELS_REGISTRY:
        model_ok = True
        for url, rel_path in files:
            local_path = CURRENT_DIR / "models" / rel_path
            is_installed = local_path.exists()
            if not is_installed and rel_path.endswith('.zip'):
                extracted_dir = CURRENT_DIR / "models" / Path(rel_path).stem
                is_installed = extracted_dir.exists()
            if not is_installed:
                model_ok = False
                break
        if model_ok:
            installed_list.append(m_name)
        else:
            missing_list.append(m_name)
    total = len(VOICE_MODELS_REGISTRY)
    if not missing_list:
        return f"✅ All {total} voice models are installed!"
    result = f"⚠️ Installed: {len(installed_list)}/{total} | Missing: {len(missing_list)}\n\n"
    result += "Missing models:\n"
    result += "\n".join(f"  ❌ {name}" for name in missing_list)
    result += "\n\n💡 Click «⬇️⬇️ Update ALL models» to download the missing ones."
    return result

def check_voice_model_status(model_id):
    """Проверяет, какие файлы модели уже скачаны.
    Возвращает (установлено_всего, всего_файлов, список_статусов).
    Для Vosk (zip) проверяет распакованную папку, а не сам zip."""
    for m_id, m_name, files in VOICE_MODELS_REGISTRY:
        if m_id == model_id:
            statuses = []
            installed = 0
            for url, rel_path in files:
                local_path = CURRENT_DIR / "models" / rel_path
                exists = local_path.exists()
                # Vosk: zip удаляется после распаковки — проверяем папку
                if not exists and rel_path.endswith('.zip'):
                    extracted_dir = CURRENT_DIR / "models" / Path(rel_path).stem
                    exists = extracted_dir.exists()
                if exists: installed += 1
                statuses.append((rel_path, exists))
            return installed, len(files), statuses
    return 0, 0, []

def update_voice_model(model_id):
    """Скачивает/обновляет одну голосовую модель.
    Возвращает (строка_статус, размер_в_байтах).
    размер_в_байтах — сумма размеров всех файлов модели (0 если не удалось определить)."""
    if not model_id:
        return "⚠️ Select a model to update", 0

    target = None
    for m_id, m_name, files in VOICE_MODELS_REGISTRY:
        if m_id == model_id:
            target = (m_name, files)
            break
    if target is None:
        return f"❌ Model {model_id} not found in the registry", 0

    m_name, files = target
    results = []
    all_ok = True
    model_total_bytes = 0

    for url, rel_path in files:
        local_path = CURRENT_DIR / "models" / rel_path
        # Определяем размер файла ДО скачивания (с диска или из HTTP-заголовка)
        file_size = _get_file_size(url, local_path)
        model_total_bytes += file_size
        size_str = _format_size(file_size)

        # Если файл уже есть — пропускаем (не перезаписываем без необходимости)
        if local_path.exists():
            results.append(f"✅ {rel_path} — already installed ({size_str})")
            continue
        # Vosk zip: проверяем распакованную папку
        if rel_path.endswith('.zip'):
            extracted_dir = CURRENT_DIR / "models" / Path(rel_path).stem
            if extracted_dir.exists():
                results.append(f"✅ {rel_path} — already installed ({size_str})")
                continue
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            m, status = download_model(url, local_path)
            if m is None:
                results.append(f"❌ {rel_path} — error: {status}")
                all_ok = False
            else:
                results.append(f"⬇️ {rel_path} — downloaded successfully ({size_str})")
        except Exception as e:
            results.append(f"❌ {rel_path} — {e}")
            all_ok = False

    # Особый случай: Vosk — нужно распаковать zip
    if model_id in ("vosk_010",) and all_ok:
        from zipfile import ZipFile
        zip_path = CURRENT_DIR / "models" / files[0][1]
        try:
            if zip_path.exists():
                with ZipFile(str(zip_path), "r") as zf:
                    zf.extractall(str(CURRENT_DIR / "models"))
                zip_path.unlink(missing_ok=True)
                results.append("📦 Archive unpacked")
        except Exception as e:
            results.append(f"⚠️ Unpack error: {e}")

    # Авто-удаление старых версий файлов после успешного обновления
    if all_ok and model_id in OLD_FILES_CLEANUP:
        for old_rel_path in OLD_FILES_CLEANUP[model_id]:
            old_path = CURRENT_DIR / "models" / old_rel_path
            if old_path.exists():
                try:
                    old_size = old_path.stat().st_size
                    old_path.unlink()
                    results.append(f"🗑 Removed old version: {old_rel_path} ({_format_size(old_size)})")
                except Exception as e:
                    results.append(f"⚠️ Could not remove old version {old_rel_path}: {e}")

    header = f"{'✅' if all_ok else '⚠️'} {m_name} ({_format_size(model_total_bytes)}):\n"
    return header + "\n".join(results), model_total_bytes

def update_all_voice_models():
    """ГЕНЕРАТОР: скачивает/обновляет ВСЕ голосовые модели подряд с живым прогрессом.
    Yield'ит накопленный статус после каждой модели — UI не зависает.
    Показывает размеры каждого файла и общий объём скачивания.
    Поддерживает остановку через stop_model_update()."""
    global _stop_model_update
    _stop_model_update = False  # Сбрасываем флаг в начале

    total = len(VOICE_MODELS_REGISTRY)
    all_results = []
    ok_count = 0
    total_bytes = 0

    # Первый yield — мгновенный отклик, чтобы UI не выглядел зависшим
    yield f"📦 Starting update of {total} models...\n⏳ Determining file sizes...\n\n"

    for i, (m_id, m_name, _) in enumerate(VOICE_MODELS_REGISTRY, 1):
        # Проверка флага остановки
        if _stop_model_update:
            _stop_model_update = False  # Сбрасываем для следующего запуска
            stop_line = (
                f"\n\n{'═' * 40}"
                f"\n🛑 Stopped by the user!"
                f"\n📊 Processed: {i-1}/{total} models | ✅ {ok_count} up to date"
                f"\n💾 Total size: {_format_size(total_bytes)}"
            )
            yield "\n\n".join(all_results) + stop_line
            return

        status, model_bytes = update_voice_model(m_id)
        total_bytes += model_bytes
        all_results.append(f"─── [{i}/{total}] {m_name} ───\n{status}")
        if "✅" in status and "❌" not in status:
            ok_count += 1

        # Живой прогресс: показываем накопленный лог + счётчик + общий объём
        progress_line = (
            f"\n\n{'─' * 40}"
            f"\n📊 Progress: {i}/{total} models | ✅ {ok_count} up to date"
            f"\n💾 Total size: {_format_size(total_bytes)}"
        )
        yield "\n\n".join(all_results) + progress_line

    summary = (
        f"\n\n═══════════════════"
        f"\n🎉 Done: {ok_count}/{total} models are up to date"
        f"\n💾 Total size of all models: {_format_size(total_bytes)}"
    )
    yield "\n\n".join(all_results) + summary

def check_all_voice_models():
    """ГЕНЕРАТОР: проверяет актуальность ВСЕХ голосовых моделей без скачивания.
    Показывает какие файлы установлены / отсутствуют, их размеры на диске
    и требуемый объём для докачки отсутствующих. Ничего не скачивает."""
    total = len(VOICE_MODELS_REGISTRY)
    all_results = []
    installed_count = 0
    missing_count = 0
    installed_bytes = 0
    missing_bytes = 0

    # Первый yield — мгновенный отклик
    yield f"🔍 Checking {total} voice models...\n⏳ Checking installed files and sizes of missing ones...\n\n"

    for i, (m_id, m_name, files) in enumerate(VOICE_MODELS_REGISTRY, 1):
        lines = []
        model_installed = True
        model_installed_bytes = 0
        model_missing_bytes = 0

        for url, rel_path in files:
            local_path = CURRENT_DIR / "models" / rel_path
            is_installed = local_path.exists()
            # Vosk: zip удаляется после распаковки — проверяем папку
            if not is_installed and rel_path.endswith('.zip'):
                extracted_dir = CURRENT_DIR / "models" / Path(rel_path).stem
                is_installed = extracted_dir.exists()

            # Размер: с диска если установлен, иначе из HTTP content-length
            file_size = _get_file_size(url, local_path)
            size_str = _format_size(file_size)

            if is_installed:
                lines.append(f"  ✅ {rel_path} — installed ({size_str})")
                model_installed_bytes += file_size
            else:
                lines.append(f"  ❌ {rel_path} — missing (need to download {size_str})")
                model_missing_bytes += file_size
                model_installed = False

        installed_bytes += model_installed_bytes
        missing_bytes += model_missing_bytes

        if model_installed:
            status_icon = "✅"
            installed_count += 1
        else:
            status_icon = "❌"
            missing_count += 1

        model_total = model_installed_bytes + model_missing_bytes
        header = f"{status_icon} {m_name} ({_format_size(model_total)})"
        if not model_installed:
            header += f" — need to download {_format_size(model_missing_bytes)}"

        all_results.append(f"─── [{i}/{total}] {header} ───\n" + "\n".join(lines))

        # Живой прогресс после каждой модели
        # «0 Б» если действительно ничего, иначе форматированный размер ("?" если размер неизвестен)
        installed_size_str = "0 B" if installed_bytes == 0 else _format_size(installed_bytes)
        missing_size_str = "0 B" if missing_bytes == 0 else _format_size(missing_bytes)
        progress_line = (
            f"\n\n{'─' * 40}"
            f"\n📊 Checked: {i}/{total} | ✅ {installed_count} installed | ❌ {missing_count} missing"
            f"\n💾 On disk: {installed_size_str} | Need to download: {missing_size_str}"
        )
        yield "\n\n".join(all_results) + progress_line

    # Финальная сводка
    installed_size_str = "0 B" if installed_bytes == 0 else _format_size(installed_bytes)
    summary = (
        f"\n\n═══════════════════"
        f"\n🔍 Check complete!"
        f"\n✅ Installed: {installed_count}/{total} models ({installed_size_str} on disk)"
    )
    if missing_count > 0:
        missing_size_str = "0 B" if missing_bytes == 0 else _format_size(missing_bytes)
        summary += (
            f"\n❌ Missing: {missing_count}/{total} models"
            f"\n⬇️ Need to download: {missing_size_str}"
            f"\n\n💡 Click «⬇️⬇️ Update ALL models» to download the missing ones."
        )
    else:
        summary += "\n\n🎉 All models are up to date! Nothing to download."
    yield "\n\n".join(all_results) + summary