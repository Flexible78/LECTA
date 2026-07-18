import torch
import gc
import numpy as np
from pathlib import Path
from libs.utils import download_model, models_path
from libs.tts.vosk_backend import Model, Synth
from libs.tts.f5_backend import F5Model, F5Synth

# ── КАРТА КОРОТКИХ ИМЁН МОДЕЛЕЙ (для отображения в таблице аудио) ──
# Используется tts_tab.py при сохранении и показе MP3.
MODEL_SHORT_NAMES = {
    2: "Vosk0.10",
    3: "Silero5_5",
    4: "SileroCIS",
    5: "F5-Misha",
    6: "F5-ESp",
    7: "SileroEN",
}

def get_model_short_name(ver):
    """Возвращает короткое имя текущей TTS-модели по её версии (ver)."""
    return MODEL_SHORT_NAMES.get(ver, "?")

# ── ГЛОБАЛЬНЫЙ ПЕРЕКЛЮЧАТЕЛЬ УСТРОЙСТВА GPU/CPU ──
# Управляется из app.py через set_tts_device()
# "auto" = cuda если доступна иначе cpu
_tts_device_mode = "auto"

def get_device():
    """Возвращает текущее устройство torch в зависимости от режима."""
    if _tts_device_mode == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def set_tts_device(mode):
    """Устанавливает режим устройства: 'auto' (GPU) или 'cpu' (RAM)."""
    global _tts_device_mode
    _tts_device_mode = mode
    return f"{'🎮 GPU (CUDA)' if mode != 'cpu' else '💾 CPU (RAM)'} режим активирован"

device = "cuda" if torch.cuda.is_available() else "cpu"
now_dir = Path.cwd()

class TTSModel:
    def __init__(self):
        self.model = None
        self.ver = None
        self.device = torch.device(device)
        self.f5synth = None  # Кэшируем F5Synth чтобы не перезагружать вокодер каждый раз

    def load(self, ver):
        if self.model is not None:
            del self.model
            self.model = None
            self.f5synth = None  # Сбрасываем кэш при смене модели
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        self.ver = ver

        if ver == 2:
            model_ver = '0.10'
            try:
                self.model = Model(model_ver)
                return ver, "Модель успешно загружена!"
            except Exception as e:
                return None, f"Ошибка инициализации: {e}"
        elif ver in [3, 4, 7]:
            # Обновляем устройство при каждой загрузке (пользователь мог переключить GPU/CPU)
            self.device = torch.device(get_device())
            if ver == 3:
                model_name = 'v5_5_ru.pt'
                lang = 'ru'
            elif ver == 4:
                model_name = 'v5_cis_base_nostress.pt'
                lang = 'ru'
            elif ver == 7:
                model_name = 'v3_en.pt'
                lang = 'en'
                
            model_dir =  models_path / "silero"
            model_path = model_dir / model_name
            if not model_path.is_file():
                model_dir.mkdir(exist_ok=True)
                print(f'Загрузка Silero TTS ({lang})...')
                model_url = f"https://models.silero.ai/models/tts/{lang}/{model_name}"
                m, status = download_model(model_url, model_path)
                if m is None:
                    return m, status
            try:
                package = torch.package.PackageImporter(str(model_path))
                self.model = package.load_pickle("tts_models", "model")
                self.model.to(self.device)
                return ver, "Модель успешно загружена!"
            except Exception as e:
                return None, f"Ошибка загрузки модели: {e}"
        else:
            try:
                self.model = F5Model()
                self.model.load(model_ver=ver)
                return ver, "Модель успешно загружена!"
            except Exception as e:
                return None, f"Ошибка инициализации: {e}"
    
    def synth_audio(self, text, speaker_id, speed=1, noise=16, ref_audio=None, ref_text=''):
        if self.ver in [3, 4, 7]:
            np_audio = self.model.apply_tts(text, speaker=speaker_id, sample_rate=48000)
            np_audio = np_audio.detach().numpy()
            np_audio = (np_audio * 32767).astype(np.int16)
            return np_audio, 48000
        elif self.ver == 5 or self.ver == 6:
            # Кэшируем F5Synth — иначе вокодер перезагружается с диска на каждый вызов!
            if self.f5synth is None or self.f5synth.model is not self.model:
                self.f5synth = F5Synth(self.model)
            audio_wave, sample_rate = self.f5synth.synth_audio(
                text,
                speaker_id=speaker_id,
                speed=speed,
                noise=noise,
                ref_audio=ref_audio,
                ref_text=ref_text
            )
            return audio_wave, sample_rate
        else:
            return Synth(self.model).synth_audio(text, speaker_id=speaker_id, speech_rate=speed), 22050

    def speakers_list(self):
        if self.ver is None:
            return []
        else:
            return sorted(self.model.speakers)

synth = TTSModel()