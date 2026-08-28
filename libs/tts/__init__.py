import os
import re
import torch
import gc
import threading
import numpy as np
from pathlib import Path
from libs.utils import download_model, models_path
from libs.tts.vosk_backend import Model, Synth
from libs.tts.f5_backend import F5Model, F5Synth

def _model_not_found_hint():
    """English guidance shown when model files are missing."""
    try:
        resolved = str(models_path)
    except Exception:
        resolved = os.getenv("LECTA_MODELS_DIR", "models")
    return (
        "\n\nModel files not found. To resolve this:\n"
        "  1) Set LECTA_MODELS_DIR to the folder containing your voice models "
        f"(current resolved path: '{resolved}'),\n"
        "  2) Check the models_path value in the settings file,\n"
        "  3) Or place the 'models' folder next to the LECTA application."
    )

# ── SHORT MODEL-NAME MAP (for the audio table display) ──
# Used by tts_tab.py when saving and displaying MP3s.
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

# ── GLOBAL GPU/CPU DEVICE SWITCH ──
# Controlled from app.py via set_tts_device()
# "auto" = cuda if available, otherwise cpu
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
    return f"{'GPU (CUDA)' if mode != 'cpu' else 'CPU (RAM)'} mode activated"

device = "cuda" if torch.cuda.is_available() else "cpu"
now_dir = Path.cwd()

# --- STRESS MARKS ---
# "+" is an internal stress mark and must NEVER be pronounced.
# It is stripped for every engine (Silero, Vosk, F5); only a real
# math sign between digits is preserved.
STRESS_AWARE_VERS = ()


def strip_stress_marks(text):
    if not text:
        return text
    text = re.sub(r"(?<=\d)\s*\+\s*(?=\d)", "<<PLUS>>", str(text))
    text = text.replace("+", "")
    return text.replace("<<PLUS>>", "+")


class TTSModel:
    def __init__(self):
        self.model = None
        self.ver = None
        self.device = torch.device(device)
        self.f5synth = None  # Cache F5Synth so the vocoder isn't reloaded every time
        self._f5synth_lock = threading.Lock()  # guard the lazy F5Synth init during parallel synthesis

    def load(self, ver):
        if self.model is not None:
            del self.model
            self.model = None
            self.f5synth = None  # Reset the cache when the model changes
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        self.ver = ver

        if ver == 2:
            model_ver = '0.10'
            try:
                self.model = Model(model_ver)
                return ver, "Model loaded successfully!"
            except Exception as e:
                if isinstance(e, FileNotFoundError) or "No such file" in str(e):
                    return None, f"Failed to load model: {e}.{_model_not_found_hint()}"
                return None, f"Initialization error: {e}"
        elif ver in [3, 4, 7]:
            # Refresh the device on every load (the user may have switched GPU/CPU)
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
                print(f'Loading Silero TTS ({lang})...')
                model_url = f"https://models.silero.ai/models/tts/{lang}/{model_name}"
                m, status = download_model(model_url, model_path)
                if m is None:
                    return m, status
            try:
                package = torch.package.PackageImporter(str(model_path))
                self.model = package.load_pickle("tts_models", "model")
                self.model.to(self.device)
                return ver, "Model loaded successfully!"
            except Exception as e:
                if isinstance(e, FileNotFoundError) or "No such file" in str(e):
                    return None, f"Failed to load model: {e}.{_model_not_found_hint()}"
                return None, f"Model loading error: {e}"
        else:
            try:
                self.model = F5Model()
                self.model.load(model_ver=ver)
                return ver, "Model loaded successfully!"
            except Exception as e:
                if isinstance(e, FileNotFoundError) or "No such file" in str(e):
                    return None, f"Failed to load model: {e}.{_model_not_found_hint()}"
                return None, f"Initialization error: {e}"
    
    def synth_audio(self, text, speaker_id, speed=1, noise=16, ref_audio=None, ref_text=''):
        text = strip_stress_marks(text)  # "+" is never read aloud
        if self.ver in [3, 4, 7]:
            np_audio = self.model.apply_tts(text, speaker=speaker_id, sample_rate=48000)
            np_audio = np_audio.detach().numpy()
            np_audio = (np_audio * 32767).astype(np.int16)
            return np_audio, 48000
        elif self.ver == 5 or self.ver == 6:
            # Cache F5Synth — otherwise the vocoder reloads from disk on every call!
            # Double-checked locking: on CPU several threads can enter at once.
            if self.f5synth is None or self.f5synth.model is not self.model:
                with self._f5synth_lock:
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