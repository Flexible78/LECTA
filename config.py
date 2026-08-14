import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse

APP_NAME = "TTS-Server"

# URL of the LECTA GitHub repository.
# NOTE: The repository is currently private. This link will become
# active once the repo is published. See BF3 of the public-release task.
LECTA_REPO_URL = "https://github.com/Flexible78/LECTA"


# Number of simultaneous TTS synthesis tasks. Reduced from 8 to 4 to keep GPU
# temperature under control (8 parallel F5-TTS on 6 GB VRAM caused 88 °C).
# Override with the LECTA_TTS_WORKERS environment variable.
TTS_WORKERS = int(os.getenv("LECTA_TTS_WORKERS", "4"))

# Pause between files in batch mode, seconds (0 = disabled).
TTS_COOLDOWN_SEC = int(os.getenv("LECTA_TTS_COOLDOWN_SEC", "0"))

# Soft GPU temperature threshold, °C. Synthesis pauses when reached.
TTS_GPU_TEMP_LIMIT_C = int(os.getenv("LECTA_GPU_TEMP_LIMIT", "83"))

# Temperature to resume synthesis after a cooldown pause, °C.
# Must be noticeably lower than the limit.
TTS_GPU_TEMP_RESUME_C = int(os.getenv("LECTA_GPU_TEMP_RESUME", "76"))

# GPU power limit in watts. Requires nvidia-smi AND administrator rights,
# therefore not used by default. Change manually only.
TTS_GPU_POWER_LIMIT_W = None
# Path.home() может вернуть относительный путь (например 'tmp'), если
# USERPROFILE/appdata заданы как относительные в Start.cmd (портабельная сборка).
# Разрешаем относительно каталога config.py (fb2tts/), чтобы путь всегда был абсолютным.
_BASE_DIR = Path(__file__).resolve().parent
_home = Path.home()
if not _home.is_absolute():
    _home = (_BASE_DIR / _home).resolve()
USER_SETTINGS_FILE = _home / ".config" / APP_NAME / "user_settings.json"
USER_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

@dataclass
class AppConfig:
    port: int = 7860
    share: bool = False
    debug: bool = False
    server_name: str = "0.0.0.0"
    log_level: str = "INFO"
    punctuation: bool = False
    translit: bool = True
    ch_size: int = 200
    gender: bool = False
    profanity: bool = False
    sound_effect: bool = False
    single_vowel: bool = False
    sp_rate: float = 1
    back_sound_sel: str = ''
    bitrate: int = 96
    noise_lvl: int = 16
    use_sound_effect: bool = False
    models_path: str = os.getenv("LECTA_MODELS_DIR", "models")
    data_path: str = 'data'
    # Edge TTS cloud flags (сохраняются между перезапусками)
    use_edge_english: bool = False
    use_edge_russian: bool = False
    use_edge_hebrew: bool = True
    dict_mode: bool = False
    completion_sound: str = "complete.wav"

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description="LECTA TTS server launch options")
        parser.add_argument("--port", type=int, default=int(os.getenv("LECTA_PORT", "7860")), help="Port to run the server on")
        parser.add_argument("--share", action="store_true", help="Create a public share link")
        parser.add_argument("--debug", action="store_true", help="Debug mode")
        parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Server bind address")
        args = parser.parse_args()
        
        return AppConfig(
            port=args.port,
            share=args.share,
            debug=args.debug,
            server_name=args.server_name,
            log_level="DEBUG" if args.debug else "INFO"
        )
    
    @staticmethod
    def load_user_settings() -> 'AppConfig':
        if USER_SETTINGS_FILE.exists():
            with open(USER_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)

            fields = {f.name for f in AppConfig.__dataclass_fields__.values()}
            filtered = {k: v for k, v in data.items() if k in fields}
            return AppConfig(**filtered)
        return AppConfig()
    
    @staticmethod
    def save_user_settings(args):
        current = AppConfig.load_user_settings()
        settings = asdict(current)
        settings.update(args)

        fields = {f.name for f in AppConfig.__dataclass_fields__.values()}
        filtered = {k: v for k, v in settings.items() if k in fields}

        # Пересоздаём каталог на случай если clean_tmp_folder() удалил tmp/.config/
        USER_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(USER_SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(filtered, f, ensure_ascii=False, indent=4)

config = AppConfig.load_user_settings()
