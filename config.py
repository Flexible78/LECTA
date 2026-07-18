import json
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse

APP_NAME = "TTS-Server"
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
    models_path: str = 'models'
    data_path: str = 'data'
    # Edge TTS cloud flags (сохраняются между перезапусками)
    use_edge_english: bool = False
    use_edge_russian: bool = False
    use_edge_hebrew: bool = True
    dict_mode: bool = False
    completion_sound: str = "complete.wav"

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description="Параметры запуска TTS-сервера")
        parser.add_argument("--port", type=int, default=7860, help="Порт для запуска сервера")
        parser.add_argument("--share", action="store_true", help="Создать публичную ссылку")
        parser.add_argument("--debug", action="store_true", help="Режим отладки")
        parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Адрес сервера")
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
