import argparse
import gc
import json
import logging
import re
from pathlib import Path

import librosa
import numpy as np
import requests
import torch
import torchaudio
from config import config

now_dir = Path.cwd()
data_path = Path(config.data_path)
models_path = Path(config.models_path)
data_path.mkdir(parents=True, exist_ok=True)
ab_name = ""
logging.basicConfig(level=logging.ERROR)


def convert(seconds):
    min, sec = divmod(seconds, 60)
    min = f"{int(min)} m. " if min else ""
    sec = f"{int(sec)} s."

    return min + sec


def get_data_list(d_path=data_path, r_glob=None, sort=None):
    base_path = Path(d_path)
    if not base_path.exists():
        return []
    if r_glob:
        files = [file.stem for file in base_path.rglob(r_glob) if file.is_file()]
        if sort:
            files.sort(key=sort)
        return files

    _EXCLUDE_DIRS = {"tts_cache", "__pycache__"}
    return [
        item.name
        for item in base_path.iterdir()
        if item.is_dir() and item.name not in _EXCLUDE_DIRS
    ]


def change_pitch(audio_array, sample_rate, semitones):
    audio_array = audio_array.astype(np.float32) / 32768.0

    if np.max(np.abs(audio_array)) > 1.0:
        audio_array = audio_array / np.max(np.abs(audio_array))
    actual_semitones = (semitones - 50) / 50 * 12
    y_shifted = librosa.effects.pitch_shift(
        y=audio_array, sr=sample_rate, n_steps=actual_semitones
    )

    y_shifted_int16 = (y_shifted * 32767.0).astype(np.int16)
    return y_shifted_int16


def download_model(model_url, target_path):
    # Преобразуем target_path в Path если это еще не Path
    if not isinstance(target_path, Path):
        target_path = Path(target_path)
    # Проверяем, существует ли файл
    if target_path.exists():
        return True, False

    try:
        response = requests.get(model_url, stream=True, timeout=5)
        response.raise_for_status()
        expected_size = int(response.headers.get("content-length", 0))

        # Создаем родительские директории если их нет
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        actual_size = target_path.stat().st_size
        if actual_size > 0 and (expected_size == 0 or actual_size == expected_size):
            return True, True
        else:
            if target_path.exists():
                target_path.unlink()
            return None, "Error: Размер неверный!"

    except Exception as e:
        if target_path.exists():
            target_path.unlink()
        return None, f"Error: {e}"
