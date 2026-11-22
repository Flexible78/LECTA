"""Shared helpers for LECTA tests (stdlib unittest, real FFmpeg audio)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.audio_tools import FFMPEG, CREATE_NO_WINDOW  # noqa: E402


def _run(cmd):
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8",
        errors="replace", creationflags=CREATE_NO_WINDOW, timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}\n{proc.stderr}")
    return proc


def tone_clip(path: Path, *, freq: int = 220, seconds: float = 0.7,
              sample_rate: int = 24000, volume: float = 0.25) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _run([FFMPEG, "-hide_banner", "-nostats", "-y", "-f", "lavfi",
          "-i", f"sine=frequency={freq}:sample_rate={sample_rate}:duration={seconds}",
          "-af", f"volume={volume}", "-c:a", "libmp3lame", "-b:a", "128k",
          "-ar", str(sample_rate), "-ac", "1", str(path)])
    return path


def silence(path: Path, *, seconds: float, sample_rate: int = 24000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _run([FFMPEG, "-hide_banner", "-nostats", "-y", "-f", "lavfi",
          "-i", f"anullsrc=r={sample_rate}:cl=mono", "-t", f"{seconds}",
          "-c:a", "libmp3lame", "-b:a", "128k", "-ar", str(sample_rate), "-ac", "1",
          str(path)])
    return path


def build_speech_with_pauses(path: Path, *, pauses=(3.0, 0.0, 2.0),
                             clip_seconds: float = 0.7,
                             sample_rate: int = 24000) -> Path:
    """Build a file of tones separated by *pauses* seconds of real silence."""
    tmp = path.parent / f"._{path.stem}_parts"
    tmp.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, pause in enumerate(pauses):
        clip = tone_clip(tmp / f"clip{i}.mp3", freq=200 + 40 * i,
                         seconds=clip_seconds, sample_rate=sample_rate)
        parts.append(clip)
        if pause > 0:
            parts.append(silence(tmp / f"gap{i}.mp3", seconds=pause,
                                 sample_rate=sample_rate))
    list_file = tmp / "list.txt"
    list_file.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in parts) + "\n", encoding="utf-8")
    _run([FFMPEG, "-hide_banner", "-nostats", "-y", "-f", "concat", "-safe", "0",
          "-i", str(list_file), "-c:a", "libmp3lame", "-b:a", "128k",
          "-ar", str(sample_rate), "-ac", "1", str(path)])
    for p in parts:
        p.unlink(missing_ok=True)
    list_file.unlink(missing_ok=True)
    try:
        tmp.rmdir()
    except OSError:
        pass
    return path


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
