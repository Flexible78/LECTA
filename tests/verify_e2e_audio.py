"""Real end-to-end check of the multilingual pipeline + pause cleaner.

Uses live Microsoft Edge TTS for RU/EN/HE (no GPU model needed), writes an MP3
through FFmpeg, then produces BEFORE/AFTER samples and a measurement report.

Run:
    python tests/verify_e2e_audio.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import libs.multilingual_router as router
from libs.audio_tools import (
    FFMPEG, CREATE_NO_WINDOW, analyze_audio, clean_audio_file, probe_audio,
    summarize_results, verify_audio_file,
)

SAMPLE_DIR = ROOT / "samples" / "audio_repair_20261115"
TEXT = (
    "Сегодня проверяем новый образец русской речи. Мы читаем книгу спокойно и естественно.\n"
    "We are testing a fresh English phrase for clear speech. It should flow naturally.\n"
    "אנחנו בודקים כעת משפט חדש בעברית כדי לוודא דיבור ברור.\n"
    "Повторно проверяем русский текст, короткие паузы и сохранность согласных."
)


def write_mp3(audio: np.ndarray, sample_rate: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.with_suffix(".raw")
    pcm = np.clip(audio, -1.0, 1.0)
    (pcm * 32767).astype(np.int16).tofile(raw)
    subprocess.run(
        [FFMPEG, "-hide_banner", "-nostats", "-y", "-f", "s16le", "-ar", str(sample_rate),
         "-ac", "1", "-i", str(raw), "-c:a", "libmp3lame", "-b:a", "192k",
         "-ar", str(sample_rate), "-ac", "1", str(path)],
        check=True, capture_output=True, creationflags=CREATE_NO_WINDOW,
    )
    raw.unlink(missing_ok=True)


def local_fallback(text, disable_norm=False):
    raise RuntimeError("local TTS intentionally unavailable in this online check")


def main() -> int:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    report = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": []}

    router.apply_generation_settings(preset="compact")
    t0 = time.time()
    sr, audio = router.process_multilingual_text(
        TEXT, local_fallback,
        use_edge_en=True, use_edge_he=True, use_edge_ru=True, dict_mode=False,
    )
    elapsed = time.time() - t0
    generated = SAMPLE_DIR / "generated_mixed_ru_en_he_compact.mp3"
    write_mp3(audio, sr, generated)

    analysis = analyze_audio(str(generated), threshold_db=-38, min_silence_sec=0.35)
    check = verify_audio_file(str(generated))
    report["steps"].append({
        "step": "generation (Edge RU+EN+HE, compact)",
        "seconds": round(elapsed, 1),
        "duration": round(analysis.info.duration, 2),
        "silence_total": round(analysis.total_silence, 2),
        "longest_pause": round(analysis.longest_silence, 2),
        "silences": len(analysis.silences),
        "valid": check["ok"],
        "warnings": analysis.warnings,
    })

    before = SAMPLE_DIR / "BEFORE_02_PITCH_AND_HR_HE_EN_RU.mp3"
    existing = ROOT / "data" / "02_PITCH_AND_HR_HE_EN_RU" / "mp3" / "02_PITCH_AND_HR_HE_EN_RU.mp3"
    if existing.is_file():
        if not before.exists():
            before.write_bytes(existing.read_bytes())
        clean = clean_audio_file(before, preset="compact", quality_kbps=192)
        after = Path(clean.output) if clean.output else None
        if after and after.exists():
            target = SAMPLE_DIR / "AFTER_02_PITCH_AND_HR_HE_EN_RU_compact.mp3"
            target.write_bytes(after.read_bytes())
            after.unlink(missing_ok=True)
        report["steps"].append({
            "step": "cleaner on the real broken file",
            "before": round(clean.before_duration, 1),
            "after": round(clean.after_duration, 1),
            "removed": round(clean.removed, 1),
            "pauses": clean.pauses_shortened,
            "status": clean.status,
            "after_file": str(target) if after else "",
            "valid": verify_audio_file(str(target))["ok"] if after else False,
        })

    clean_generated = clean_audio_file(generated, preset="compact", quality_kbps=192)
    compact_path = Path(clean_generated.output) if clean_generated.output else None
    compact_sample = SAMPLE_DIR / "AFTER_generated_mixed_ru_en_he_compact.mp3"
    if compact_path and compact_path.is_file():
        compact_sample.write_bytes(compact_path.read_bytes())
        compact_path.unlink(missing_ok=True)
    report["steps"].append({
        "step": "cleaner on freshly generated audio",
        **summarize_results([clean_generated]),
        "status": clean_generated.status,
        "after_sample": str(compact_sample) if compact_sample.is_file() else "",
        "valid": verify_audio_file(str(compact_sample))["ok"] if compact_sample.is_file() else False,
    })

    # Keep the reproducible BEFORE asset and exactly one generated BEFORE/AFTER
    # pair; don't leave collision-safe intermediate cleaner outputs in samples.
    for duplicate in SAMPLE_DIR.glob("generated_mixed_ru_en_he_compact_compact*.mp3"):
        duplicate.unlink(missing_ok=True)

    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (SAMPLE_DIR / "e2e_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())