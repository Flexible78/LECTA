"""Run the real ``gr_tabs.tts_tab.tts()`` pipeline end to end.

Unlike ``verify_e2e_audio.py`` (which calls the router library directly) this
exercises the generator that the Gradio button calls: XML parsing, chunk
batching, segment assembly, final trim/normalize and atomic verified MP3
export.  It uses a throw-away project directory and deletes it afterwards, so
no real project is touched.

Run:
    python tests/verify_gui_pipeline.py
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import AppConfig
import libs.multilingual_router as router
import gr_tabs.tts_tab as tt
from libs.audio_tools import analyze_audio, probe_audio, verify_audio_file

PROJECT = "E2E_GUI_MIXED_RU_EN_HE"
XML = (
    '<root autor="LECTA E2E" album="LECTA E2E">'
    "<line>Это проверка русской речи. This is an English sentence inside the "
    "same paragraph. זהו משפט בעברית לבדיקת ההגייה וההפסקות. Снова русский текст. "
    "Проверяем, что паузы короткие и речь не обрезана.</line>"
    "</root>"
)


def main() -> int:
    tmp_root = Path(tempfile.mkdtemp(prefix="lecta_gui_pipeline_"))
    data = tmp_root / "data"
    base = data / PROJECT
    (base / "xml").mkdir(parents=True)
    (base / "mp3").mkdir()
    # An empty FB2 marker keeps the auto-parse branch from failing; XML already exists.
    (base / f"{PROJECT}.fb2").write_text("<FictionBook/>", encoding="utf-8")
    (base / "xml" / "1.xml").write_text(XML, encoding="utf-8")

    original_data = tt.data_path
    saved = AppConfig.load_user_settings()
    saved_flags = {
        "use_edge_english": router.USE_EDGE_FOR_ENGLISH,
        "use_edge_hebrew": router.USE_EDGE_FOR_HEBREW,
        "use_edge_russian": router.USE_EDGE_FOR_RUSSIAN,
        "dict_mode": router.DICTIONARY_MODE,
    }
    report: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "project": PROJECT}
    try:
        tt.data_path = data
        AppConfig.save_user_settings({
            "use_edge_english": True,
            "use_edge_hebrew": True,
            "use_edge_russian": True,
            "dict_mode": False,
            "generation_preset": "compact",
        })
        router.USE_EDGE_FOR_ENGLISH = True
        router.USE_EDGE_FOR_HEBREW = True
        router.USE_EDGE_FOR_RUSSIAN = True
        router.DICTIONARY_MODE = False
        router.apply_generation_settings(preset="compact")

        t0 = time.time()
        steps = 0
        status = ""
        for row in tt.tts(
            PROJECT, True, "ru-RU-DmitryNeural", 1, "", 192, 10, False, False, False
        ):
            steps += 1
            if len(row) > 1:
                status = row[1]
        elapsed = time.time() - t0

        mp3 = base / "mp3" / f"{PROJECT}.mp3"
        report.update({"elapsed_sec": round(elapsed, 1), "yield_steps": steps,
                       "final_status": status, "mp3_exists": mp3.is_file()})
        if mp3.is_file():
            info = probe_audio(str(mp3))
            analysis = analyze_audio(str(mp3), threshold_db=-38, min_silence_sec=0.2)
            report.update({
                "duration": round(info.duration, 2),
                "bytes": mp3.stat().st_size,
                "valid": verify_audio_file(str(mp3))["ok"],
                "silence_total": round(analysis.total_silence, 2),
                "longest_pause": round(analysis.longest_silence, 2),
                "pauses": [round(s.duration, 2) for s in analysis.silences],
                "warnings": analysis.warnings,
                "success_factor": "ALL FILES SUCCESSFULLY" in status,
            })
    finally:
        tt.data_path = original_data
        AppConfig.save_user_settings(asdict(saved))
        router.USE_EDGE_FOR_ENGLISH = saved_flags["use_edge_english"]
        router.USE_EDGE_FOR_HEBREW = saved_flags["use_edge_hebrew"]
        router.USE_EDGE_FOR_RUSSIAN = saved_flags["use_edge_russian"]
        router.DICTIONARY_MODE = saved_flags["dict_mode"]
        router.apply_generation_settings({
            "generation_preset": saved.generation_preset,
            "pause_intra_ms": saved.pause_intra_ms,
            "pause_sentence_ms": saved.pause_sentence_ms,
            "pause_paragraph_ms": saved.pause_paragraph_ms,
            "pause_language_ms": saved.pause_language_ms,
            "pause_learning_ms": saved.pause_learning_ms,
            "explicit_break_max_ms": saved.explicit_break_max_ms,
            "edge_trim_ms": saved.edge_trim_ms,
            "micro_fade_ms": saved.micro_fade_ms,
        })
        shutil.rmtree(tmp_root, ignore_errors=True)

    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = ROOT / "samples" / "audio_repair_20261115" / "gui_pipeline_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    ok = bool(report.get("mp3_exists") and report.get("valid") and report.get("success_factor"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
