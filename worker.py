#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LECTA background worker — runs the FULL pipeline (parse + TTS + packaging)
in a separate, detached process.

Why: the Gradio GUI can crash or the browser can be closed, but this process
keeps running and finishes the job. At the end it writes a package of 3 files
(mp3 + txt + source fb2) next to the original source file and opens it in
Explorer.

Usage:
    python worker.py <job.json>

job.json:
{
  "projects": ["book1", "book2", ...],
  "parse": {"sound_effect": false, "punctuation": false, "translit": true, "ch_size": 400},
  "tts": {
      "model_ver": 5, "device": "auto",
      "spk_sel": "...", "sp_rate": 1.0, "back_sound_sel": "",
      "bitrate": 96, "noise_lvl": 10, "use_sound_effect": false,
      "use_accents": true, "repl": true,
      "use_edge_en": false, "use_edge_he": true, "use_edge_ru": false, "dict_mode": false
  },
  "status_file": "tmp/bg_status.json",
  "log_file": "tmp/bg_worker.log"
}
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# ── UTF-8 console/file safety (emoji in logs must never crash the worker) ──
for _stream in (sys.stdout, sys.stderr):
    try:
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Gradio must never be used here — stub the handful of functions that the
# shared tts_tab module calls, so we can reuse its battle-tested pipeline
# (batch_tts_all_projects, thermal throttle, caching) fully headless.
import gradio as gr


def _noop(*args, **kwargs):
    return None


gr.update = _noop
gr.Info = _noop
gr.Warning = _noop
gr.Error = _noop

import shutil
import traceback

from pydub import AudioSegment

from libs.fb2_processor import FB2Processor
import libs.multilingual_router as router
from libs.tts import set_tts_device, synth
from libs.utils import data_path, get_data_list
from gr_tabs.tts_tab import (
    _concat_audio_segments,
    batch_tts_all_projects,
    parse_percent_from_html,
)

logger = logging.getLogger("LECTA-Worker")
logger.setLevel(logging.INFO)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_sh)
logger.propagate = False

STATUS_FILE = None
STATUS = {
    "state": "running",
    "stage": "",
    "project": "",
    "pct": 0,
    "message": "",
    "error": None,
    "package_dir": None,
}


def _save_status():
    """Atomically write the current status so the GUI can read it any time."""
    if not STATUS_FILE:
        return
    try:
        tmp = STATUS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(STATUS, ensure_ascii=False), encoding="utf-8")
        tmp.replace(STATUS_FILE)
    except Exception:
        pass


def _set_status(**kw):
    STATUS.update(kw)
    _save_status()


# ═══ STEP 1: PARSE (FB2 → XML) ═══
def run_parse(project, parse_cfg, replace=False):
    _set_status(stage="parse", project=project, pct=0, message=f"⏳ Parsing: {project}...")
    try:
        proc = FB2Processor()
        for pct, msg in proc.process_book(
            ab_path=project,
            replace=replace,
            sound_effect=parse_cfg.get("sound_effect", False),
            punctuation=parse_cfg.get("punctuation", False),
            translit=parse_cfg.get("translit", True),
            ch_size=parse_cfg.get("ch_size", 200),
        ):
            _set_status(stage="parse", project=project, pct=pct, message=msg)
            if pct >= 100:
                break
    except Exception as e:
        logger.error("Parse error %s: %s", project, e, exc_info=True)
        _set_status(stage="parse", project=project, message=f"⚠️ Parse error: {e}")


# ═══ STEP 2: TTS (XML → MP3) — reuses the exact same batch pipeline as the GUI ═══
def run_tts(tts_cfg, projects):
    _set_status(stage="tts", message="🎙 Starting TTS...")
    gen = batch_tts_all_projects(
        tts_cfg.get("spk_sel") or "",
        tts_cfg.get("sp_rate", 1.0),
        tts_cfg.get("back_sound_sel") or "",
        tts_cfg.get("bitrate", 96),
        tts_cfg.get("noise_lvl", 10),
        tts_cfg.get("use_sound_effect", False),
        tts_cfg.get("use_accents", True),
        tts_cfg.get("repl", True),
        projects,
    )
    for result in gen:
        try:
            _, log_msg, tts_html, _, _ = result
        except Exception:
            continue
        pct = parse_percent_from_html(str(tts_html)) if tts_html else 0
        _set_status(stage="tts", pct=pct, message=str(log_msg)[:300])


# ═══ STEP 3: PACKAGE (3 files next to the source + open Explorer) ═══
def _is_temp_dir(p: Path) -> bool:
    """Heuristic: gradio File uploads / system temp dirs should not be the
    target folder — fall back to Downloads instead."""
    try:
        import tempfile as _tf

        sys_tmp = Path(_tf.gettempdir()).resolve()
        if p.resolve() == sys_tmp or sys_tmp in p.resolve().parents:
            return True
        low = str(p).lower()
        if "gradio" in low or "\\tmp\\" in low or "/tmp/" in low:
            return True
    except Exception:
        pass
    return False


def _extract_book_text(project) -> str:
    """Full readable text of the book: paragraphs from the fb2 source,
    falling back to the parsed XML."""
    base = data_path / project
    paras = []
    fb2 = base / f"{project}.fb2"
    if fb2.exists():
        try:
            from lxml import etree

            root = etree.parse(str(fb2)).getroot()
            for el in root.iter():
                tag = el.tag if isinstance(el.tag, str) else ""
                if tag.lower().endswith("p") and el.text and el.text.strip():
                    paras.append(el.text.strip())
        except Exception:
            paras = []
    if not paras:
        xml_dir = base / "xml"
        if xml_dir.exists():
            from lxml import etree

            for xf in sorted(xml_dir.glob("*.xml")):
                try:
                    root = etree.parse(str(xf)).getroot()
                    for el in root.iter():
                        tag = el.tag if isinstance(el.tag, str) else ""
                        if tag in ("p", "title", "subtitle", "cite") and el.text and el.text.strip():
                            paras.append(el.text.strip())
                except Exception:
                    continue
    return "\n\n".join(paras)


def build_package(project) -> Path | None:
    """Create the 3-file package (mp3 + txt + source) next to the source file,
    or in Downloads if the source is unavailable. Returns the output dir."""
    _set_status(stage="package", project=project, message=f"📦 Building package: {project}...")
    base = data_path / project

    # 1) Determine the source file and target folder
    src = None
    src_info = base / "_source_path.txt"
    if src_info.exists():
        try:
            p = src_info.read_text(encoding="utf-8").strip()
            if p and Path(p).exists() and not _is_temp_dir(Path(p)):
                src = Path(p)
        except Exception:
            pass
    if src is None:
        cand = base / f"{project}.fb2"
        if cand.exists():
            src = cand

    if src is not None:
        out_root = src.parent
    else:
        out_root = Path.home() / "Downloads"
        try:
            out_root.mkdir(parents=True, exist_ok=True)
        except Exception:
            out_root = base

    out_dir = out_root / f"{project} — аудиокнига"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error("Package dir error: %s", e)
        return None

    # 2) File 1: combined mp3 (all chapters, sorted)
    mp3_dir = base / "mp3"
    mp3s = sorted(mp3_dir.glob("*.mp3")) if mp3_dir.exists() else []
    mp3s = [m for m in mp3s if "_PARTIAL" not in m.name]
    if mp3s:
        try:
            segments = [AudioSegment.from_file(str(m)) for m in mp3s]
            combined = _concat_audio_segments(segments)
            combined.export(
                str(out_dir / f"{project}.mp3"), format="mp3", bitrate="192"
            )
        except Exception as e:
            logger.error("MP3 combine error: %s", e)

    # 3) File 2: full text
    try:
        text = _extract_book_text(project)
        if text.strip():
            (out_dir / f"{project}.txt").write_text(text, encoding="utf-8")
    except Exception as e:
        logger.error("Text export error: %s", e)

    # 4) File 3: source book file
    if src is not None and src.exists():
        try:
            suffix = src.suffix.lower() or ".fb2"
            shutil.copy2(src, out_dir / f"{project}{suffix}")
        except Exception as e:
            logger.error("Source copy error: %s", e)

    files_made = sorted(p.name for p in out_dir.iterdir())
    _set_status(
        stage="package",
        project=project,
        pct=100,
        message=f"✅ Package ready: {out_dir}",
        package_dir=str(out_dir),
    )
    logger.info("Package files: %s", files_made)
    return out_dir


def open_in_explorer(path: Path):
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            import subprocess

            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:
        logger.error("Explorer open error: %s", e)


# ═══ MAIN ═══
def main():
    global STATUS_FILE
    if len(sys.argv) < 2:
        print("Usage: python worker.py <job.json>")
        return 1
    job_path = Path(sys.argv[1])
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Cannot read job file: {e}")
        return 1

    STATUS_FILE = Path(job.get("status_file") or "tmp/bg_status.json")
    try:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    projects = job.get("projects") or []
    if not projects:
        projects = sorted(get_data_list())
    parse_cfg = job.get("parse") or {}
    tts_cfg = job.get("tts") or {}

    # Apply cloud flags exactly like the GUI does
    try:
        router.USE_EDGE_FOR_ENGLISH = bool(tts_cfg.get("use_edge_en", router.USE_EDGE_FOR_ENGLISH))
        router.USE_EDGE_FOR_HEBREW = bool(tts_cfg.get("use_edge_he", router.USE_EDGE_FOR_HEBREW))
        router.USE_EDGE_FOR_RUSSIAN = bool(tts_cfg.get("use_edge_ru", router.USE_EDGE_FOR_RUSSIAN))
        router.DICTIONARY_MODE = bool(tts_cfg.get("dict_mode", router.DICTIONARY_MODE))
    except Exception:
        pass

    _set_status(stage="init", message=f"🚀 Worker started: {len(projects)} project(s)")

    try:
        # Load the TTS model (this process is independent — it loads its own)
        ver = tts_cfg.get("model_ver")
        if ver:
            _set_status(stage="init", message=f"⏳ Loading TTS model {ver}...")
            try:
                set_tts_device(str(tts_cfg.get("device") or "auto"))
                res = synth.load(int(ver))
                if isinstance(res, tuple) and res[0] is None:
                    logger.warning("Model load warning: %s", res[1])
            except Exception as e:
                logger.error("Model load error (continuing anyway): %s", e)

        # 1) Parse
        repl = bool(tts_cfg.get("repl", True))
        for project in projects:
            run_parse(project, parse_cfg, replace=repl)

        # 2) TTS
        run_tts(tts_cfg, projects)

        # 3) Package
        packaged = []
        for project in projects:
            try:
                d = build_package(project)
                if d:
                    packaged.append(d)
            except Exception as e:
                logger.error("Package error %s: %s", project, e)
                traceback.print_exc()

        _set_status(
            state="done",
            stage="done",
            pct=100,
            message="🎉 Done! All projects processed and packaged.",
            package_dir=str(packaged[0]) if packaged else None,
        )
        logger.info("Worker finished: %d package(s)", len(packaged))

        # Open Explorer — even if the GUI is already closed
        for d in packaged:
            open_in_explorer(d)
        return 0
    except Exception as e:
        logger.error("FATAL: %s", e, exc_info=True)
        _set_status(state="error", message=f"❌ Error: {e}", error=str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
