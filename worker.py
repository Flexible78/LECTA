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


# Kept as a function (not module-level side effect): importing worker for tests
# must not monkey-patch the real gradio module used by the GUI.
_GRADIO_ORIGINALS: dict[str, object] = {}


def install_headless_gradio_stubs() -> None:
    """Replace the few Gradio UI helpers tts_tab calls with no-ops."""
    for name in ("update", "Info", "Warning", "Error"):
        if name not in _GRADIO_ORIGINALS:
            _GRADIO_ORIGINALS[name] = getattr(gr, name, None)
        setattr(gr, name, _noop)

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
import gr_tabs.tts_tab as tts_module

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
        ok = False
        for pct, msg in proc.process_book(
            ab_path=project,
            replace=replace,
            sound_effect=parse_cfg.get("sound_effect", False),
            punctuation=parse_cfg.get("punctuation", False),
            translit=parse_cfg.get("translit", True),
            ch_size=parse_cfg.get("ch_size", 200),
        ):
            text = str(msg or "")
            if text.lstrip().startswith("❌"):
                _set_status(stage="parse", project=project, message=text)
                raise RuntimeError(text.lstrip("❌ ").strip())
            ok = True
            _set_status(stage="parse", project=project, pct=pct, message=msg)
            if pct >= 100:
                break
        return ok
    except Exception as e:
        logger.error("Parse error %s: %s", project, e, exc_info=True)
        _set_status(stage="parse", project=project, message=f"⚠️ Parse error: {e}")
        return False


# ═══ STEP 2: TTS (XML → MP3) — reuses the exact same batch pipeline as the GUI ═══
def run_tts(tts_cfg, projects):
    _set_status(stage="tts", message="🎙 Starting TTS...")
    tts_module.LAST_BATCH_FAILURES = []
    tts_module.LAST_BATCH_ERRORS = []
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
    return list(tts_module.LAST_BATCH_FAILURES)


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


def _natural_key(name: str):
    """Sort '2.mp3' before '10.mp3' instead of lexicographic order."""
    import re as _re

    return [int(t) if t.isdigit() else t.lower()
            for t in _re.split(r"(\d+)", name)]


def _project_mp3s(project) -> list[Path]:
    mp3_dir = data_path / project / "mp3"
    if not mp3_dir.exists():
        return []
    files = [p for p in mp3_dir.glob("*.mp3")
             if p.is_file() and "_PARTIAL" not in p.name
             and not p.name.startswith(".")]
    return sorted(files, key=lambda p: _natural_key(p.name))


def _combine_mp3s(mp3s: list[Path], target: Path) -> tuple[bool, str]:
    """Concatenate chapters with ffmpeg (stream copy, falls back to re-encode).

    Returns (ok, message).  The output is verified before the caller publishes it.
    """
    from libs.audio_tools import FFMPEG, CREATE_NO_WINDOW, verify_audio_file

    if not mp3s:
        return False, "No chapter MP3 files were produced"
    list_file = target.with_name(f".{target.stem}.concat.txt")
    lines = []
    for p in mp3s:
        # ffmpeg concat demuxer escaping
        safe = str(p.resolve()).replace("'", "'\\''")
        lines.append(f"file '{safe}'")
    try:
        list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as e:
        return False, f"Cannot write concat list: {e}"
    expected = 0.0
    try:
        from libs.audio_tools import probe_audio
        invalid = []
        for p in mp3s:
            info = probe_audio(str(p))
            if not info.ok:
                invalid.append(f"{p.name}: {info.error or 'unreadable'}")
            expected += info.duration
        if invalid:
            return False, "Unreadable chapter(s): " + "; ".join(invalid[:3])
        cmd = [FFMPEG, "-hide_banner", "-nostats", "-y", "-f", "concat",
               "-safe", "0", "-i", str(list_file), "-vn",
               "-c:a", "libmp3lame", "-b:a", "192k", "-ar", "24000", "-ac", "1",
               "-id3v2_version", "3", "-f", "mp3", str(target)]
        proc = __import__("subprocess").run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", creationflags=CREATE_NO_WINDOW, timeout=7200)
        if proc.returncode != 0:
            return False, f"ffmpeg concat failed: {(proc.stderr or '')[-300:]}"
        check = verify_audio_file(str(target), expected_duration=expected, min_duration=1.0)
        if not check.get("ok"):
            return False, "Combined MP3 failed verification: " + "; ".join(check.get("errors") or [])
        return True, f"{len(mp3s)} chapter(s), {expected:.1f}s"
    except Exception as e:
        return False, f"Concat error: {e}"
    finally:
        try:
            list_file.unlink(missing_ok=True)
        except OSError:
            pass


def build_package(project, failures: list[str] | None = None) -> Path | None:
    """Build a verified 3-file package atomically.

    Refuses to package a project whose TTS reported failures (otherwise the
    package would silently contain a truncated or missing audio track).
    Returns the published directory, or raises RuntimeError.
    """
    import uuid

    _set_status(stage="package", project=project, message=f"📦 Building package: {project}...")
    base = data_path / project

    project_failures = [f for f in (failures or []) if f.startswith(f"{project}::")]
    if project_failures:
        details = "; ".join(f.split("::", 1)[-1] for f in project_failures[:3])
        raise RuntimeError(f"TTS failed for {project} — package not created ({details})")

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

    final_dir = out_root / f"{project} — аудиокнига"
    staging = out_root / f".lecta_pkg_{uuid.uuid4().hex[:8]}"
    try:
        staging.mkdir(parents=True, exist_ok=False)
    except Exception as e:
        raise RuntimeError(f"Cannot create staging folder: {e}")

    problems: list[str] = []
    try:
        # 1) MP3
        mp3s = _project_mp3s(project)
        mp3_target = staging / f"{project}.mp3"
        ok, msg = _combine_mp3s(mp3s, mp3_target)
        if not ok:
            problems.append(msg)
        else:
            logger.info("Package MP3: %s", msg)

        # 2) Full text
        text = _extract_book_text(project)
        if not text.strip():
            problems.append("Book text could not be extracted")
        else:
            (staging / f"{project}.txt").write_text(text, encoding="utf-8")

        # 3) Source file
        if src is None or not src.exists():
            problems.append("Source book file is missing")
        else:
            suffix = src.suffix.lower() or ".fb2"
            shutil.copy2(src, staging / f"{project}{suffix}")

        if problems:
            raise RuntimeError("; ".join(problems))

        # Publish atomically: keep the old package until the new one is complete.
        backup = out_root / f".lecta_old_{uuid.uuid4().hex[:8]}"
        moved_old = False
        if final_dir.exists():
            try:
                final_dir.rename(backup)
                moved_old = True
            except Exception as e:
                raise RuntimeError(f"Cannot replace existing package: {e}")
        try:
            staging.rename(final_dir)
        except Exception as e:
            if moved_old:
                try:
                    backup.rename(final_dir)
                except Exception:
                    pass
            raise RuntimeError(f"Cannot publish package: {e}")
        if moved_old:
            shutil.rmtree(backup, ignore_errors=True)

        files_made = sorted(p.name for p in final_dir.iterdir())
        logger.info("Package files: %s", files_made)
        _set_status(
            stage="package", project=project, pct=100,
            message=f"✅ Package ready: {final_dir}",
            package_dir=str(final_dir),
        )
        return final_dir
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def open_in_explorer(path: Path):
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            import subprocess

            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:
        logger.error("Explorer open error: %s", e)


def play_done_sound():
    """🔊 Короткий системный сигнал завершения (работает и в detached-режиме)."""
    try:
        if sys.platform == "win32":
            import winsound

            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        else:
            print("\a", end="", flush=True)
    except Exception as e:
        logger.error("Done sound error: %s", e)


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
        router.apply_generation_settings(tts_cfg.get("audio") or {})
    except Exception:
        pass

    _set_status(stage="init", message=f"🚀 Worker started: {len(projects)} project(s)")

    # Install only in the standalone process, after tests/importers have access
    # to the unmodified Gradio module.
    install_headless_gradio_stubs()

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
        parse_failures = []
        for project in projects:
            if not run_parse(project, parse_cfg, replace=repl):
                parse_failures.append(project)

        # 2) TTS
        tts_failures = run_tts(tts_cfg, projects)

        # 3) Package — only verified projects are published.
        packaged = []
        package_errors = []
        for project in projects:
            if project in parse_failures:
                package_errors.append(f"{project}: parsing failed")
                continue
            try:
                d = build_package(project, failures=tts_failures)
                if d:
                    packaged.append(d)
            except Exception as e:
                logger.error("Package error %s: %s", project, e)
                package_errors.append(f"{project}: {e}")

        if package_errors:
            _set_status(
                state="error",
                stage="done",
                pct=100,
                message="⚠️ Finished with errors: " + "; ".join(package_errors[:3]),
                error="; ".join(package_errors),
                package_dir=str(packaged[0]) if packaged else None,
            )
            auto_open = bool(tts_cfg.get("auto_open", True))
            if auto_open:
                for d in packaged:
                    open_in_explorer(d)
            if bool(job.get("sound_done", True)):
                play_done_sound()
            return 1

        _set_status(
            state="done",
            stage="done",
            pct=100,
            message="🎉 Done! All projects processed and packaged.",
            package_dir=str(packaged[0]) if packaged else None,
        )
        logger.info("Worker finished: %d package(s)", len(packaged))

        # Open Explorer (if auto_open checkbox is enabled) — even if the GUI is already closed
        auto_open = bool(tts_cfg.get("auto_open", True))
        if auto_open:
            for d in packaged:
                open_in_explorer(d)
        else:
            logger.info("Auto-open disabled by checkbox — skipping Explorer")

        # 🔊 Звуковой сигнал завершения (отключаемый чекбоксом)
        if bool(job.get("sound_done", True)):
            play_done_sound()
        else:
            logger.info("Sound signal disabled by checkbox")
        return 0
    except Exception as e:
        logger.error("FATAL: %s", e, exc_info=True)
        _set_status(state="error", message=f"❌ Error: {e}", error=str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
