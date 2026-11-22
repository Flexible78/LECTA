"""Standalone silence cleaner UI for LECTA projects and MP3 files."""
from __future__ import annotations

import os
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gradio as gr

from config import AppConfig
from libs.audio_tools import (
    PRESETS,
    analyze_audio,
    clean_audio_file,
    make_preview,
    outputs_dir,
    project_mp3_files,
    results_report,
    results_table,
)
from libs.utils import data_path, get_data_list

_CANCEL = threading.Event()
_CANCEL_LOCK = threading.Lock()
_LAST_OUTPUT_DIR = ""


def _path_of(item) -> str:
    if isinstance(item, str):
        return item
    if hasattr(item, "name"):
        return str(item.name)
    return str(item or "")


def resolve_sources(uploads, projects, folder_text):
    paths = []
    for item in (uploads or []):
        p = Path(_path_of(item))
        if p.is_file() and p.suffix.lower() == ".mp3":
            paths.append(p)
    for name in projects or []:
        if name:
            paths.extend(Path(p) for p in project_mp3_files(data_path / name))
    for raw in (folder_text or "").splitlines():
        raw = raw.strip().strip('"').strip("'")
        if not raw:
            continue
        p = Path(raw)
        if p.is_dir():
            paths.extend(p.rglob("*.mp3"))
        elif p.is_file() and p.suffix.lower() == ".mp3":
            paths.append(p)
    unique, seen = [], set()
    for p in paths:
        try:
            key = str(p.resolve()).casefold()
        except OSError:
            key = str(p).casefold()
        if key not in seen and p.is_file():
            seen.add(key)
            unique.append(p)
    return unique


def load_sources(uploads, projects, folder_text):
    sources = resolve_sources(uploads, projects, folder_text)
    choices = [(f"{p.name} — {p.parent}", str(p)) for p in sources]
    values = [str(p) for p in sources]
    return (gr.update(choices=choices, value=values),
            gr.update(choices=choices, value=values[0] if values else None),
            f"Loaded {len(sources)} MP3 file(s).")


def _settings(preset, threshold, min_silence, keep_ms, edge_ms, trim_edges):
    return {
        "preset": preset or "compact",
        "threshold_db": float(threshold),
        "min_silence_sec": float(min_silence),
        "keep_ms": int(keep_ms),
        "edge_keep_ms": int(edge_ms),
        "trim_edges": bool(trim_edges),
    }


def _persist_cleaner_defaults(preset, trim_edges, quality_kbps=None):
    """Remember the cleaner choices so the next launch starts the same way."""
    payload = {
        "cleaner_preset": preset or "compact",
        "cleaner_trim_edges": bool(trim_edges),
    }
    if quality_kbps is not None:
        try:
            payload["cleaner_quality_kbps"] = int(quality_kbps)
        except (TypeError, ValueError):
            pass
    try:
        AppConfig.save_user_settings(payload)
    except Exception:
        pass


def _analyze_row(path, threshold, min_silence):
    try:
        a = analyze_audio(path, threshold_db=float(threshold), min_silence_sec=float(min_silence))
    except Exception as exc:
        return [Path(path).name, f"❌ {exc}", "—", "—", "—", "—", "—"]
    return [
        Path(path).name,
        "✅ Ready" if a.info.ok else f"❌ {a.info.error}",
        f"{a.info.duration:.1f}s",
        f"{a.total_silence:.1f}s",
        f"{a.longest_silence:.1f}s",
        f"{a.leading_silence:.1f}s / {a.trailing_silence:.1f}s",
        f"{a.peak_dbfs:.1f} dBFS",
    ]


def analyze_selected(files, preset, threshold, min_silence):
    if not files:
        return [], "Choose or load one or more MP3 files first."
    pr = PRESETS.get(preset, PRESETS["compact"])
    # One ffmpeg decode per file; probing several files in parallel keeps big
    # audiobook batches from taking minutes.
    workers = min(4, max(1, os.cpu_count() or 2), len(files))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda p: _analyze_row(p, threshold, min_silence), files))
    return rows, f"Analyzed {len(rows)} file(s). {pr.description}"


def preview_selected(path, preset, threshold, min_silence, keep_ms, edge_ms, trim_edges):
    if not path:
        return None, None, "Choose an MP3 file first."
    payload = make_preview(
        path, **_settings(preset, threshold, min_silence, keep_ms, edge_ms, trim_edges)
    )
    if not payload.get("ok"):
        return None, None, payload.get("message", "Preview could not be created.")
    return payload["before"], payload["after"], payload["message"]


def cancel_cleaning():
    _CANCEL.set()
    return "Cancellation requested — the current FFmpeg operation will stop safely."


def clean_selected(files, preset, threshold, min_silence, keep_ms, edge_ms,
                   trim_edges, quality_kbps):
    global _LAST_OUTPUT_DIR
    if not files:
        yield "", "Choose or load one or more MP3 files first.", [], None, "", ""
        return
    # A new run explicitly clears a stale cancel flag; cancellation after that
    # is observed by clean_audio_file and the loop below.
    with _CANCEL_LOCK:
        _CANCEL.clear()
    _persist_cleaner_defaults(preset, trim_edges, quality_kbps)
    opts = _settings(preset, threshold, min_silence, keep_ms, edge_ms, trim_edges)
    results = []
    total = len(files)
    for i, path in enumerate(files, 1):
        if _CANCEL.is_set():
            break
        yield f"Processing {i}/{total}: {Path(path).name}", "Cleaning…", results_table(results), None, "", ""
        result = clean_audio_file(
            path, **opts, quality_kbps=int(quality_kbps), cancel_event=_CANCEL
        )
        results.append(result)
        yield f"Processed {i}/{total}: {Path(path).name}", results_report(results), results_table(results), None, "", ""
    if _CANCEL.is_set():
        from libs.audio_tools import CleanResult
        results.append(CleanResult(source="", status="cancelled", message="Batch cancelled"))
    created = [Path(r.output) for r in results if r.status == "done" and r.output]
    archive = None
    folder = str(created[0].parent) if created else ""
    if created:
        archive_path = outputs_dir() / f"cleaned_batch_{uuid.uuid4().hex[:8]}.zip"
        try:
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                used = set()
                for p in created:
                    arcname = p.name
                    if arcname.casefold() in used:
                        arcname = f"{p.parent.name}/{p.name}"
                    used.add(arcname.casefold())
                    zf.write(p, arcname=arcname)
            archive = str(archive_path)
            _LAST_OUTPUT_DIR = folder
        except Exception as exc:
            archive = None
            folder = ""
            yield "Finished with archive error", f"{results_report(results)}\n\nCould not create ZIP: {exc}", results_table(results), None, "", ""
            return
    else:
        _LAST_OUTPUT_DIR = ""
    yield "Batch complete" if not _CANCEL.is_set() else "Batch cancelled", results_report(results), results_table(results), archive, folder, ""


def open_output_folder(folder):
    target = folder or _LAST_OUTPUT_DIR
    if not target or not Path(target).is_dir():
        return "No output folder is available yet."
    try:
        if os.name == "nt":
            os.startfile(str(Path(target)))
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(Path(target))])
        return f"Opened: {target}"
    except Exception as exc:
        return f"Could not open folder: {exc}"


def preset_values(preset_key):
    """Slider defaults for the selected cleaner profile."""
    p = PRESETS.get(preset_key, PRESETS["compact"])
    return (
        gr.update(value=p.threshold_db),
        gr.update(value=p.min_silence_sec),
        gr.update(value=p.keep_ms),
        gr.update(value=p.edge_keep_ms),
    )


def audio_tools_tab():
    try:
        _cfg = AppConfig.load_user_settings()
        _preset_default = getattr(_cfg, "cleaner_preset", "compact") or "compact"
        _quality_default = int(getattr(_cfg, "cleaner_quality_kbps", 192) or 192)
        _trim_default = bool(getattr(_cfg, "cleaner_trim_edges", True))
    except Exception:
        _preset_default, _quality_default, _trim_default = "compact", 192, True
    _p = PRESETS.get(_preset_default, PRESETS["compact"])
    with gr.Tab("✂️ Remove pauses", id="audio-tools"):
        gr.Markdown(
            "## MP3 pause cleaner\n"
            "Analyze and preview before saving. Original audio is never overwritten; cleaned copies are saved beside the source. "
            "Compact is the safe default. Aggressive can cut soft consonants—preview it first."
        )
        with gr.Row():
            with gr.Column(scale=1):
                uploads = gr.File(
                    label="MP3 files (one or many)", file_count="multiple", type="filepath",
                    file_types=[".mp3"],
                )
                projects = gr.CheckboxGroup(
                    label="Or add MP3s from LECTA projects", choices=sorted(get_data_list()),
                    interactive=True,
                )
                folder_text = gr.Textbox(
                    label="Optional folder or file paths (one per line)", lines=2,
                    placeholder="Paste a folder containing MP3s, or an MP3 path",
                )
                load_btn = gr.Button("Load selected audio", variant="secondary")
                file_picker = gr.Dropdown(label="Audio to preview", choices=[], interactive=True)
                batch_files = gr.CheckboxGroup(label="Files to clean", choices=[], interactive=True,
                                               value=[])
                source_status = gr.Markdown("Select files, projects, or a folder, then load audio.")
                with gr.Accordion("Cleaning and quality settings", open=True):
                    preset = gr.Radio(
                        choices=[("Compact (default)", "compact"), ("Natural", "natural"),
                                 ("Aggressive — preview first", "aggressive")],
                        value=_preset_default, label="Pause profile",
                    )
                    threshold = gr.Slider(-55, -25, value=_p.threshold_db, step=1, label="Silence threshold (dBFS)",
                                          info="Lower is more conservative; protects quiet speech.")
                    min_silence = gr.Slider(0.2, 1.5, value=_p.min_silence_sec, step=0.05, label="Minimum gap to shorten (s)")
                    keep_ms = gr.Slider(40, 700, value=_p.keep_ms, step=10, label="Pause to keep (ms)")
                    edge_ms = gr.Slider(20, 300, value=_p.edge_keep_ms, step=10, label="Leading/trailing audio to keep (ms)")
                    trim_edges = gr.Checkbox(value=_trim_default, label="Trim excessive silence at file edges")
                    quality = gr.Slider(64, 320, value=_quality_default, step=16, label="Output MP3 bitrate (kbps)")
                with gr.Row():
                    analyze_btn = gr.Button("Analyze", variant="secondary")
                    preview_btn = gr.Button("Preview before / after", variant="secondary")
                with gr.Row():
                    clean_btn = gr.Button("Clean selected batch", variant="primary")
                    cancel_btn = gr.Button("Cancel", variant="stop")
            with gr.Column(scale=1):
                analysis_table = gr.Dataframe(
                    headers=["File", "Status", "Duration", "Silence", "Longest gap", "Edge silence", "Peak"],
                    datatype=["str"] * 7, value=[], interactive=False, wrap=True,
                    label="Silence analysis",
                )
                preview_status = gr.Markdown("Before/after previews appear here.")
                with gr.Row():
                    before_audio = gr.Audio(label="Before", type="filepath", interactive=False)
                    after_audio = gr.Audio(label="After", type="filepath", interactive=False)
                progress = gr.Textbox(label="Batch progress", lines=1, interactive=False)
                report = gr.Textbox(label="Results", lines=5, interactive=False)
                results = gr.Dataframe(
                    headers=["File", "Result", "Before", "After", "Removed", "Pauses", "Output / message"],
                    datatype=["str"] * 7, value=[], interactive=False, wrap=True,
                    label="Cleaned files",
                )
                with gr.Row():
                    download_zip = gr.DownloadButton(label="Download cleaned ZIP", visible=True)
                    open_folder_btn = gr.Button("Open output folder", variant="secondary")
                output_folder = gr.Textbox(label="Output folder", interactive=False)
                folder_status = gr.Markdown("")

        preset.change(preset_values, inputs=preset,
                     outputs=[threshold, min_silence, keep_ms, edge_ms])
        load_btn.click(load_sources, inputs=[uploads, projects, folder_text], outputs=[batch_files, file_picker, source_status])
        analyze_btn.click(
            analyze_selected,
            inputs=[batch_files, preset, threshold, min_silence],
            outputs=[analysis_table, preview_status],
        )
        preview_btn.click(
            preview_selected,
            inputs=[file_picker, preset, threshold, min_silence, keep_ms, edge_ms, trim_edges],
            outputs=[before_audio, after_audio, preview_status],
        )
        clean_btn.click(
            clean_selected,
            inputs=[batch_files, preset, threshold, min_silence, keep_ms, edge_ms, trim_edges, quality],
            outputs=[progress, report, results, download_zip, output_folder, folder_status],
        )
        cancel_btn.click(cancel_cleaning, outputs=folder_status, queue=False)
        open_folder_btn.click(open_output_folder, inputs=output_folder, outputs=folder_status, queue=False)
        projects.change(lambda _v: gr.update(choices=sorted(get_data_list())),
                        inputs=projects, outputs=projects)
