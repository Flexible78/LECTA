"""LECTA audio tools: silence analysis, safe pause cleanup and MP3 validation.

The module is intentionally UI-free.  Both generated audio (TTS assembly) and
the standalone "Remove Pauses" tool use the same primitives so there is exactly
one implementation of silence detection / trimming.

Design rules
------------
* Never touch the source file.  Cleaned audio is written to a new file using a
  temporary name and an atomic rename.
* Only cut inside detected silence.  Speech is never re-ordered or removed.
* Use FFmpeg for decoding/encoding (already shipped with LECTA on Windows) and
  always pass arguments as a list, never through a shell string.
* A cleaned file is rejected if it is shorter than planned, unreadable, fully
  silent, or lacks an audio stream.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

# ════════════════════════════════════════════════════════════════════
#  FFmpeg discovery
# ════════════════════════════════════════════════════════════════════

_APP_DIR = Path(__file__).resolve().parent.parent          # fb2tts/
_PROJECT_DIR = _APP_DIR.parent                              # LECTA/


def _find_binary(name: str) -> str:
    exe = f"{name}.exe" if os.name == "nt" else name
    candidates: list[Path] = []
    env_dir = os.environ.get("LECTA_FFMPEG_DIR")
    if env_dir:
        candidates.append(Path(env_dir) / exe)
    candidates += [
        _PROJECT_DIR / "ffmpeg" / exe,
        _APP_DIR / "ffmpeg" / exe,
        Path.cwd() / "ffmpeg" / exe,
    ]
    which = shutil.which(name)
    if which:
        candidates.append(Path(which))
    for c in candidates:
        if c.is_file():
            return str(c)
    # Last resort: ffmpeg on PATH (subprocess will raise a clear error)
    return exe


FFMPEG = _find_binary("ffmpeg")
FFPROBE = _find_binary("ffprobe")

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# ════════════════════════════════════════════════════════════════════
#  Data models
# ════════════════════════════════════════════════════════════════════


@dataclass
class AudioInfo:
    path: str
    ok: bool = False
    error: str = ""
    duration: float = 0.0
    codec: str = ""
    sample_rate: int = 0
    channels: int = 0
    bit_rate: int = 0
    size_bytes: int = 0
    has_cover: bool = False
    container: str = ""

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "ok": self.ok,
            "error": self.error,
            "duration": round(self.duration, 3),
            "codec": self.codec,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_rate": self.bit_rate,
            "size_bytes": self.size_bytes,
            "has_cover": self.has_cover,
            "container": self.container,
        }


@dataclass
class SilenceInterval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.end, 3),
                "duration": round(self.duration, 3)}


@dataclass
class AudioAnalysis:
    info: AudioInfo
    threshold_db: float
    min_silence_sec: float
    silences: list[SilenceInterval] = field(default_factory=list)
    total_silence: float = 0.0
    leading_silence: float = 0.0
    trailing_silence: float = 0.0
    longest_silence: float = 0.0
    peak_dbfs: float = -99.0
    mean_dbfs: float = -99.0
    warnings: list[str] = field(default_factory=list)

    @property
    def audible_duration(self) -> float:
        return max(0.0, self.info.duration - self.total_silence)

    @property
    def internal_silences(self) -> list[SilenceInterval]:
        eps = 0.06
        return [s for s in self.silences
                if s.start > eps and s.end < self.info.duration - eps]

    def excessive(self, max_keep_sec: float) -> list[SilenceInterval]:
        return [s for s in self.silences if s.duration > max_keep_sec + 0.05]

    def to_dict(self) -> dict:
        return {
            "info": self.info.to_dict(),
            "threshold_db": self.threshold_db,
            "min_silence_sec": self.min_silence_sec,
            "silence_count": len(self.silences),
            "total_silence": round(self.total_silence, 3),
            "leading_silence": round(self.leading_silence, 3),
            "trailing_silence": round(self.trailing_silence, 3),
            "longest_silence": round(self.longest_silence, 3),
            "audible_duration": round(self.audible_duration, 3),
            "peak_dbfs": round(self.peak_dbfs, 2),
            "mean_dbfs": round(self.mean_dbfs, 2),
            "warnings": list(self.warnings),
            "silences": [s.to_dict() for s in self.silences],
        }


@dataclass(frozen=True)
class CleanPreset:
    key: str
    label: str
    threshold_db: float
    min_silence_sec: float
    keep_ms: int
    edge_keep_ms: int
    description: str
    aggressive: bool = False


PRESETS: dict[str, CleanPreset] = {
    "compact": CleanPreset(
        key="compact",
        label="Compact (default)",
        threshold_db=-38.0,
        min_silence_sec=0.35,
        keep_ms=110,
        edge_keep_ms=50,
        description="Shortens clearly excessive gaps; keeps a short natural breath.",
    ),
    "natural": CleanPreset(
        key="natural",
        label="Natural",
        threshold_db=-40.0,
        min_silence_sec=0.45,
        keep_ms=220,
        edge_keep_ms=90,
        description="Keeps conversational rhythm and comfortable sentence gaps.",
    ),
    "aggressive": CleanPreset(
        key="aggressive",
        label="Aggressive (preview first!)",
        threshold_db=-32.0,
        min_silence_sec=0.30,
        keep_ms=45,
        edge_keep_ms=20,
        description="Maximum removal. May clip soft consonants — preview before saving.",
        aggressive=True,
    ),
}

DEFAULT_THRESHOLD_DB = PRESETS["compact"].threshold_db
DEFAULT_MIN_SILENCE_SEC = PRESETS["compact"].min_silence_sec
DEFAULT_KEEP_MS = PRESETS["compact"].keep_ms
DEFAULT_QUALITY_KBPS = 192


@dataclass
class CleanPlan:
    analysis: AudioAnalysis
    keep_ms: int
    edge_keep_ms: int
    trim_edges: bool
    removals: list[tuple[float, float]] = field(default_factory=list)
    kept: list[tuple[float, float]] = field(default_factory=list)
    removed_total: float = 0.0
    expected_duration: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "removals": [[round(a, 3), round(b, 3)] for a, b in self.removals],
            "kept": [[round(a, 3), round(b, 3)] for a, b in self.kept],
            "removed_total": round(self.removed_total, 3),
            "expected_duration": round(self.expected_duration, 3),
            "keep_ms": self.keep_ms,
            "edge_keep_ms": self.edge_keep_ms,
            "trim_edges": self.trim_edges,
            "warnings": list(self.warnings),
        }


@dataclass
class CleanResult:
    source: str
    output: str = ""
    status: str = "pending"     # done | skipped | failed | cancelled
    message: str = ""
    before_duration: float = 0.0
    after_duration: float = 0.0
    removed: float = 0.0
    pauses_shortened: int = 0
    warnings: list[str] = field(default_factory=list)
    plan: CleanPlan | None = None
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "output": self.output,
            "status": self.status,
            "message": self.message,
            "before_duration": round(self.before_duration, 3),
            "after_duration": round(self.after_duration, 3),
            "removed": round(self.removed, 3),
            "pauses_shortened": self.pauses_shortened,
            "warnings": list(self.warnings),
            "error": self.error,
        }


# ════════════════════════════════════════════════════════════════════
#  Low-level helpers
# ════════════════════════════════════════════════════════════════════


def _run(cmd: Sequence[str], *, timeout: float | None = 900,
         cancel_event: threading.Event | None = None) -> tuple[int, str, str]:
    """Run a subprocess safely; no shell, Unicode-safe argument list."""
    try:
        proc = subprocess.Popen(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATE_NO_WINDOW,
        )
    except FileNotFoundError:
        return 127, "", f"Executable not found: {cmd[0]}"
    except Exception as e:  # pragma: no cover - OS specific
        return 1, "", str(e)

    deadline = None if not timeout else (__import__("time").monotonic() + timeout)
    while True:
        if cancel_event is not None and cancel_event.is_set():
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                proc.communicate(timeout=2)
            except Exception:
                pass
            return -2, "", "cancelled"
        try:
            out, err = proc.communicate(timeout=0.2)
            return proc.returncode, out or "", err or ""
        except subprocess.TimeoutExpired:
            if deadline is not None and __import__("time").monotonic() > deadline:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.communicate(timeout=2)
                except Exception:
                    pass
                return -1, "", "timeout"
            continue


def probe_audio(path: str | os.PathLike) -> AudioInfo:
    p = str(Path(path))
    info = AudioInfo(path=p)
    if not Path(p).is_file():
        info.error = "File not found"
        return info
    try:
        info.size_bytes = Path(p).stat().st_size
    except OSError as e:
        info.error = f"Cannot read file: {e}"
        return info
    cmd = [FFPROBE, "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", p]
    rc, out, err = _run(cmd, timeout=60)
    if rc != 0:
        info.error = (err or "ffprobe failed").strip().splitlines()[-1][:300]
        return info
    try:
        data = json.loads(out or "{}")
    except Exception as e:
        info.error = f"ffprobe returned invalid JSON: {e}"
        return info

    audio = next((s for s in data.get("streams", [])
                  if s.get("codec_type") == "audio"), None)
    if audio is None:
        info.error = "No audio stream (this is not a playable audio file)"
        return info
    fmt = data.get("format", {})
    info.codec = audio.get("codec_name", "")
    info.sample_rate = int(audio.get("sample_rate") or 0)
    info.channels = int(audio.get("channels") or 0)
    try:
        info.bit_rate = int(audio.get("bit_rate") or fmt.get("bit_rate") or 0)
    except (TypeError, ValueError):
        info.bit_rate = 0
    try:
        info.duration = float(fmt.get("duration") or audio.get("duration") or 0.0)
    except (TypeError, ValueError):
        info.duration = 0.0
    info.container = fmt.get("format_name", "")
    info.has_cover = any(s.get("codec_type") == "video" for s in data.get("streams", []))
    if info.duration <= 0:
        info.error = "Audio stream has zero duration"
        return info
    info.ok = True
    return info


_SIL_START = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SIL_END = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")
_MEAN_VOL = re.compile(r"mean_volume:\s*(-?\d+(?:\.\d+)?)")
_MAX_VOL = re.compile(r"max_volume:\s*(-?\d+(?:\.\d+)?)")


def _parse_float(pattern: re.Pattern, text: str, default: float) -> float:
    m = pattern.search(text)
    if not m:
        return default
    try:
        return float(m.group(1))
    except ValueError:
        return default


def _silences_and_levels(path: str, threshold_db: float, min_silence_sec: float,
                         duration: float | None,
                         cancel_event: threading.Event | None = None
                         ) -> tuple[list[SilenceInterval], float, float]:
    """One decode pass: silencedetect and volumedetect share the same stream."""
    if duration is None:
        info = probe_audio(path)
        duration = info.duration if info.ok else 0.0
    filt = (f"silencedetect=noise={threshold_db}dB:d={max(0.05, float(min_silence_sec))},"
            "volumedetect")
    cmd = [FFMPEG, "-hide_banner", "-nostats", "-i", path,
           "-vn", "-af", filt, "-f", "null", "-"]
    rc, _out, err = _run(cmd, timeout=3600, cancel_event=cancel_event)
    if rc not in (0, -2) and not err.strip():
        raise RuntimeError(f"ffmpeg silence detection failed (code {rc})")
    intervals: list[SilenceInterval] = []
    current_start: float | None = None
    for line in err.splitlines():
        if "silence_start:" in line:
            current_start = _parse_float(_SIL_START, line, 0.0)
        elif "silence_end:" in line:
            end = _parse_float(_SIL_END, line, 0.0)
            start = 0.0 if current_start is None else current_start
            intervals.append(SilenceInterval(max(0.0, start), max(start, end)))
            current_start = None
    if current_start is not None and duration and current_start < duration:
        intervals.append(SilenceInterval(max(0.0, current_start), duration))
    # Clip and merge touching ranges
    clipped: list[SilenceInterval] = []
    for s in intervals:
        s2 = SilenceInterval(max(0.0, min(s.start, duration or s.start)),
                             max(0.0, min(s.end, duration or s.end)))
        if s2.duration <= 0.01:
            continue
        if clipped and abs(s2.start - clipped[-1].end) < 0.02:
            clipped[-1] = SilenceInterval(clipped[-1].start, max(clipped[-1].end, s2.end))
        else:
            clipped.append(s2)
    mean = _parse_float(_MEAN_VOL, err, -99.0)
    peak = _parse_float(_MAX_VOL, err, -99.0)
    return clipped, mean, peak


def detect_silences(path: str, threshold_db: float = DEFAULT_THRESHOLD_DB,
                    min_silence_sec: float = DEFAULT_MIN_SILENCE_SEC,
                    duration: float | None = None,
                    cancel_event: threading.Event | None = None) -> list[SilenceInterval]:
    """Detect silent ranges using ffmpeg's silencedetect (streaming, no RAM use)."""
    intervals, _mean, _peak = _silences_and_levels(
        path, threshold_db, min_silence_sec, duration, cancel_event)
    return intervals


def _volume_stats(path: str, cancel_event: threading.Event | None = None) -> tuple[float, float]:
    """Return (mean_dbfs, peak_dbfs) or (-99, -99) when unavailable."""
    cmd = [FFMPEG, "-hide_banner", "-nostats", "-i", path,
           "-vn", "-af", "volumedetect", "-f", "null", "-"]
    rc, _out, err = _run(cmd, timeout=3600, cancel_event=cancel_event)
    if not err:
        return -99.0, -99.0
    mean = _parse_float(re.compile(r"mean_volume:\s*(-?\d+(?:\.\d+)?)"), err, -99.0)
    peak = _parse_float(re.compile(r"max_volume:\s*(-?\d+(?:\.\d+)?)"), err, -99.0)
    return mean, peak


def analyze_audio(path: str, threshold_db: float = DEFAULT_THRESHOLD_DB,
                  min_silence_sec: float = DEFAULT_MIN_SILENCE_SEC,
                  cancel_event: threading.Event | None = None) -> AudioAnalysis:
    info = probe_audio(path)
    warnings: list[str] = []
    if not info.ok:
        return AudioAnalysis(info=info, threshold_db=threshold_db,
                             min_silence_sec=min_silence_sec,
                             warnings=[info.error or "Invalid audio file"])
    silences, mean, peak = _silences_and_levels(
        path, threshold_db, min_silence_sec, info.duration, cancel_event)
    total = sum(s.duration for s in silences)
    leading = next((s.duration for s in silences if s.start <= 0.05), 0.0)
    trailing = next((s.duration for s in silences
                     if s.end >= info.duration - 0.05), 0.0)
    longest = max((s.duration for s in silences), default=0.0)
    analysis = AudioAnalysis(
        info=info, threshold_db=threshold_db, min_silence_sec=min_silence_sec,
        silences=silences, total_silence=total, leading_silence=leading,
        trailing_silence=trailing, longest_silence=longest,
        peak_dbfs=peak, mean_dbfs=mean, warnings=warnings,
    )
    if peak < -50:
        warnings.append("Recording is almost fully silent (peak below -50 dBFS).")
    elif peak < -25:
        warnings.append("Recording is very quiet (peak below -25 dBFS).")
    if info.duration < 0.6:
        warnings.append("Audio is shorter than 0.6 s.")
    if not silences and info.duration > 2:
        warnings.append("No silence above the threshold was detected.")
    return analysis


# ════════════════════════════════════════════════════════════════════
#  Cleaning plan
# ════════════════════════════════════════════════════════════════════


def build_clean_plan(analysis: AudioAnalysis, *, keep_ms: int,
                     edge_keep_ms: int, trim_edges: bool = True,
                     min_silence_sec: float | None = None,
                     max_keep_ms: int | None = None) -> CleanPlan:
    """Turn detected silence into removal ranges.

    * Leading silence keeps its *tail* (the moment before speech starts).
    * Internal silence keeps its *tail* (so the pause feels like a breath
      before the next phrase, not a cut in the middle of a pause).
    * Trailing silence keeps its *head*.
    * Silences shorter than ``min_silence_sec`` are left completely alone.
    """
    duration = analysis.info.duration
    min_sil = analysis.min_silence_sec if min_silence_sec is None else min_silence_sec
    keep = max(0, int(keep_ms))
    edge_keep = max(0, int(edge_keep_ms))
    if max_keep_ms is not None:
        keep = min(keep, max(0, int(max_keep_ms)))
        edge_keep = min(edge_keep, max(0, int(max_keep_ms)))
    removals: list[tuple[float, float]] = []
    warnings: list[str] = []

    for sil in analysis.silences:
        if sil.duration < min_sil - 0.01:
            continue
        is_leading = sil.start <= 0.05
        is_trailing = sil.end >= duration - 0.05
        if is_leading:
            if not trim_edges:
                continue
            keep_here = edge_keep / 1000.0
            cut_end = sil.end - keep_here
            if cut_end > sil.start + 0.005:
                removals.append((sil.start, cut_end))
        elif is_trailing:
            if not trim_edges:
                continue
            keep_here = edge_keep / 1000.0
            cut_start = sil.start + keep_here
            if cut_start < sil.end - 0.005:
                removals.append((cut_start, sil.end))
        else:
            keep_here = keep / 1000.0
            if sil.duration <= keep_here + 0.03:
                continue
            cut_start = sil.start
            cut_end = sil.end - keep_here
            if cut_end - cut_start > 0.005:
                removals.append((cut_start, cut_end))

    removals.sort()
    # Build kept intervals as the complement of removals.
    kept: list[tuple[float, float]] = []
    cursor = 0.0
    for a, b in removals:
        a, b = max(0.0, a), min(duration, b)
        if a > cursor + 0.0005:
            kept.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < duration - 0.0005:
        kept.append((cursor, duration))
    kept = [(a, b) for a, b in kept if b - a > 0.002]
    removed_total = sum(b - a for a, b in removals)
    expected = max(0.0, duration - removed_total)
    if analysis.info.duration > 0 and removed_total / max(0.001, duration) > 0.65:
        warnings.append("More than 65% of the file is silence — review the result carefully.")
    if expected < 0.8 and duration >= 2:
        warnings.append("Result would be shorter than 0.8 s — rejected for safety.")
    return CleanPlan(analysis=analysis, keep_ms=keep, edge_keep_ms=edge_keep,
                     trim_edges=trim_edges, removals=removals, kept=kept,
                     removed_total=removed_total, expected_duration=expected,
                     warnings=warnings)


# ════════════════════════════════════════════════════════════════════
#  Render
# ════════════════════════════════════════════════════════════════════


def _filter_graph(kept: Sequence[tuple[float, float]], sample_rate: int = 24000) -> str:
    n = len(kept)
    if n == 0:
        return "[0:a]anull[out]"
    lines = [
        f"[0:a]aformat=sample_fmts=fltp:sample_rates={sample_rate}:channel_layouts=mono,"
        f"asplit={n}" + "".join(f"[s{i}]" for i in range(n))
    ]
    for i, (a, b) in enumerate(kept):
        dur = max(0.0, b - a)
        fade = min(0.005, dur / 4.0) if dur > 0.02 else 0.0
        chain = f"[s{i}]atrim=start={a:.4f}:end={b:.4f},asetpts=PTS-STARTPTS"
        if fade > 0:
            chain += f",afade=t=in:st=0:d={fade:.4f}"
            chain += f",afade=t=out:st={max(0.0, dur - fade):.4f}:d={fade:.4f}"
        lines.append(chain + f"[a{i}]")
    lines.append("".join(f"[a{i}]" for i in range(n)) +
                 f"concat=n={n}:v=0:a=1[out]")
    return ";\n".join(lines)


def render_plan(source: str, output: str, plan: CleanPlan, *,
                quality_kbps: int = DEFAULT_QUALITY_KBPS,
                sample_rate: int = 24000,
                cancel_event: threading.Event | None = None) -> None:
    """Render the plan to ``output`` (temporary file).  One encode, no shell."""
    output = str(output)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    graph = _filter_graph(plan.kept, sample_rate=sample_rate)
    fd, script = tempfile.mkstemp(prefix="lecta_filter_", suffix=".txt",
                                  dir=str(Path(output).parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(graph)
        cmd = [
            FFMPEG, "-hide_banner", "-nostats", "-y",
            "-i", source,
            "-filter_complex_script", script,
            "-map", "[out]",
        ]
        has_cover = bool(plan.analysis.info.has_cover)
        if has_cover:
            cmd += ["-map", "0:v?", "-c:v", "copy", "-disposition:v", "attached_pic"]
        cmd += [
            "-map_metadata", "0",
            "-c:a", "libmp3lame", "-b:a", f"{int(quality_kbps)}k",
            "-ar", str(int(sample_rate)), "-ac", "1",
            "-id3v2_version", "3",
            "-f", "mp3", output,
        ]
        rc, _out, err = _run(cmd, cancel_event=cancel_event)
        if rc != 0:
            raise RuntimeError(f"ffmpeg encode failed (code {rc}): {err.strip()[-400:]}")
    finally:
        try:
            os.unlink(script)
        except OSError:
            pass


# ════════════════════════════════════════════════════════════════════
#  Validation
# ════════════════════════════════════════════════════════════════════


def verify_audio_file(path: str, *, expected_duration: float | None = None,
                      min_duration: float = 0.3,
                      cancel_event: threading.Event | None = None) -> dict:
    """Decode the whole file and check that it is playable and audible."""
    info = probe_audio(path)
    result = {"ok": False, "info": info.to_dict(), "errors": [], "warnings": []}
    errors: list[str] = result["errors"]
    warnings: list[str] = result["warnings"]
    if not info.ok:
        errors.append(info.error or "Invalid audio")
        return result
    if info.duration < min_duration:
        errors.append(f"Output is too short ({info.duration:.2f}s)")
    if expected_duration is not None:
        tolerance = max(1.5, expected_duration * 0.02)
        if abs(info.duration - expected_duration) > tolerance:
            errors.append(
                f"Duration mismatch: expected {expected_duration:.2f}s, "
                f"got {info.duration:.2f}s")
    # Full decode + loudness in a single pass (used to decode the file twice).
    rc, _out, err = _run([FFMPEG, "-hide_banner", "-nostats", "-nostdin", "-i", path,
                          "-af", "volumedetect", "-f", "null", "-"], timeout=3600,
                         cancel_event=cancel_event)
    if rc != 0:
        errors.append(f"Decoder error: {(err or '').strip()[-300:]}")
    peak = _parse_float(_MAX_VOL, err or "", -99.0)
    if peak < -50:
        errors.append("Output contains no audible signal (fully silent)")
    if peak > 0.2:
        warnings.append("Output may be clipped (peak above 0 dBFS)")
    result["ok"] = not errors
    result["peak_dbfs"] = round(peak, 2)
    return result


# ════════════════════════════════════════════════════════════════════
#  Filenames / outputs
# ════════════════════════════════════════════════════════════════════


def outputs_dir() -> Path:
    p = _APP_DIR / "data" / "_audio_tools"
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_cleaned_path(source: str | os.PathLike, preset_key: str = "compact",
                         output_dir: str | os.PathLike | None = None,
                         overwrite: bool = False) -> Path:
    src = Path(source)
    folder = Path(output_dir) if output_dir else src.parent
    suffix = f"_{preset_key}"
    base = folder / f"{src.stem}{suffix}.mp3"
    if overwrite or not base.exists():
        return base
    i = 2
    while True:
        cand = folder / f"{src.stem}{suffix}_{i}.mp3"
        if not cand.exists():
            return cand
        i += 1


def _check_writable(folder: Path) -> bool:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / f".lecta_write_{uuid.uuid4().hex[:8]}.tmp"
        probe.write_bytes(b"ok")
        probe.unlink()
        return True
    except Exception:
        return False


# ════════════════════════════════════════════════════════════════════
#  Public cleaning API
# ════════════════════════════════════════════════════════════════════


def clean_audio_file(
    source: str | os.PathLike,
    *,
    preset: str = "compact",
    threshold_db: float | None = None,
    min_silence_sec: float | None = None,
    keep_ms: int | None = None,
    edge_keep_ms: int | None = None,
    trim_edges: bool = True,
    quality_kbps: int = DEFAULT_QUALITY_KBPS,
    output_path: str | os.PathLike | None = None,
    output_dir: str | os.PathLike | None = None,
    overwrite: bool = False,
    cancel_event: threading.Event | None = None,
    dry_run: bool = False,
    progress: Callable[[str], None] | None = None,
) -> CleanResult:
    src = str(Path(source))
    pr = PRESETS.get(preset, PRESETS["compact"])
    threshold = pr.threshold_db if threshold_db is None else float(threshold_db)
    min_sil = pr.min_silence_sec if min_silence_sec is None else float(min_silence_sec)
    keep = pr.keep_ms if keep_ms is None else int(keep_ms)
    edge_keep = pr.edge_keep_ms if edge_keep_ms is None else int(edge_keep_ms)
    result = CleanResult(source=src)

    def note(msg: str) -> None:
        if progress:
            try:
                progress(msg)
            except Exception:
                pass

    if cancel_event is not None and cancel_event.is_set():
        result.status = "cancelled"
        result.message = "Cancelled before processing"
        return result

    note("Reading audio…")
    analysis = analyze_audio(src, threshold_db=threshold,
                             min_silence_sec=min_sil,
                             cancel_event=cancel_event)
    result.before_duration = analysis.info.duration
    result.plan = None
    if not analysis.info.ok:
        result.status = "failed"
        result.error = analysis.info.error
        result.message = f"❌ {analysis.info.error}"
        return result
    if analysis.info.duration < 0.3:
        result.status = "failed"
        result.message = "❌ Audio is too short to process"
        return result

    plan = build_clean_plan(analysis, keep_ms=keep, edge_keep_ms=edge_keep,
                            trim_edges=trim_edges)
    result.plan = plan
    result.pauses_shortened = len(plan.removals)
    result.warnings = list(analysis.warnings) + list(plan.warnings)
    if plan.removed_total < 0.02:
        result.status = "skipped"
        result.message = "✅ No excessive pauses found — nothing to remove"
        return result
    if plan.expected_duration < 0.8 and analysis.info.duration >= 2:
        result.status = "failed"
        result.message = "❌ Safety check: result would be too short. Choose Natural preset."
        return result
    if dry_run:
        result.status = "planned"
        result.message = (f"Planned: −{plan.removed_total:.1f}s, "
                          f"{len(plan.removals)} pauses")
        return result

    # Destination: same folder as source by default, never overwrite unless asked.
    if output_path is not None:
        dest = Path(output_path)
    else:
        preferred = Path(output_dir) if output_dir else Path(src).parent
        if not _check_writable(preferred):
            preferred = outputs_dir()
        dest = default_cleaned_path(src, preset_key=pr.key, output_dir=preferred,
                                    overwrite=overwrite)
    if dest.exists() and not overwrite:
        dest = default_cleaned_path(src, preset_key=pr.key,
                                    output_dir=dest.parent, overwrite=False)
    tmp = dest.with_name(f".{dest.stem}.{uuid.uuid4().hex[:8]}.tmp.mp3")
    note("Removing excessive pauses…")
    try:
        render_plan(src, str(tmp), plan, quality_kbps=quality_kbps,
                    sample_rate=analysis.info.sample_rate or 24000,
                    cancel_event=cancel_event)
    except Exception as e:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        if cancel_event is not None and cancel_event.is_set():
            result.status = "cancelled"
            result.message = "Cancelled"
            return result
        result.status = "failed"
        result.error = str(e)
        result.message = f"❌ Encoding failed: {e}"
        return result

    note("Validating output…")
    check = verify_audio_file(str(tmp), expected_duration=plan.expected_duration,
                              cancel_event=cancel_event)
    if not check["ok"]:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        result.status = "failed"
        result.error = "; ".join(check["errors"])
        result.message = "❌ Output rejected by integrity check: " + result.error
        return result

    try:
        os.replace(tmp, dest)
    except Exception as e:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        result.status = "failed"
        result.error = str(e)
        result.message = f"❌ Could not save file: {e}"
        return result

    after_info = probe_audio(str(dest))
    result.output = str(dest)
    result.after_duration = after_info.duration
    result.removed = max(0.0, result.before_duration - result.after_duration)
    result.status = "done"
    result.message = (f"✅ Saved {dest.name}: {result.before_duration:.1f}s → "
                      f"{result.after_duration:.1f}s (−{result.removed:.1f}s)")
    result.warnings += check.get("warnings", [])
    return result


def clean_batch(
    sources: Iterable[str | os.PathLike],
    *,
    preset: str = "compact",
    threshold_db: float | None = None,
    min_silence_sec: float | None = None,
    keep_ms: int | None = None,
    edge_keep_ms: int | None = None,
    trim_edges: bool = True,
    quality_kbps: int = DEFAULT_QUALITY_KBPS,
    output_dir: str | os.PathLike | None = None,
    overwrite: bool = False,
    cancel_event: threading.Event | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[CleanResult]:
    results: list[CleanResult] = []
    files = [str(s) for s in sources if s]
    for i, src in enumerate(files, 1):
        if cancel_event is not None and cancel_event.is_set():
            break
        label = Path(src).name
        if progress:
            try:
                progress(f"[{i}/{len(files)}] {label}")
            except Exception:
                pass
        results.append(clean_audio_file(
            src, preset=preset, threshold_db=threshold_db,
            min_silence_sec=min_silence_sec, keep_ms=keep_ms,
            edge_keep_ms=edge_keep_ms, trim_edges=trim_edges,
            quality_kbps=quality_kbps, output_dir=output_dir,
            overwrite=overwrite, cancel_event=cancel_event))
    return results


def summarize_results(results: Sequence[CleanResult]) -> dict:
    done = [r for r in results if r.status == "done"]
    skipped = [r for r in results if r.status == "skipped"]
    failed = [r for r in results if r.status == "failed"]
    cancelled = [r for r in results if r.status == "cancelled"]
    return {
        "processed": len(results),
        "done": len(done),
        "skipped": len(skipped),
        "failed": len(failed),
        "cancelled": len(cancelled),
        "before": sum(r.before_duration for r in results),
        "after": sum(r.after_duration for r in done),
        "removed": sum(r.removed for r in done),
        "pauses": sum(r.pauses_shortened for r in done),
    }


def results_table(results: Sequence[CleanResult]) -> list[list[str]]:
    rows = []
    for r in results:
        name = Path(r.source).name
        if r.status == "done":
            state = "✅ Cleaned"
        elif r.status == "skipped":
            state = "ℹ️ No change"
        elif r.status == "cancelled":
            state = "⏹ Cancelled"
        else:
            state = "❌ Failed"
        rows.append([
            name, state,
            f"{r.before_duration:.1f}s",
            f"{r.after_duration:.1f}s" if r.after_duration else "—",
            f"−{r.removed:.1f}s" if r.removed else "—",
            str(r.pauses_shortened),
            r.output or r.message,
        ])
    return rows


def results_report(results: Sequence[CleanResult]) -> str:
    s = summarize_results(results)
    lines = [
        f"Files processed: {s['processed']}  ·  cleaned: {s['done']}  ·  "
        f"unchanged: {s['skipped']}  ·  failed: {s['failed']}"
        + (f"  ·  cancelled: {s['cancelled']}" if s["cancelled"] else ""),
    ]
    if s["done"]:
        lines.append(
            f"Duration: {s['before']:.1f}s → {s['after']:.1f}s "
            f"(removed {s['removed']:.1f}s across {s['pauses']} pauses)")
    warnings: list[str] = []
    for r in results:
        warnings.extend(f"{Path(r.source).name}: {w}" for w in r.warnings)
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"• {w}" for w in warnings[:12])
        if len(warnings) > 12:
            lines.append(f"…and {len(warnings) - 12} more")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════
#  Before / after preview
# ════════════════════════════════════════════════════════════════════


def _map_time(t: float, removals: Sequence[tuple[float, float]]) -> float:
    shift = 0.0
    for a, b in removals:
        if t <= a:
            break
        shift += min(t, b) - a
    return max(0.0, t - shift)


def _extract_clip(source: str, dest: str, start: float, duration: float,
                  cancel_event: threading.Event | None = None) -> bool:
    cmd = [FFMPEG, "-hide_banner", "-nostats", "-y", "-ss", f"{start:.3f}",
           "-t", f"{duration:.3f}", "-i", source, "-vn",
           "-c:a", "libmp3lame", "-b:a", "128k", "-ar", "24000", "-ac", "1",
           "-f", "mp3", dest]
    rc, _out, _err = _run(cmd, cancel_event=cancel_event)
    return rc == 0 and Path(dest).is_file() and Path(dest).stat().st_size > 512


def make_preview(source: str | os.PathLike, *,
                 preset: str = "compact",
                 threshold_db: float | None = None,
                 min_silence_sec: float | None = None,
                 keep_ms: int | None = None,
                 edge_keep_ms: int | None = None,
                 trim_edges: bool = True,
                 cancel_event: threading.Event | None = None) -> dict:
    """Create short before/after clips around the longest removable pause."""
    src = str(Path(source))
    result = clean_audio_file(
        src, preset=preset, threshold_db=threshold_db,
        min_silence_sec=min_silence_sec, keep_ms=keep_ms,
        edge_keep_ms=edge_keep_ms, trim_edges=trim_edges,
        output_dir=outputs_dir() / "preview_tmp", overwrite=True,
        cancel_event=cancel_event)
    payload = {"ok": False, "before": "", "after": "", "message": "",
               "analysis": None, "clean": result}
    if result.status not in ("done",):
        payload["message"] = result.message
        return payload
    plan = result.plan
    analysis = plan.analysis
    # Pick the longest internal silence (fall back to any silence)
    target = max(analysis.silences, key=lambda s: s.duration, default=None)
    if target is None or target.duration < 0.4:
        payload["message"] = "No long pause available for a preview"
        return payload
    before_start = max(0.0, target.start - 3.0)
    before_end = min(analysis.info.duration, target.end + 3.0)
    before_dur = max(1.0, before_end - before_start)
    after_start = _map_time(before_start, plan.removals)
    after_dur = max(1.0, before_dur - (target.duration - plan.keep_ms / 1000.0))
    out = outputs_dir() / "preview"
    out.mkdir(parents=True, exist_ok=True)
    stamp = uuid.uuid4().hex[:6]
    before_path = out / f"{Path(src).stem}_before_{stamp}.mp3"
    after_path = out / f"{Path(src).stem}_after_{stamp}.mp3"
    if not _extract_clip(src, str(before_path), before_start, before_dur, cancel_event):
        payload["message"] = "Could not create the before clip"
        return payload
    if not _extract_clip(result.output, str(after_path), after_start, after_dur, cancel_event):
        payload["message"] = "Could not create the after clip"
        return payload
    payload.update({
        "ok": True,
        "before": str(before_path),
        "after": str(after_path),
        "message": (f"Preview around a {target.duration:.1f}s pause "
                    f"(kept {plan.keep_ms} ms)"),
        "analysis": analysis.to_dict(),
        "target": target.to_dict(),
        "plan": plan.to_dict(),
        "clean": result.to_dict(),
    })
    return payload


# ════════════════════════════════════════════════════════════════════
#  Project helpers
# ════════════════════════════════════════════════════════════════════


def project_mp3_files(project_dir: str | os.PathLike) -> list[str]:
    p = Path(project_dir)
    if p.is_file():
        p = p.parent
    files = sorted(p.glob("*.mp3"))
    return [str(f) for f in files
            if f.is_file() and "_PARTIAL" not in f.name
            and not f.name.startswith(".")]


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:04.1f}" if h else f"{m}:{s:04.1f}"


__all__ = [
    "FFMPEG", "FFPROBE", "PRESETS", "DEFAULT_THRESHOLD_DB",
    "DEFAULT_MIN_SILENCE_SEC", "DEFAULT_KEEP_MS", "DEFAULT_QUALITY_KBPS",
    "AudioInfo", "AudioAnalysis", "SilenceInterval", "CleanPreset",
    "CleanPlan", "CleanResult",
    "probe_audio", "detect_silences", "analyze_audio", "build_clean_plan",
    "clean_audio_file", "clean_batch", "make_preview", "verify_audio_file",
    "summarize_results", "results_table", "results_report",
    "default_cleaned_path", "outputs_dir", "project_mp3_files", "format_seconds",
]