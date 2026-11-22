"""Multilingual TTS routing and loss-free segment assembly.

Design rules (2026-11 audio repair)
-----------------------------------
* A failed cloud/local synthesis is an **error**, never a silent placeholder.
  The caller aborts the file instead of writing a 2.5-second "pause" into it.
* Every segment is edge-trimmed at a very low threshold (-48 dBFS) so quiet
  Hebrew/Russian/English consonants survive.
* Segments are joined with explicit, policy-driven gaps.  Speech is never
  cross-faded into speech.
* Pause policy is configurable from Settings and is persisted by ``AppConfig``.
"""
import asyncio
import concurrent.futures
import hashlib
import io
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from pydub import AudioSegment

from config import TTS_WORKERS

#: Microsoft throttles bursts of parallel Edge requests ("No audio was
#: received").  Keeping this low makes synthesis slower but reliable.
EDGE_MAX_WORKERS = max(1, int(os.getenv("LECTA_EDGE_WORKERS", "3")))
#: Edge retry attempts per segment before the split-in-half recovery.
EDGE_RETRIES = max(1, int(os.getenv("LECTA_EDGE_RETRIES", "4")))

_TTS_CACHE_DIR = Path(__file__).resolve().parent.parent / 'data' / 'tts_cache'
_TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

USE_EDGE_FOR_ENGLISH = False
USE_EDGE_FOR_HEBREW = True
USE_EDGE_FOR_RUSSIAN = False
DICTIONARY_MODE = False

# ════════════════════════════════════════════════════════════════════
#  Pause policy (all values in milliseconds unless noted)
# ════════════════════════════════════════════════════════════════════

#: Gap between fragments of one sentence (word / punctuation split).
PAUSE_INTRA_MS = 40
#: Gap after ``.``, ``!``, ``?`` inside one paragraph.
PAUSE_SENTENCE_MS = 280
#: Gap between paragraphs of the same language.
PAUSE_PARAGRAPH_MS = 450
#: Gap when the language changes inside a paragraph (RU -> EN term etc.).
PAUSE_LANGUAGE_MS = 160
#: Extra breathing room used by the learning/dictionary mode.
PAUSE_LEARNING_MS = 900
#: Longest silence tolerated *inside* one synthesized fragment.  A TTS model
#: (F5-TTS especially) breathes on its own at commas and full stops, and Edge
#: returns whole sentences with its built-in breaks.  Those pauses bypassed the
#: join policy entirely, which is why narration sounded halting even with the
#: "compact" profile.  Longer runs are shortened to this value.
PAUSE_INTERNAL_MS = 250
#: Envelope frame used by the silence analysis (ms).
ENVELOPE_FRAME_MS = 20
#: A frame quieter than this (relative to the fragment peak) counts as silence.
SILENCE_REL_DB = -45.0
#: Upper bound for an explicit ``<break time="N">`` from parsed XML.
#: The historic format is deciseconds (``30`` -> 3.0 s).
EXPLICIT_BREAK_MAX_MS = 1800
#: Silence kept around a trimmed segment edge.
EDGE_TRIM_MS = 30
#: Tiny fade used only to prevent a click at a cut point (never over speech).
MICRO_FADE_MS = 8
#: Absolute trim threshold.  -48 dBFS keeps quiet consonants.
SILENCE_THRESHOLD = 0.004

#: Named profiles.  Compact is the default and deliberately avoids the old
#: "learning pauses" that turned normal narration into long dead air.
GENERATION_PRESETS = {
    "compact": {
        "pause_intra_ms": 40,
        "pause_sentence_ms": 280,
        "pause_paragraph_ms": 450,
        "pause_language_ms": 160,
        "pause_learning_ms": 900,
        "pause_internal_ms": 250,
        "explicit_break_max_ms": 1400,
        "edge_trim_ms": 30,
        "micro_fade_ms": 8,
    },
    "natural": {
        "pause_intra_ms": 70,
        "pause_sentence_ms": 380,
        "pause_paragraph_ms": 600,
        "pause_language_ms": 240,
        "pause_learning_ms": 1100,
        "pause_internal_ms": 400,
        "explicit_break_max_ms": 2200,
        "edge_trim_ms": 35,
        "micro_fade_ms": 8,
    },
    "learning": {
        "pause_intra_ms": 130,
        "pause_sentence_ms": 650,
        "pause_paragraph_ms": 950,
        "pause_language_ms": 500,
        "pause_learning_ms": 1500,
        "pause_internal_ms": 700,
        "explicit_break_max_ms": 3500,
        "edge_trim_ms": 40,
        "micro_fade_ms": 10,
    },
}
DEFAULT_GENERATION_PRESET = "compact"

_GENERATION_FIELDS = tuple(GENERATION_PRESETS["compact"].keys())


def generation_settings() -> dict:
    """Return the current policy as a JSON-serialisable dict."""
    return {
        "generation_preset": _current_preset,
        **{name: int(globals()[name.upper()]) for name in _GENERATION_FIELDS},
    }


def apply_generation_settings(values=None, *, preset: str | None = None) -> dict:
    """Apply explicit values and/or a named preset.  Unknown values are ignored."""
    values = dict(values or {})
    chosen = preset or values.get("generation_preset")
    if chosen in GENERATION_PRESETS:
        _apply_named_preset_only(chosen)
    for name in _GENERATION_FIELDS:
        if name in values and values[name] is not None:
            try:
                globals()[name.upper()] = max(0, int(float(values[name])))
            except (TypeError, ValueError):
                pass
    return generation_settings()


def _apply_named_preset_only(preset: str) -> None:
    global _current_preset
    _current_preset = preset
    for name, value in GENERATION_PRESETS[preset].items():
        globals()[name.upper()] = int(value)


_current_preset = DEFAULT_GENERATION_PRESET


def explicit_break_ms(value, *, cap_ms: int | None = None) -> int:
    """Convert a legacy ``<break time="N">`` value (deciseconds) to ms."""
    try:
        raw = float(value)
    except (TypeError, ValueError):
        raw = PAUSE_PARAGRAPH_MS / 100.0
    cap = EXPLICIT_BREAK_MAX_MS if cap_ms is None else max(0, int(cap_ms))
    return int(max(0.0, min(raw * 100.0, cap)))


# ════════════════════════════════════════════════════════════════════
#  Logging
# ════════════════════════════════════════════════════════════════════

class ColorFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(levelname)s - %(message)s"
    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


logger = logging.getLogger("Router")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(ColorFormatter())
    logger.addHandler(ch)


# ════════════════════════════════════════════════════════════════════
#  Basic audio helpers
# ════════════════════════════════════════════════════════════════════

def create_silence(duration_sec, sample_rate):
    return np.zeros(max(0, int(sample_rate * duration_sec)), dtype=np.float32)


def normalize_audio(audio_np):
    if audio_np is None or getattr(audio_np, 'size', 0) == 0:
        return None
    audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=0.0, neginf=0.0)
    if str(audio_np.dtype).startswith('int') or np.max(np.abs(audio_np)) > 10.0:
        audio_np = audio_np.astype(np.float32) / 32768.0
    else:
        audio_np = audio_np.astype(np.float32)
    return np.clip(audio_np, -1.0, 1.0)


def resample_audio(audio_np, orig_sr, target_sr):
    audio_np = normalize_audio(audio_np)
    if audio_np is None:
        return None
    if orig_sr == target_sr:
        return audio_np
    try:
        import torch
        import torchaudio

        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
        resampled = torchaudio.functional.resample(audio_tensor, orig_sr, target_sr)
        return resampled.squeeze(0).numpy()
    except Exception as e:
        logger.error(f"Resample error: {e}")
        raise


def _frame_rms(audio_np: np.ndarray, sample_rate: int, frame_ms: int = ENVELOPE_FRAME_MS):
    """Return (per-frame RMS envelope, frame width in samples)."""
    width = max(1, int(sample_rate * max(1, frame_ms) / 1000.0))
    count = len(audio_np) // width
    if count <= 0:
        return np.zeros(0, dtype=np.float64), width
    frames = audio_np[: count * width].astype(np.float64).reshape(count, width)
    return np.sqrt(np.mean(frames * frames, axis=1)), width


def _silence_threshold(env: np.ndarray) -> float:
    """Threshold that follows the fragment's own level (relative to its peak)."""
    peak = float(env.max()) if getattr(env, "size", 0) else 0.0
    return max(SILENCE_THRESHOLD, peak * (10.0 ** (SILENCE_REL_DB / 20.0)))


def _silent_runs(env: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Half-open frame ranges (start, end) of near-silence runs."""
    if getattr(env, "size", 0) == 0:
        empty = np.zeros(0, dtype=int)
        return empty, empty
    silent = env < _silence_threshold(env)
    padded = np.concatenate(([False], silent, [False]))
    edges = np.diff(padded.astype(np.int8))
    return np.where(edges == 1)[0], np.where(edges == -1)[0]


def squeeze_internal_silence(audio_np, sample_rate, max_gap_ms=None,
                             *, frame_ms: int = ENVELOPE_FRAME_MS):
    """Shorten dead air *inside* one synthesized fragment to ``max_gap_ms``.

    The join policy only controls the gaps LECTA inserts between fragments.
    Everything a TTS engine breathes out on its own (F5-TTS pauses on commas
    and full stops, Edge returns whole sentences with its own breaks) stayed in
    the audio untouched, so the narration kept ~0.5 s of air every few words no
    matter which pause profile was selected.  Long silence runs are cut down to
    the configured limit here; the first part of the run is kept so the breath
    still leads into the next word.
    """
    if audio_np is None or getattr(audio_np, 'size', 0) == 0:
        return audio_np
    limit_ms = PAUSE_INTERNAL_MS if max_gap_ms is None else int(max_gap_ms)
    if limit_ms <= 0:
        return audio_np
    env, width = _frame_rms(audio_np, sample_rate, frame_ms)
    if env.size < 3:
        return audio_np
    starts, ends = _silent_runs(env)
    if len(starts) == 0:
        return audio_np
    limit = max(1, int(round(limit_ms / float(max(1, frame_ms)))))
    keep = np.ones(len(audio_np), dtype=bool)
    changed = False
    for start, end in zip(starts, ends):
        if (end - start) <= limit:
            continue
        # Keep the head of the run (a breath belongs before the next word) and
        # drop the rest.  Runs never overlap, so a single mask is enough.
        cut_from = (start + limit) * width
        cut_to = min(len(audio_np), end * width)
        if cut_to > cut_from:
            keep[cut_from:cut_to] = False
            changed = True
    if not changed:
        return audio_np
    return audio_np[keep]


def _fade_edges(audio_np: np.ndarray, sample_rate: int, fade_ms: int) -> np.ndarray:
    """Apply a very short fade to both edges (anti-click only)."""
    n = min(int(sample_rate * max(0, fade_ms) / 1000.0), len(audio_np) // 4)
    if n <= 1:
        return audio_np
    out = audio_np.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def gentle_trim_and_fade(audio_np, sample_rate, pad_sec=None, fade_sec=None):
    """Trim only true edge silence; return ``None`` when there is no speech.

    The caller must treat ``None`` as a synthesis failure.  Returning a few
    zero samples here was the source of lost words in the old implementation.
    """
    if pad_sec is None:
        pad_sec = EDGE_TRIM_MS / 1000.0
    if fade_sec is None:
        fade_sec = MICRO_FADE_MS / 1000.0
    audio_np = normalize_audio(audio_np)
    if audio_np is None or len(audio_np) == 0:
        return None
    if float(np.max(np.abs(audio_np))) < SILENCE_THRESHOLD:
        return None
    # Envelope-based detection: TTS engines leave a low hiss (around -45 dBFS)
    # after the last word, which an absolute 0.004 threshold never removed.
    # Those tails were added to every configured gap, so a 450 ms paragraph
    # pause was heard as 700-900 ms of dead air.
    env, width = _frame_rms(audio_np, sample_rate)
    if env.size == 0:
        return None
    active = np.where(env >= _silence_threshold(env))[0]
    if len(active) == 0:
        return None
    pad = max(0, int(sample_rate * pad_sec))
    start = max(0, int(active[0]) * width - pad)
    end = min(len(audio_np), int(active[-1] + 1) * width + pad)
    trimmed = audio_np[start:end]
    if len(trimmed) < max(1, int(sample_rate * 0.02)):
        return None
    return _fade_edges(trimmed, sample_rate, int(fade_sec * 1000))


def _join(a, b, sample_rate, gap_ms=0):
    if a is None or getattr(a, 'size', 0) == 0:
        return b
    if b is None or getattr(b, 'size', 0) == 0:
        return a
    gap = max(0, int(sample_rate * int(gap_ms) / 1000.0))
    if gap:
        return np.concatenate([a, np.zeros(gap, dtype=np.float32), b])
    return np.concatenate([a, b])


def crossfade_join(a, b, sample_rate, overlap_sec=None):
    """Backward-compatible name.

    Speech is no longer cross-faded.  ``overlap_sec`` is interpreted as a
    neutral gap so old callers remain harmless during migration.
    """
    gap_ms = 0 if overlap_sec is None else int(float(overlap_sec) * 1000)
    return _join(a, b, sample_rate, gap_ms)


def split_by_words(long_text, max_chars):
    words = long_text.split()
    word_chunks = []
    curr = ""
    for w in words:
        if len(curr) + len(w) + 1 <= max_chars:
            curr += w + " "
        else:
            if curr:
                word_chunks.append(curr.strip())
            curr = w + " "
    if curr:
        word_chunks.append(curr.strip())
    return word_chunks


def smart_split(text, max_chars):
    sentences = re.split(r'(?<=[.!?\n])\s+', text)
    chunks = []
    current_chunk = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            sub_sentences = re.split(r'(?<=[,;:—])\s+', s)
            sub_chunk = ""
            for sub in sub_sentences:
                sub = sub.strip()
                if not sub:
                    continue
                if len(sub) > max_chars:
                    if sub_chunk:
                        chunks.append(sub_chunk.strip())
                        sub_chunk = ""
                    chunks.extend(split_by_words(sub, max_chars))
                else:
                    if len(sub_chunk) + len(sub) + 1 <= max_chars:
                        sub_chunk += sub + " "
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk.strip())
                        sub_chunk = sub + " "
            if sub_chunk:
                chunks.append(sub_chunk.strip())
        else:
            if len(current_chunk) + len(s) + 1 <= max_chars:
                current_chunk += s + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = s + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def auto_split_mixed_languages(text):
    if '|' in text:
        return [p.strip() for p in text.split('|') if p.strip()]
    chunks = []
    current_chunk = ""
    current_type = None
    for char in text:
        if '\u0590' <= char <= '\u05FF':
            char_type = 'heb'
        elif char.isalpha():
            if re.match(r'[а-яА-ЯёЁ]', char):
                char_type = 'ru'
            else:
                char_type = 'en'
        else:
            char_type = current_type if current_type else 'other'
        if current_type is None:
            current_type = char_type
        if char_type == current_type or char_type == 'other':
            current_chunk += char
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = char
            current_type = char_type
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks


# ═══ SEGMENT CACHE ═══
_segment_cache = {}


def _cache_key(voice, target_sr, text):
    return (voice, target_sr, hashlib.md5(text.encode('utf-8')).hexdigest())


def _cache_key_str(voice, target_sr, text):
    return f"{voice}_{target_sr}_{hashlib.md5(text.encode('utf-8')).hexdigest()}"


def _load_from_disk_cache(key_str):
    cache_file = _TTS_CACHE_DIR / f"{key_str}.npy"
    if cache_file.exists():
        try:
            return np.load(str(cache_file), allow_pickle=False)
        except Exception as e:
            logger.warning(f"[CACHE] Disk load error: {e}")
    return None


def _save_to_disk_cache(key_str, audio_np):
    try:
        import uuid

        cache_file = _TTS_CACHE_DIR / f"{key_str}.npy"
        # Unique temp name: parallel Edge workers used to collide on one .tmp
        # file, which made whole segments disappear from the disk cache.
        tmp = cache_file.with_name(f".{key_str}.{uuid.uuid4().hex[:8]}.tmp.npy")
        np.save(str(tmp), audio_np)
        tmp.replace(cache_file)
    except Exception as e:
        logger.warning(f"[CACHE] Disk save error: {e}")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def clear_segment_cache():
    global _segment_cache
    _segment_cache.clear()
    logger.info("[CACHE] Segment cache cleared")


# ═══ EDGE TTS EXECUTOR ═══
_edge_executor = None


def _run_async_safely(coroutine_func, timeout=120):
    global _edge_executor
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine_func())
    if _edge_executor is None:
        _edge_executor = concurrent.futures.ThreadPoolExecutor(max_workers=TTS_WORKERS)
        logger.debug(f"[EDGE TTS] Executor created (max_workers={TTS_WORKERS})")
    future = _edge_executor.submit(asyncio.run, coroutine_func())
    return future.result(timeout=timeout)


def _clean_edge_text(text: str) -> str:
    # "+" is a stress mark: Edge TTS must never pronounce it.
    text = re.sub(r"(?<=\d)\s*\+\s*(?=\d)", "<<PLUS>>", text or "")
    text = text.replace("+", "").replace("<<PLUS>>", "+")
    return text.replace('"', '').replace("'", '').strip()


def get_edge_audio(text, voice="he-IL-AvriNeural", target_sr=24000,
                   max_retries=EDGE_RETRIES, _depth=0):
    """Synthesize one Edge segment or raise ``RuntimeError``.

    Persistent failures split the text in half and retry each half, which
    recovers from transient per-request limits without ever inserting silence.
    """
    clean_text = _clean_edge_text(text)
    if not clean_text:
        raise RuntimeError("Empty text for Edge TTS")

    ck = _cache_key(voice, target_sr, clean_text)
    ck_str = _cache_key_str(voice, target_sr, clean_text)

    if ck in _segment_cache:
        logger.info(f"[CACHE HIT] memory: {clean_text[:30]}...")
        return _segment_cache[ck].copy()

    disk_audio = _load_from_disk_cache(ck_str)
    if disk_audio is not None and getattr(disk_audio, "size", 0) > 0:
        if float(np.max(np.abs(disk_audio))) >= SILENCE_THRESHOLD:
            _segment_cache[ck] = disk_audio.copy()
            logger.info(f"[CACHE HIT] disk: {clean_text[:30]}...")
            return disk_audio.copy()

    last_error = None
    for attempt in range(max(1, int(max_retries))):
        try:
            logger.info(
                f"[CLOUD] Voice: {voice} | Text: {clean_text[:50]}..."
                + (f" (attempt {attempt + 1}/{max_retries})" if attempt else "")
            )
            import edge_tts

            async def _synthesize():
                communicate = edge_tts.Communicate(clean_text, voice)
                audio_bytes = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_bytes += chunk["data"]
                return audio_bytes

            t0 = time.time()
            audio_bytes = _run_async_safely(_synthesize, timeout=45)
            elapsed = time.time() - t0
            if elapsed > 5.0:
                logger.warning(f"[EDGE SLOW] {elapsed:.1f}s for: {clean_text[:80]}...")
            if not audio_bytes:
                raise RuntimeError("Empty response from Edge TTS")

            audio_seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
            audio_seg = audio_seg.set_frame_rate(target_sr).set_channels(1)
            samples = np.array(audio_seg.get_array_of_samples())
            if audio_seg.sample_width == 2:
                audio_np = samples.astype(np.float32) / 32768.0
            elif audio_seg.sample_width == 4:
                audio_np = samples.astype(np.float32) / 2147483648.0
            else:
                audio_np = samples.astype(np.float32)
            audio_np = gentle_trim_and_fade(audio_np, target_sr)
            if audio_np is None or len(audio_np) < int(target_sr * 0.04):
                raise RuntimeError("Edge TTS returned no audible speech")
            # Edge bakes its own breaks into the sentence; keep only the
            # configured amount of air inside the fragment.
            audio_np = squeeze_internal_silence(audio_np, target_sr)

            _segment_cache[ck] = audio_np.copy()
            _save_to_disk_cache(ck_str, audio_np)
            logger.info(f"[CLOUD OK] {len(audio_np) / target_sr:.1f}s audio")
            return audio_np
        except Exception as e:
            last_error = e
            logger.warning(f"Edge TTS error ({voice}, attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                # Exponential-ish backoff with jitter avoids retry stampedes.
                time.sleep(0.8 + attempt * 1.2 + random.uniform(0, 0.4))

    # Last resort: split long text.  Short text is a genuine failure.
    words = clean_text.split()
    if _depth < 2 and len(words) >= 6:
        mid = len(words) // 2
        left = " ".join(words[:mid])
        right = " ".join(words[mid:])
        logger.warning("[EDGE] Splitting failed segment and retrying halves")
        try:
            a = get_edge_audio(left, voice, target_sr, max_retries=2, _depth=_depth + 1)
            b = get_edge_audio(right, voice, target_sr, max_retries=2, _depth=_depth + 1)
            return _join(a, b, target_sr, gap_ms=PAUSE_INTRA_MS)
        except Exception as split_error:
            last_error = f"{last_error} / split retry: {split_error}"

    raise RuntimeError(
        f"Edge TTS failed for voice {voice} after {max_retries} attempts: {last_error}"
    )


# ════════════════════════════════════════════════════════════════════
#  Language-aware text units
# ════════════════════════════════════════════════════════════════════

@dataclass
class _Unit:
    index: int
    text: str
    lang: str            # he | en | ru
    paragraph: int
    ends_sentence: bool


_SENTENCE_END = re.compile(r'[.!?…。！？]\s*$')
_CLAUSE_END = re.compile(r'[,;:—–-]\s*$')
#: A period/question/exclamation followed by a new sentence.  The next
#: character must look like the start of a sentence (not a digit, not a
#: lowercase abbreviation continuation).
_SENTENCE_SPLIT = re.compile(
    r'(?<=[.!?…。！？])\s+(?=[A-ZА-ЯЁ\u0590-\u05FF"«„\'(\d])'
)


def _split_sentences(text: str) -> list[str]:
    """Split one language fragment into sentences.

    Edge TTS inserts its own ~1 s break after a period inside one request.
    Sending sentence-sized units lets LECTA apply its own (much shorter)
    sentence gap instead, so no long dead air is baked into the audio.
    """
    parts: list[str] = []
    for chunk in _SENTENCE_SPLIT.split(text or ""):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts or ([text.strip()] if (text or "").strip() else [])


def _split_units(text: str) -> list[_Unit]:
    lines = [ln.strip() for ln in re.split(r'\n+', text or '') if ln and ln.strip()]
    units: list[_Unit] = []
    idx = 0
    for p_idx, line in enumerate(lines):
        for part in auto_split_mixed_languages(line):
            part = part.strip()
            if not part:
                continue
            for sentence in _split_sentences(part):
                if not re.search(r'[a-zA-Zа-яА-ЯёЁ0-9\u0590-\u05FF]', sentence):
                    continue
                if re.search(r'[\u0590-\u05FF]', sentence):
                    lang = 'he'
                elif re.search(r'[a-zA-Z]', sentence) and not re.search(r'[а-яА-ЯёЁ]', sentence):
                    lang = 'en'
                else:
                    lang = 'ru'
                units.append(_Unit(
                    index=idx,
                    text=sentence,
                    lang=lang,
                    paragraph=p_idx,
                    ends_sentence=bool(_SENTENCE_END.search(sentence)),
                ))
                idx += 1
    return units


def _gap_between(prev: _Unit | None, cur: _Unit) -> int:
    """Choose a natural pause at a segment boundary."""
    if prev is None:
        return 0
    if cur.paragraph != prev.paragraph:
        return max(PAUSE_PARAGRAPH_MS, PAUSE_LANGUAGE_MS if cur.lang != prev.lang else 0)
    if cur.lang != prev.lang:
        return PAUSE_LANGUAGE_MS
    if prev.ends_sentence:
        return PAUSE_SENTENCE_MS
    if _CLAUSE_END.search(prev.text):
        return max(PAUSE_INTRA_MS, PAUSE_SENTENCE_MS // 2)
    return PAUSE_INTRA_MS


def _learning_gap(unit: _Unit, previous_lang: str | None) -> int:
    if DICTIONARY_MODE:
        return max(PAUSE_LEARNING_MS, PAUSE_LANGUAGE_MS)
    return 0


# ════════════════════════════════════════════════════════════════════
#  Public pipeline
# ════════════════════════════════════════════════════════════════════

def process_multilingual_text(text, safe_synth_func,
                             use_edge_en=None, use_edge_he=None, use_edge_ru=None,
                             dict_mode=None):
    """Synthesize mixed RU/EN/HE text.

    Returns ``(sample_rate, np_audio)`` or raises ``RuntimeError`` when any
    required segment cannot be synthesized.  Callers must not write a file on
    failure.
    """
    global DICTIONARY_MODE
    _use_en = USE_EDGE_FOR_ENGLISH if use_edge_en is None else use_edge_en
    _use_he = USE_EDGE_FOR_HEBREW if use_edge_he is None else use_edge_he
    _use_ru = USE_EDGE_FOR_RUSSIAN if use_edge_ru is None else use_edge_ru
    if dict_mode is not None:
        DICTIONARY_MODE = bool(dict_mode)

    logger.debug(f"[CONFIG] Flags: EN={_use_en} HE={_use_he} RU={_use_ru}")

    units = _split_units(text)
    if not units:
        raise RuntimeError("No speakable text after preprocessing")

    target_sr = 24000
    edge_tasks: list[tuple[_Unit, str]] = []
    local_tasks: list[_Unit] = []

    for unit in units:
        if unit.lang == 'he' and _use_he:
            edge_tasks.append((unit, "he-IL-AvriNeural"))
        elif unit.lang == 'en' and _use_en:
            edge_tasks.append((unit, "en-US-ChristopherNeural"))
        elif unit.lang == 'ru' and _use_ru:
            edge_tasks.append((unit, "ru-RU-DmitryNeural"))
        else:
            local_tasks.append(unit)

    edge_results: dict[int, np.ndarray] = {}
    local_results: dict[int, np.ndarray] = {}

    # Phase 2: parallel Edge TTS.  Any persistent failure aborts the file.
    if edge_tasks:
        n_workers = max(1, min(EDGE_MAX_WORKERS, TTS_WORKERS, len(edge_tasks)))
        logger.info(f"[ROUTER] {len(edge_tasks)} Edge tasks (workers={n_workers})")
        errors: list[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_map = {
                executor.submit(get_edge_audio, unit.text, voice, target_sr): unit
                for unit, voice in edge_tasks
            }
            for future in concurrent.futures.as_completed(future_map):
                unit = future_map[future]
                try:
                    edge_results[unit.index] = future.result()
                except Exception as e:
                    errors.append(f"{unit.lang} '{unit.text[:40]}…': {e}")
                    logger.error(f"Edge task failed (unit {unit.index}): {e}")
        if errors:
            raise RuntimeError(
                "Cloud speech synthesis failed — no audio was written. "
                + "; ".join(errors[:3])
            )

    # Phase 3: sequential local TTS
    for unit in local_tasks:
        is_english = unit.lang == 'en'
        lang_tag = "EN-LOCAL" if is_english else ("HE-LOCAL" if unit.lang == 'he' else "RU-LOCAL")
        char_limit = 145 if is_english else 110
        chunks = smart_split(unit.text, max_chars=char_limit) or [unit.text]
        logger.info(f"[LOCAL] [{lang_tag}] {unit.text[:40]}...")
        local_parts: list[np.ndarray] = []
        for i, chunk in enumerate(chunks):
            if not re.search(r'[a-zA-Zа-яА-ЯёЁ0-9\u0590-\u05FF]', chunk):
                continue
            t0 = time.time()
            sr, aud = safe_synth_func(chunk, disable_norm=is_english)
            elapsed = time.time() - t0
            if elapsed > 5.0:
                logger.warning(f"[LOCAL SLOW] {elapsed:.1f}s for: {chunk[:80]}...")
            if aud is None or getattr(aud, 'size', 0) == 0:
                raise RuntimeError(
                    f"Local TTS returned no audio for {lang_tag}: {chunk[:60]!r}"
                )
            resampled = resample_audio(aud, sr, target_sr)
            trimmed = gentle_trim_and_fade(resampled, target_sr)
            if trimmed is None:
                raise RuntimeError(
                    f"Local TTS returned silent audio for {lang_tag}: {chunk[:60]!r}"
                )
            # F5-TTS breathes on its own at commas/full stops.  Without this
            # the configured pause profile only governed the joins, so the
            # model's internal air stayed in the narration (~0.5 s every few
            # words even on the compact profile).
            trimmed = squeeze_internal_silence(trimmed, target_sr)
            if local_parts:
                # Chunks are parts of the same sentence: only a micro gap.
                local_parts.append(create_silence(PAUSE_INTRA_MS / 1000.0, target_sr))
            local_parts.append(trimmed)
        if not local_parts:
            raise RuntimeError(f"Local TTS produced no chunks for {lang_tag}")
        local_results[unit.index] = (
            np.concatenate(local_parts) if len(local_parts) > 1 else local_parts[0]
        )

    # Phase 4: assemble in original order with policy gaps.
    all_units = sorted(units, key=lambda u: u.index)
    final_audio = None
    previous: _Unit | None = None
    for unit in all_units:
        seg = edge_results.get(unit.index)
        if seg is None:
            seg = local_results.get(unit.index)
        if seg is None or getattr(seg, 'size', 0) == 0:
            raise RuntimeError(f"Missing audio for segment {unit.index}: {unit.text[:50]!r}")
        gap = _gap_between(previous, unit)
        if previous is not None and DICTIONARY_MODE and previous.lang != unit.lang:
            gap = max(gap, _learning_gap(unit, previous.lang))
        final_audio = _join(final_audio, seg, target_sr, gap_ms=gap)
        previous = unit

    if final_audio is None or len(final_audio) == 0:
        raise RuntimeError("No valid audio was produced")

    # Trim leading/trailing silence but keep a short natural pre-roll.
    abs_final = np.abs(final_audio)
    active = np.where(abs_final > SILENCE_THRESHOLD)[0]
    if len(active) == 0:
        raise RuntimeError("Assembled audio is fully silent")
    start = max(0, int(active[0]) - int(target_sr * 0.02))
    end = min(len(final_audio), int(active[-1]) + int(target_sr * 0.05))
    final_audio = _fade_edges(final_audio[start:end], target_sr, MICRO_FADE_MS)

    logger.info(
        f"[ROUTER OK] Edge:{len(edge_results)} | Local:{len(local_results)} | "
        f"Units:{len(all_units)} | {len(final_audio) / target_sr:.1f}s"
    )
    return target_sr, final_audio


__all__ = [
    "GENERATION_PRESETS", "DEFAULT_GENERATION_PRESET",
    "generation_settings", "apply_generation_settings", "explicit_break_ms",
    "gentle_trim_and_fade", "crossfade_join", "process_multilingual_text",
    "squeeze_internal_silence",
    "get_edge_audio", "clear_segment_cache", "auto_split_mixed_languages",
    "smart_split", "create_silence", "normalize_audio", "resample_audio",
]