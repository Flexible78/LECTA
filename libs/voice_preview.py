"""Voice preview for the LECTA sidebar.

Plays a short phrase in the selected voice/model without creating
a full project. Caches results to tmp/voice_preview/<model>__<speaker>.wav.

Usage:
    from libs.voice_preview import preview_voice, PREVIEW_PHRASES
"""

import logging
from pathlib import Path

from libs.utils import now_dir

logger = logging.getLogger(__name__)

# Preview phrases for each language. These are data for synthesis, NOT UI strings.
# They must contain the target language's script so the voice sounds natural.
PREVIEW_PHRASES = {
    "ru": "Здравствуйте! Приятно познакомиться. Меня зовут Лекта.",
    "en": "Hello! This is how the selected voice sounds.",
    "he": "שלום! ככה נשמע הקול שבחרת.",
}

# Cache directory (cleaned by clean_tmp_folder)
_CACHE_DIR = now_dir / "tmp" / "voice_preview"


def _get_first_speaker():
    """Return the first speaker from the currently loaded model."""
    from libs.tts import synth
    sp_list = synth.speakers_list()
    if not sp_list:
        return ""
    first = sp_list[0]
    if isinstance(first, tuple):
        return first[1] if len(first) > 1 else first[0]
    return first


def _is_synthesis_busy():
    """Check if synthesis is currently running by probing the TTS semaphore.
    Returns True if synthesis appears to be active."""
    try:
        from gr_tabs.tts_tab import _tts_sema
        import config as cfg
        # If acquire(blocking=False) fails, all permits are consumed → synthesis active
        acquired = _tts_sema.acquire(blocking=False)
        if acquired:
            _tts_sema.release()
            return False
        return True
    except Exception:
        return False


def preview_voice(model_value, speaker):
    """Return (status_md, audio_gr_update) for the selected voice.

    Args:
        model_value: The model version identifier (int from tts_state).
        speaker: Selected speaker name/ID. If empty, uses first speaker from model.

    Returns:
        (status_markdown_str, gradio.Audio.update_or_None)
    """
    import gradio as gr
    import numpy as np
    from pydub import AudioSegment, effects
    from libs.tts import synth

    # Determine model version
    model_value_int = None
    try:
        model_value_int = int(model_value) if model_value else None
    except (TypeError, ValueError):
        pass

    if model_value_int is None:
        return ("⚠️ Select a TTS model first", gr.update())

    # Prevent parallel GPU access
    if _is_synthesis_busy():
        return (
            "⚠️ Busy — synthesis is running. Try again when it finishes.",
            gr.update(),
        )

    # Detect language for preview phrase
    # Model IDs: 7=Silero_en(en), all others are Russian
    lang = "en" if model_value_int == 7 else "ru"
    phrase = PREVIEW_PHRASES.get(lang, PREVIEW_PHRASES["en"])

    # Resolve speaker: use first from model if empty or invalid
    spk = speaker if speaker and str(speaker).strip() else _get_first_speaker()
    safe_speaker = str(spk).replace("/", "_").replace("\\", "_").replace(":", "_")

    cache_key = f"{model_value_int}__{safe_speaker}__{lang}"
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"{cache_key}.wav"

    # Check cache
    if cache_file.exists():
        return (
            "✅ Ready (cached)",
            gr.update(value=str(cache_file), autoplay=True, visible=True),
        )

    # Load model if needed
    if synth.ver != model_value_int:
        try:
            synth.load(model_value_int)
        except Exception as e:
            logger.exception("Failed to load model for preview")
            return (f"❌ Failed to load model: {e}", gr.update())

    # Synthesize
    try:
        res = synth.synth_audio(phrase, speaker_id=spk, speed=1.0, noise=16)
        if res is None:
            return ("❌ Preview failed: synthesis returned None", gr.update())

        a, b = res
        if isinstance(a, (int, float)):
            sr, aud_raw = int(a), b
        elif isinstance(b, (int, float)):
            sr, aud_raw = int(b), a
        else:
            sr, aud_raw = 24000, a

        if aud_raw is None or getattr(aud_raw, "size", 0) == 0:
            return ("❌ Preview failed: empty audio", gr.update())

        np_audio = np.nan_to_num(aud_raw, nan=0.0, posinf=0.0, neginf=0.0)
        np_audio = np.clip(np_audio, -1.0, 1.0)
        audio_int16 = (np_audio * 32767).astype(np.int16)
        audio_seg = AudioSegment(
            audio_int16.tobytes(), frame_rate=sr, sample_width=2, channels=1
        )
        audio_seg = effects.normalize(audio_seg)
        audio_seg.export(str(cache_file), format="wav")

        return (
            "✅ Ready",
            gr.update(value=str(cache_file), autoplay=True, visible=True),
        )
    except Exception as e:
        logger.exception("Voice preview failed")
        return (f"❌ Preview failed: {e}", gr.update())
