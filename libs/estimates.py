# libs/estimates.py — производственные оценки времени синтеза (для подсказок в UI)
"""Empirical cost model for LECTA synthesis.

The numbers come from measurements on the reference machine (RTX 3060 Laptop,
6 GB VRAM, torch 2.9 + cu128, ESpeech-TTS F5-TTS, speed 1.0, RuAccent on):

    preprocess + RuAccent            ~2 s per 1000 characters
    F5-TTS synthesis                 ~1.5 s * steps per 1000 characters
        steps  8 -> ~12 s / 1000 chars      steps 14 -> ~23 s / 1000 chars
        steps 10 -> ~21 s / 1000 chars      steps 20 -> ~32 s / 1000 chars
                                         steps 32 -> ~48 s / 1000 chars

Cloud voices (Microsoft Edge) cost no GPU time at all: roughly 1-2 s per
request, so generation depends mostly on the network.

These helpers are used only for UI hints — never for scheduling or control
flow, so a wrong estimate can never break a job.
"""

from __future__ import annotations

#: Seconds of GPU work per inference step per 1000 characters (F5-TTS).
SECONDS_PER_STEP_PER_KCHAR = 1.5
#: RuAccent + text normalisation cost per 1000 characters.
ACCENT_SECONDS_PER_KCHAR = 2.0
#: Edge TTS: seconds of wall-clock per request (sentence-sized fragment).
EDGE_SECONDS_PER_REQUEST = 1.5
#: Average characters in one Edge request (the router splits by sentences).
EDGE_CHARS_PER_REQUEST = 120

#: Typical Russian book size used for the "what does that mean for a book"
#: column in the help overlay.
TYPICAL_BOOK_CHARS = 300_000


def tts_seconds(chars: int, steps: int = 12, *, accents: bool = True,
                cloud: bool = False) -> float:
    """Rough wall-clock seconds for synthesizing ``chars`` characters."""
    chars = max(0, int(chars))
    if cloud:
        return EDGE_SECONDS_PER_REQUEST * max(1.0, chars / EDGE_CHARS_PER_REQUEST)
    seconds = SECONDS_PER_STEP_PER_KCHAR * max(1, int(steps)) * chars / 1000.0
    if accents:
        seconds += ACCENT_SECONDS_PER_KCHAR * chars / 1000.0
    return seconds


def humanize_seconds(seconds: float) -> str:
    """'75' -> '1 h 15 min', '40' -> '40 s', '600' -> '10 min'."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{int(round(seconds))} s"
    minutes = int(round(seconds / 60.0))
    if minutes < 60:
        return f"{minutes} min"
    hours, rest = divmod(minutes, 60)
    if rest == 0:
        return f"{hours} h"
    return f"{hours} h {rest} min"


def per_kchar(steps: int, *, accents: bool = True) -> str:
    """Human label for the steps slider, e.g. '~23 s per 1000 chars'."""
    return humanize_seconds(tts_seconds(1000, steps, accents=accents))


def steps_table_rows(steps_list=None, *, chars: int = TYPICAL_BOOK_CHARS,
                     accents: bool = True):
    """Rows for the help overlay table: (steps, per 1000 chars, per book)."""
    rows = []
    for steps in (steps_list or (4, 8, 10, 14, 20, 32)):
        rows.append((
            int(steps),
            per_kchar(steps, accents=accents),
            humanize_seconds(tts_seconds(chars, steps, accents=accents)),
        ))
    return rows


def steps_hint_html(steps: int, *, accents: bool = True,
                    chars: int = TYPICAL_BOOK_CHARS) -> str:
    """Live hint shown under the inference-steps slider."""
    steps = max(1, int(steps))
    one_k = per_kchar(steps, accents=accents)
    book = humanize_seconds(tts_seconds(chars, steps, accents=accents))
    quality = ("draft quality — fastest" if steps <= 8
               else "good balance (recommended)" if steps <= 14
               else "better prosody" if steps <= 20
               else "maximum quality, slowest")
    return (
        "<div class='lecta-hint'>"
        f"<b>{steps} steps</b> &rarr; about <b>{one_k}</b> per 1000 characters &middot; "
        f"a {chars // 1000}k-character book &asymp; <b>{book}</b> &middot; {quality}"
        "</div>"
    )
