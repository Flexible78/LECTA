"""GPU temperature monitoring for LECTA TTS.

Provides a single function get_gpu_temp() that reads the GPU temperature
via nvidia-smi, with 5-second caching. On machines without NVIDIA GPUs
or when nvidia-smi is unavailable, returns None silently.
"""

import subprocess
import time
import os
import sys

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

# Cache: (timestamp, temperature)
_temp_cache = (0.0, None)
_CACHE_TTL = 5.0  # seconds — must not poll more often than this


def get_gpu_temp() -> int | None:
    """Return GPU temperature in °C, or None if unavailable.

    Reads from nvidia-smi with a 5-second cache. On platforms without
    nvidia-smi, on timeout, or on any error, returns None silently.
    """
    global _temp_cache
    now = time.monotonic()
    if now - _temp_cache[0] < _CACHE_TTL:
        return _temp_cache[1]

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            creationflags=_CREATE_NO_WINDOW,
        )
        if result.returncode != 0:
            _temp_cache = (now, None)
            return None
        output = result.stdout.strip()
        if not output:
            _temp_cache = (now, None)
            return None
        # Take the first line (first GPU)
        temp_str = output.splitlines()[0].strip()
        temp = int(temp_str)
        _temp_cache = (now, temp)
        return temp
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, Exception):
        _temp_cache = (now, None)
        return None
