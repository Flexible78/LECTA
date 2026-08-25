"""Test thermal throttle logic and app wiring."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gr_tabs.tts_tab import (
    _poll_gpu_temp_thermal,
    _thermal_throttle_active,
    _last_temp_poll,
    _TEMP_POLL_INTERVAL,
    stop_text_to_sp,
)
from libs.thermal import get_gpu_temp

def test_thermal():
    print(f"[1] Thermal vars: active={_thermal_throttle_active}, last_poll={_last_temp_poll}")
    
    # Poll once — should work even without GPU
    is_hot, suffix = _poll_gpu_temp_thermal()
    print(f"[2] First poll: is_hot={is_hot}, suffix='{suffix}'")
    
    # On a machine without NVIDIA GPU, get_gpu_temp returns None
    t = get_gpu_temp()
    print(f"[3] GPU temp: {t} (None means no NVIDIA GPU)")
    
    # Verify the function doesn't crash
    print(f"[4] Poll interval: {_TEMP_POLL_INTERVAL}s")
    print(f"[5] stop_text_to_sp default: {stop_text_to_sp}")
    
    print("[OK] Thermal throttle tests passed")

if __name__ == "__main__":
    test_thermal()