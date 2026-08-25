"""Smoke test: parse a test project and build the 3-file package
(without running heavy TTS). Verifies worker build_package logic."""
import json
import shutil
import sys
from pathlib import Path

# UTF-8 console safety
for _s in (sys.stdout, sys.stderr):
    try:
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from libs.fb2_processor import FB2Processor
from libs.utils import data_path

PROJECT = "test_no_cover"

# 1) Parse
proc = FB2Processor()
last_pct = 0
for pct, msg in proc.process_book(ab_path=PROJECT, replace=True):
    last_pct = pct
print(f"[TEST] Parse progress: {last_pct}% {msg}")

# 2) Create fake mp3 dir + file so package has an mp3 (skip real TTS)
mp3_dir = data_path / PROJECT / "mp3"
mp3_dir.mkdir(parents=True, exist_ok=True)
sample = mp3_dir / "1.mp3"
if not sample.exists():
    # tiny silent mp3 via pydub
    from pydub import AudioSegment
    AudioSegment.silent(duration=1000).export(str(sample), format="mp3", bitrate="96")

# 3) Build package
from worker import build_package
out_dir = build_package(PROJECT)
print(f"[TEST] Package dir: {out_dir}")
if out_dir is None:
    print("[TEST] FAIL: no package dir")
    sys.exit(1)
files = sorted(p.name for p in out_dir.iterdir())
print(f"[TEST] Package files: {files}")
assert any(f.endswith(".mp3") for f in files), "no mp3"
assert any(f.endswith(".txt") for f in files), "no txt"
assert any(f.endswith(".fb2") for f in files), "no fb2"
print("[TEST] OK — 3-file package built")
