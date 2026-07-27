#!/usr/bin/env python3
"""
Check for user-facing Russian strings in the LECTA codebase.

Exits with code 0 if no Russian strings are found, or code 1 with a list
of matches if any are present.

Excluded directories (hardcoded):
  - dict/
  - libs/russian/
  - libs/russian.py
  - libs/accent/ruaccent/
  - libs/tts_preprocessor.py
  - libs/tts/vosk_backend/g2p.py
  - libs/tts/f5_backend/
  - libs/f5_tts_backend.py
  - libs/favicon/
  - backup/
  - venv/
  - .git/
  - __pycache__/
  - *.egg-info/
"""

import io
import os
import re
import sys
from pathlib import Path

# Fix stdout encoding on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Directories and files to exclude
EXCLUDED_DIRS = {
    "dict",
    "backup",
    "venv",
    ".git",
    "__pycache__",
    "favicon",
    "ruaccent",
    "f5_backend",
    "vosk_backend",
    "site-packages",
}

EXCLUDED_FILES = {
    "russian.py",
    "tts_preprocessor.py",
    "f5_tts_backend.py",
    "g2p.py",
    "check_cyrillic.py",
    "ui_assets.py",  # internal i18n dictionary (GRADIO_RU_EN_MAP), not user-facing
    "test_apply_patterns.py",  # test file with Cyrillic test data
}

EXCLUDED_PREFIXES = (
    str(Path("libs") / "russian"),
    str(Path("libs") / "accent" / "ruaccent"),
    str(Path("libs") / "tts" / "f5_backend"),
    str(Path("libs") / "tts" / "vosk_backend" / "g2p.py"),
    str(Path("libs") / "f5_tts_backend.py"),
    str(Path("libs") / "tts_preprocessor.py"),
    str(Path("libs") / "favicon"),
    str(Path("dict")),
)

# Pattern for Russian/Cyrillic characters
CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")


def should_exclude(file_path: Path) -> bool:
    """Check if a file should be excluded from Cyrillic checking."""
    # Check by name
    if file_path.name in EXCLUDED_FILES:
        return True
    # Check by parent directory
    for part in file_path.parts:
        if part in EXCLUDED_DIRS:
            return True
    # Check by relative path prefix
    try:
        rel = file_path.relative_to(Path.cwd())
        for prefix in EXCLUDED_PREFIXES:
            if str(rel).startswith(str(prefix)):
                return True
    except ValueError:
        pass
    return False


def is_relevant_line(line: str) -> bool:
    """Check if a line contains Russian text that is user-facing.
    
    We focus on strings that would appear in the UI:
      - gr.Info(...), gr.Warning(...), gr.Error(...)
      - gr.update(label=..., value=..., info=..., placeholder=...)
      - label=, info=, placeholder=, value= in component constructors
      - gr.Markdown(...), gr.HTML(...), gr.Button(...)
      - Returned status strings (function returns with Cyrillic)
      - Plain string literals containing Cyrillic
    """
    # Skip comments and docstrings
    stripped = line.strip()
    if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
        return False
    
    # Look for Russian characters
    if not CYRILLIC_RE.search(line):
        return False
    
    # Only flag lines that look like they might be UI strings
    ui_indicators = [
        "gr.Info(", "gr.Warning(", "gr.Error(", "gr.Markdown(",
        "gr.HTML(", "gr.update(", "label=",
        "info=", "placeholder=", "value=",
        "gr.Textbox(", "gr.Button(", "gr.Dropdown(",
        "gr.Checkbox(", "gr.Radio(", "gr.Slider(",
        "gr.DataFrame(", "gr.File(", "gr.Audio(",
        "gr.Accordion(", "gr.Group(",
    ]
    
    for indicator in ui_indicators:
        if indicator in line:
            return True
    
    # Also flag status/result strings returned by handlers
    # Pattern: return that contains Cyrillic text
    if 'return' in line and ('"' in line or "'" in line):
        return True
    
    # Flag f-strings with Cyrillic content (common in status messages)
    if ('f"' in line or "f'" in line) and CYRILLIC_RE.search(line):
        return True
    
    # Flag yield statements with Cyrillic (generator status messages)
    if 'yield' in line and CYRILLIC_RE.search(line):
        return True
    
    # Catch Cyrillic inside non-raw string literals in data structures
    # (e.g. model names in tuples/lists that feed into UI dropdowns).
    # First strip inline comments so we don't flag Cyrillic in # comments.
    # Then look for Cyrillic in regular (non-raw) string quotes.
    code_only = line.split('#')[0]
    if CYRILLIC_RE.search(code_only):
        # Non-raw double-quoted strings: "..." containing Cyrillic (not r"... or R"...)
        if re.search(r'(?<![rR])(?<![\w])"[^"]*[а-яА-ЯёЁ][^"]*"', code_only):
            return True
        # Non-raw single-quoted strings: '...' containing Cyrillic (not r'... or R'...)
        if re.search(r"(?<![rR])(?<![\w])'[^']*[а-яА-ЯёЁ][^']*'", code_only):
            return True
    
    return False


def main():
    root = Path.cwd()
    found_issues = []
    
    for py_file in sorted(root.rglob("*.py")):
        if should_exclude(py_file):
            continue
        
        rel_path = py_file.relative_to(root)
        
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        
        in_docstring = False
        for i, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            # Track multi-line docstring state.
            # - Pure """ or ''' lines toggle state (open ↔ close).
            # - """text (opening with content, no closer) sets state to True.
            # - """text.""" (single-line) does not change state.
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if stripped in ('"""', "'''"):
                    in_docstring = not in_docstring
                elif not (stripped.endswith('"""') or stripped.endswith("'''")):
                    in_docstring = True
                continue
            if in_docstring:
                # Handle closing """ at the end of a content line (e.g. "text.""")
                if stripped.endswith('"""') or stripped.endswith("'''"):
                    in_docstring = False
                continue
            
            if is_relevant_line(line):
                found_issues.append((rel_path, i, line.strip()))
    
    if found_issues:
        print("FAIL: Found Cyrillic in user-facing strings:\n")
        for path, line_no, text in found_issues:
            print(f"  {path}:{line_no}: {text[:120]}")
        print(f"\nTotal: {len(found_issues)} issue(s)")
        sys.exit(1)
    else:
        print("PASS: No user-facing Cyrillic strings found.")
        sys.exit(0)


if __name__ == "__main__":
    main()
