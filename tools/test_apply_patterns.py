#!/usr/bin/env python3
"""
Unit tests for gr_tabs/parse_tab.apply_patterns()

Run: .venv\Scripts\python.exe tools\test_apply_patterns.py

Tests: deletion, replacement via =>, Whole line mode, bad regex,
empty pattern list, Cyrillic & Hebrew patterns.
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gr_tabs.parse_tab import apply_patterns


def test_deletion():
    content = "Line one with garbage\nLine two clean\nLine three garbage"
    patterns = "garbage"
    result, report = apply_patterns(content, patterns, False, False)
    assert "garbage" not in result, f"Expected 'garbage' removed, got: {result}"
    assert "removed" in report, f"Expected 'removed' in report, got: {report}"
    print("  PASS: test_deletion")


def test_replace_with_arrow():
    content = "Hello world\nHello there\nGoodbye world"
    patterns = "Hello => Hi"
    result, report = apply_patterns(content, patterns, False, False)
    assert "Hello" not in result, f"Expected 'Hello' replaced, got: {result}"
    assert "Hi world" in result, f"Expected 'Hi world', got: {result}"
    assert "replaced" in report, f"Expected 'replaced' in report, got: {report}"
    print("  PASS: test_replace_with_arrow")


def test_whole_line():
    content = "Keep this line\nDelete this garbage line\nAlso keep\nLine with garbage inside"
    patterns = "garbage"
    result, report = apply_patterns(content, patterns, True, False)
    assert "garbage" not in result, f"Expected no 'garbage' lines, got: {result}"
    assert "Keep this line" in result, f"Expected 'Keep this line' kept, got: {result}"
    print("  PASS: test_whole_line")


def test_bad_regex():
    content = "Some text here"
    patterns = "[invalid(regex"
    result, report = apply_patterns(content, patterns, False, True)
    assert result == content, f"Bad regex should not modify content, got: {result}"
    assert "Bad regex" in report, f"Expected 'Bad regex' in report, got: {report}"
    print("  PASS: test_bad_regex")


def test_empty_patterns():
    content = "Some text"
    patterns = ""
    result, report = apply_patterns(content, patterns, False, False)
    assert result == content, f"Empty patterns should not modify content"
    assert "No patterns" in report, f"Expected 'No patterns' msg"
    print("  PASS: test_empty_patterns")


def test_cyrillic_and_hebrew():
    content = "<p>отвэзетэ,</p>\n<p>שלום</p>\n<p>normal text</p>"
    patterns = "<p>отвэзетэ,</p>\n<p>שלום</p>"
    result, report = apply_patterns(content, patterns, False, False)
    assert "отвэзетэ" not in result, f"Cyrillic not removed: {result}"
    assert "שלום" not in result, f"Hebrew not removed: {result}"
    assert "normal text" in result, f"Normal text should remain: {result}"
    print("  PASS: test_cyrillic_and_hebrew")


def test_multiple_patterns():
    content = "foo bar baz\nfoo qux baz\nhello world"
    patterns = "foo\nbaz"
    result, report = apply_patterns(content, patterns, False, False)
    assert "foo" not in result, f"Expected 'foo' removed"
    assert "baz" not in result, f"Expected 'baz' removed"
    assert "hello world" in result, f"Expected 'hello world' kept"
    assert "2 patterns" in report or "patterns" in report
    print("  PASS: test_multiple_patterns")


def test_not_saved_note():
    content = "some text with junk"
    patterns = "junk"
    result, report = apply_patterns(content, patterns, False, False)
    assert "Not saved yet" in report, f"Should warn about saving"
    print("  PASS: test_not_saved_note")


def test_regex_mode():
    content = "abc123def\nabc456def\nxyz789uvw"
    patterns = r"abc\d+def"
    result, report = apply_patterns(content, patterns, False, True)
    assert "abc123def" not in result, f"Expected regex match removed"
    assert "abc456def" not in result, f"Expected regex match removed"
    assert "xyz789uvw" in result, f"Expected non-match kept: {result}"
    print("  PASS: test_regex_mode")


if __name__ == "__main__":
    print("Testing apply_patterns()...\n")
    failures = 0
    tests = [
        test_deletion,
        test_replace_with_arrow,
        test_whole_line,
        test_bad_regex,
        test_empty_patterns,
        test_cyrillic_and_hebrew,
        test_multiple_patterns,
        test_not_saved_note,
        test_regex_mode,
    ]
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL: {t.__name__} - {e}")
            failures += 1
        except Exception as e:
            print(f"  ERROR: {t.__name__} - {e}")
            failures += 1
    print(f"\n{'='*40}")
    if failures:
        print(f"❌ {failures} test(s) FAILED")
        sys.exit(1)
    else:
        print(f"✅ All {len(tests)} tests PASSED")
        sys.exit(0)
