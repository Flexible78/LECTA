# -*- coding: utf-8 -*-
"""Смоук-тест _queue_styler-логики (копия логики из app.py) на установленном pandas."""
import pandas as pd


def _status_bg(status):
    s = (status or "").strip()
    if s.startswith("❌") or s.startswith("⚠️"):
        return "rgba(244,63,94,0.30)"
    if s.startswith("🔊"):
        return "rgba(56,189,248,0.22)"
    if s.startswith("⏳"):
        return "rgba(245,158,11,0.26)"
    if s.startswith("✅"):
        return "rgba(16,185,129,0.26)"
    return "rgba(148,163,184,0.10)"


rows = [
    ["book1.fb2", "2.3 MB", "✅ converted", "C:/books/book1.fb2"],
    ["book2.pdf", "1.1 MB", "🔊 45%", "C:/books/book2.pdf"],
    ["book3.epub", "3.7 MB", "⏳ converting [3/6]", "C:/books/book3.epub"],
    ["bad.doc", "0.5 MB", "❌ unsupported", "C:/books/bad.doc"],
    ["new.txt", "0.1 MB", "⏳ pending", "C:/books/new.txt"],
]

df = pd.DataFrame(rows, columns=["File", "Size", "Status", "Path"])
print("pandas:", pd.__version__)


def _paint(row):
    color = _status_bg(row["Status"])
    return [f"background-color: {color}"] * len(df.columns)


styler = df.style.apply(_paint, axis=1)
html = styler.to_html()
assert "background-color: rgba(244,63,94" in html, "red row not rendered!"
assert "background-color: rgba(16,185,129" in html, "green row not rendered!"
assert "background-color: rgba(245,158,11" in html, "amber row not rendered!"
assert "background-color: rgba(56,189,248" in html, "blue row not rendered!"

# Пустая таблица — не должна падать
empty = pd.DataFrame(columns=["File", "Size", "Status", "Path"])
assert empty.empty

print("QUEUE_STYLER_OK")
