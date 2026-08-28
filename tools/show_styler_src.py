# -*- coding: utf-8 -*-
"""Показать фрагмент исходника Dataframe.postprocess вокруг обработки Styler."""
import inspect
import re

from gradio.components.dataframe import Dataframe

src = inspect.getsource(Dataframe.postprocess)
lines = src.splitlines()
for i, line in enumerate(lines):
    if "styler" in line.lower():
        lo = max(0, i - 6)
        hi = min(len(lines), i + 10)
        print(f"--- context around line {i} ---")
        for j in range(lo, hi):
            print(f"{j:4d} {lines[j]}")
        print()
