# -*- coding: utf-8 -*-
"""Проверка поддержки pandas Styler в gr.Dataframe установленного Gradio."""
import inspect

import gradio as gr

from gradio.components.dataframe import Dataframe

src = inspect.getsource(Dataframe.postprocess)
print("gradio version:", gr.__version__)
print("postprocess mentions Styler:", "Styler" in src)
low = src.lower()
print("postprocess mentions styler:", "styler" in low)

# Также посмотрим сигнатуру preprocess/postprocess и docstring
print("---- docstring ----")
print((Dataframe.postprocess.__doc__ or "")[:600])
