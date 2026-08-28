# -*- coding: utf-8 -*-
"""Быстрая проверка соответствия yield-кортежей и outputs в Gradio-обёртках."""
import ast
import sys


def check_yields(path, func_name, expected):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    bad = 0
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            found = True
            for sub in ast.walk(node):
                if isinstance(sub, ast.Yield):
                    val = sub.value
                    if isinstance(val, ast.Tuple):
                        n = len(val.elts)
                    else:
                        n = 1
                    line = sub.lineno
                    status = "OK " if n == expected else "BAD"
                    if n != expected:
                        bad += 1
                    print(f"{path}:{line}  yield -> {n} values (expect {expected})  {status}")
    if not found:
        print(f"{path}: function {func_name} NOT FOUND")
        bad += 1
    return bad


def main():
    errors = 0
    # app.py: process_file_wrapper -> 8 outputs
    errors += check_yields("fb2tts/app.py", "process_file_wrapper", 8)
    # tts_tab.py: batch_tts_all_projects -> 5 outputs
    errors += check_yields("fb2tts/gr_tabs/tts_tab.py", "batch_tts_all_projects", 5)
    if errors == 0:
        print("ALL_YIELD_CHECKS_OK")
    else:
        print(f"YIELD_MISMATCHES: {errors}")
        sys.exit(1)


if __name__ == "__main__":
    main()
