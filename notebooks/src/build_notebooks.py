"""Собирает .ipynb из исходников notebooks/src/*.py (ячейки разделены «# %%»).

Запуск из корня репозитория: python notebooks/src/build_notebooks.py
"""
import json
import re
from pathlib import Path

SRC = Path(__file__).parent
DST = SRC.parent


def cells_from(text: str):
    cells = []
    for chunk in re.split(r"^# %%", text, flags=re.M):
        if not chunk.strip():
            continue
        if chunk.startswith(" [markdown]"):
            body = chunk[len(" [markdown]"):].strip("\n")
            lines = [re.sub(r"^# ?", "", ln) for ln in body.splitlines()]
            cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)})
        else:
            code = chunk.strip("\n")
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": code})
    for c in cells:
        src = c["source"]
        c["source"] = [ln + "\n" for ln in src.split("\n")[:-1]] + [src.split("\n")[-1]]
    return cells


def write(name: str, text: str):
    nb = {"cells": cells_from(text), "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                                 "name": "python3"},
                                                  "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (DST / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("→", DST / name)


for f in sorted(SRC.glob("lab_*.py")):
    text = f.read_text(encoding="utf-8")
    if f.stem == "lab_00_data":
        kit = (DST.parent / "labkit.py").read_text(encoding="utf-8")
        first, rest = text.split("\n# %%\n", 1)
        text = (first + "\n# %%\n%%writefile /kaggle/working/labkit.py\n" + kit.rstrip("\n")
                + "\n\n# %%\n" + rest)
    if f.stem == "lab_04_vlm":
        for key, suffix in [("qwen25vl_3b", "qwen3b"), ("qwen25vl_7b", "qwen7b"), ("internvl3_2b", "internvl")]:
            write(f"lab_04_vlm_{suffix}.ipynb", text.replace('MODEL_KEY = "qwen25vl_3b"', f'MODEL_KEY = "{key}"', 1))
    else:
        write(f"{f.stem}.ipynb", text)
