"""Kaggle 提出用 Notebook とメタデータの生成。"""

import json
from pathlib import Path


def write_cpu_notebook(
    output: Path, source: str, *, kernel_id: str, title: str, competition: str
) -> None:
    """非公開・インターネット無効の提出 Notebook を書き出す。"""
    output.mkdir(parents=True, exist_ok=True)
    notebook: dict[str, object] = {
        "cells": [
            {
                "cell_type": "code",
                "id": "inference",
                "metadata": {},
                "source": source.splitlines(keepends=True),
                "execution_count": None,
                "outputs": [],
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    metadata = {
        "id": kernel_id,
        "title": title,
        "code_file": "submit.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": [competition],
    }
    for name, value in (("submit.ipynb", notebook), ("kernel-metadata.json", metadata)):
        (output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
