"""Kaggle 提出用 Notebook とメタデータの生成。"""

import json
from pathlib import Path


def _payloads(
    source: str, kernel_id: str, title: str, competition: str
) -> tuple[tuple[str, dict[str, object]], ...]:
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
    metadata: dict[str, object] = {
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
    return (("submit.ipynb", notebook), ("kernel-metadata.json", metadata))


def write_cpu_notebook(
    output: Path, source: str, *, kernel_id: str, title: str, competition: str
) -> None:
    """非公開・インターネット無効の提出 Notebook を書き出す。"""
    output.mkdir(parents=True, exist_ok=True)
    for name, value in _payloads(source, kernel_id, title, competition):
        (output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")


def validate_cpu_notebook(
    output: Path, source: str, *, kernel_id: str, title: str, competition: str
) -> str:
    """生成したコードと実行設定を照合する。"""
    for name, expected in _payloads(source, kernel_id, title, competition):
        actual = json.loads((output / name).read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"Notebook のコードまたは実行設定が不正です: {name}")
    compile(source, "submit.ipynb", "exec")
    return source
