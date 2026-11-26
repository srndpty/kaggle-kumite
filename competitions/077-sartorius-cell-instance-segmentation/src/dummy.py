"""採点疎通用の非公開 Kaggle Notebook を生成する。"""

import argparse
import json
from pathlib import Path

SLUG = "sartorius-cell-instance-segmentation"
NOTEBOOK_SOURCE = """\
import csv
from pathlib import Path

slug = "sartorius-cell-instance-segmentation"
image_dirs = sorted(path / "test" for path in Path("/kaggle/input").rglob(slug))
image_dirs = [path for path in image_dirs if path.is_dir()]
if len(image_dirs) != 1:
    raise ValueError(f"Expected one official test directory, got {image_dirs}")
ids = [path.stem for path in sorted(image_dirs[0].glob("*.png"))]
if not ids or len(set(ids)) != len(ids):
    raise ValueError("Missing or duplicate test IDs")
with Path("submission.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(("id", "predicted"))
    writer.writerows((image_id, "1 1") for image_id in ids)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": NOTEBOOK_SOURCE.splitlines(keepends=True),
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    metadata = {
        "id": f"{args.owner}/kumite-077-dummy",
        "title": "Kumite 077 Dummy",
        "code_file": "submit.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": [SLUG],
        "dataset_sources": [],
        "kernel_sources": [],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in (("submit.ipynb", notebook), ("kernel-metadata.json", metadata)):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
