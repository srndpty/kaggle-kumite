"""採点疎通用の Kaggle Notebook を生成する。"""

import argparse
import json
from pathlib import Path

SLUG = "nbme-score-clinical-patient-notes"
DUMMY_SOURCE = """import csv
from pathlib import Path

(sample,) = Path('/kaggle/input').rglob('sample_submission.csv')
with sample.open(newline='', encoding='utf-8') as stream:
    rows = list(csv.DictReader(stream))
ids = [row['id'] for row in rows]
assert ids and len(set(ids)) == len(ids)
with Path('/kaggle/working/submission.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.writer(stream)
    writer.writerow(('id', 'location'))
    writer.writerows((row_id, '') for row_id in ids)
"""


def build_dummy(owner: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "id": "dummy",
                "metadata": {},
                "source": DUMMY_SOURCE.splitlines(keepends=True),
                "execution_count": None,
                "outputs": [],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    metadata = {
        "id": f"{owner}/kumite-nbme-dummy",
        "title": "Kumite NBME Dummy",
        "code_file": "submit.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": [SLUG],
    }
    for name, value in (("submit.ipynb", notebook), ("kernel-metadata.json", metadata)):
        (output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_dummy(args.owner, args.output)
