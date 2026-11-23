"""Kaggle の非公開 CPU Notebook を生成する。"""

import argparse
import json
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = "commonlitreadabilityprize"

DUMMY = """
from pathlib import Path

import pandas as pd

data = next(path.parent for path in Path('/kaggle/input').rglob('sample_submission.csv')
            if (path.parent / 'train.csv').is_file())
submission = pd.read_csv(data / 'sample_submission.csv')
submission['target'] = pd.read_csv(data / 'train.csv')['target'].mean()
submission.to_csv('submission.csv', index=False)
"""


def build(output: Path, owner: str, trial: str, source: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    notebook = {
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
    title = f"Kumite CommonLit {trial}"
    metadata = {
        "id": f"{owner}/{title.lower().replace(' ', '-')}",
        "title": title,
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


def inference_source(trial: str) -> str:
    spec = find_spec("kumite.text_features")
    if spec is None or spec.origin is None:
        raise RuntimeError("kumite.text_features が見つかりません")
    features = Path(spec.origin).read_text(encoding="utf-8")
    model = (ROOT / "src" / "model.py").read_text(encoding="utf-8-sig")
    model = model.replace("from kumite.text_features import build_features", "")
    return f"""
{features}

{model}

data = next(path.parent for path in Path('/kaggle/input').rglob('sample_submission.csv')
            if all((path.parent / name).is_file() for name in FILES))
train, test, sample = load(data)
pred, _ = predict(train['excerpt'].tolist(), train['target'].to_numpy(),
                  test['excerpt'].tolist(), {trial!r})
write_submission(test, sample, pred, Path('submission.csv'))
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner")
    parser.add_argument("output", type=Path)
    parser.add_argument("--trial", default="E00")
    args = parser.parse_args()
    source = DUMMY if args.trial == "E00" else inference_source(args.trial)
    build(args.output, args.owner, args.trial, source)
