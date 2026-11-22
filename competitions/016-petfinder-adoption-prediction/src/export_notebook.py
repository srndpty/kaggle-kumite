"""実験コードを埋め込んだ非公開提出用 Notebook を生成する。"""

import argparse
import json
from pathlib import Path

from train import ROOT, TRIALS

INPUT = """
from pathlib import Path

mounts = [parent / "petfinder-adoption-prediction"
          for parent in (Path("/kaggle/input"), Path("/kaggle/input/competitions"))]
roots = [mount for mount in mounts if (mount / "test" / "sample_submission.csv").is_file()]
if len(roots) != 1:
    raise ValueError(f"公式データのマウントを特定できません: {mounts}")
root = roots[0]
"""

DUMMY = """
import pandas as pd

sample = pd.read_csv(root / "test" / "sample_submission.csv")
if sample.empty or sample["PetID"].isna().any() or not sample["PetID"].is_unique:
    raise ValueError("sample_submission の PetID が無効です")
sample.assign(AdoptionSpeed=2).to_csv("submission.csv", index=False)
"""

PREDICT = """
from importlib.metadata import version
from platform import python_version
from dataset import TARGET
from kumite.provenance import array_sha256
from model import FIXED, classify, evaluate, learn_thresholds
from train import prepare, submission

print("Python", python_version())
print({package: version(package) for package in ("catboost", "numpy", "pandas", "scikit-learn", "scipy")})
assignments, ids, sample, x, query, _ = prepare(root, variant)
y = assignments[TARGET].to_numpy()
folds = assignments["fold"].to_numpy()
print("split_hash", array_sha256(folds))
# 提出時の閾値は全学習行の OOF で決める。内側 CV はローカルの評価にだけ使う。
fit_variant = "baseline" if variant == "threshold" else variant
result = evaluate(x, y, folds, query, fit_variant, workers=2)
thresholds = learn_thresholds(y, result.oof[:, 0]) if variant == "threshold" else FIXED
submission(sample, ids, classify(result.query, thresholds)).to_csv("submission.csv", index=False)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=TRIALS)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = INPUT + DUMMY
    if args.variant != "dummy":
        repo = ROOT.parents[1]
        shared = (
            "__init__",
            "checks",
            "crossfit",
            "cv_run",
            "experiments",
            "paths",
            "provenance",
            "runs",
            "text_features",
        )
        files = {
            **{f"{name}.py": ROOT / "src" / f"{name}.py" for name in ("dataset", "model", "train")},
            **{f"kumite/{name}.py": repo / "src" / "kumite" / f"{name}.py" for name in shared},
        }
        embedded = {name: path.read_text(encoding="utf-8") for name, path in files.items()}
        source = (
            "import sys\nfrom pathlib import Path\nfrom tempfile import mkdtemp\n\n"
            "package = Path(mkdtemp())\n"
            f"files = {embedded!r}\n"
            "for name, contents in files.items():\n"
            "    path = package / name\n"
            "    path.parent.mkdir(parents=True, exist_ok=True)\n"
            "    path.write_text(contents, encoding='utf-8')\n"
            "sys.path.insert(0, str(package))\n"
            f"variant = {args.variant!r}\n" + PREDICT
        )
        source = INPUT + source
    cell = {
        "cell_type": "code",
        "source": source.splitlines(keepends=True),
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "id": "submission",
    }
    notebook = {
        "cells": [cell],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
