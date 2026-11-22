"""実験コードを埋め込んだ非公開提出用 Notebook を生成する。"""

import argparse
import json
from hashlib import sha256
from pathlib import Path

from dataset import ID
from select_model import read_oof, verify_oof
from train import ROOT, RUN_VARIANTS, FrozenSplit, prepare

from kumite.experiments import find_record
from kumite.runs import RunStore

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
import json
import numpy as np
from dataset import TARGET
from kumite.provenance import array_sha256
from model import FIXED, classify, evaluate, learn_thresholds
from train import prepare, submission

print("Python", python_version())
print({package: version(package) for package in ("catboost", "numpy", "pandas", "scikit-learn", "scipy")})
assignments, ids, sample, x, query, data_hash, train_hash = prepare(root, variant, frozen=frozen)
y = assignments[TARGET].to_numpy()
folds = assignments["fold"].to_numpy()
print("split_hash", array_sha256(folds))
# 提出時の閾値は全学習行の OOF で決める。内側 CV はローカルの評価にだけ使う。
fit_variant = "baseline" if variant == "threshold" else variant
result = evaluate(x, y, folds, query, fit_variant, workers=2)
thresholds = learn_thresholds(y, result.oof[:, 0]) if variant == "threshold" else FIXED
submission(sample, ids, classify(result.query, thresholds)).to_csv("submission.csv", index=False)
np.save("test_scores.npy", result.query)
Path("replay.json").write_text(json.dumps({
    "source_run": source_run, "bundle_sha256": bundle_sha256,
    "train_hash": train_hash, "data_hash": data_hash, "split_hash": array_sha256(folds),
    "thresholds": thresholds.tolist(),
    "versions": {package: version(package) for package in ("catboost", "numpy", "pandas", "scikit-learn", "scipy")},
    "python": python_version(),
}, indent=2), encoding="utf-8")
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", choices=["dummy", *RUN_VARIANTS])
    parser.add_argument("output", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--local", action="store_true", help="ローカルデータを入力として固定する")
    args = parser.parse_args()
    mount = (
        f"from pathlib import Path\nroot = Path({str(args.data.resolve())!r})\n"
        if args.local
        else INPUT
    )
    source = mount + DUMMY
    metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    if args.run != "dummy":
        store = RunStore(ROOT)
        records = store.records()
        baseline = find_record(records, "E01")
        selected = find_record(records, args.run)
        frame = read_oof(store, baseline)
        official, public_ids, _, _, _, data_hash, train_hash = prepare(args.data, "baseline")
        verify_oof(baseline, frame, official, data_hash)
        if selected != baseline:
            verify_oof(selected, read_oof(store, selected), official, data_hash)
        frozen: FrozenSplit = {
            "pet_ids": frame[ID].tolist(),
            "public_test_ids": public_ids.tolist(),
            "folds": frame["fold"].tolist(),
            "split_hash": baseline.split_hash,
            "train_hash": train_hash,
            "data_hash": data_hash,
            # Kaggle の採点時にはテスト一式が公式仕様により差し替わる。
            "allow_replaced_test": not args.local,
        }
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
        bundle = sha256(json.dumps([embedded, PREDICT], sort_keys=True).encode()).hexdigest()
        metadata["kumite"] = {
            "source_run": selected.run_id,
            "source_code_version": selected.code_version,
            "bundle_sha256": bundle,
            "baseline_oof_sha256": baseline.artifacts["oof.csv"],
            "source_oof_sha256": selected.artifacts["oof.csv"],
            "train_hash": train_hash,
            "data_hash": data_hash,
            "split_hash": baseline.split_hash,
        }
        source = (
            "import sys\nfrom pathlib import Path\nfrom tempfile import mkdtemp\n\n"
            "package = Path(mkdtemp())\n"
            f"files = {embedded!r}\n"
            "for name, contents in files.items():\n"
            "    path = package / name\n"
            "    path.parent.mkdir(parents=True, exist_ok=True)\n"
            "    path.write_text(contents, encoding='utf-8')\n"
            "import hashlib, json\n"
            f"predict = {PREDICT!r}\n"
            f"bundle_sha256 = {bundle!r}\n"
            "if hashlib.sha256(json.dumps([files, predict], sort_keys=True).encode()).hexdigest() != bundle_sha256:\n"
            "    raise ValueError('Notebook のコードハッシュが一致しません')\n"
            "sys.path.insert(0, str(package))\n"
            f"variant = {selected.data_version!r}\n"
            f"source_run = {selected.run_id!r}\n"
            f"frozen = {dict(frozen)!r}\n"
            "exec(predict)\n"
        )
        source = mount + source
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
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
