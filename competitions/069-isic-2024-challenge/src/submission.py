"""検証済みモデルから提出 Notebook と検査用サンプルを作る。"""

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from model import features, fit, predict


def validate_submission(output: pd.DataFrame, sample: pd.DataFrame) -> None:
    columns = ["isic_id", "target"]
    if list(output.columns) != columns or list(sample.columns) != columns:
        raise ValueError("提出列が isic_id,target と一致しません")
    ids = output["isic_id"]
    if ids.empty or ids.isna().any() or not ids.is_unique or not ids.equals(sample["isic_id"]):
        raise ValueError("提出 ID の件数、内容、順序が公式サンプルと一致しません")
    p = output["target"].to_numpy(dtype=float)
    if not (np.isfinite(p) & (p >= 0) & (p <= 1)).all():
        raise ValueError("提出確率が不正です")


def submission(
    model: lgb.Booster, test: pd.DataFrame, sample: pd.DataFrame, variant: str
) -> pd.DataFrame:
    prediction = pd.Series(predict(model, features(test, variant)), index=test["isic_id"])
    if not prediction.index.is_unique or set(prediction.index) != set(sample["isic_id"]):
        raise ValueError("テスト病変 ID とサンプルが一致しません")
    output = sample.assign(target=prediction.reindex(sample["isic_id"]).to_numpy())
    validate_submission(output, sample)
    return output


def notebook(variant: str, digest: str) -> dict[str, object]:
    model_code = Path(__file__).with_name("model.py").read_text(encoding="utf-8-sig")
    submit_code = Path(__file__).read_text(encoding="utf-8-sig")
    submit_code = submit_code[
        submit_code.index("def validate_submission(") : submit_code.index("def notebook(")
    ]
    entry = f"""
from pathlib import Path
import hashlib

root = Path('/kaggle/input')
def one(name):
    paths = list(root.rglob(name))
    if len(paths) != 1:
        raise ValueError(f'Expected one {{name}}, found {{len(paths)}}')
    return paths[0]

model_path = one('model.txt')
if hashlib.sha256(model_path.read_bytes()).hexdigest() != {digest!r}:
    raise ValueError('Model SHA-256 mismatch')
model = lgb.Booster(model_file=str(model_path))
test = pd.read_csv(one('test-metadata.csv'))
sample = pd.read_csv(one('sample_submission.csv'))
submission(model, test, sample, {variant!r}).to_csv('submission.csv', index=False)
"""
    cells = [
        {
            "cell_type": "code",
            "id": name,
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": code.splitlines(keepends=True),
        }
        for name, code in (("model", model_code), ("submission", submit_code), ("entry", entry))
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_export(
    path: Path,
    train: pd.DataFrame,
    y: np.ndarray,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    variant: str,
) -> None:
    model_path = path / "model.txt"
    fit(features(train, variant), y, variant).save_model(str(model_path))
    model = lgb.Booster(model_file=str(model_path))
    submission(model, test, sample, variant).to_csv(path / "sample-predictions.csv", index=False)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    (path / "submit.ipynb").write_text(json.dumps(notebook(variant, digest)), encoding="utf-8")


def validate_export(path: Path, sample: pd.DataFrame) -> None:
    validate_submission(pd.read_csv(path / "sample-predictions.csv"), sample)
    payload = json.loads((path / "submit.ipynb").read_text(encoding="utf-8"))
    cells = payload["cells"]
    if payload["nbformat"] != 4 or [cell["id"] for cell in cells] != [
        "model",
        "submission",
        "entry",
    ]:
        raise ValueError("提出 Notebook の構造が不正です")
    for cell in cells:
        compile("".join(cell["source"]), cell["id"], "exec")
