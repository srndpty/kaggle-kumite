"""実験結果の保存と読み込み。train.py と calibrate.py で共通に使う。"""

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from cv import N_SPLITS, SEED, SPLIT_VERSION
from dataset import CLASSES, COMPETITION_DIR, OUTPUT_DIR
from sklearn.metrics import log_loss

from kumite.experiments import LOG_FILE_NAME, ExperimentRecord, append_record

COMPETITION_ID = "001"
RUNS_DIR = OUTPUT_DIR / "runs"


@dataclass(frozen=True)
class Experiment:
    model: str
    hypothesis: str
    changed_one_thing: str
    features: str = "raw"
    params: dict[str, Any] = field(default_factory=dict)


def multi_logloss(y: np.ndarray | pd.Series, pred: np.ndarray) -> float:
    return float(log_loss(y, pred, labels=range(len(CLASSES))))


def save_run(
    run_id: str,
    exp: Experiment,
    y: pd.Series,
    folds: np.ndarray,
    oof: np.ndarray,
    test_pred: np.ndarray,
    test_ids: pd.Series,
    runtime_sec: float,
    fold_infos: list[dict[str, Any]],
) -> None:
    """スコアを計算して表示し、予測・提出ファイル・実験ログを書き出す。"""
    fold_scores = [multi_logloss(y[folds == k], oof[folds == k]) for k in range(N_SPLITS)]
    oof_score = multi_logloss(y, oof)
    for k, (score, info) in enumerate(zip(fold_scores, fold_infos, strict=True)):
        print(f"fold {k}: {score:.5f} {info}")
    print(f"OOF logloss: {oof_score:.5f}（fold 標準偏差 {np.std(fold_scores):.5f}）")

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "oof.npy", oof)
    np.save(run_dir / "test.npy", test_pred)
    submission = pd.DataFrame(test_pred, columns=CLASSES)
    submission.insert(0, "id", test_ids)
    submission.to_csv(run_dir / "submission.csv", index=False)
    (run_dir / "fold_info.json").write_text(json.dumps(fold_infos), encoding="utf-8")

    append_record(
        COMPETITION_DIR / LOG_FILE_NAME,
        ExperimentRecord(
            competition_id=COMPETITION_ID,
            run_id=run_id,
            data_version=exp.features,
            split_version=SPLIT_VERSION,
            seed=SEED,
            hypothesis=exp.hypothesis,
            changed_one_thing=exp.changed_one_thing,
            model=exp.model,
            params=exp.params,
            fold_scores=[round(s, 5) for s in fold_scores],
            oof_score=round(oof_score, 5),
            runtime_sec=round(runtime_sec, 1),
        ),
    )
    print(f"出力: {run_dir}")


def load_predictions(run_id: str) -> tuple[np.ndarray, np.ndarray]:
    """(OOF 予測, テスト予測) を返す。"""
    run_dir = RUNS_DIR / run_id
    return np.load(run_dir / "oof.npy"), np.load(run_dir / "test.npy")
