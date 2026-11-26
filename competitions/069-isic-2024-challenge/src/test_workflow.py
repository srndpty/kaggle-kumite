from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from model import CATEGORICAL, NUMERIC, features
from sklearn.metrics import roc_auc_score
from submission import validate_submission
from validation import make_splits, pauc

from kumite.checks import check_label_independence
from kumite.crossfit import split_roles
from kumite.experiments import ExperimentRecord
from kumite.provenance import array_sha256
from kumite.quality_gates import dense_oof


def test_pauc_matches_official_conversion() -> None:
    y = np.tile([0, 1], 20)
    prediction = np.random.default_rng(42).integers(0, 6, len(y)) / 5
    scaled = roc_auc_score(1 - y, -prediction, max_fpr=0.2)
    assert pauc(y, prediction) == pytest.approx(0.02 + (scaled - 0.5) * 0.36)
    assert pauc(y, np.full(len(y), 0.5)) == pytest.approx(0.02)
    assert pauc(y, y.astype(float)) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="両クラス"):
        pauc(np.zeros(len(y)), prediction)


def test_patient_split_and_label_independence() -> None:
    y = np.tile([0, 1], 20)
    patients = np.repeat(np.arange(20), 2)
    frame = pd.DataFrame(
        {
            **{name: np.arange(len(y), dtype=float) for name in NUMERIC},
            **{name: ["value"] * len(y) for name in CATEGORICAL},
            "patient_id": patients,
            "iddx_1": np.where(y, "malignant", "benign"),
        }
    )
    splits = make_splits(frame, y)
    assert ((split_roles(len(y), splits) == 2).sum(axis=0) == 1).all()
    for train, valid in splits:
        assert not set(patients[train]) & set(patients[valid])
        changed = y.copy()
        changed[valid] = 1 - changed[valid]
        check_label_independence(
            lambda labels: [
                features(frame.assign(target=labels), "E02")
                .select_dtypes(include="number")
                .to_numpy()
            ],
            y,
            changed,
        )
        pd.testing.assert_frame_equal(
            features(frame, "E02").iloc[valid],
            features(frame.iloc[valid], "E02"),
        )


def test_submission_rejects_wrong_ids() -> None:
    sample = pd.DataFrame({"isic_id": ["a", "b"], "target": [0.5, 0.5]})
    validate_submission(sample, sample)
    with pytest.raises(ValueError, match="提出 ID"):
        validate_submission(sample.iloc[::-1].reset_index(drop=True), sample)


def test_oof_rejects_missing_prediction(tmp_path: Path) -> None:
    y = np.array([0, 1, 0, 1])
    first, second = np.array([0, 1]), np.array([2, 3])
    splits = [(first, second), (second, first)]
    path = tmp_path / "oof.npy"
    np.save(path, [0.1, 0.9, np.nan, 0.8])
    record = ExperimentRecord(
        competition_id="069",
        run_id="B00",
        metric="pAUC_TPR80",
        greater_is_better=True,
        data_version="test",
        data_hash="test",
        split_version="patient-v1",
        split_hash=array_sha256(split_roles(len(y), splits)),
        seed=42,
        hypothesis="基準",
        changed_one_thing="なし",
        model="test",
        fold_scores=[0.2, 0.2],
        oof_score=0.2,
        code_version="test",
    )
    assess = dense_oof(y, splits, pauc, expected_shape=y.shape, data_hash="test")
    with pytest.raises(ValueError, match="検証行は有限値"):
        assess(lambda _: path, record)
