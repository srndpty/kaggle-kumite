"""多ラベル分類の分割、予測と提出の回帰テスト。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from model import fit_model, log_loss, predict, write_submission
from train import make_folds


def test_group_folds_cover_all_rows() -> None:
    groups = np.repeat(np.arange(10), 3)
    folds = make_folds(groups)
    assert set(folds) == set(range(5))
    assert all(len(set(folds[groups == group])) == 1 for group in np.unique(groups))
    assert np.array_equal(folds, make_folds(groups))


def test_single_class_uses_smoothed_prior() -> None:
    y = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]])
    frame = pd.DataFrame(
        {"cp_type": ["trt_cp"] * 4, "cp_time": [24, 48] * 2, "cp_dose": ["D1"] * 4}
    )
    result = predict(frame, y, frame.iloc[:2], "E01")
    np.testing.assert_allclose(result[:, :2], [[0.1, 0.9], [0.1, 0.9]])


def test_endpoints_follow_official_clipping() -> None:
    assert log_loss(np.array([[0, 1]]), np.array([[0, 1]])) == pytest.approx(1e-15)
    assert log_loss(np.array([[1]]), np.array([[0]])) == pytest.approx(-np.log(1e-15))


def test_submission_follows_sample_order_and_zeros_controls(tmp_path: Path) -> None:
    train = pd.DataFrame(
        {
            "cp_type": ["trt_cp", "trt_cp", "ctl_vehicle"],
            "cp_time": [24, 48, 24],
            "cp_dose": ["D1", "D2", "D1"],
            "g-0": [0, 1, 0],
            "c-0": [1, 0, 1],
        },
        index=pd.Index(["a", "b", "c"], name="sig_id"),
    )
    labels = pd.DataFrame({"z": [1, 1, 0], "a": [0, 1, 0]}, index=train.index)
    y = labels.to_numpy()
    pred = predict(train, y, train, "E03")
    assert (pred[2] == 0).all()
    changed = y.copy()
    changed[-1] = 1
    np.testing.assert_allclose(predict(train, changed, train, "E03"), pred)
    train.to_csv(tmp_path / "test_features.csv")
    labels.iloc[::-1].to_csv(tmp_path / "sample_submission.csv")
    output = tmp_path / "submission.csv"
    state = fit_model(train, y, "E03")
    state["labels"] = np.asarray(labels.columns, dtype=str)
    write_submission(tmp_path, state, output)
    saved = pd.read_csv(output)
    assert saved.columns.tolist() == ["sig_id", "z", "a"]
    assert saved["sig_id"].tolist() == ["c", "b", "a"]
    np.testing.assert_allclose(saved.iloc[:, 1:].to_numpy(), pred[::-1])
