"""採点、グループ分割、閾値と提出形式の回帰テスト。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dataset import GROUP, ID, TARGET, connected_groups, image_features
from model import FIXED, classify, fold_thresholds, make_folds, qwk
from PIL import Image
from select_model import paired_interval
from train import submission

from kumite.checks import check_label_independence, shift_labels
from kumite.crossfit import folds_to_splits


def test_qwk_and_rounding() -> None:
    y = np.arange(5)
    assert qwk(y, y) == 1
    assert qwk(y, y[::-1]) < qwk(y, np.array([1, 0, 2, 3, 4]))
    np.testing.assert_array_equal(classify(np.r_[-1, FIXED, 5], FIXED), [0, 1, 2, 3, 4, 4])


def test_image_missing_and_connected_groups(tmp_path: Path) -> None:
    frame = pd.DataFrame({ID: ["a", "b", "c"], GROUP: ["x", "y", "z"]})
    paths = [tmp_path / f"{pet}-1.jpg" for pet in ("a", "b")]
    for path in paths:
        Image.new("L", (12, 6), 30).save(path)
    features, hashes, unreadable = image_features(paths, frame[ID], summarize=True)
    assert unreadable == 0
    assert features["readable_images"].tolist() == [1, 1, 0]
    assert features["width"].isna().tolist() == [False, False, True]
    groups = connected_groups(frame, hashes)
    assert groups[0] == groups[1] != groups[2]


def test_outer_labels_do_not_affect_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    groups = np.repeat(np.arange(30), 5)
    y = np.tile(np.arange(5), 30)
    x = pd.DataFrame({"signal": y + np.linspace(0, 0.9, len(y))})
    folds = make_folds(y, groups, 5)
    for fit, valid in folds_to_splits(folds):
        assert not set(groups[fit]) & set(groups[valid])

    def predict(
        x: pd.DataFrame, y: np.ndarray, query: pd.DataFrame, *, regression: bool
    ) -> np.ndarray:
        return query["signal"].to_numpy() + y.mean()

    monkeypatch.setattr("model.fit_scores", predict)
    train, valid = folds_to_splits(folds)[0]

    def build(labels: np.ndarray) -> list[np.ndarray]:
        thresholds = fold_thresholds(x, labels, groups, train)
        return [thresholds, classify(x.iloc[valid]["signal"].to_numpy(), thresholds)]

    mask = folds == 0
    check_label_independence(build, y, shift_labels(y, mask, n_classes=5))


def test_submission_reorders_and_rejects_invalid_predictions() -> None:
    sample = pd.DataFrame({ID: ["b", "a"]})
    ids = pd.Series(["a", "b"], name=ID)
    result = submission(sample, ids, np.array([1, 3]))
    assert result.to_dict("list") == {ID: ["b", "a"], TARGET: [3, 1]}
    with pytest.raises(ValueError):
        submission(sample, ids, np.array([1, 1.2]))


def test_bootstrap_pairs_the_same_groups() -> None:
    y = np.tile(np.arange(5), 10)
    frame = pd.DataFrame(
        {TARGET: y, "prediction": np.full(len(y), 2), "group": np.repeat(np.arange(10), 5)}
    )
    assert paired_interval(frame, frame["prediction"].to_numpy()) == [0.0, 0.0]
    assert paired_interval(frame, y) == [1.0, 1.0]
