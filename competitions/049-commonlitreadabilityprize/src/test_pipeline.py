"""グループ分割・学習境界・提出契約の回帰テスト。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from model import predict, write_submission
from splits import groups_for, make_folds

from kumite.checks import check_label_independence, perturb_targets
from kumite.crossfit import FoldOutput, cross_fit
from kumite.text_features import build_features


def test_grouped_folds() -> None:
    long = "abcdefghi " * 20
    frame = pd.DataFrame(
        {
            "id": [str(i) for i in range(25)],
            "excerpt": [
                long,
                long.upper(),
                long + "abcdefghi",
                *[np.random.default_rng(i).bytes(20).hex() for i in range(22)],
            ],
            "url_legal": [
                None,
                None,
                "https://example.test/book",
                "https://example.test/book#page",
                *([None] * 21),
            ],
            "target": np.arange(25),
        }
    )
    groups = groups_for(frame)
    assert len(set(groups[:4])) == 1
    assert groups[4] != groups[0]
    folds = make_folds(frame)
    assert folds["id"].equals(frame["id"])
    assert folds.groupby("group")["fold"].nunique().max() == 1
    result = cross_fit(
        folds["fold"].to_numpy(),
        lambda _training, validation: FoldOutput(validation.astype(float), np.empty(0)),
    )
    np.testing.assert_array_equal(result.oof, np.arange(len(frame)))


def test_vocabulary_and_label_independence() -> None:
    train = ["shared word", "shared word", "shared word"]
    valid = ["validationonly"] * 3
    query = ["queryonly"] * 3
    _, vectorizers = build_features(train, valid, query)
    assert set(vectorizers[0].vocabulary_) == {"shared", "word"}
    y = np.arange(6, dtype=float)
    mask = np.arange(6) >= 3
    for trial in ("E00", "E01", "E02", "E03", "E04", "E05"):
        check_label_independence(
            lambda labels, trial=trial: predict(train, labels[~mask], valid, trial),
            y,
            perturb_targets(y, mask),
        )


def test_submission_order_and_finiteness(tmp_path: Path) -> None:
    test = pd.DataFrame({"id": ["b", "a"]})
    sample = pd.DataFrame({"id": ["a", "b"], "target": [0.0, 0.0]})
    path = tmp_path / "submission.csv"
    write_submission(test, sample, np.array([2.0, 1.0]), path)
    pd.testing.assert_frame_equal(
        pd.read_csv(path), pd.DataFrame({"id": ["a", "b"], "target": [1.0, 2.0]})
    )
    with pytest.raises(ValueError, match="有限"):
        write_submission(test, sample, np.array([np.inf, 1.0]), path)
    with pytest.raises(ValueError, match="ID"):
        write_submission(test, sample.assign(id=["a", "c"]), np.array([2.0, 1.0]), path)
