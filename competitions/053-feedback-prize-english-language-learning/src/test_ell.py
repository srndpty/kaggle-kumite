from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import train as training
from model import TARGETS, duplicate_key, fit_model, write_submission
from train import mcrmse


def test_mcrmse_averages_column_roots() -> None:
    actual = np.zeros((2, 6))
    prediction = np.tile(np.arange(6), (2, 1))
    assert mcrmse(actual, prediction) == 2.5


def test_duplicate_key_normalizes_width_case_and_whitespace() -> None:
    assert duplicate_key(" Ａbc\tDEF\n") == duplicate_key("abc def")


def test_folds_keep_duplicate_essays_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    texts = [text for i in range(25) for text in (f"Essay {i}", f"  essay\t{i} ")]
    frame = pd.DataFrame({"text_id": np.arange(len(texts)), "full_text": texts})
    for target in TARGETS:
        frame[target] = np.repeat(np.linspace(1, 5, 25), 2)
    frame.to_csv(data / "train.csv", index=False)
    monkeypatch.setattr(training, "COMP", tmp_path)
    _, _, groups, splits, _ = training.load_cv()
    assert len(set(groups)) == 25
    for fit, valid in splits:
        assert set(groups[fit]).isdisjoint(groups[valid])


def test_fold_model_only_uses_training_labels() -> None:
    texts = ["fox and cat", "fox and dog", "dog and cat", "unseen word"]
    labels = np.tile(np.arange(4).reshape(-1, 1), (1, 6)).astype(float) + 1
    before = fit_model(texts[:3], labels[:3], "base")(texts[3:])
    labels[3] = 5
    after = fit_model(texts[:3], labels[:3], "base")(texts[3:])
    np.testing.assert_array_equal(before, after)


def test_submission_preserves_sample_order_and_checks_ids(tmp_path: Path) -> None:
    train = pd.DataFrame(
        {"text_id": ["t1", "t2", "t3"], "full_text": ["cat fox", "cat dog", "dog fox"]}
    )
    test = pd.DataFrame({"text_id": ["002", "001"], "full_text": ["cat fox", "dog fox"]})
    sample = pd.DataFrame({"text_id": ["001", "002"]})
    for target in TARGETS:
        train[target] = [2.0, 3.0, 4.0]
        sample[target] = 0.0
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    sample.to_csv(tmp_path / "sample_submission.csv", index=False)
    output = tmp_path / "submission.csv"
    write_submission(tmp_path, output, "base")
    submitted = pd.read_csv(output, dtype={"text_id": str})
    assert list(submitted.columns) == ["text_id", *TARGETS]
    assert submitted["text_id"].tolist() == ["001", "002"]
    assert ((submitted[list(TARGETS)] >= 1) & (submitted[list(TARGETS)] <= 5)).all().all()
    sample["text_id"] = ["002", "003"]
    sample.to_csv(tmp_path / "sample_submission.csv", index=False)
    with pytest.raises(ValueError, match="ID"):
        write_submission(tmp_path, output, "base")
