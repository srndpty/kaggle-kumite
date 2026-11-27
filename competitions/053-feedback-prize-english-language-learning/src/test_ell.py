import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import train as training
from export import validate_notebook, write_notebook
from model import SLUG, TARGETS, duplicate_key, fit_model, write_submission
from train import mcrmse


def test_mcrmse_averages_column_roots() -> None:
    actual = np.zeros((2, 6))
    prediction = np.tile(np.arange(6), (2, 1))
    assert mcrmse(actual, prediction) == 2.5


@pytest.mark.parametrize("bad", [np.nan, np.inf, "3", 3 + 1j])
def test_mcrmse_rejects_invalid_values(bad: object) -> None:
    valid = np.full((2, 6), 3.0)
    invalid = np.full((2, 6), bad)
    for actual, prediction in ((valid, invalid), (invalid, valid)):
        with pytest.raises(ValueError, match="有限な実数"):
            mcrmse(actual, prediction)


def test_mcrmse_rejects_empty_and_wrong_shape() -> None:
    for invalid in (np.empty((0, 6)), np.ones((1, 5)), np.ones(6)):
        with pytest.raises(ValueError, match="６列"):
            mcrmse(invalid, invalid)


@pytest.mark.parametrize(
    ("labels", "block", "message"),
    [
        (np.ones((2, 5)), "base", "６列"),
        (np.ones((1, 6)), "base", "同じ行数"),
        (np.full((2, 6), np.nan), "base", "有限な実数"),
        (np.full((2, 6), np.inf), "base", "有限な実数"),
        (np.full((2, 6), "3"), "base", "有限な実数"),
        (np.full((2, 6), 5.5), "base", "範囲"),
        (np.ones((2, 6)), "typo", "未知の特徴量設定"),
    ],
)
def test_fit_model_rejects_invalid_inputs(labels: np.ndarray, block: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fit_model(["cat fox", "cat dog"], labels, block)


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


@pytest.fixture
def submission_data(tmp_path: Path) -> Path:
    data = tmp_path / "input" / "competitions" / SLUG
    data.mkdir(parents=True)
    train = pd.DataFrame(
        {
            "text_id": ["t1", "t2", "t3"],
            "full_text": ["cat fox", "cat dog", "dog fox and cat fox"],
        }
    )
    test = pd.DataFrame({"text_id": ["002", "001"], "full_text": ["cat fox", "dog fox"]})
    sample = pd.DataFrame({"text_id": ["001", "002"]})
    for i, target in enumerate(TARGETS):
        train[target] = [2.0, 3.0 + i / 10, 4.0]
        sample[target] = 0.0
    train.to_csv(data / "train.csv", index=False)
    test.to_csv(data / "test.csv", index=False)
    sample.to_csv(data / "sample_submission.csv", index=False)
    return data


def test_submission_preserves_sample_order_and_checks_ids(
    tmp_path: Path, submission_data: Path
) -> None:
    output = tmp_path / "submission.csv"
    write_submission(submission_data, output, "base")
    submitted = pd.read_csv(output, dtype={"text_id": str})
    assert list(submitted.columns) == ["text_id", *TARGETS]
    assert submitted["text_id"].tolist() == ["001", "002"]
    assert ((submitted[list(TARGETS)] >= 1) & (submitted[list(TARGETS)] <= 5)).all().all()
    sample_path = submission_data / "sample_submission.csv"
    sample = pd.read_csv(sample_path)
    sample["text_id"] = ["002", "003"]
    sample.to_csv(sample_path, index=False)
    with pytest.raises(ValueError, match="ID"):
        write_submission(submission_data, output, "base")


def test_generated_notebook_matches_local_inference(tmp_path: Path, submission_data: Path) -> None:
    notebook = tmp_path / "notebook"
    working = tmp_path / "working"
    working.mkdir()
    submission = working / "submission.csv"
    kernel = "test/ell"
    code = write_notebook(
        notebook, "bigram", kernel, input_root=tmp_path / "input", submission=submission
    )
    actual_code = validate_notebook(notebook, code, kernel)
    subprocess.run([sys.executable, "-c", actual_code], check=True, cwd=working, timeout=60)
    local = tmp_path / "local.csv"
    write_submission(submission_data, local, "bigram")
    actual = pd.read_csv(submission, dtype={"text_id": str})
    expected = pd.read_csv(local, dtype={"text_id": str})
    assert list(actual.columns) == ["text_id", *TARGETS]
    assert actual["text_id"].tolist() == expected["text_id"].tolist() == ["001", "002"]
    predictions = actual[list(TARGETS)].to_numpy()
    assert np.isfinite(predictions).all()
    assert ((predictions >= 1) & (predictions <= 5)).all()
    np.testing.assert_allclose(predictions, expected[list(TARGETS)].to_numpy(), rtol=0, atol=1e-12)

    metadata_path = notebook / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["enable_internet"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="実行設定"):
        validate_notebook(notebook, code, kernel)
