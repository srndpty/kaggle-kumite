"""質問分割・順位採点・提出形式の回帰テスト。"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from groups import question_groups
from quest import (
    correlations,
    guarded_quantile_align,
    predict,
    prediction_diagnostics,
    quantile_align,
    require_scorable_preview,
    submission,
    validate_submission,
)
from run import notebook_source
from sklearn.model_selection import GroupKFold

from kumite.checks import check_label_independence


def test_transitive_question_groups() -> None:
    data = pd.DataFrame(
        {
            "url": ["https://one.test/questions/1/slug", "https://one.test/questions/1", "", ""],
            "question_title": ["first", "second", " SECOND ", "other"],
            "question_body": ["body", "text", "TEXT", "different"],
        }
    )
    groups = question_groups(data)
    assert groups.tolist() == [0, 0, 0, 3]
    for train, valid in GroupKFold(n_splits=2).split(data, groups=groups):
        assert not set(groups[train]) & set(groups[valid])


def test_spearman_ties_and_constants() -> None:
    y = np.array([[1, 2, 1], [1, 2, 2], [3, 2, 3]])
    pred = np.array([[0, 1, 3], [0, 1, 2], [1, 1, 1]])
    np.testing.assert_allclose(correlations(y, pred), [1, 0, -1])


@pytest.mark.parametrize("variant", ["E03", "E04", "E05"])
def test_validation_labels_are_hidden(variant: str) -> None:
    data = pd.DataFrame(
        {
            "question_title": ["alpha beta", "alpha gamma", "delta beta", "gamma beta"],
            "question_body": ["title text"] * 4,
            "answer": ["answer text"] * 4,
        }
    )
    y = np.arange(12, dtype=float).reshape(4, 3) / 12
    perturbed = y.copy()
    perturbed[-1] = 0
    check_label_independence(
        lambda labels: [predict(data.iloc[:3], data.iloc[3:], labels[:3], variant)[0]],
        y,
        perturbed,
    )


def test_submission() -> None:
    columns = ["qa_id", *(f"target_{index}" for index in range(30))]
    sample = pd.DataFrame([[1, *([0] * 30)], [2, *([0] * 30)]], columns=columns)
    test = pd.DataFrame({"qa_id": [2, 1]})
    predictions = np.stack([np.full(30, 0.2), np.full(30, 0.1)])
    actual = submission(sample, test, predictions)
    assert actual.columns.tolist() == columns
    assert actual["qa_id"].tolist() == [1, 2]
    assert actual["target_0"].tolist() == [0.1, 0.2]
    with pytest.raises(ValueError):
        submission(sample, test.iloc[:1], predictions)
    with pytest.raises(ValueError):
        submission(sample, pd.DataFrame({"qa_id": [1, 1]}), predictions)
    with pytest.raises(ValueError):
        submission(sample[columns[::-1]], test, predictions)
    with pytest.raises(ValueError):
        submission(sample, test, predictions + 1)
    for wrong in ([2.0, 1.0], ["2", "1"], [2, None]):
        with pytest.raises(ValueError, match="Invalid submission IDs"):
            submission(sample, pd.DataFrame({"qa_id": wrong}), predictions)


def test_validate_submission_checks_real_test_ids(tmp_path: Path) -> None:
    columns = ["qa_id", *(f"target_{index}" for index in range(30))]
    sample = pd.DataFrame([[1, *([0] * 30)], [2, *([0] * 30)]], columns=columns)
    path = tmp_path / "submission.csv"
    sample.to_csv(path, index=False)
    validate_submission(path, sample, pd.DataFrame({"qa_id": [2, 1]}))
    with pytest.raises(ValueError, match="Submission and test IDs differ"):
        validate_submission(path, sample, pd.DataFrame({"qa_id": [3, 1]}))
    sample.assign(target_0=float("inf")).to_csv(path, index=False)
    with pytest.raises(ValueError, match="nonfinite"):
        validate_submission(path, sample, pd.DataFrame({"qa_id": [2, 1]}))


def test_exported_notebook_source(tmp_path: Path) -> None:
    slug = "google-quest-challenge"
    inputs = tmp_path / slug
    inputs.mkdir()
    columns = [f"target_{index}" for index in range(30)]
    train = pd.DataFrame(
        {
            "qa_id": [3, 4, 5],
            "question_title": ["alpha beta", "alpha gamma", "delta beta"],
            "question_body": ["question text"] * 3,
            "answer": ["answer text"] * 3,
            **{column: [0.1, 0.2, 0.4] for column in columns},
        }
    )
    train.to_csv(inputs / "train.csv", index=False)
    train.drop(columns=columns).iloc[:2].to_csv(inputs / "test.csv", index=False)
    sample = train[["qa_id", *columns]].iloc[1::-1]
    sample.to_csv(inputs / "sample_submission.csv", index=False)
    output = tmp_path / "submission.csv"
    code = notebook_source("E01")
    patched = code.replace('"/kaggle/input"', repr(tmp_path.as_posix())).replace(
        '"/kaggle/working/submission.csv"', repr(output.as_posix())
    )
    exec(compile(patched, "submit.ipynb", "exec"), {})
    actual = pd.read_csv(output)
    assert actual.columns.equals(sample.columns)
    assert actual["qa_id"].tolist() == [4, 3]
    assert np.isfinite(actual[columns].to_numpy()).all()


def test_quantile_alignment_preserves_prediction_ties() -> None:
    labels = np.column_stack([np.array([0, 0, 0, 1, 1, 1]), np.linspace(0, 1, 6)])
    pred = np.array([[0.1, 0.2], [0.1, 0.3], [0.8, 0.4], [0.9, 0.5]])
    aligned = quantile_align(labels, pred)
    np.testing.assert_array_equal(aligned[:, 0], [0, 0, 1, 1])
    np.testing.assert_array_equal(aligned[:, 1], pred[:, 1])


@pytest.mark.parametrize("align", [quantile_align, guarded_quantile_align])
def test_quantile_alignment_is_deterministic_and_order_equivariant(
    align: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> None:
    labels = np.column_stack([np.arange(200) == 199, np.arange(200) >= 100]).astype(float)
    pred = np.column_stack([np.linspace(0.1, 0.9, 200)] * 2)
    order = np.random.default_rng(42).permutation(len(pred))
    expected = align(labels, pred)
    np.testing.assert_array_equal(align(labels, pred), expected)
    np.testing.assert_array_equal(align(labels, pred[order])[np.argsort(order)], expected)


def test_guarded_quantiles_and_preview_gate() -> None:
    labels = np.column_stack([np.arange(200) == 199, np.arange(200) >= 100]).astype(float)
    pred = np.column_stack([np.linspace(0.1, 0.9, 200)] * 2)
    original = quantile_align(labels, pred)
    guarded = guarded_quantile_align(labels, pred)
    columns = pd.Index(["rare", "balanced"])
    with pytest.raises(ValueError, match=r"rare \(99.50%\)"):
        require_scorable_preview(columns, original)
    require_scorable_preview(columns, guarded)
    np.testing.assert_array_equal(guarded[:, 1], original[:, 1])
    assert (np.diff(guarded[:, 0]) > 0).all()
    assert prediction_diagnostics(columns, guarded, labels)["undefined_count"] == 0
    constant = np.zeros_like(guarded)
    assert prediction_diagnostics(columns, constant, labels)["undefined_count"] == 2
    with pytest.raises(ValueError, match="rare"):
        require_scorable_preview(columns, constant)
