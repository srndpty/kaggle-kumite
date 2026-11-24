"""課題分割、指標、提出の回帰テスト。"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import train as training
from model import write_submission
from train import make_splits, mcrmse

from kumite.crossfit import split_roles
from kumite.cv_run import Evaluation


def test_prompt_splits_exclude_duplicates_and_validate_every_row() -> None:
    rows = pd.DataFrame(
        {
            "prompt_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "text": ["Same!", "alpha", "same", "beta", "gamma", "delta", "epsilon", "zeta"],
        }
    )
    splits = make_splits(rows)
    roles = split_roles(len(rows), splits)
    assert ((roles == 2).sum(axis=0) == 1).all()
    for fit, valid in splits:
        assert set(rows.iloc[fit]["prompt_id"]).isdisjoint(rows.iloc[valid]["prompt_id"])
    assert len(splits) == 4
    assert roles[0, 2] == roles[1, 0] == 0


def test_mcrmse_averages_after_square_root() -> None:
    actual = np.zeros((2, 2))
    predicted = np.array([[0, 2], [0, 4]])
    assert mcrmse(actual, predicted) == pytest.approx(np.sqrt(10) / 2)


def test_submission_order_and_values(tmp_path: Path) -> None:
    test = pd.DataFrame({"student_id": ["01", "02"]})
    sample = tmp_path / "template.csv"
    sample.write_text("student_id,content,wording\n02,0,0\n01,0,0\n", encoding="utf-8")
    output = tmp_path / "submission.csv"
    write_submission(test, np.array([[1.0, 2.0], [3.0, 4.0]]), output, sample)
    actual = pd.read_csv(output, dtype={"student_id": str})
    assert list(actual) == ["student_id", "content", "wording"]
    assert actual["student_id"].tolist() == ["02", "01"]
    assert actual["content"].tolist() == [3.0, 1.0]
    with pytest.raises(ValueError, match="提出予測"):
        write_submission(test, np.array([[float("inf"), 2.0], [3.0, 4.0]]), output, sample)


@pytest.mark.parametrize(
    ("ids", "message"),
    [
        (["01"], "提出ID集合"),
        (["01", "02", "03"], "提出ID集合"),
        (["01", "01", "02"], "student_id に欠損または重複"),
    ],
)
def test_submission_rejects_invalid_ids(tmp_path: Path, ids: list[str], message: str) -> None:
    sample = tmp_path / "template.csv"
    sample.write_text("student_id,content,wording\n01,0,0\n02,0,0\n", encoding="utf-8")
    output = tmp_path / "submission.csv"
    with pytest.raises(ValueError, match=message):
        write_submission(pd.DataFrame({"student_id": ids}), np.zeros((len(ids), 2)), output, sample)
    assert not output.exists()


@pytest.mark.parametrize(
    ("prediction", "message"),
    [
        (np.zeros((4, 1)), "OOF の形状"),
        (np.full((4, 2), np.nan), "OOF に非有限値"),
        (np.full((4, 2), np.inf), "OOF に非有限値"),
    ],
)
def test_selection_rejects_invalid_oof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prediction: np.ndarray, message: str
) -> None:
    rows = pd.DataFrame(
        {
            "prompt_id": ["a", "b", "c", "d"],
            "text": ["abc", "de", "fgh i", "j"],
            "prompt_text": ["original text"] * 4,
            "prompt_question": ["question"] * 4,
            "content": [0.0] * 4,
            "wording": [0.0] * 4,
        }
    )
    config = Evaluation("050", "MCRMSE", False, mcrmse, "data", training.PRIMARY_SPLIT, "split", 42)
    record = SimpleNamespace(
        metric=config.metric,
        greater_is_better=False,
        data_hash=config.data_hash,
        split_version=config.split_version,
        split_hash=config.split_hash,
        params={"trial": "E00"},
        oof_score=0.0,
    )
    artifact = tmp_path / "oof.npy"
    np.save(artifact, prediction)
    store = SimpleNamespace(record=lambda _: record, artifact_path=lambda *_: artifact)
    monkeypatch.setattr(training, "RunStore", lambda _: store)
    monkeypatch.setattr(training, "evaluation", lambda *_: config)
    with pytest.raises(ValueError, match=message):
        training.select(rows, ["E00"])


def test_selection_rejects_duplicate_runs() -> None:
    with pytest.raises(ValueError, match="重複"):
        training.select(pd.DataFrame(), ["E00", "E00"])
