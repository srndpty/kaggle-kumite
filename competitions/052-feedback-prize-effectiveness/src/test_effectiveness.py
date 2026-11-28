import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from export import write_notebook
from model import (
    CLASSES,
    ESSAY,
    ID,
    ROLE,
    SLUG,
    TARGET,
    TEXT,
    fit_model,
    other_discourses,
)
from train import check_candidates, group_keys, logloss, make_splits, run

from kumite.kaggle_notebook import validate_cpu_notebook
from kumite.provenance import file_sha256
from kumite.quality_gates import SelectionPlan
from kumite.runs import RunStore


def test_context_excludes_target_and_other_essays() -> None:
    frame = pd.DataFrame(
        {ID: ["a", "b", "c"], ESSAY: ["x", "x", "y"], TEXT: ["one", "two", "three"]}
    )
    assert other_discourses(frame) == ["two", "one", ""]


def test_group_keys_join_duplicate_essay_text(tmp_path: Path) -> None:
    for key, text in (("x", " Ａ statement "), ("y", "a   statement"), ("z", "other essay")):
        (tmp_path / f"{key}.txt").write_text(text, encoding="utf-8")
    others = [str(i) for i in range(12)]
    for key in others:
        (tmp_path / f"{key}.txt").write_text(f"unique essay {key}", encoding="utf-8")
    frame = pd.DataFrame({ESSAY: ["x", "x", "y", "z", *others]})
    groups, _ = group_keys(frame, tmp_path)
    assert groups[0] == groups[1] == groups[2] != groups[3]
    labels = np.arange(len(groups)) % 3
    splits, _ = make_splits(labels, groups)
    assert all(set(groups[fit]).isdisjoint(groups[valid]) for fit, valid in splits)


def test_logloss_uses_explicit_class_order() -> None:
    scores = np.asarray([[0.2, 0.7, 0.1], [0.1, 0.3, 0.6]])
    np.testing.assert_allclose(logloss(np.array([0, 2]), scores), -np.log(0.2 * 0.6) / 2)


def test_context_notebook_matches_local_predictions(tmp_path: Path) -> None:
    data = tmp_path / "input" / "competitions" / SLUG
    data.mkdir(parents=True)
    train = pd.DataFrame(
        {
            ID: list("abcdef"),
            ESSAY: ["x", "x", "y", "y", "z", "z"],
            TEXT: [
                "claim support",
                "support example",
                "claim evidence",
                "evidence example",
                "claim review",
                "review example",
            ],
            ROLE: ["Claim", "Evidence"] * 3,
            TARGET: list(CLASSES) * 2,
        }
    )
    test = train.iloc[:3].drop(columns=TARGET).copy()
    sample = pd.DataFrame({ID: ["b", "c", "a"], **dict.fromkeys(CLASSES, 0.0)})
    train.to_csv(data / "train.csv", index=False)
    test.to_csv(data / "test.csv", index=False)
    sample.to_csv(data / "sample_submission.csv", index=False)
    fitted = fit_model(train, "context")
    assert fitted.n_iter and not fitted.convergence_warning
    predict = fitted.predict
    fake_labels = test.assign(**{TARGET: "ignored"})
    np.testing.assert_array_equal(predict(test), predict(fake_labels))
    notebook = tmp_path / "export"
    submitted = tmp_path / "submission.csv"
    kernel = "test/arguments"
    source = write_notebook(
        notebook,
        "context",
        kernel,
        train_sha256=file_sha256(data / "train.csv"),
        input_root=tmp_path / "input",
        submission=submitted,
    )
    actual_code = validate_cpu_notebook(
        notebook, source, kernel_id=kernel, title="arguments", competition=SLUG
    )
    subprocess.run([sys.executable, "-c", actual_code], check=True, timeout=60)
    actual = pd.read_csv(submitted)
    expected = pd.DataFrame(predict(test), index=test[ID], columns=list(CLASSES)).loc[sample[ID]]
    assert list(actual.columns) == [ID, *CLASSES]
    assert actual[ID].tolist() == ["b", "c", "a"]
    np.testing.assert_allclose(actual[list(CLASSES)], expected[list(CLASSES)], atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual[list(CLASSES)].sum(axis=1), 1, atol=1e-12, rtol=0)
    submitted.unlink()
    train.assign(**{TEXT: "modified"}).to_csv(data / "train.csv", index=False)
    failure = subprocess.run(
        [sys.executable, "-c", actual_code], capture_output=True, text=True, timeout=60
    )
    assert failure.returncode != 0
    assert "Training data hash mismatch" in failure.stderr
    assert not submitted.exists()


@pytest.mark.parametrize(
    ("trial", "run_id"),
    [
        ("E00", "unknown"),
        ("E01", "E00-final"),
        ("E01", "E01-final"),
    ],
)
def test_cv_rejects_unregistered_trial(tmp_path: Path, trial: str, run_id: str) -> None:
    path = tmp_path / "plan.json"
    path.write_text('{"candidates":["E00-final"],"tolerance":0.001}', encoding="utf-8")
    with pytest.raises(ValueError, match="run ID"):
        run(trial, run_id, path)


def test_selection_and_export_reject_wrong_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore(tmp_path)
    monkeypatch.setattr(RunStore, "record", lambda *_: SimpleNamespace(params={"block": "role"}))
    with pytest.raises(ValueError, match="構成が一致しません"):
        check_candidates(store, SelectionPlan(("E00-final",), 0.001))
