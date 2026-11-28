import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
    write_submission,
)
from train import group_keys, logloss, make_splits

from kumite.kaggle_notebook import validate_cpu_notebook


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
    predict = fit_model(train, "context")
    fake_labels = test.assign(**{TARGET: "ignored"})
    np.testing.assert_array_equal(predict(test), predict(fake_labels))
    notebook = tmp_path / "export"
    submitted = tmp_path / "submission.csv"
    kernel = "test/arguments"
    source = write_notebook(
        notebook, "context", kernel, input_root=tmp_path / "input", submission=submitted
    )
    actual_code = validate_cpu_notebook(
        notebook, source, kernel_id=kernel, title="arguments", competition=SLUG
    )
    subprocess.run([sys.executable, "-c", actual_code], check=True, timeout=60)
    local = tmp_path / "local.csv"
    write_submission(data, local, "context")
    actual = pd.read_csv(submitted)
    expected = pd.read_csv(local)
    assert list(actual.columns) == [ID, *CLASSES]
    assert actual[ID].tolist() == ["b", "c", "a"]
    np.testing.assert_allclose(actual[list(CLASSES)], expected[list(CLASSES)], atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual[list(CLASSES)].sum(axis=1), 1, atol=1e-12, rtol=0)
