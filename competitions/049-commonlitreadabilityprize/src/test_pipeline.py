"""グループ分割・学習境界・提出契約の回帰テスト。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import train as pipeline
from model import write_submission
from splits import groups_for, make_folds
from train import rmse, run_cv, select

from kumite.checks import perturb_targets
from kumite.crossfit import FoldOutput, cross_fit
from kumite.cv_run import Evaluation
from kumite.runs import RunStore
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


def test_validation_text_does_not_change_vocabulary_or_idf() -> None:
    train = ["shared trainingonly"] * 3 + ["shared word"] * 3
    _, baseline = build_features(train, train, train)
    _, changed = build_features(train, ["shared validationonly"] * 3, ["queryonly"] * 3)
    assert set(changed[0].vocabulary_) == {"shared", "trainingonly", "word"}
    assert changed[0].vocabulary_ == baseline[0].vocabulary_
    np.testing.assert_array_equal(changed[0].idf_, baseline[0].idf_)


@pytest.fixture
def cv_inputs(tmp_path: Path) -> tuple[RunStore, pd.DataFrame, pd.DataFrame, Evaluation]:
    train = pd.DataFrame(
        {
            "id": [f"row-{i}" for i in range(12)],
            "excerpt": ["shared training text"] * 6 + ["shared validation text"] * 6,
            "target": np.arange(12, dtype=float),
        }
    )
    folds = pd.DataFrame({"id": train["id"], "group": np.arange(12), "fold": [0] * 6 + [1] * 6})
    return (
        RunStore(tmp_path),
        train,
        folds,
        Evaluation("049", "RMSE", False, rmse, "d", "v", "s", 42),
    )


def test_fixed_cv_is_independent_of_validation_labels(
    cv_inputs: tuple[RunStore, pd.DataFrame, pd.DataFrame, Evaluation],
) -> None:
    store, train, folds, evaluation = cv_inputs
    mask = folds["fold"].eq(0).to_numpy()
    changed = train.assign(target=perturb_targets(train["target"].to_numpy(), mask))
    run_cv(store, train, folds, evaluation, "E04", "E04-original")
    run_cv(store, changed, folds, evaluation, "E04", "E04-perturbed")
    original = store.load_array("E04-original", "oof.npy")
    perturbed = store.load_array("E04-perturbed", "oof.npy")
    np.testing.assert_array_equal(original[mask], perturbed[mask])
    assert not np.array_equal(original[~mask], perturbed[~mask])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("metric", "MAE", "評価契約"),
        ("greater_is_better", True, "評価契約"),
        ("params", {"trial": "E02", "alpha": 1.0, "solver": "lsqr"}, "trial"),
        ("params", {"trial": "E01", "alpha": 2.0, "solver": "lsqr"}, "モデル設定"),
        ("oof_score", 0.0, "OOF RMSE"),
        ("folds", None, "保存した fold"),
    ],
)
def test_select_rejects_inconsistent_evidence(
    cv_inputs: tuple[RunStore, pd.DataFrame, pd.DataFrame, Evaluation],
    field: str,
    value: object,
    message: str,
) -> None:
    store, train, folds, evaluation = cv_inputs
    run_cv(store, train, folds, evaluation, "E01", "E01-test")
    run_cv(store, train, folds, evaluation, "E04", "E04-test")
    runs = ["E01-test", "E04-test"]
    select(store, train, folds, evaluation, runs)
    if field == "folds":
        folds = folds.assign(group=folds["group"] + 1)
    else:
        records = [
            json.loads(line) for line in store.log_path.read_text(encoding="utf-8").splitlines()
        ]
        records[0][field] = value
        store.log_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
    with pytest.raises(ValueError, match=message):
        select(store, train, folds, evaluation, runs)


def test_selection_rejects_changed_inference_source(
    cv_inputs: tuple[RunStore, pd.DataFrame, pd.DataFrame, Evaluation],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, train, folds, evaluation = cv_inputs
    monkeypatch.setattr(pipeline, "ROOT", store.competition_dir)
    sources = {"text_features": "original text features", "model": "original model"}
    monkeypatch.setattr(pipeline, "inference_sources", lambda: sources)
    monkeypatch.setattr(pipeline, "load_evaluation", lambda _: (folds, evaluation))
    runs = ["E01-test", "E04-test"]
    for run in runs:
        run_cv(store, train, folds, evaluation, run.partition("-")[0], run)
    path = store.competition_dir / "selection.json"
    selected = select(store, train, folds, evaluation, runs)
    path.write_text(json.dumps(selected), encoding="utf-8")
    assert pipeline.verify_selection(path, train, sources).run_id == selected["run_id"]
    with pytest.raises(ValueError, match="推論ソース"):
        pipeline.verify_selection(path, train, {**sources, "text_features": "changed"})


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
