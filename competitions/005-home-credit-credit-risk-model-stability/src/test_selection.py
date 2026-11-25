"""最終選定と Notebook 生成時の証跡検査。"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pipeline import PARAMS, fold_weeks, stability
from train import METRIC, SPLIT_VERSION, TRIALS, Evaluation, export, select_final, verify_cv

from kumite.experiments import ExperimentRecord
from kumite.provenance import array_sha256
from kumite.runs import RunStore


@pytest.fixture
def candidates(tmp_path: Path) -> tuple[RunStore, Evaluation]:
    store = RunStore(tmp_path)
    weeks = np.repeat(np.arange(50), 2)
    target = np.tile([0, 1], 50)
    fold = np.full(len(weeks), -1, dtype=np.int8)
    for index, (_, valid) in enumerate(fold_weeks(weeks, target)):
        fold[valid] = index
    expected = Evaluation("data", weeks, target, fold, b"# snapshot\n")
    for trial in TRIALS:
        write_run(store, expected, trial)
    return store, expected


def write_run(
    store: RunStore, expected: Evaluation, run_id: str, *, wrong_fold: bool = False
) -> None:
    fold = expected.fold.copy()
    if wrong_fold:
        fold[fold == 0] = 1
    pred = np.where(expected.fold >= 0, expected.target, np.nan)
    scores = []
    for index in range(3):
        rows = expected.fold == index
        scores.append(stability(expected.weeks[rows], expected.target[rows], pred[rows]))
    valid = expected.fold >= 0
    pooled = stability(expected.weeks[valid], expected.target[valid], pred[valid])
    record = ExperimentRecord(
        competition_id="005",
        run_id=run_id,
        metric=METRIC,
        greater_is_better=True,
        data_version=run_id.split("-")[0],
        data_hash=expected.data_hash,
        split_version=SPLIT_VERSION,
        split_hash=array_sha256(fold),
        seed=PARAMS["seed"],
        hypothesis="",
        changed_one_thing="",
        model="LightGBM",
        params=PARAMS,
        fold_scores=[score["score"] for score in scores],
        oof_score=float(np.mean([score["score"] for score in scores])),
        code_version="test",
    )

    def save(output: Path) -> None:
        np.save(output / "oof.npy", pred)
        np.save(output / "fold.npy", fold)
        (output / "metrics.json").write_text(
            json.dumps({"folds": scores, "pooled": pooled}), encoding="utf-8"
        )
        (output / "pipeline.py").write_bytes(expected.source)

    with store.reserve(run_id) as reservation:
        store.publish(reservation, record, save)


def test_checks_all_candidates_before_using_fixed_tie_order(
    candidates: tuple[RunStore, Evaluation],
) -> None:
    store, expected = candidates
    assert select_final(store, expected, list(reversed(TRIALS))).run_id == "E00"
    path = store.runs_dir / "E04" / "oof.npy"
    pred = np.load(path, allow_pickle=False)
    pred[-1] = 0.3
    np.save(path, pred)
    with pytest.raises(ValueError, match=r"E04: oof\.npy の SHA256"):
        select_final(store, expected, list(TRIALS))


def test_rejects_different_fold_even_with_consistent_artifact_hash(
    candidates: tuple[RunStore, Evaluation],
) -> None:
    store, expected = candidates
    write_run(store, expected, "E04-other", wrong_fold=True)
    with pytest.raises(ValueError, match="fold 割当"):
        select_final(store, expected, [*list(TRIALS)[:-1], "E04-other"])


def test_export_rejects_source_change_and_submission_run(
    candidates: tuple[RunStore, Evaluation], tmp_path: Path
) -> None:
    store, expected = candidates
    selected = store.record("E04")
    with pytest.raises(ValueError, match=r"CV 時の pipeline\.py"):
        export(
            store,
            tmp_path,
            "E04",
            "owner",
            "F00",
            selected,
            expected._replace(source=b"# changed\n"),
        )
    submission = replace(selected, run_id="F00-old", fold_scores=[])
    with pytest.raises(ValueError, match="CV run の評価条件"):
        export(store, tmp_path, "E04", "owner", "F00", submission, expected)
    assert not (store.runs_dir / "F00").exists()


def test_rejects_metrics_that_do_not_reproduce(
    candidates: tuple[RunStore, Evaluation],
) -> None:
    store, expected = candidates
    record = replace(store.record("E04"), oof_score=0.5)
    with pytest.raises(ValueError, match="OOF の再計算スコア"):
        verify_cv(store, record, "E04", expected)
