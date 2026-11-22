"""実験の取り違えと Notebook の証跡欠落を検出する。"""

import json
from dataclasses import replace
from pathlib import Path

import external
import kaggle_notebook
import numpy as np
import pandas as pd
import pytest
from dataset import SEED, SPLIT_VERSION, train_content_hash
from evidence import METRIC, checked_oof, notebook_sources, source_hashes
from metrics import IDENTITIES, score
from model import binary_loss, soft_binary_examples

from kumite.experiments import ExperimentRecord
from kumite.provenance import array_sha256
from kumite.runs import RunStore


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("trial", "trial"),
        ("metric", "評価契約"),
        ("direction", "評価契約"),
        ("oof", "OOF スコア"),
        ("fold", "fold ハッシュ"),
    ],
)
def test_checked_oof_rejects_mismatched_evidence(tmp_path: Path, fault: str, message: str) -> None:
    target = np.tile([False, True, False, True], 3)
    group = np.tile([False, False, True, True], 3)
    labels = np.column_stack([target, *([group] * len(IDENTITIES))])
    prediction = target.astype(float)
    original_folds = np.repeat(np.arange(3, dtype=np.int8), 4)
    ids = pd.Series(np.arange(len(target)), name="id")
    store = RunStore(tmp_path)
    record = ExperimentRecord(
        competition_id="042",
        run_id="E03-r2",
        metric=METRIC,
        greater_is_better=True,
        data_version="official-v1",
        data_hash="data",
        split_version=SPLIT_VERSION,
        split_hash=array_sha256(original_folds),
        seed=SEED,
        hypothesis="",
        changed_one_thing="",
        model="",
        params={"trial": "E03"},
        fold_scores=[],
        oof_score=score(labels, prediction),
        code_version="fixture",
    )
    if fault == "trial":
        record = replace(record, params={"trial": "E02"})
    if fault == "metric":
        record = replace(record, metric="ROC AUC")
    if fault == "direction":
        record = replace(record, greater_is_better=False)

    def write(out: Path) -> None:
        # RunStore が成果物をハッシュした時点ですでに別実験が混ざっている状況。
        folds = np.roll(original_folds, 1) if fault == "fold" else original_folds
        pd.DataFrame({"id": ids, "fold": folds}).to_csv(out / "folds.csv", index=False)
        np.save(out / "oof.npy", 1 - prediction if fault == "oof" else prediction)

    with store.reserve(record.run_id) as reservation:
        store.publish(reservation, record, write)
    with pytest.raises(ValueError, match=message):
        checked_oof(store, store.record(record.run_id), "data", record.split_hash, ids, labels)


def test_soft_bce_matches_weighted_binary_bce() -> None:
    target = np.array([0, 0.25, 0.5, 0.9, 1])
    logits = np.array([-100.0, -1.0, 0, 2.0, 100])
    rows, outcomes, weights = soft_binary_examples(target)
    actual = np.bincount(rows, weights=weights * binary_loss(logits[rows], outcomes))
    np.testing.assert_allclose(actual, binary_loss(logits, target), atol=1e-14)


@pytest.mark.parametrize("column", ["comment_text", "target", IDENTITIES[0]])
def test_train_content_hash_tracks_input_columns(column: str) -> None:
    row = {"id": 1, "comment_text": "Text", "target": 0.2, **dict.fromkeys(IDENTITIES, np.nan)}
    frame = pd.DataFrame([row])
    changed = frame.assign(**{column: "Changed" if column == "comment_text" else 0.8})
    assert train_content_hash(frame) != train_content_hash(changed)


@pytest.mark.parametrize("manifest_present", [False, True])
def test_notebook_rejects_missing_or_changed_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest_present: bool
) -> None:
    monkeypatch.setattr(kaggle_notebook, "ROOT", tmp_path)
    store = RunStore(tmp_path)
    record = ExperimentRecord(
        competition_id="042",
        run_id="E01",
        metric=METRIC,
        greater_is_better=True,
        data_version="official-v1",
        data_hash="data",
        split_version=SPLIT_VERSION,
        split_hash="folds",
        seed=SEED,
        hypothesis="",
        changed_one_thing="",
        model="",
        params={"trial": "E01"},
        fold_scores=[],
        oof_score=0.5,
        code_version="fixture",
    )

    def write(out: Path) -> None:
        if manifest_present:
            sources = notebook_sources()
            sources["model.py"] += "\n# 学習時のソース\n"
            (out / "notebook_evidence.json").write_text(
                json.dumps({"source_sha256": source_hashes(sources)}), encoding="utf-8"
            )

    with store.reserve(record.run_id) as reservation:
        store.publish(reservation, record, write)
    with pytest.raises(ValueError, match="ソース" if manifest_present else "証跡"):
        kaggle_notebook.export("owner", "E01", tmp_path / "notebook")


@pytest.mark.parametrize("split", ["public", "private"])
def test_external_refuses_changed_data_before_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split: str
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "selection.json").write_text(
        json.dumps(
            {
                "baseline": "E01",
                "selected": "E03",
                "data_hash": "original",
                "submission_hash": "submission",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(external, "files_sha256", lambda _: "changed")
    from types import SimpleNamespace

    monkeypatch.setattr(
        RunStore, "record", lambda *_: SimpleNamespace(artifacts={"submission.csv": "submission"})
    )
    with pytest.raises(ValueError, match="公式データ"):
        external.evaluate(tmp_path, split)
