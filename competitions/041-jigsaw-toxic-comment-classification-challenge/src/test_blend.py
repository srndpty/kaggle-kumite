"""後処理で元runの分割・列順・成果物を取り違えない。"""

from dataclasses import replace

import blend
import numpy as np
import pandas as pd
import pytest
from dataset import LABELS
from pipeline import train

from kumite.experiments import ExperimentRecord
from kumite.provenance import array_sha256
from kumite.runs import RunStore


@pytest.mark.parametrize("corruption", [None, "labels", "split", "artifact"])
def test_source_integrity(tmp_path, monkeypatch, corruption):
    folds = np.array([0, 1, 2, 0, 1, 2])
    ids = np.array([f"id{i}" for i in range(6)])
    params = {"experiment": "E03", "parent": "E01", "labels": list(LABELS)}
    record = ExperimentRecord(
        competition_id="041",
        run_id="E03",
        metric="macro_auc",
        greater_is_better=True,
        data_version="v1",
        data_hash="data",
        split_version="text-group-multilabel-3fold-v1-s42",
        split_hash=array_sha256(folds),
        seed=42,
        hypothesis="h",
        changed_one_thing="one",
        model="tfidf_lr",
        params=params,
        fold_scores=[0.8] * 3,
        oof_score=0.8,
        code_version="code",
    )
    if corruption == "labels":
        record = replace(record, params={**params, "labels": list(reversed(LABELS))})
    if corruption == "split":
        record = replace(record, split_hash="other")

    def write(out):
        np.save(out / "oof.npy", np.full((6, 6), 0.5))
        np.save(out / "test.npy", np.full((2, 6), 0.5))
        pd.DataFrame({"id": ids, "fold": folds}).to_csv(out / "folds.csv", index=False)

    store = RunStore(tmp_path)
    with store.reserve("E03") as reservation:
        store.publish(reservation, record, write)
    monkeypatch.setattr(blend, "STORE", store)
    if corruption == "artifact":
        np.save(store.runs_dir / "E03" / "oof.npy", np.zeros((6, 6)))

    def load():
        return blend.load_source("E03", "E03", 42, "data", folds, ids, 2)

    if corruption:
        with pytest.raises(ValueError):
            load()
    else:
        assert load()[0].shape == (6, 6)


def test_training_cannot_silently_run_blend_as_lr():
    with pytest.raises(ValueError, match=r"blend\.py"):
        train("E06", "E06")
