"""保存された CV と選定を使う経路の異常系。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import train as pipeline
import zillow

from kumite.crossfit import split_roles
from kumite.experiments import ExperimentRecord
from kumite.provenance import array_sha256
from kumite.runs import RunStore


@pytest.fixture
def evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[pd.DataFrame, RunStore]:
    store = RunStore(tmp_path)
    monkeypatch.setattr(pipeline, "STORE", store)
    monkeypatch.setattr(pipeline, "data_hash", lambda: "official")
    dates = pd.date_range("2016-12-01", "2017-08-01", freq="MS").repeat(2)
    raw = pd.DataFrame({zillow.ID: np.arange(len(dates)), zillow.DATE: dates, zillow.TARGET: 0.0})
    path = tmp_path / "transactions.csv"
    raw.to_csv(path, index=False)
    monkeypatch.setattr(zillow, "TRANSACTIONS", {2016: path})
    splits = zillow.time_splits(raw[zillow.DATE])
    roles = split_roles(len(raw), splits)
    valid = (roles == 2).any(axis=0)
    for run_id, score in (("E02", 0.5), ("E04", 0.25)):
        pred = np.where(valid, score, np.nan)
        record = ExperimentRecord(
            competition_id="14",
            run_id=run_id,
            metric="MAE",
            greater_is_better=False,
            data_version=pipeline.feature_version(run_id),
            data_hash="official",
            split_version=pipeline.SPLIT_VERSION,
            split_hash=array_sha256(roles),
            seed=42,
            hypothesis="",
            changed_one_thing="",
            model="regression_l1",
            params=pipeline.trial_params(run_id),
            fold_scores=[score] * 4,
            oof_score=score,
            code_version="test",
        )

        def write(out: Path, oof: np.ndarray = pred) -> None:
            np.save(out / "oof.npy", oof)
            np.save(out / "split_roles.npy", roles)

        with store.reserve(run_id) as reservation:
            store.publish(reservation, record, write)
    return raw, store


def test_final_checks_selection_and_current_data(
    tmp_path: Path,
    evidence: tuple[pd.DataFrame, RunStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, _ = evidence
    path = tmp_path / "selection.json"
    pipeline.select(raw, ["E02", "E04"], path)
    original = path.read_bytes()
    assert pipeline.verify_selection(raw, original).run_id == "E04"
    payload = json.loads(original)
    path.write_text(json.dumps({**payload, "selected": "E02"}), encoding="utf-8")
    with pytest.raises(ValueError, match="選定記録"):
        pipeline.final(raw, "final-test", None, path)
    path.write_bytes(original)
    monkeypatch.setattr(pipeline, "data_hash", lambda: "modified")
    with pytest.raises(ValueError, match="入力データ"):
        pipeline.final(raw, "final-test", None, path)


def change_log(store: RunStore, **changes: object) -> None:
    rows = [json.loads(line) for line in store.log_path.read_text(encoding="utf-8").splitlines()]
    rows[0].update(changes)
    store.log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.mark.parametrize("modify_file", [False, True])
def test_select_rejects_changed_oof(
    tmp_path: Path,
    evidence: tuple[pd.DataFrame, RunStore],
    modify_file: bool,
) -> None:
    raw, store = evidence
    if modify_file:
        np.save(store.artifact_path("E02", "oof.npy"), np.zeros(len(raw)))
    else:
        change_log(store, oof_score=0.01)
    with pytest.raises(ValueError, match=r"ハッシュ|保存済み MAE"):
        pipeline.select(raw, ["E02", "E04"], tmp_path / "selection.json")


def test_analyze_rejects_other_split_and_reordered_rows(
    evidence: tuple[pd.DataFrame, RunStore],
) -> None:
    raw, store = evidence
    reordered = raw.iloc[[1, 0, *range(2, len(raw))]]
    with pytest.raises(ValueError, match="取引行の順序"):
        pipeline.analyze(reordered, "E02", "E04")
    change_log(store, split_hash="different")
    with pytest.raises(ValueError, match="分割"):
        pipeline.analyze(raw, "E02", "E04")
