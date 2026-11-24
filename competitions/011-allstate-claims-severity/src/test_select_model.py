import json
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import select_model

from kumite.crossfit import split_roles
from kumite.experiments import ExperimentRecord
from kumite.provenance import array_sha256, file_sha256, files_sha256
from kumite.runs import RunStore


@pytest.fixture
def runs(tmp_path, monkeypatch):
    train = pd.DataFrame({"id": range(10), "loss": np.arange(10.0), "feature": range(10)})
    x = train[["feature"]]
    y = train["loss"].to_numpy()
    version, splits = select_model.make_splits(x)
    data = tmp_path / "data"
    data.mkdir()
    for name in select_model.FILES:
        (data / name).touch()
    monkeypatch.setattr(select_model, "ROOT", tmp_path)
    monkeypatch.setattr(select_model, "load", lambda _: (train, train, train))
    store = RunStore(tmp_path)
    for run_id in select_model.CANDIDATES:
        record = ExperimentRecord(
            competition_id="011",
            run_id=run_id,
            data_version="test",
            split_version=version,
            data_hash=files_sha256(data / name for name in select_model.FILES),
            split_hash=array_sha256(split_roles(len(y), splits)),
            seed=select_model.SEED,
            metric="MAE",
            greater_is_better=False,
            hypothesis="",
            changed_one_thing="",
            model="test",
            params={"variant": run_id},
            fold_scores=[0.0] * 5,
            oof_score=0.0,
            code_version="test",
        )
        with store.reserve(run_id) as reservation:
            store.publish(reservation, record, lambda out: np.save(out / "oof.npy", y))
    return store, y


@pytest.mark.parametrize("problem", ["shape", "nan", "inf", "score", "hash"])
def test_invalid_oof_is_rejected(runs, problem):
    store, y = runs
    record = store.record("E00")
    path = store.artifact_path("E00", "oof.npy")
    pred = y.copy()
    if problem == "shape":
        pred = pred[:, None]
    elif problem in {"nan", "inf"}:
        pred[0] = float(problem)
    else:
        pred[0] += 1
    np.save(path, pred)
    if problem != "hash":
        record.artifacts["oof.npy"] = file_sha256(path)
        records = [record, *store.records()[1:]]
        store.log_path.write_text(
            "".join(json.dumps(asdict(row)) + "\n" for row in records), encoding="utf-8"
        )
    expected = {"shape": "形状", "nan": "NaN", "inf": "Inf", "score": "OOF MAE", "hash": "ハッシュ"}
    with pytest.raises(ValueError, match=expected[problem]):
        select_model.select(list(select_model.CANDIDATES))


@pytest.mark.parametrize("ids", [["E00", "E01", "E02", "E03"], ["E00"] * 5])
def test_incomplete_or_duplicate_candidates_are_rejected(ids):
    with pytest.raises(ValueError, match=r"E00〜E04|重複"):
        select_model.select(ids)


def test_cli_writes_new_record_and_refuses_overwrite(runs, tmp_path, monkeypatch):
    store, _ = runs
    output = tmp_path / "review.json"
    monkeypatch.setattr(
        sys, "argv", ["select_model.py", *select_model.CANDIDATES, "--output", str(output)]
    )
    select_model.main()
    content = output.read_bytes()
    result = json.loads(content)
    assert result["selected"] == "E00"
    assert result["data_hash"] == store.record("E00").data_hash
    assert result["split_hash"] == store.record("E00").split_hash
    assert result["oof_sha256"] == {row.run_id: row.artifacts["oof.npy"] for row in store.records()}
    with pytest.raises(SystemExit, match="1"):
        select_model.main()
    assert output.read_bytes() == content
