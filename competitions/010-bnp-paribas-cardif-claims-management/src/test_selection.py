"""選定時に不整合のある実験を拒否する。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import select_model
import train
from select_model import expected_params, select, verified_losses
from train import SEED, SPLIT_VERSION, score

from kumite.experiments import ExperimentRecord
from kumite.provenance import array_sha256
from kumite.runs import RunStore


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("nan", "NaN"),
        ("range", "範囲外"),
        ("shape", "形状"),
        ("score", "スコア"),
        ("data", "データハッシュ"),
        ("params", "実験設定"),
        ("split", "fold"),
        ("artifact", "ハッシュ"),
    ],
)
def test_invalid_run_is_rejected(tmp_path: Path, fault: str, message: str) -> None:
    store = RunStore(tmp_path)
    y = np.array([0, 1])
    roles = np.array([[2, 1], [1, 2]])
    pred = np.array([0.25, 0.75])
    oof_score = score(y, pred)
    split_hash = array_sha256(roles)
    params = expected_params("E00", "E00")
    if fault == "nan":
        pred[0] = np.nan
    if fault == "range":
        pred[0] = -0.25
    if fault == "shape":
        pred = pred[:, None]
    if fault == "params":
        params["n_estimators"] = 1

    def artifacts(path: Path) -> None:
        np.save(path / "oof.npy", pred)
        np.save(path / "split_roles.npy", roles)

    with store.reserve("E00") as reservation:
        output = store.publish(
            reservation,
            ExperimentRecord(
                competition_id="010",
                run_id="E00",
                metric="LogLoss",
                greater_is_better=False,
                data_version="kaggle-v1",
                data_hash="other" if fault == "data" else "data",
                split_version=SPLIT_VERSION,
                split_hash="other" if fault == "split" else split_hash,
                seed=SEED,
                hypothesis="",
                changed_one_thing="",
                model="LightGBM",
                params=params,
                fold_scores=[],
                oof_score=1.0 if fault == "score" else oof_score,
                code_version="test",
            ),
            artifacts,
        )
    if fault == "artifact":
        np.save(output / "oof.npy", np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match=message):
        verified_losses(store, "E00", y, "data", roles, split_hash, "E00")


def test_existing_selection_is_preserved(tmp_path: Path) -> None:
    output = tmp_path / "selection.json"
    output.write_text("historical", encoding="utf-8")
    with pytest.raises(FileExistsError):
        select(tmp_path / "data", output)
    assert output.read_text(encoding="utf-8") == "historical"


def test_select_retrained_runs_without_historical_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(select_model, "ROOT", tmp_path)
    pd.DataFrame(
        {
            "ID": range(60),
            "target": [0, 1] * 30,
            "value": range(60),
            "category": [None, "known", "other"] * 20,
        }
    ).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"ID": [60], "value": [60], "category": [None]}).to_csv(
        tmp_path / "test.csv", index=False
    )
    pd.DataFrame({"ID": [60], "PredictedProb": [0.5]}).to_csv(
        tmp_path / "sample_submission.csv", index=False
    )
    train.run(tmp_path, "base-replay", "E00")
    train.run(tmp_path, "missing-replay", "E03")

    output = tmp_path / "selection.json"
    with pytest.raises(ValueError, match="実験設定"):
        select(tmp_path, output, baseline_id="base-replay", candidates=[("missing-replay", "E01")])
    assert not output.exists()
    select(tmp_path, output, baseline_id="base-replay", candidates=[("missing-replay", "E03")])
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["baseline"] == "base-replay"
    assert list(record["runs"]) == ["base-replay", "missing-replay"]
    assert record["runs"]["missing-replay"]["variant"] == "E03"
