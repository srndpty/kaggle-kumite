"""選定時に不整合のある実験を拒否する。"""

from pathlib import Path

import numpy as np
import pytest
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
    params = expected_params("E00")
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
        verified_losses(store, "E00", y, "data", roles, split_hash)


def test_existing_selection_is_preserved(tmp_path: Path) -> None:
    output = tmp_path / "selection.json"
    output.write_text("historical", encoding="utf-8")
    with pytest.raises(FileExistsError):
        select(tmp_path / "data", output)
    assert output.read_text(encoding="utf-8") == "historical"
