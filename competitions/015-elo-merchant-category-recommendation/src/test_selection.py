"""採用判定が実験契約と保存済み成果物を照合することを確かめる。"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import select_model
from train import make_folds, rmse

from kumite.bootstrap import BootstrapDiff
from kumite.experiments import ExperimentRecord
from kumite.provenance import array_sha256
from kumite.runs import RunStore


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    y = np.tile([-33.0, 1.0, 3.0], 5)
    cards = pd.DataFrame(
        {"card_id": list(range(len(y))), "target": y, "first_active_month": "2017-01"}
    )
    monkeypatch.setattr(select_model, "load_cards", lambda _: (cards, cards, cards))
    monkeypatch.setattr(select_model, "files_sha256", lambda _: "official")
    monkeypatch.setattr(
        select_model, "group_bootstrap_diff", lambda *_: BootstrapDiff(-1, -2, 1, 0.8)
    )
    store = RunStore(tmp_path)

    def save(run_id, seed=42, **changes):
        trial = run_id.split("-")[0]
        prediction = y + (2 if trial == "E01" else 1)
        record = ExperimentRecord(
            competition_id="015",
            run_id=run_id,
            metric="RMSE",
            greater_is_better=False,
            data_version="official-v1",
            data_hash="official",
            split_version=f"card-extreme-v1-seed{seed}",
            split_hash=array_sha256(make_folds(y, seed)),
            seed=seed,
            hypothesis="fixture",
            changed_one_thing=trial,
            model="LightGBM",
            params={"trial": trial, "num_leaves": 31},
            fold_scores=[rmse(y, prediction)] * 5,
            oof_score=rmse(y, prediction),
            code_version="fixture",
        )

        def write(out):
            np.save(out / "oof.npy", prediction)
            (out / "submission.csv").write_text("card_id,target\na,1\n", encoding="utf-8")

        with store.reserve(run_id) as reservation:
            store.publish(reservation, replace(record, **changes), write)

    return store, save


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"oof_score": 0.5}, "記録済み OOF RMSE"),
        ({"params": {"trial": "E03"}}, "実験定義"),
        ({"metric": "MAE"}, "評価条件"),
        ({"greater_is_better": True}, "評価条件"),
        ({"seed": 43}, "seed"),
    ],
)
def test_rejects_inconsistent_record(recorded, changes, message):
    store, save = recorded
    save("E01")
    save("E05", **changes)
    with pytest.raises(ValueError, match=message):
        select_model.select(store.competition_dir, ["E01", "E05"], None)


def test_rejects_tampered_oof(recorded):
    store, save = recorded
    save("E01")
    save("E05")
    path = store.artifact_path("E05", "oof.npy")
    prediction = np.load(path)
    prediction[0] += 1
    np.save(path, prediction)
    with pytest.raises(ValueError, match="ハッシュ"):
        select_model.select(store.competition_dir, ["E01", "E05"], None)


@pytest.mark.parametrize("num_leaves", [31, 63])
def test_repeat_contract(recorded, num_leaves):
    store, save = recorded
    save("E01")
    save("E05")
    save("E01-s43", seed=43)
    save("E05-s43", seed=43, params={"trial": "E05", "num_leaves": num_leaves})
    runs = ["E01", "E05"]
    repeat = ["E01-s43", "E05-s43"]
    if num_leaves == 31:
        assert select_model.select(store.competition_dir, runs, repeat)["final_run"] == "E05"
    else:
        with pytest.raises(ValueError, match="seed 以外"):
            select_model.select(store.competition_dir, runs, repeat)
