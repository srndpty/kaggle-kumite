"""最終モデル選定の再検証・成果物照合を検査する。"""

from dataclasses import replace

import analyze
import pytest
import select_model

from kumite.experiments import ExperimentRecord
from kumite.runs import RunStore


@pytest.mark.parametrize("second_delta, expected", [(0.01, "E02"), (-0.01, "E01")])
def test_requires_both_seeds(tmp_path, monkeypatch, second_delta, expected):
    store = RunStore(tmp_path)
    monkeypatch.setattr(select_model, "files_sha256", lambda _: "data")
    monkeypatch.setattr(analyze, "STORE", store)
    monkeypatch.setattr(select_model, "STORE", store)
    monkeypatch.setattr(select_model, "COMPETITION_DIR", tmp_path)
    for seed, delta in ((42, 0.01), (43, second_delta)):
        for experiment, score in (("E01", 0.8), ("E02", 0.8 + delta)):
            run_id = experiment + ("" if seed == 42 else "-s43")
            record = ExperimentRecord(
                competition_id="041",
                run_id=run_id,
                metric="macro_auc",
                greater_is_better=True,
                data_version="v1",
                data_hash="data",
                split_version=str(seed),
                split_hash=str(seed),
                seed=seed,
                hypothesis="h",
                changed_one_thing="one",
                model="lr",
                params={
                    "experiment": experiment,
                    "parent": "E00" if experiment == "E01" else "E01",
                },
                fold_scores=[score] * 3,
                oof_score=score,
                runtime_sec=1.0,
                code_version="code",
            )

            def write(out):
                (out / "submission.csv").write_text("id,p\na,0.5\n", encoding="utf-8")

            with store.reserve(run_id) as reservation:
                store.publish(reservation, record, write)
    assert select_model.select("E02", "E01")["selected"] == expected
    with pytest.raises(FileExistsError):
        select_model.select("E02", "E01")


def test_comparison_rejects_other_metric(monkeypatch):
    first = ExperimentRecord(
        competition_id="041",
        run_id="a",
        metric="macro_auc",
        greater_is_better=True,
        data_version="v1",
        data_hash="data",
        split_version="s",
        split_hash="s",
        seed=42,
        hypothesis="h",
        changed_one_thing="one",
        model="lr",
        fold_scores=[0.8] * 3,
        oof_score=0.8,
        code_version="code",
    )

    class Store:
        def record(self, run_id):
            return first if run_id == "a" else replace(first, metric="logloss", run_id="b")

    monkeypatch.setattr(analyze, "STORE", Store())
    with pytest.raises(ValueError):
        analyze.compare(["a", "b"])
