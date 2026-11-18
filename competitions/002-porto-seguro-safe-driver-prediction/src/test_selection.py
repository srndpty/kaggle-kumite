"""実験名の取り違えや設定不整合を選定時に拒否する。"""

from dataclasses import replace

import numpy as np
import pytest
import select_model
from pipeline import EXPERIMENTS

from kumite.experiments import ExperimentRecord
from kumite.provenance import files_sha256
from kumite.runs import RunStore


@pytest.mark.parametrize(
    "corruption",
    [
        None,
        "experiment",
        "parent",
        "weighted",
        "data_version",
        "model",
        "seed",
        "param_seed",
    ],
)
def test_selection_validates_saved_definition(tmp_path, monkeypatch, corruption) -> None:
    source = tmp_path / "source.csv"
    source.write_text("data", encoding="utf-8")
    data_hash = files_sha256((source,))
    store = RunStore(tmp_path)
    monkeypatch.setattr(select_model, "STORE", store)
    monkeypatch.setattr(select_model, "DATA_FILES", (source,))
    monkeypatch.setattr(select_model, "COMPETITION_DIR", tmp_path)
    for experiment, score in (("E01", 0.2), ("E04", 0.25), ("E06", 0.26)):
        for seed in (42, 43, 44):
            exp = EXPERIMENTS[experiment]
            run_id = experiment + ("" if seed == 42 else f"-s{seed}")
            record = ExperimentRecord(
                competition_id="002",
                run_id=run_id,
                metric="normalized_gini",
                greater_is_better=True,
                data_version=exp.data_version,
                data_hash=data_hash,
                split_version=f"seed{seed}",
                split_hash=str(seed),
                seed=seed,
                hypothesis=exp.hypothesis,
                changed_one_thing=exp.change,
                model=exp.model,
                params={
                    "experiment": experiment,
                    "parent": exp.parent,
                    "weighted": exp.weighted,
                    "seed": seed,
                },
                fold_scores=[score] * 5,
                oof_score=score,
                code_version="test",
            )
            if run_id == "E06" and corruption:
                if corruption == "experiment":
                    # E06 という名前で E01 を実行した状況を再現する。
                    wrong = EXPERIMENTS["E01"]
                    record = replace(
                        record,
                        model=wrong.model,
                        data_version=wrong.data_version,
                        params={
                            "experiment": "E01",
                            "parent": wrong.parent,
                            "weighted": wrong.weighted,
                            "seed": seed,
                        },
                    )
                elif corruption in ("parent", "weighted", "param_seed"):
                    key = "seed" if corruption == "param_seed" else corruption
                    value = {"parent": "E00", "weighted": True, "param_seed": 99}[corruption]
                    record = replace(record, params={**record.params, key: value})
                elif corruption == "seed":
                    record = replace(record, seed=99)
                elif corruption == "model":
                    record = replace(record, model="prior")
                else:
                    record = replace(record, data_version="wrong")

            def write(out) -> None:
                np.save(out / "oof.npy", np.array([0.1, 0.9]))
                (out / "submission.csv").write_text("id,target\n1,0.5\n", encoding="utf-8")

            with store.reserve(run_id) as reservation:
                store.publish(reservation, record, write)
    if corruption:
        with pytest.raises(ValueError, match="実験定義"):
            select_model.select()
        assert not (tmp_path / "outputs" / "analysis" / "selection.json").exists()
    else:
        result = select_model.select()
        assert result["selected_run"] == "E06"
        assert (tmp_path / "outputs" / "analysis" / "selection.json").exists()
