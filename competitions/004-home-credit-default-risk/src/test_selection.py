"""最良候補・別seed確認・証跡・上書き防止を実際のRunStoreで検証する。"""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import select_model
from pipeline import EXPERIMENTS, parameters

from kumite.experiments import ExperimentRecord
from kumite.provenance import files_sha256
from kumite.runs import RunStore


def fixture_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delta43: float = 0.01
) -> RunStore:
    data = tmp_path / "input.txt"
    data.write_text("input", encoding="utf-8")
    monkeypatch.setattr(select_model, "DATA_FILES", (data,))
    store = RunStore(tmp_path)
    for seed in (42, 43):
        for experiment in ("E01", "E02"):
            exp = EXPERIMENTS[experiment]
            run_id = experiment if seed == 42 else f"{experiment}-s{seed}"
            score = 0.75 + ((0.01 if seed == 42 else delta43) if experiment == "E02" else 0)
            with store.reserve(run_id) as reservation:
                record = ExperimentRecord(
                    competition_id="004",
                    run_id=run_id,
                    metric="roc_auc",
                    greater_is_better=True,
                    data_version=f"application-v1|history={exp.history}|missing_count={exp.missing_count}",
                    data_hash=files_sha256((data,)),
                    split_version=f"seed{seed}",
                    split_hash=f"hash{seed}",
                    seed=seed,
                    hypothesis=exp.hypothesis,
                    changed_one_thing=exp.change,
                    model="lightgbm_binary",
                    params={
                        **parameters(seed),
                        "experiment": experiment,
                        "definition": asdict(exp),
                        "benchmark": False,
                    },
                    fold_scores=[score] * 3,
                    oof_score=score,
                    runtime_sec=1.0,
                    code_version="fixture",
                )

                def write(output: Path) -> None:
                    np.save(output / "oof.npy", np.array([0.1, 0.9]))
                    np.save(output / "test.npy", np.array([0.3]))
                    (output / "folds.csv").write_text(
                        "SK_ID_CURR,fold\n1,0\n2,1\n", encoding="utf-8"
                    )
                    (output / "submission.csv").write_text(
                        "SK_ID_CURR,TARGET\n3,0.3\n", encoding="utf-8"
                    )
                    (output / "features.json").write_text("{}", encoding="utf-8")

                store.publish(reservation, record, write)
    return store


@pytest.mark.parametrize("delta43,selected", [(0.01, "E02"), (-0.01, "E01"), (0.0, "E01")])
def test_seed_confirmation_and_no_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delta43: float, selected: str
) -> None:
    store = fixture_store(tmp_path, monkeypatch, delta43)
    output = tmp_path / "selection.json"
    result = select_model.select("E02", "E01", store, output)
    assert result["selected_run"] == selected
    assert result["accepted"] == (delta43 > 0)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        select_model.select("E02", "E01", store, output)
    assert output.read_bytes() == before


def test_selection_rejects_nonbest_and_modified_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = fixture_store(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="最良候補"):
        select_model.select("E01", "E00", store, tmp_path / "selection.json")
    (store.runs_dir / "E02-s43" / "folds.csv").write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="ハッシュ"):
        select_model.select("E02", "E01", store, tmp_path / "selection.json")
    assert not (tmp_path / "selection.json").exists()


def test_selection_rejects_definition_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    store = fixture_store(tmp_path, monkeypatch)
    record = replace(store.record("E02"), data_version="different")
    with pytest.raises(ValueError, match="実験定義"):
        select_model.validate_record(record, "E02", 42)
