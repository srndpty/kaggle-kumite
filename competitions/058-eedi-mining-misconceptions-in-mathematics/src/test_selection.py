import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import train
from ranking import LABEL

from kumite.cv_run import Evaluation
from kumite.experiments import ExperimentRecord, append_record
from kumite.provenance import file_sha256
from kumite.runs import RunStore


@pytest.fixture
def selection_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, list[str], pd.DataFrame, Evaluation]:
    frame = pd.DataFrame({LABEL: [0] * 5, "text": ["label"] * 5, "group": list("abcde")})
    evaluation = Evaluation("058", "MAP@25", True, train.map25, "data", "v1", "split", 42)
    store = RunStore(tmp_path)
    monkeypatch.setattr(train, "STORE", store)
    monkeypatch.setattr(train, "load_cv", lambda: (frame, evaluation))
    monkeypatch.setattr(train, "load_mapping", lambda _: (np.arange(100), ["label"] * 100))
    runs = list(train.TRIALS)
    candidates = np.tile(np.roll(np.arange(100), 2), (len(frame), 1))
    for index, run in enumerate(runs):
        folder = store.runs_dir / run
        folder.mkdir(parents=True)
        pred = np.tile(np.roll(np.arange(100), 4 - index), (len(frame), 1))
        np.save(folder / "oof.npy", pred)
        np.save(folder / "candidates.npy", candidates)
        params = {
            "trial": run,
            "source_hash": file_sha256(Path(train.__file__).with_name("ranking.py")),
            "e5_revision": train.E5_REVISION,
            "e5_hash": "weights",
        }
        append_record(
            store.log_path,
            ExperimentRecord(
                competition_id="058",
                run_id=run,
                metric="MAP@25",
                greater_is_better=True,
                data_version="official-v1",
                data_hash="data",
                split_version="v1",
                split_hash="split",
                seed=42,
                hypothesis="test",
                changed_one_thing="test",
                model=run,
                params=params,
                fold_scores=[],
                oof_score=train.map25(frame[LABEL].to_numpy(), pred),
                code_version="test",
                artifacts={file.name: file_sha256(file) for file in folder.iterdir()},
            ),
        )
    return tmp_path / "selection.json", runs, frame, evaluation


@pytest.mark.parametrize("tampering", ["score", "winner", "candidate"])
def test_export_rejects_tampered_selection(
    selection_fixture: tuple[Path, list[str], pd.DataFrame, Evaluation], tampering: str
) -> None:
    path, runs, _, _ = selection_fixture
    train.select(runs, path, "E00")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if tampering == "score":
        saved["comparisons"][1]["score"] = 0
    elif tampering == "winner":
        saved["selected_run"] = "E03"
    else:
        saved["comparisons"].pop()
        saved["selected_run"] = "E03"
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match=r"選定記録|全 trial"):
        train.export_run(saved["selected_run"], "final", "owner", "dataset", path)


def test_export_rejects_changed_source(
    selection_fixture: tuple[Path, list[str], pd.DataFrame, Evaluation],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = train.STORE.record("E00")
    monkeypatch.setattr(
        RunStore,
        "record",
        lambda *_: replace(record, params={**record.params, "source_hash": "old"}),
    )
    with pytest.raises(ValueError, match="推論ソース"):
        train.export_run("E00", "final", "owner", None, None)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"metric": "MAP@10"}, "CV 評価条件"),
        ({"greater_is_better": False}, "CV 評価条件"),
        ({"split_version": "v2"}, "CV 評価条件"),
        ({"seed": 0}, "CV 評価条件"),
        ({"params": {"trial": "E99"}}, "trial"),
        ({"params": {"trial": "E02", "e5_revision": "old"}}, "E5 リビジョン"),
        (
            {"params": {"trial": "E01", "e5_revision": train.E5_REVISION, "e5_hash": "weights"}},
            "重複",
        ),
        (
            {"params": {"trial": "E02", "e5_revision": train.E5_REVISION, "e5_hash": "other"}},
            "E5 ファイル",
        ),
    ],
)
def test_selection_rejects_incompatible_runs(
    selection_fixture: tuple[Path, list[str], pd.DataFrame, Evaluation],
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, Any],
    message: str,
) -> None:
    _, runs, frame, evaluation = selection_fixture
    read_record = RunStore.record

    def altered_record(store: RunStore, run: str) -> ExperimentRecord:
        record = read_record(store, run)
        return replace(record, **changes) if run == "E02" else record

    monkeypatch.setattr(RunStore, "record", altered_record)
    with pytest.raises(ValueError, match=message):
        train.build_selection(runs, "E00", frame, evaluation)
