"""選定・再実行・提出で不正な証跡を拒否する。"""

import sys
from pathlib import Path

import notebook
import numpy as np
import pandas as pd
import pytest
import train
from evidence import load_baseline, manifest, source_hashes
from notebook import competition_data, verified_scores, write_submission
from sklearn.metrics import f1_score
from train import SLUG, write_evidence

from kumite.cv_run import Evaluation
from kumite.experiments import ExperimentRecord
from kumite.runs import RunStore

Y = np.array([0, 1])
EVAL = Evaluation("046", "F1", True, f1_score, "data", "v1", "split", 42)


def publish(
    store: RunStore, trial: str, options: dict[str, bool], *, probabilities: tuple[float, float]
) -> None:
    record = ExperimentRecord(
        competition_id="046",
        run_id=trial,
        metric="F1",
        greater_is_better=True,
        data_version="official-csv",
        data_hash="data",
        split_version="v1",
        split_hash="split",
        seed=42,
        hypothesis="",
        changed_one_thing="",
        model="TF-IDF + LR",
        params=options,
        fold_scores=[1.0],
        oof_score=1.0,
        code_version="test",
    )

    def write(out: Path) -> None:
        np.save(out / "oof.npy", Y)
        write_evidence(
            out,
            {
                "oof": np.array(probabilities),
                "coverage": np.ones(2),
                "data_hash": "data",
                "split_hash": "split",
                "tune_0": np.array([0.6]),
            },
            manifest(trial, options, source_hashes()),
        )

    with store.reserve(trial) as reservation:
        store.publish(reservation, record, write)


def test_selection_refuses_tampered_oof_or_changed_source(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    publish(store, "E02", {"chars": True}, probabilities=(0.2, 0.8))
    hashes = source_hashes()
    assert verified_scores(store, EVAL, Y, hashes, ("E02",), "")[0] == {"E02": 1.0}
    changed = {**hashes, "text_features.py": "changed"}
    with pytest.raises(ValueError, match="CV ソース"):
        verified_scores(store, EVAL, Y, changed, ("E02",), "")
    np.save(store.artifact_path("E02", "oof.npy"), 1 - Y)
    with pytest.raises(ValueError, match="ハッシュ"):
        verified_scores(store, EVAL, Y, hashes, ("E02",), "")


def test_baseline_requires_e00_settings_and_matching_decisions(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    publish(store, "E02", {"chars": True}, probabilities=(0.2, 0.8))
    publish(store, "E00", {}, probabilities=(0.2, 0.8))
    with pytest.raises(ValueError, match="試行設定"):
        load_baseline(store, "E02", EVAL, Y, source_hashes(), [1])
    cached = load_baseline(store, "E00", EVAL, Y, source_hashes(), [1])
    np.testing.assert_array_equal(cached["oof"], [0.2, 0.8])


def test_baseline_rejects_inconsistent_threshold(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    publish(store, "E00", {}, probabilities=(0.8, 0.2))
    with pytest.raises(ValueError, match=r"閾値 0\.5 の再構成"):
        load_baseline(store, "E00", EVAL, Y, source_hashes(), [1])


def test_competition_mount_ignores_unrelated_csvs_and_rejects_ambiguity(tmp_path: Path) -> None:
    data = tmp_path / "competitions" / SLUG
    other = tmp_path / "other"
    for path in (data, other):
        path.mkdir(parents=True)
        for name in ("train.csv", "test.csv", "sample_submission.csv"):
            (path / name).touch()
    assert competition_data(tmp_path) == data
    (tmp_path / SLUG).mkdir()
    with pytest.raises(ValueError, match="一意"):
        competition_data(tmp_path)


@pytest.mark.parametrize("ids", ["one,hello\none,world\n", ",hello\ntwo,world\n"])
def test_submission_rejects_duplicate_or_missing_qid(tmp_path: Path, ids: str) -> None:
    (tmp_path / "test.csv").write_text("qid,question_text\n" + ids, encoding="utf-8")
    (tmp_path / "sample_submission.csv").write_text(
        "qid,prediction\none,0\ntwo,0\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="ID"):
        write_submission(tmp_path, tmp_path / "submission.csv")


def test_cv_saves_equal_probabilities_and_freezes_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame(
        {
            "qid": [f"train{i}" for i in range(48)],
            "question_text": [f"{'Apple' if i % 2 else 'Orange'} fruit {i}" for i in range(48)],
            "target": [i % 2 for i in range(48)],
        }
    ).to_csv(data / "train.csv", index=False)
    (data / "test.csv").write_text("qid,question_text\ntest1,Apple fruit\n", encoding="utf-8")
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(notebook, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["train", *train.TRIALS])
    train.main()
    store = RunStore(tmp_path)
    with (
        np.load(store.artifact_path("E00", "probabilities.npz")) as baseline,
        np.load(store.artifact_path("E01", "probabilities.npz")) as tuned,
    ):
        for name in ("oof", "tune_0", "tune_1", "tune_2"):
            np.testing.assert_array_equal(baseline[name], tuned[name])

    selection = tmp_path / "selection.json"
    args = ["--model", "final", "--selection", str(selection)]
    first = tmp_path / "first"
    monkeypatch.setattr(sys, "argv", ["notebook", "owner", str(first), *args, "--freeze"])
    notebook.main()
    frozen = selection.read_bytes()
    with pytest.raises(FileExistsError):
        notebook.main()
    replay = tmp_path / "replay"
    monkeypatch.setattr(sys, "argv", ["notebook", "owner", str(replay), *args])
    notebook.main()
    assert selection.read_bytes() == frozen
    assert (replay / "submit.ipynb").read_bytes() == (first / "submit.ipynb").read_bytes()
