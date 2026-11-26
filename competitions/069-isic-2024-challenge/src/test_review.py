import hashlib
import json
import shutil
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from analyze import verified_predictions
from model import CATEGORICAL, NUMERIC, training_rows
from relocate import relocate_notebook
from submission import notebook
from validation import pauc
from workflow import sampling_evidence

from kumite.crossfit import split_roles
from kumite.experiments import ExperimentRecord
from kumite.provenance import array_sha256
from kumite.quality_gates import SourceSpec, dense_oof, publish_gated_cv, reserve_cv
from kumite.runs import RunStore


def test_sampling_stays_inside_training_fold() -> None:
    labels = np.tile([0] * 19 + [1], 3)
    train = np.arange(20, 60)
    valid = np.arange(20)
    kept = train[training_rows(labels[train], "E04")]
    np.testing.assert_array_equal(kept, train[training_rows(labels[train], "E04")])
    assert not set(kept) & set(valid)
    assert set(train[labels[train] == 1]) <= set(kept)
    ids = pd.Series([f"ISIC_{row:03}" for row in kept])
    evidence = sampling_evidence(ids, labels[kept])
    assert evidence == sampling_evidence(ids.iloc[::-1], labels[kept][::-1])
    assert evidence == {
        "seed": 42,
        "positive_count": 2,
        "sampled_negative_count": 10,
        "positive_rate": 2 / 12,
        "isic_ids_sha256": hashlib.sha256(
            json.dumps(sorted(ids.tolist()), separators=(",", ":")).encode()
        ).hexdigest(),
    }


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("nonfinite", "検証行は有限値"),
        ("out_of_range", "確率が不正"),
        ("missing", "oof.npy"),
        ("split", "検証分割のハッシュ"),
        ("score", "スコア"),
        ("run_id", "run_id"),
    ],
)
def test_analysis_checks_published_evidence(tmp_path: Path, problem: str, message: str) -> None:
    y = np.array([0, 1, 0, 1])
    a, b = np.array([0, 1]), np.array([2, 3])
    splits = [(a, b), (b, a)]
    predictions = np.array([0.1, 0.9, 0.2, 0.8])
    assessment = dense_oof(y, splits, pauc, expected_shape=y.shape, data_hash="fixture")
    store = RunStore(tmp_path)
    code = tmp_path / "model.py"
    code.write_text("# fixture", encoding="utf-8")
    source = SourceSpec(tmp_path, (code.name,), {})
    score = 0.1 if problem == "score" else pauc(y, predictions)
    record = ExperimentRecord(
        competition_id="069",
        run_id="E04",
        metric="pAUC_TPR80",
        greater_is_better=True,
        data_version="fixture",
        data_hash="fixture",
        split_version="patient-v1",
        split_hash=array_sha256(split_roles(len(y), splits)),
        seed=42,
        hypothesis="fixture",
        changed_one_thing="fixture",
        model="fixture",
        code_version="fixture",
        fold_scores=[score, score],
        oof_score=score,
    )
    if problem == "nonfinite":
        predictions[0] = np.nan
    if problem == "out_of_range":
        predictions[0] = -0.1
    with reserve_cv(store, "E04", source, {}) as (reservation, start):
        # 壊れた内容でもファイルのハッシュ自体は一致する証跡を作る。
        publish_gated_cv(
            store,
            reservation,
            record,
            lambda path: np.save(path / "oof.npy", predictions),
            source=source,
            before_training=start,
            assess=lambda *_: ([score, score], score),
        )
    if problem in {"missing", "split", "run_id"}:
        np.testing.assert_array_equal(verified_predictions(store, "E04", assessment), predictions)
    if problem == "missing":
        store.artifact_path("E04", "oof.npy").unlink()
    if problem == "split":
        assessment = dense_oof(y, splits[::-1], pauc, expected_shape=y.shape, data_hash="fixture")
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        verified_predictions(store, "../E04" if problem == "run_id" else "E04", assessment)


def test_relocated_notebook_executes_both_layouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.txt"
    train = pd.DataFrame(
        {
            **{field: np.arange(8, dtype=float) for field in NUMERIC},
            **{field: pd.Categorical(["value"] * 8) for field in CATEGORICAL},
        }
    )
    lgb.train(
        {"objective": "binary", "verbosity": -1, "num_threads": 1, "min_data_in_leaf": 1},
        lgb.Dataset(train, label=[0, 0, 0, 0, 1, 1, 1, 1]),
        num_boost_round=2,
    ).save_model(str(model))
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    original, relocated = tmp_path / "original.ipynb", tmp_path / "relocated.ipynb"
    original.write_text(json.dumps(notebook("B00", digest)), encoding="utf-8")
    relocate_notebook(original, relocated)
    payload = json.loads(relocated.read_text(encoding="utf-8"))
    sample = pd.DataFrame({"isic_id": ["b", "a"], "target": [0.5, 0.5]})
    outputs = []
    monkeypatch.chdir(tmp_path)
    for layout, comp_prefix, dataset_prefix in (
        ("flat", "", ""),
        ("categorized", "competitions", "datasets/shion5"),
    ):
        root = tmp_path / layout
        competition = root / comp_prefix / "isic-2024-challenge"
        dataset = root / dataset_prefix / "kumite-isic-2024-f00-model"
        competition.mkdir(parents=True)
        dataset.mkdir(parents=True)
        shutil.copyfile(model, dataset / model.name)
        train.iloc[[0, 7]].assign(isic_id=["a", "b"]).to_csv(
            competition / "test-metadata.csv", index=False
        )
        sample.to_csv(competition / "sample_submission.csv", index=False)
        namespace: dict[str, object] = {}
        for cell in payload["cells"]:
            code = "".join(cell["source"]).replace("'/kaggle/input'", repr(root.as_posix()))
            exec(compile(code, cell["id"], "exec"), namespace)
        outputs.append(pd.read_csv(tmp_path / "submission.csv"))
    pd.testing.assert_frame_equal(*outputs)
    assert outputs[0]["isic_id"].tolist() == ["b", "a"]
    assert outputs[0]["target"].iloc[0] > outputs[0]["target"].iloc[1]
