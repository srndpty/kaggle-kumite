"""キャッシュ改変・run取り違え・事後分析の数式を検査する。"""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import dataset
import numpy as np
import pandas as pd
import predict as prediction
import pytest
import torch
import verify
from dataset import CLASSES, RATE
from network import SmallCNN
from review_diagnostics import oof_decomposition
from select_model import RUN_DEFINITIONS, validate_run_artifacts, validate_run_definition
from split_diagnostic import fold_profiles
from supervision import FINE_CLASSES

from kumite.experiments import ExperimentRecord
from kumite.provenance import file_sha256
from kumite.runs import RunStore


@pytest.fixture
def prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    output = tmp_path / "outputs"
    path = output / "prepared"
    path.mkdir(parents=True)
    monkeypatch.setattr(dataset, "OUTPUT", output)
    train = pd.DataFrame(dict(id=["a", "b"], label=[0, 11]))
    test = pd.DataFrame(dict(id=["c"]))
    for name, frame in (("train", train), ("test", test)):
        frame.to_csv(path / f"{name}.csv", index=False)
        np.save(path / f"{name}.npy", np.zeros((len(frame), RATE), dtype=np.float32))
    (path / "summary.json").write_text(
        json.dumps(
            dict(
                data_hash=dataset.identity(train, test),
                caches={name: file_sha256(path / f"{name}.npy") for name in ("train", "test")},
            )
        ),
        encoding="utf-8",
    )
    return path


def test_deep_cache_valid(prepared: Path) -> None:
    frame, test, waves, query, _ = dataset.load_prepared(deep=True)
    assert waves.shape == (len(frame), RATE) and query.shape == (len(test), RATE)


@pytest.mark.parametrize("name", ["train", "test"])
@pytest.mark.parametrize("entry", ["predict", "verify"])
def test_cache_value_tampering_rejected_before_inference(
    prepared: Path, monkeypatch: pytest.MonkeyPatch, name: str, entry: str
) -> None:
    array = np.load(prepared / f"{name}.npy", mmap_mode="r+")
    array[0, 0] = 0.25
    array.flush()
    del array
    dataset.load_prepared()  # 形状・dtype・manifestは変わっていない。
    with pytest.raises(ValueError, match="音声cacheのハッシュ不一致"):
        dataset.load_prepared(deep=True)

    def unexpected(*args, **kwargs):
        raise AssertionError("推論・fold処理に進んではいけません")

    monkeypatch.setattr(
        prediction, "RunStore", lambda root: SimpleNamespace(record=lambda run: None)
    )
    monkeypatch.setattr(verify, "RunStore", lambda root: SimpleNamespace(records=unexpected))
    monkeypatch.setattr(prediction, "feature_batches", unexpected)
    monkeypatch.setattr(verify, "folds_to_splits", unexpected)
    with pytest.raises(ValueError, match="音声cacheのハッシュ不一致"):
        if entry == "predict":
            prediction.reproduce("E05-r2")
        else:
            verify.main()


def record(run: str) -> ExperimentRecord:
    experiment, seed = RUN_DEFINITIONS[run]
    return ExperimentRecord(
        competition_id="083",
        run_id=run,
        metric="accuracy",
        greater_is_better=True,
        data_version="audio-v1",
        data_hash="data",
        split_version="speaker",
        split_hash="folds",
        seed=seed,
        hypothesis="仮説",
        changed_one_thing="単独変更",
        model="SmallCNN",
        fold_scores=[0.8] * 3,
        oof_score=0.8,
        code_version="code",
        params=dict(
            experiment=experiment,
            benchmark=False,
            random_split=False,
            classes=list(CLASSES),
            training_classes=list(FINE_CLASSES if experiment == "E05" else CLASSES),
            config={"epochs": 20},
        ),
    )


@pytest.mark.parametrize("run", list(RUN_DEFINITIONS))
def test_allowed_run_definitions(run: str) -> None:
    validate_run_definition(record(run))


@pytest.mark.parametrize(
    "field,value",
    [
        ("experiment", "E01"),
        ("training_classes", list(CLASSES)),
        ("classes", list(reversed(CLASSES))),
        ("benchmark", True),
        ("random_split", True),
    ],
)
@pytest.mark.parametrize("run", ["E05-r2", "E05-s43"])
def test_run_params_mismatch(run: str, field: str, value: object) -> None:
    original = record(run)
    with pytest.raises(ValueError):
        validate_run_definition(replace(original, params={**original.params, field: value}))


@pytest.mark.parametrize(
    "changes", [{"run_id": "E05-r3"}, {"model": "別モデル"}, {"seed": 43}, {"metric": "別指標"}]
)
def test_run_identity_mismatch(changes: dict) -> None:
    with pytest.raises(ValueError):
        validate_run_definition(replace(record("E05-r2"), **changes))


@pytest.mark.parametrize("run,outputs,linear", [("E05-r2", 12, False), ("E01", 12, True)])
def test_checkpoint_disagrees_with_declaration(
    tmp_path: Path, run: str, outputs: int, linear: bool
) -> None:
    original = record(run)
    checkpoint = dict(state_dict=SmallCNN(outputs).state_dict(), linear=linear)

    def write(path: Path) -> None:
        (path / "config.json").write_text(json.dumps(original.params["config"]), encoding="utf-8")
        for k in range(3):
            torch.save(checkpoint, path / f"fold{k}.pt")

    store = RunStore(tmp_path)
    with store.reserve(run) as reservation:
        store.publish(reservation, original, write)
    with pytest.raises(ValueError, match="checkpoint"):
        validate_run_artifacts(store, original)


def test_oof_contributions_and_silence_sensitivity() -> None:
    y = np.array([10, 0, 0, 1])
    probabilities = {"base": np.eye(12)[[10, 0, 1, 1]], "candidate": np.eye(12)[[0, 0, 0, 1]]}
    result = oof_decomposition(y, probabilities, 0.1)
    for values in result.values():
        assert (
            sum(row["accuracy_contribution"] for row in values["classes"].values())
            == values["oof_accuracy"]
        )
        own_prior = next(row for row in values["sensitivity"] if row["silence_share"] == 0.25)
        assert own_prior["weighted_accuracy"] == values["oof_accuracy"]
    assert result["base"]["groups"]["non_silence"]["binary_detection_recall"] == 1
    sensitivity = {
        run: next(
            row["weighted_accuracy"] for row in values["sensitivity"] if row["silence_share"] == 0.1
        )
        for run, values in result.items()
    }
    assert sensitivity["base"] == pytest.approx(0.7)
    assert sensitivity["candidate"] == pytest.approx(0.9)


def test_fold_profiles_use_window_rms_and_speech_duration() -> None:
    rows = [
        dict(
            path=f"train/{word}/s{speaker}_nohash_0.wav",
            group=f"s{speaker}",
            label=i,
            rate=RATE,
            samples=RATE,
        )
        for speaker in range(3)
        for i, word in enumerate(CLASSES[:10])
    ]
    rows += [
        dict(
            path=f"noise/{source}.wav",
            group=f"noise{source}",
            label=10,
            rate=RATE,
            samples=RATE * 90,
        )
        for source in range(3)
        for _ in range(source + 1)
    ]
    frame = pd.DataFrame(rows)
    folds = np.array([int(group[-1]) for group in frame.group])
    profiles = fold_profiles(frame, folds, np.arange(len(frame), dtype=np.float64) / 100)
    for k, profile in enumerate(profiles):
        for role in ("train", "valid"):
            values = profile[role]
            assert sum(values["class_counts"].values()) == values["n_rows"]
            assert values["speech_duration_seconds"]["max"] == 1
            assert (
                sum(row["windows"] for row in values["background_recordings"])
                == values["silence_windows"]
            )
        assert profile["valid"]["n_speakers"] == 1
        assert profile["valid"]["silence_windows"] == k + 1
