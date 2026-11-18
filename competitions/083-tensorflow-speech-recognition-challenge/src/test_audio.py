"""公式データなしで音声の契約と本番学習経路を検査する。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from dataset import (
    CLASSES,
    RATE,
    label_index,
    make_folds,
    one_second,
    read_wave,
    submission,
    validate_folds,
)
from network import Features, fit_model, mix_noise, predict, shift_wave
from scipy.io import wavfile
from select_model import choose_candidate

from kumite.checks import check_label_independence
from kumite.experiments import ExperimentRecord


def test_wave_contract(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    wavfile.write(path, 8000, np.full((4000, 2), 16384, dtype=np.int16))
    rate, wave = read_wave(path)
    assert rate == 8000 and wave.shape == (4000,)
    normalized = one_second(wave, rate)
    assert normalized.shape == (RATE,)
    assert abs(float(normalized[:7000].mean()) - 0.5) < 0.001
    assert np.all(normalized[8000:] == 0)
    assert np.array_equal(
        one_second(np.arange(18000, dtype=np.float32), RATE), np.arange(1000, 17000)
    )


@pytest.mark.parametrize("value", [np.nan, np.inf, 1.1])
def test_invalid_wave(tmp_path: Path, value: float) -> None:
    path = tmp_path / "invalid.wav"
    wavfile.write(path, RATE, np.array([value], dtype=np.float32))
    with pytest.raises(ValueError):
        read_wave(path)


def toy_manifest() -> pd.DataFrame:
    rows = [
        dict(id=f"{speaker}/{label}.wav", label=label, group=f"s{speaker}")
        for speaker in range(9)
        for label in (*range(10), 11)
    ]
    rows += [
        dict(id=f"noise{source}@{offset}", label=10, group=f"noise:{source}")
        for source in range(6)
        for offset in range(3)
    ]
    return pd.DataFrame(rows)


def test_group_folds_and_recreation() -> None:
    frame = toy_manifest()
    folds = make_folds(frame, 42)
    assert np.array_equal(folds, make_folds(frame, 42))
    assert frame.assign(fold=folds).groupby("group").fold.nunique().max() == 1
    assert not np.array_equal(folds, make_folds(frame, 43))
    broken = folds.copy()
    broken[0] = (broken[0] + 1) % 3
    with pytest.raises(ValueError):
        validate_folds(frame, broken)


def test_random_diagnostic_is_separate() -> None:
    frame = toy_manifest()
    folds = make_folds(frame, 42, True)
    assert frame.assign(fold=folds).groupby("group").fold.nunique().max() > 1
    with pytest.raises(ValueError):
        validate_folds(frame, folds)


def test_labels_and_submission_order() -> None:
    assert label_index("cat") == 11 and label_index("yes") == 0
    assert label_index("silence") == 11  # 発話フォルダはsilenceとして扱わない
    test = pd.DataFrame({"id": ["b.wav", "a.wav"]})
    probabilities = np.eye(12)[[0, 11]]
    result = submission(test, probabilities)
    assert result.to_dict("list") == {"fname": ["b.wav", "a.wav"], "label": ["yes", "unknown"]}
    with pytest.raises(ValueError):
        submission(test, np.full((2, 12), np.nan))


def test_shift_does_not_wrap() -> None:
    wave = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(
        shift_wave(wave, torch.tensor([1, -1])),
        torch.tensor([[0.0, 1.0, 2.0, 3.0], [2.0, 3.0, 4.0, 0.0]]),
    )


def test_feature_shape_and_linear_difference() -> None:
    waves = torch.randn(2, RATE)
    mel, linear = Features()(waves), Features(True)(waves)
    assert mel.shape == linear.shape == (2, 1, 40, 101)
    assert torch.isfinite(mel).all() and torch.isfinite(linear).all()
    assert not torch.allclose(mel, linear)


def test_noise_is_finite_for_silence() -> None:
    wave = torch.zeros(3, RATE)
    mixed = mix_noise(wave, wave, torch.Generator().manual_seed(42))
    assert torch.isfinite(mixed).all() and torch.equal(wave, mixed)


@pytest.mark.parametrize("experiment", ["E01", "E03", "E04", "E05"])
def test_actual_training_ignores_valid_labels(experiment: str) -> None:
    rng = np.random.default_rng(1)
    waves = torch.tensor(rng.normal(0, 0.1, (36, RATE)), dtype=torch.float32)
    y = np.tile(np.arange(len(CLASSES)), 3)
    train_idx = np.arange(24)
    features = Features()
    with torch.no_grad():
        x = features(waves)

    def build(labels: np.ndarray) -> tuple[np.ndarray, ...]:
        model, mean, std, _ = fit_model(
            x, waves, labels, train_idx, features, experiment, 42, 1, 12
        )
        return (
            np.concatenate([p.detach().numpy().ravel() for p in model.parameters()]),
            predict(model, x[24:], mean, std),
            mean.numpy(),
            std.numpy(),
        )

    perturbed = y.copy()
    perturbed[24:] = (perturbed[24:] + 1) % 12
    check_label_independence(build, y, perturbed, atol=0)
    original = build(y)[0]
    changed = y.copy()
    changed[:24] = (changed[:24] + 1) % 12
    assert not np.array_equal(original, build(changed)[0])
    if experiment == "E04":
        altered_waves = waves.clone()
        altered_waves[24:] = 0.9
        model, _, _, _ = fit_model(x, altered_waves, y, train_idx, features, experiment, 42, 1, 12)
        altered_weights = np.concatenate([p.detach().numpy().ravel() for p in model.parameters()])
        assert np.array_equal(original, altered_weights)


def test_selection_excludes_diagnostics_and_requires_complete_trials() -> None:
    from dataclasses import replace

    template = ExperimentRecord(
        competition_id="083",
        run_id="E01",
        metric="accuracy",
        greater_is_better=True,
        data_version="v1",
        data_hash="data",
        split_version="speaker",
        split_hash="folds",
        seed=42,
        hypothesis="比較",
        changed_one_thing="単独変更",
        model="CNN",
        fold_scores=[0.8, 0.8, 0.8],
        oof_score=0.8,
        code_version="code",
        runtime_sec=10,
    )
    records = [
        replace(
            template,
            run_id=f"E{k:02}",
            params={"experiment": f"E{k:02}"},
            fold_scores=[0.8 + k / 100] * 3,
        )
        for k in range(1, 6)
    ]
    diagnostic = replace(
        records[-1],
        run_id="diagnostic",
        params={"experiment": "E05", "random_split": True},
        fold_scores=[1.0] * 3,
    )
    assert choose_candidate([*records, diagnostic]).run_id == "E05"
    failure = replace(
        records[-1],
        run_id="failed",
        metric="execution_success",
        params={"experiment": "E05", "failed": True},
        fold_scores=[],
        oof_score=0,
    )
    assert choose_candidate([*records, failure]).run_id == "E05"
    with pytest.raises(ValueError):
        choose_candidate(records[:-1])
    with pytest.raises(ValueError):
        choose_candidate([*records, replace(records[-1], run_id="E05-r2")])


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    from network import SmallCNN

    torch.manual_seed(42)
    model = SmallCNN().eval()
    features = torch.randn(3, 1, 40, 101)
    original = predict(model, features, torch.zeros(1), torch.ones(1))
    path = tmp_path / "fold.pt"
    torch.save(dict(state_dict=model.state_dict(), mean=torch.zeros(1), std=torch.ones(1)), path)
    loaded = torch.load(path, weights_only=True)
    restored = SmallCNN()
    restored.load_state_dict(loaded["state_dict"])
    assert np.array_equal(original, predict(restored, features, loaded["mean"], loaded["std"]))


def test_prepare_manifest_cache_and_recreation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dataset

    data, output = tmp_path / "data", tmp_path / "outputs"
    monkeypatch.setattr(dataset, "DATA", data)
    monkeypatch.setattr(dataset, "OUTPUT", output)
    rng = np.random.default_rng(5)
    for speaker in range(9):
        for word in (*CLASSES[:10], "cat"):
            path = data / "train" / "audio" / word / f"s{speaker}_nohash_0.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(path, RATE, rng.integers(-1000, 1000, RATE, dtype=np.int16))
    noise = data / "train" / "audio" / "_background_noise_"
    noise.mkdir()
    for source in range(6):
        wavfile.write(
            noise / f"{source}.wav", RATE, rng.integers(-100, 100, RATE * 2, dtype=np.int16)
        )
    test_dir = data / "test" / "audio"
    test_dir.mkdir(parents=True)
    for name in ("z.wav", "a.wav"):
        wavfile.write(test_dir / name, RATE, rng.integers(-1000, 1000, RATE, dtype=np.int16))
    pd.DataFrame({"fname": ["z.wav", "a.wav"], "label": ["silence"] * 2}).to_csv(
        data / "sample_submission.csv", index=False
    )
    dataset.prepare()
    frame, test, waves, test_waves, summary = dataset.load_prepared()
    assert test.id.tolist() == ["z.wav", "a.wav"]
    assert waves.shape == (111, RATE) and test_waves.shape == (2, RATE)
    assert summary["class_counts"]["silence"] == 12
    from kumite.provenance import array_sha256

    assert array_sha256(make_folds(frame, 42)) == summary["split_hash"]
    for i in (0, 20, 100):
        row = frame.iloc[i]
        rate, wave = read_wave(data / row["path"])
        if int(row["offset"]) >= 0:
            start = int(row["offset"]) * rate
            wave = wave[start : start + rate]
        assert np.array_equal(waves[i], one_second(wave, rate))
    with pytest.raises(FileExistsError):
        dataset.prepare()


def test_fine_supervision_and_probability_aggregation() -> None:
    from supervision import FINE_CLASSES, collapse_probabilities, fine_labels

    frame = pd.DataFrame({"word": ["yes", "silence", "bird", "zero"], "label": [0, 10, 11, 11]})
    labels = fine_labels(frame)
    assert np.asarray(FINE_CLASSES)[labels].tolist() == frame.word.tolist()
    probabilities = np.zeros((2, 31))
    probabilities[:, 0] = 0.4
    probabilities[:, 11:] = 0.6 / 20
    collapsed = collapse_probabilities(probabilities)
    assert np.all(probabilities.argmax(1) == 0)
    assert np.all(collapsed.argmax(1) == 11)
    assert np.allclose(collapsed.sum(1), 1)
    rounded = np.zeros((1, 31), dtype=np.float32)
    rounded[0, 11:13] = [0.8, 0.20000012]
    from kumite.checks import validate_probabilities

    corrected = collapse_probabilities(rounded)
    validate_probabilities(corrected, 1, 12)
    assert corrected[0, 11] == 1
    with pytest.raises(ValueError):
        fine_labels(pd.DataFrame({"word": ["novel"], "label": [11]}))
    with pytest.raises(ValueError):
        fine_labels(pd.DataFrame({"word": ["bird"], "label": [0]}))
    with pytest.raises(ValueError):
        collapse_probabilities(np.zeros((2, 30)))
