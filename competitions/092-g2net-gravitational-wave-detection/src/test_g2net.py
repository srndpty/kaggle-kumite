import csv
import io
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import run
import torch
from dataset import SAMPLE_RATE, SHAPE, make_splits, waveform_path
from model import fit_whitening, transform
from prepare import prepare
from run import CONFIGS, fit_fold, score, validate_csv, write_csv

from kumite.checks import check_label_independence
from kumite.crossfit import FoldOutput


def test_grouped_splits() -> None:
    y = np.tile([0, 0, 1, 1], 6)
    groups = np.repeat(np.arange(12), 2)
    splits = make_splits(y, groups)
    assert len(splits) == 3
    for train, valid in splits:
        assert not np.intersect1d(groups[train], groups[valid]).size


def test_auc() -> None:
    y = np.array([0, 1, 0, 1])
    assert score(y, np.array([0.2, 0.7, 0.1, 0.8])) == 1.0
    assert score(y, np.full(4, 0.5)) == 0.5
    with pytest.raises(ValueError):
        score(y, np.array([0.1, np.nan, 0.8, 0.9]))


def test_fold_predictions_ignore_validation_labels() -> None:
    rng = np.random.default_rng(42)
    waves = rng.normal(scale=1e-20, size=(12, *SHAPE))
    y = np.tile([0, 1], 6)
    train, valid = np.arange(8), np.arange(8, 12)
    config = replace(CONFIGS["E03"], shuffle_ligo=True)
    device = torch.device("cpu")

    def build(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        result = fit_fold(0, train, valid, waves, waves[:2], labels, config, device)
        return result.valid, result.query

    changed = y.copy()
    changed[valid] = 1 - y[valid]
    check_label_independence(build, y, changed, atol=0)


def test_submission_ids_and_probabilities(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    ids = ["0000000000", "0000000001"]
    write_csv(path, ids, np.array([0.2, 0.7]))
    validate_csv(path, ids)
    with pytest.raises(ValueError):
        validate_csv(path, ids[::-1])
    with pytest.raises(ValueError):
        validate_csv(path, ids[:1])
    with path.open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow((ids[0], "nan"))
    with pytest.raises(ValueError):
        validate_csv(path, ids)


def test_cv_selection_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_root = run.ROOT
    data = tmp_path / "packed"
    for name in ("quality-gates.toml", "selection-plans.json"):
        (tmp_path / name).write_bytes((real_root / name).read_bytes())
    rng = np.random.default_rng(42)
    test_ids = [f"{index:010x}" for index in range(30, 32)]
    archive = tmp_path / "official.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for split, ids, filename in (
            ("train", [f"{index:010x}" for index in range(12)], "training_labels.csv"),
            ("test", test_ids, "sample_submission.csv"),
        ):
            for key in reversed(ids):
                wave = io.BytesIO()
                np.save(wave, rng.normal(scale=1e-20, size=SHAPE))
                bundle.writestr(waveform_path(Path(), split, key).as_posix(), wave.getvalue())
            write_csv(tmp_path / filename, ids, np.arange(len(ids)) % 2)
            bundle.write(tmp_path / filename, filename)
    prepare(archive, data)

    spec = run.source(CONFIGS["B00"], real_root / "selection-plans.json")
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "DATA", data)
    monkeypatch.setattr(run, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(run, "source", lambda *_: spec)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda _: None)

    def predict(
        fold: int,
        train: np.ndarray,
        valid: np.ndarray,
        waves: np.ndarray,
        test: np.ndarray,
        y: np.ndarray,
        config: run.Config,
        device: torch.device,
    ) -> FoldOutput:
        return fit_fold(fold, train, valid, waves, test, y, config, torch.device("cpu"))

    monkeypatch.setattr(run, "fit_fold", predict)
    run.main(["cv", "baseline", "B00-initial"])
    run.main(["select", "baseline"])
    run.main(["export", "baseline", "F00-smoke"])
    paths = list((tmp_path / "outputs" / "exports").rglob("submission.csv"))
    assert len(paths) == 1
    validate_csv(paths[0], test_ids)


@pytest.mark.parametrize(
    "invalid",
    [
        np.full((1, *SHAPE), np.nan),
        np.full((1, *SHAPE), np.inf),
        np.zeros((1, 2, SHAPE[1])),
        np.zeros((1, SHAPE[0], SHAPE[1] - 1)),
        np.full((1, *SHAPE), np.finfo(np.float64).max),
    ],
)
def test_whitening_rejects_invalid_waveforms(invalid: np.ndarray) -> None:
    with pytest.raises(ValueError):
        fit_whitening([invalid])
    with pytest.raises(ValueError):
        transform(invalid, 3, True, None)


def test_whitening_handles_zero_and_strong_single_frequency() -> None:
    zero = np.zeros((1, *SHAPE))
    with pytest.raises(ValueError, match="学習行"):
        fit_whitening([])
    np.testing.assert_array_equal(transform(zero, 3, True, fit_whitening([zero])), zero)
    sinusoid = 1e75 * np.sin(2 * np.pi * 100 * np.arange(SHAPE[1]) / SAMPLE_RATE)
    wave = np.broadcast_to(sinusoid, (2, *SHAPE))
    whitening = fit_whitening([wave])
    assert np.isfinite(whitening).all()
    result = transform(wave, 3, True, whitening)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert np.any(result)


@pytest.mark.parametrize("bad_value", [0.0, -1.0, np.nan, np.inf])
def test_transform_rejects_invalid_whitening(bad_value: float) -> None:
    wave = np.zeros((1, *SHAPE))
    whitening = fit_whitening([wave])
    whitening[0, 100] = bad_value
    with pytest.raises(ValueError, match="白色化係数"):
        transform(wave, 3, True, whitening)
