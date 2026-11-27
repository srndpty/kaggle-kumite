import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import train as pipeline
from dataset import RATE, SPECIES, WINDOW, Event, aggregate, crop, starts, training_examples
from model import read_crop
from train import check_scores, pcm_fingerprint, validate_submission, write_submission

from kumite.group_folds import multilabel_group_folds
from kumite.lwlrap import lwlrap
from kumite.quality_gates import SelectionPlan, SourceSpec, dense_oof, verify_cv_run
from kumite.runs import RunStore


def test_synthetic_recording_covers_tail_and_short_clip(tmp_path: Path) -> None:
    wave = np.zeros(2 * WINDOW + 10, dtype=np.float32)
    wave[-1] = 0.75
    audio = tmp_path / "a.flac"
    sf.write(audio, wave, RATE, format="FLAC")
    grid = starts(len(wave), WINDOW)
    assert grid == [0, WINDOW, WINDOW + 10]
    assert read_crop(audio, grid[-1])[-1] == 0.75
    assert len(crop(wave[:10], 0)) == WINDOW
    other = tmp_path / "b.flac"
    sf.write(other, wave, RATE, format="FLAC")
    assert pcm_fingerprint(audio) == pcm_fingerprint(other)


def test_fold_does_not_split_duplicates() -> None:
    labels = np.zeros((7, 24), dtype=np.bool_)
    labels[:, 0] = True
    groups = ["a", "a", "b", "c", "d", "e", "f"]
    folds = multilabel_group_folds(groups, labels, n_folds=3, seed=42)
    assert set(folds) == {0, 1, 2}
    assert folds[0] == folds[1]


def test_false_events_do_not_override_same_species_positives() -> None:
    keys = ["training", "valid", "background"]
    tp = {"training": [Event(0, RATE, 2 * RATE)], "valid": [Event(2, RATE, 2 * RATE)]}
    fp = {"training": [Event(0, RATE, 2 * RATE), Event(1, RATE, 2 * RATE)]}
    positives, background, hard = training_examples(
        keys, np.full(3, 3 * WINDOW), np.array([0, 2]), tp, fp, 42, True
    )
    assert [item.species for item in positives] == [(0,)]
    assert len(background) == len(hard) == 1
    assert hard[0].species == (1,)
    assert all(item.row != 1 for item in [*positives, *background, *hard])


def test_aggregate_and_lwlrap() -> None:
    scores = np.asarray([[0.5, 1], [0.9, 0.2], [0.1, 0.4], [0.3, 0.3]])
    np.testing.assert_allclose(aggregate(scores, 1), [0.9, 1])
    np.testing.assert_allclose(aggregate(scores, 3), [1.7 / 3, 1.7 / 3])
    truth = np.asarray([[True, False], [True, True]], dtype=np.bool_)
    assert lwlrap(truth, scores[:2]) == 2.5 / 3
    check_scores(np.ones((2, len(SPECIES))), 2)


def test_submission_format_rejects_reordered_ids_and_invalid_scores(tmp_path: Path) -> None:
    output = tmp_path / "submission.csv"
    ids = ["002", "001"]
    write_submission(output, ids, np.full((2, 24), 0.5))
    validate_submission(output, ids)
    with pytest.raises(ValueError, match="ID"):
        validate_submission(output, list(reversed(ids)))
    with pytest.raises(ValueError, match="値"):
        write_submission(output, ids, np.full((2, 24), np.inf))


def test_small_audio_cv_publishes_and_rescores_oof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    (data / "train").mkdir(parents=True)
    (data / "test").mkdir()
    for i in range(6):
        wave = np.full(2 * WINDOW, i / 10, dtype=np.float32)
        sf.write(data / "train" / f"r{i}.flac", wave, RATE)
    sf.write(data / "test" / "query.flac", np.zeros(WINDOW, dtype=np.float32), RATE)
    for filename in ("train_tp.csv", "train_fp.csv"):
        with (data / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["recording_id", "species_id", "t_min", "t_max"])
            if filename == "train_tp.csv":
                writer.writerows((f"r{i}", i % 2, 1, 2) for i in range(6))
    pipeline.write_submission(data / "sample_submission.csv", ["query"], np.zeros((1, 24)))
    marker = tmp_path / "quality-gates.toml"
    marker.write_text("version = 1\n", encoding="utf-8")
    monkeypatch.setattr(pipeline, "DATA", data)
    monkeypatch.setattr(pipeline, "COMP", tmp_path)
    monkeypatch.setattr(pipeline, "UPDATES", 1)
    monkeypatch.setattr(
        pipeline, "source", lambda *_: SourceSpec(tmp_path, (marker.name,), {"updates": 1})
    )
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        pipeline.run("E00", "E00", marker, SelectionPlan(("E00",), 0.0), torch.device("cpu"))
    finally:
        torch.set_num_threads(threads)
    *_, labels, splits, evaluation = pipeline.load_cv()
    assess = dense_oof(
        labels, splits, lwlrap, expected_shape=labels.shape, data_hash=evaluation.data_hash
    )
    store = RunStore(tmp_path)
    assert verify_cv_run(store, "E00", assess).oof_score > 0
    assert store.load_array("E00", "test.npy").shape == (1, 24)
