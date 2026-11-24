import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
import soundfile
import torch
import train
from dataset import WINDOW, Clip, eligible_noisy, split_groups, windows
from dummy import SOURCE
from export import notebook_source
from model import Tagger, fit, predict

from kumite.provenance import file_sha256
from kumite.runs import RunStore


def test_group_split_and_noisy_exclusion() -> None:
    clips = [Clip(Path(str(i)), group, ("tag",), 1) for i, group in enumerate("aabcd")]
    labels = np.ones((len(clips), 1), dtype=bool)
    folds = split_groups(clips, labels)
    assert folds[0] == folds[1]
    assert set(folds) == {0, 1, 2}
    assert eligible_noisy(clips[1:], clips[:1]) == clips[2:]


def test_short_and_tail_windows_and_fit() -> None:
    short = np.ones(100, dtype=np.float32)
    long = np.arange(WINDOW + 123, dtype=np.float32)
    assert not windows(short)[0, 100:].any()
    np.testing.assert_array_equal(windows(long)[-1], long[-WINDOW:])
    torch.set_num_threads(2)
    device = torch.device("cpu")
    model = fit(
        [short],
        np.array([[True, False, True]]),
        [],
        np.empty((0, 3), dtype=bool),
        updates=1,
        seed=42,
        device=device,
        deadline=time.monotonic() + 30,
        time_mask=True,
    )
    scores = predict(model, short, device, maximum=False)
    assert scores.shape == (3,)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_notebook_inference_uses_mounted_clips(tmp_path: Path) -> None:
    audio = tmp_path / "test"
    audio.mkdir()
    soundfile.write(audio / "hidden.wav", np.zeros(128, dtype=np.float32), 16000)
    with ZipFile(tmp_path / "test.zip", "w") as archive:
        archive.write(audio / "hidden.wav", "test/hidden.wav")
    pd.DataFrame({"fname": ["public.wav"], "tag": [0.0]}).to_csv(
        tmp_path / "sample_submission.csv", index=False
    )
    checkpoint = tmp_path / "freesound-fold0.pt"
    torch.save(Tagger(1).state_dict(), checkpoint)
    source = notebook_source(["tag"], [file_sha256(checkpoint)], False)
    source = source.replace("/kaggle/input", tmp_path.as_posix())
    source = source.replace("/kaggle/working", tmp_path.as_posix())
    exec(compile(source, "<exported notebook>", "exec"), {})
    submission = pd.read_csv(tmp_path / "submission.csv")
    assert list(submission.columns) == ["fname", "tag"]
    assert submission["fname"].tolist() == ["hidden.wav"]
    assert submission["tag"].between(0.0, 1.0).all()


def test_p0_preserves_template_order(tmp_path: Path) -> None:
    tags = [f"tag_{i}" for i in range(80)]
    names = ["second.wav", "first.wav"]
    frame = pd.DataFrame(np.zeros((2, 80)), columns=tags)
    frame.insert(0, "fname", names)
    frame.to_csv(tmp_path / "sample_submission.csv", index=False)
    with ZipFile(tmp_path / "test.zip", "w") as archive:
        for name in reversed(names):
            archive.writestr(f"test/{name}", b"")
    source = SOURCE.replace("/kaggle/input", tmp_path.as_posix())
    source = source.replace("/kaggle/working", tmp_path.as_posix())
    exec(compile(source, "<p0 notebook>", "exec"), {})
    submission = pd.read_csv(tmp_path / "submission.csv")
    assert list(submission.columns) == ["fname", *tags]
    assert submission["fname"].tolist() == names
    assert (np.diff(submission[tags].to_numpy(), axis=1) < 0).all()


def test_cv_run_publishes_reproducible_oof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train, "ROOT", tmp_path)
    root = tmp_path / "data"
    root.mkdir()
    names = [f"curated-{i}.wav" for i in range(6)]
    for i, name in enumerate([*names, "noisy.wav"]):
        wave = np.zeros(32, dtype=np.float32)
        wave[i] = 0.5
        soundfile.write(root / name, wave, 16000)
    pd.DataFrame({"fname": names, "labels": ["a", "b", "a,b"] * 2}).to_csv(
        root / "train_curated.csv", index=False
    )
    pd.DataFrame({"fname": ["noisy.wav"], "labels": ["b"]}).to_csv(
        root / "train_noisy.csv", index=False
    )
    pd.DataFrame({"fname": ["test.wav"], "a": [0.0], "b": [0.0]}).to_csv(
        root / "sample_submission.csv", index=False
    )
    train.run("E00", "E00", 1, False, 30, None)
    store = RunStore(tmp_path)
    assert store.load_array("E00", "oof.npy").shape == (6, 2)
    assert len(store.record("E00").fold_scores) == 3
