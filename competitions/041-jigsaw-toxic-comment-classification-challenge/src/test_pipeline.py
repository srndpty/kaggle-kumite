"""漏洩・提出形式・指標を小さいデータで検査する。"""

import numpy as np
import pandas as pd
import pytest
from cv import make_folds, text_key
from dataset import LABELS, submission
from metrics import macro_auc
from transformer import cpu_smoke

from kumite.checks import check_label_independence, validate_multilabel_probabilities
from kumite.text_features import build_features


def test_duplicate_groups():
    texts = [f"comment {i}" for i in range(90)] + ["  COMMENT 0  "]
    y = np.array([[i % 2, (i // 2) % 2] * 3 for i in range(91)])
    folds = make_folds(texts, y)
    assert folds[0] == folds[-1]
    assert np.array_equal(folds, make_folds(texts, y))
    assert text_key("Ａ BC") == text_key("a bc")


def test_single_class_rejected():
    with pytest.raises(ValueError, match="片クラス"):
        make_folds([str(i) for i in range(12)], np.zeros((12, 6), dtype=int))


def test_fold_vocabulary_and_label_independence():
    texts = ["hello friend"] * 4
    frames, vecs = build_features(texts, ["validationexclusive"], ["testexclusive"])
    assert "validationexclusive" not in vecs[0].vocabulary_
    assert "testexclusive" not in vecs[0].vocabulary_
    assert frames[1].nnz == frames[2].nnz == 0

    def build(_):
        return [a.toarray() for a in build_features(texts, ["hello"], ["friend"])[0]]

    check_label_independence(build, np.array([0, 1]), np.array([1, 1]))


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1])
def test_probabilities(value):
    pred = np.full((2, 6), 0.8)
    validate_multilabel_probabilities(pred, 2, 6)
    pred[0, 0] = value
    with pytest.raises(ValueError):
        validate_multilabel_probabilities(pred, 2, 6)


def test_submission_and_metric():
    sample = pd.DataFrame({"id": ["b", "a"], **{c: [0.0, 0.0] for c in LABELS}})
    pred = np.tile([[0.2], [0.8]], (1, 6))
    assert list(submission(sample, np.array(["b", "a"]), pred).columns) == ["id", *LABELS]
    with pytest.raises(ValueError):
        submission(sample, np.array(["a", "b"]), pred)
    with pytest.raises(ValueError):
        submission(sample, np.array(["b", "a"]), pred[:, :5])
    assert macro_auc(np.tile([[0], [1]], (1, 6)), pred) == 1.0


def test_tiny_transformer():
    cpu_smoke()


def test_public_labels_never_read(tmp_path, monkeypatch):
    import dataset

    train = pd.DataFrame(
        {
            "id": ["train1", "train2"],
            "comment_text": ["hello", "bye"],
            **{c: [0, 1] for c in LABELS},
        }
    )
    test = pd.DataFrame({"id": ["b", "a"], "comment_text": ["two", "one"]})
    sample = pd.DataFrame({"id": ["a", "b"], **{c: [0.0, 0.0] for c in LABELS}})
    paths = tuple(tmp_path / n for n in ("train.csv", "test.csv", "sample_submission.csv"))
    for frame, path in zip((train, test, sample), paths, strict=True):
        frame.to_csv(path, index=False)
    (tmp_path / "test_labels.csv").write_text("invalid;must never be read", encoding="utf-8")
    monkeypatch.setattr(dataset, "DATA_FILES", paths)
    _, ordered, _ = dataset.load_data()
    assert ordered.id.tolist() == ["a", "b"]
    assert ordered.comment_text.tolist() == ["one", "two"]


def test_fit_ignores_validation_labels():
    from models import fit_linear

    texts = ["friendly hello"] * 4 + ["hostile insult"] * 4
    labels = np.array([[0] * 6] * 4 + [[1] * 6] * 4 + [[0] * 6, [1] * 6])

    def build(y):
        frames, _ = build_features(texts, ["friendly", "hostile"], ["hello"])
        return list(fit_linear(frames, y[:8]))

    changed = labels.copy()
    changed[8:] = 1 - changed[8:]
    check_label_independence(build, labels, changed)


def test_cv_publishes_complete_run(tmp_path, monkeypatch):
    import pipeline

    from kumite.runs import RunStore

    train = pd.DataFrame(
        {
            "id": [f"tr{i}" for i in range(90)],
            "comment_text": [
                f"hello friendly number {i}" if i % 2 else f"hostile insult number {i}"
                for i in range(90)
            ],
            **{label: [i % 2 for i in range(90)] for label in LABELS},
        }
    )
    test = pd.DataFrame(
        {"id": ["test1", "test2"], "comment_text": ["hello friendly", "hostile insult"]}
    )
    sample = pd.DataFrame({"id": test.id, **{label: [0.0, 0.0] for label in LABELS}})
    source = tmp_path / "data.txt"
    source.write_text("synthetic fixture", encoding="utf-8")
    store = RunStore(tmp_path)
    monkeypatch.setattr(pipeline, "STORE", store)
    monkeypatch.setattr(pipeline, "DATA_FILES", (source,))
    monkeypatch.setattr(pipeline, "load_data", lambda: (train, test, sample))
    pipeline.train("E01", "E01")
    assert store.load_array("E01", "oof.npy").shape == (90, 6)
    assert store.load_array("E01", "test.npy").shape == (2, 6)
    saved = pd.read_csv(store.artifact_path("E01", "submission.csv"))
    assert saved.id.tolist() == ["test1", "test2"]
    assert store.record("E01").competition_id == "041"
    with pytest.raises(FileExistsError):
        pipeline.train("E01", "E01")
