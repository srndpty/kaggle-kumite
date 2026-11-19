from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from features import EXPERIMENTS, PairFeatures
from inputs import official_data_hash, read_verified_train
from predict import write_submission
from split import make_splits, question_nodes
from train import fit_model
from train import train as train_run


def test_question_identity_and_splits() -> None:
    frame = pd.DataFrame(
        {
            "qid1": [1, 3, 5],
            "qid2": [2, 4, 6],
            "question1": ["Ａ ", "a", None],
            "question2": [None, "other", None],
        }
    )
    nodes = question_nodes(frame)
    assert nodes[0, 0] == nodes[1, 0]
    assert len(set(nodes[[0, 2], 1])) == 2
    nodes = np.arange(400).reshape(-1, 2)
    y = np.arange(200) % 2
    version, first = make_splits(nodes, y, 42)
    assert version == "components-v1-seed42"
    _, second = make_splits(nodes, y, 42)
    for (train, valid), (again_train, again_valid) in zip(first, second, strict=True):
        np.testing.assert_array_equal(train, again_train)
        np.testing.assert_array_equal(valid, again_valid)
        assert not np.intersect1d(nodes[train], nodes[valid]).size


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_symmetric_features_and_fold_vocabulary(experiment: str) -> None:
    pairs = np.array([["where is it", "where is this"], ["no tea 2", "tea 3"]])
    features = PairFeatures(experiment)
    features.fit(pairs)
    before = features.word.vocabulary_.copy()
    np.testing.assert_allclose(features.transform(pairs), features.transform(pairs[:, ::-1]))
    features.transform(np.array([["unknown", "another"]]))
    assert before == features.word.vocabulary_
    assert "unknown" not in before


def test_largest_component_uses_node_split() -> None:
    row = np.arange(1_000)
    nodes = np.column_stack([row, row + 1])
    version, splits = make_splits(nodes, row % 2, 42)
    assert version == "nodes-v1-seed42"
    assert sum(len(valid) for _, valid in splits) < len(row)
    for train, valid in splits:
        assert not np.intersect1d(nodes[train], nodes[valid]).size


def test_fit_to_submission(tmp_path: Path) -> None:
    pairs = np.array([["same question", "same question"], ["one", "two"]])
    features, model = fit_model(pairs, np.array([1, 0]), "E01", 42)
    pred = model.predict_proba(features.transform(pairs))[:, 1]
    ids = pd.Series([9, 2])
    output = tmp_path / "submission.csv"
    write_submission(ids, pred, output)
    frame = pd.read_csv(output)
    assert frame["test_id"].to_list() == [9, 2]
    np.testing.assert_allclose(frame["is_duplicate"], pred)
    with pytest.raises(ValueError):
        write_submission(ids, np.array([0.5, np.nan]), output)


def test_small_cv_publishes_submission(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    row = np.arange(100)
    pd.DataFrame(
        {
            "id": row,
            "qid1": 2 * row,
            "qid2": 2 * row + 1,
            "question1": [f"first question {i}" for i in row],
            "question2": [f"second question {i}" for i in row],
            "is_duplicate": row % 2,
        }
    ).to_csv(data / "train.csv", index=False)
    pd.DataFrame(
        {"test_id": [9, 2, 100], "question1": ["first"] * 3, "question2": ["second"] * 3}
    ).to_csv(data / "test.csv", index=False)
    pd.DataFrame({"test_id": [9, 2], "is_duplicate": [0, 0]}).to_csv(
        data / "sample_submission.csv", index=False
    )
    train_run("E01", "E01", 42, tmp_path)
    run = tmp_path / "outputs" / "runs" / "E01"
    assert pd.read_csv(run / "submission.csv")["test_id"].to_list() == [9, 2]
    assert np.isfinite(np.load(run / "oof.npy")).all()

    verified_hash = official_data_hash(tmp_path)
    training_file = data / "train.csv"
    frame = pd.read_csv(training_file)
    frame.loc[0, "is_duplicate"] = 1
    frame.to_csv(training_file, index=False)
    with pytest.raises(ValueError, match="公式データのハッシュ"):
        read_verified_train(tmp_path, verified_hash)
