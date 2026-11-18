"""指標・分割・fold 内変換・保存の契約。"""

import numpy as np
import pandas as pd
import pytest
from cv import make_folds
from dataset import submission
from features import build_features
from metrics import normalized_gini


def test_gini_ties_and_monotone() -> None:
    y = np.array([0, 0, 1, 1])
    assert normalized_gini(y, np.array([0.0, 0.1, 0.8, 1.0])) == 1
    assert normalized_gini(y, np.array([1.0, 0.8, 0.1, 0.0])) == -1
    assert normalized_gini(y, np.ones(4)) == 0
    assert normalized_gini(y, np.array([0.0, 1.0, 1.0, 1.0])) == pytest.approx(0.5)
    for bad_y, bad_p in ((np.zeros(4), np.ones(4)), (y, np.full(4, np.nan)), (y, np.ones(3))):
        with pytest.raises(ValueError):
            normalized_gini(bad_y, bad_p)


def test_duplicate_groups_and_reproducibility() -> None:
    x = pd.DataFrame({"a": np.repeat(np.arange(30), 2)})
    y = np.tile([0, 1], 30)
    folds, name = make_folds(x, y)
    assert "group" in name
    assert set(folds) == set(range(5))
    np.testing.assert_array_equal(folds[::2], folds[1::2])
    np.testing.assert_array_equal(folds, make_folds(x, y)[0])
    unique = pd.DataFrame({"a": np.arange(60)})
    assert "group" not in make_folds(unique, y)[1]


def test_fold_dictionary_and_missing_do_not_read_validation() -> None:
    tr = pd.DataFrame({"x": [-1.0, 2.0], "a_cat": [-1, 3]})
    va = pd.DataFrame({"x": [-1.0], "a_cat": [99]})
    frames, cats = build_features(tr, va, va, missing_nan=True, missing_count=True)
    assert cats == ["a_cat"]
    assert frames[0]["a_cat"].cat.categories.tolist() == [-1, 3]
    assert pd.isna(frames[1]["a_cat"].iloc[0])
    assert pd.isna(frames[0]["x"].iloc[0])
    assert frames[0]["missing_count"].iloc[0] == 2
    other, _ = build_features(tr, va * 2, va * 3, missing_nan=True, missing_count=True)
    pd.testing.assert_frame_equal(frames[0], other[0])
    numeric, used = build_features(tr, va, va, categorical=False)
    assert used == [] and numeric[0]["a_cat"].dtype == tr["a_cat"].dtype


def test_submission_id_and_probabilities() -> None:
    sample = pd.DataFrame({"id": [2, 1], "target": [0.0, 0.0]})
    assert submission(sample, np.array([2, 1]), np.array([0.1, 0.9]))["id"].tolist() == [2, 1]
    for ids, pred in (([1, 2], [0.1, 0.9]), ([2, 1], [0.1, 1.1])):
        with pytest.raises(ValueError):
            submission(sample, np.array(ids), np.array(pred))


def test_prior_pipeline_publishes_aligned_artifacts(tmp_path, monkeypatch) -> None:
    import json

    import pipeline

    from kumite.runs import RunStore

    n = 100
    train = pd.DataFrame({"id": np.arange(n), "target": np.tile([0, 1], n // 2), "a": np.arange(n)})
    test = pd.DataFrame({"id": [101, 100], "a": [1, 2]})
    sample = test[["id"]].assign(target=0.0)
    source = tmp_path / "source.csv"
    train.to_csv(source, index=False)
    store = RunStore(tmp_path)
    monkeypatch.setattr(pipeline, "STORE", store)
    monkeypatch.setattr(pipeline, "DATA_FILES", (source,))
    monkeypatch.setattr(pipeline, "load_data", lambda: (train, test, sample))
    pipeline.run("E00", "E00")
    assert len(store.records()) == 1
    assert store.load_array("E00", "oof.npy").shape == (n,)
    pred = pd.read_csv(store.artifact_path("E00", "submission.csv"))
    assert pred["id"].tolist() == [101, 100]
    metrics = json.loads(store.artifact_path("E00", "metrics.json").read_text())
    assert metrics["mean_fold_gini"] == 0
    assert len(pd.read_csv(store.artifact_path("E00", "folds.csv"))) == n


def test_load_reorders_test_to_sample_and_rejects_duplicate_id(tmp_path, monkeypatch) -> None:
    import dataset

    paths = tuple(tmp_path / n for n in ("train.csv", "test.csv", "sample_submission.csv"))
    pd.DataFrame({"id": [1, 2], "target": [0, 1], "a": [2, 3]}).to_csv(paths[0], index=False)
    pd.DataFrame({"id": [3, 4], "a": [30, 40]}).to_csv(paths[1], index=False)
    pd.DataFrame({"id": [4, 3], "target": [0.0, 0.0]}).to_csv(paths[2], index=False)
    monkeypatch.setattr(dataset, "DATA_FILES", paths)
    _, test, _ = dataset.load_data()
    assert test["a"].tolist() == [40, 30]
    pd.DataFrame({"id": [3, 3], "a": [30, 40]}).to_csv(paths[1], index=False)
    with pytest.raises(ValueError, match="ID"):
        dataset.load_data()


def test_calc_ablation_changes_only_calc_columns() -> None:
    frame = pd.DataFrame({"ps_car_01": [-1, 2], "ps_calc_01": [1, 3], "ps_ind_01_cat": [0, 1]})
    built, cats = build_features(frame, frame, frame, categorical=False, drop_calc=True)
    assert cats == []
    for result in built:
        pd.testing.assert_frame_equal(result, frame.drop(columns="ps_calc_01"))
