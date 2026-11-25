"""逐次再生と時系列分割の境界を検査する。"""

import numpy as np
import pandas as pd
import pytest
from evaluation import mae, splits_for
from features import BASE, Replay, feature_matrix, prediction_frame


@pytest.mark.parametrize("method", ["E00", "E01", "E02", "E03", "E04"])
def test_replay_matches_offline_features(method: str) -> None:
    rows = pd.DataFrame(1.0, index=range(5), columns=BASE)
    rows["date_id"] = [0, 0, 0, 0, 1]
    rows["seconds_in_bucket"] = [0, 0, 10, 30, 0]
    rows["stock_id"] = [0, 1, 0, 0, 0]
    rows["wap"] = [1.0, 5.0, 2.0, 3.0, 10.0]
    rows["target"] = 1.0

    replay = Replay(method)
    actual = pd.concat(
        [replay.transform(batch) for _, batch in rows.groupby(["date_id", "seconds_in_bucket"])],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(actual, feature_matrix(rows, method))
    changed = rows.copy()
    changed["target"] = -99
    changed.loc[3:, "wap"] = 999
    pd.testing.assert_frame_equal(actual.iloc[:3], feature_matrix(changed, method).iloc[:3])
    if method == "E02":
        np.testing.assert_allclose(
            actual["wap_delta_10"], [np.nan, np.nan, 1, np.nan, np.nan], equal_nan=True
        )


def test_forward_folds_cover_only_labeled_rows_once() -> None:
    days = np.repeat(np.arange(100), 2)
    y = np.ones(len(days))
    y[-1] = np.nan
    splits = splits_for(days, y)
    for train, valid in splits:
        assert days[train].max() < days[valid].min()
    np.testing.assert_array_equal(
        np.concatenate([valid for _, valid in splits]), np.arange(20, 199)
    )
    assert mae(np.array([1, 4]), np.array([0, 1])) == 2.0


def test_prediction_frame_validates_order_and_finiteness() -> None:
    batch = pd.DataFrame({"row_id": ["a", "b"]})
    sample = batch.astype({"row_id": "string"}).assign(target=0.0)
    assert prediction_frame(batch, sample, np.array([1.0, 2.0]))["target"].tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="Invalid"):
        prediction_frame(batch, sample.iloc[::-1], np.zeros(2))
    with pytest.raises(ValueError, match="Invalid"):
        prediction_frame(batch, sample, np.array([np.nan, 2.0]))


def test_p0_predicts_once_per_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import SimpleNamespace

    from p0 import SOURCE

    samples = [
        pd.DataFrame({"row_id": ["first", "second"], "target": 9.0}),
        pd.DataFrame({"row_id": ["third"], "target": 9.0}),
    ]
    predictions: list[pd.DataFrame] = []

    def iter_test():
        for index, sample in enumerate(samples):
            assert len(predictions) == index
            yield None, None, sample

    env = SimpleNamespace(iter_test=iter_test, predict=predictions.append)
    monkeypatch.setitem(sys.modules, "optiver2023", SimpleNamespace(make_env=lambda: env))
    exec(compile(SOURCE, "submit.ipynb", "exec"), {})
    assert len(predictions) == len(samples)
    assert all((prediction["target"] == 0).all() for prediction in predictions)


def test_exported_notebook_replays_and_predicts_once_per_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from types import SimpleNamespace

    import lightgbm as lgb
    from final import notebook_source

    rows = pd.DataFrame(1, index=range(4), columns=BASE)
    rows["row_id"] = ["a", "b", "c", "d"]
    rows["stock_id"] = [0, 1, 0, 1]
    rows["date_id"] = [0, 0, 1, 1]
    x = feature_matrix(rows, "E00")
    model = lgb.train(
        {"objective": "regression_l1", "verbosity": -1, "num_threads": 1},
        lgb.Dataset(x, label=np.arange(len(rows))),
        num_boost_round=1,
    )
    predictions: list[pd.DataFrame] = []

    def iter_test():
        for index, (_, batch) in enumerate(rows.groupby("date_id")):
            assert len(predictions) == index
            sample = batch.loc[:, ["row_id"]].astype("string").assign(target=np.nan)
            yield batch, None, sample

    env = SimpleNamespace(iter_test=iter_test, predict=predictions.append)
    monkeypatch.setitem(sys.modules, "optiver2023", SimpleNamespace(make_env=lambda: env))
    source = notebook_source(model.model_to_string(), "E00")
    exec(compile(source, "submit.ipynb", "exec"), {})
    output = pd.concat(predictions, ignore_index=True)
    assert output["row_id"].tolist() == rows["row_id"].tolist()
    np.testing.assert_array_equal(output["target"], model.predict(x, num_threads=1))
