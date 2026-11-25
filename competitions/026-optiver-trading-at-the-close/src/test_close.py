"""逐次再生と時系列分割の境界を検査する。"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from deployment.package_kaggle import commit_deployment, deployment_manifest
from evaluation import mae, splits_for
from features import BASE, KEYS, Replay, feature_matrix, prediction_frame


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


def _clock_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [f"r{i}" for i in range(9)],
            "date_id": [0, 0, 0, 0, 0, 0, 0, 1, 1],
            "seconds_in_bucket": [0, 0, 10, 10, 60, 70, 240, 0, 0],
            "stock_id": [0, 1, 0, 1, 0, 0, 0, 0, 1],
            "imbalance_size": [4, 2, 5, 1, 3, 8, 4, 4, 2],
            "imbalance_buy_sell_flag": [1, -1, 1, -1, 1, 1, 1, 1, -1],
            "reference_price": 1.0,
            "matched_size": [6, 2, 5, 3, 7, 2, 6, 6, 2],
            "far_price": 1.0,
            "near_price": 1.0,
            "bid_price": [10, 10, 10, 10, 10, 10, 10, 20, 20],
            "bid_size": [3, 1, 2, 4, 2, 1, 3, 3, 1],
            "ask_price": [12, 14, 15, 11, 13, 16, 12, 22, 24],
            "ask_size": [1, 3, 2, 1, 2, 1, 1, 1, 3],
            "wap": [11, 13, 14, 10.5, 12, 15, 11, 21, 23],
        }
    )


def _assert_clock_features(method: str, rows: pd.DataFrame, features: pd.DataFrame) -> None:
    if method == "E00":
        assert list(features.columns) == BASE
    elif method == "E01":
        total = rows["bid_size"] + rows["ask_size"]
        auction = rows["imbalance_size"] + rows["matched_size"]
        np.testing.assert_allclose(
            features["spread"], rows["ask_price"] - rows["bid_price"], rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            features["book_ratio"],
            (rows["bid_size"] - rows["ask_size"]) / total,
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            features["auction_ratio"], rows["imbalance_size"] / auction, rtol=1e-5, atol=1e-5
        )
    elif method == "E02":
        np.testing.assert_allclose(
            features["wap_delta_10"],
            [np.nan, np.nan, 3, -2.5, np.nan, 3, np.nan, np.nan, np.nan],
            equal_nan=True,
        )
        np.testing.assert_allclose(
            features["wap_delta_60"],
            [np.nan, np.nan, np.nan, np.nan, 1, 1, np.nan, np.nan, np.nan],
            equal_nan=True,
        )
    elif method == "E03":
        np.testing.assert_allclose(features.loc[0:1, "wap_median_delta"], [-1, 1])
        np.testing.assert_allclose(features.loc[0:1, "wap_rank"], [0.5, 1])
    elif method == "E04":
        assert bool(features.loc[6, "after_240"])
        assert not bool(features.loc[5, "after_240"])
        assert not bool(features.loc[6, "after_300"])


def test_replay_rejects_invalid_batches() -> None:
    columns = list(dict.fromkeys([*KEYS, *BASE]))
    replay = Replay("E00")
    with pytest.raises(ValueError, match="unique"):
        replay.transform(pd.DataFrame(columns=columns))
    rows = _clock_rows()
    missing = rows.iloc[[0]].copy()
    missing["stock_id"] = np.nan
    with pytest.raises(ValueError, match="unique"):
        replay.transform(missing)
    with pytest.raises(ValueError, match="unique"):
        replay.transform(pd.concat([rows.iloc[[0]], rows.iloc[[0]]], ignore_index=True))
    with pytest.raises(ValueError, match="unique"):
        replay.transform(rows.iloc[[0, 4]])
    replay.transform(rows.iloc[[0]])
    with pytest.raises(ValueError, match="chronological"):
        replay.transform(rows.iloc[[1]])
    with pytest.raises(ValueError, match="unique"):
        Replay("E02").transform(pd.concat([rows.iloc[[0]], rows.iloc[[0]]], ignore_index=True))
    replay.transform(rows.iloc[[2]])


@pytest.mark.parametrize("method", ["E00", "E01", "E02", "E03", "E04"])
def test_clock_replay_matches_method_features(method: str) -> None:
    rows = _clock_rows()
    replay = Replay(method)
    actual = pd.concat(
        [
            replay.transform(batch)
            for _, batch in rows.groupby(["date_id", "seconds_in_bucket"], sort=True)
        ],
        ignore_index=True,
    )
    expected = feature_matrix(rows, method)
    pd.testing.assert_frame_equal(actual, expected)
    _assert_clock_features(method, rows, actual)


def _train_clock(method: str, rows: pd.DataFrame) -> tuple[lgb.Booster, pd.DataFrame]:
    features = feature_matrix(rows, method)
    _assert_clock_features(method, rows, features)
    model = lgb.train(
        {
            "objective": "regression_l1",
            "verbosity": -1,
            "num_threads": 1,
            "min_data_in_leaf": 1,
            "deterministic": True,
            "seed": 1,
            "force_col_wise": True,
        },
        lgb.Dataset(features, label=np.arange(len(rows), dtype=float)),
        num_boost_round=3,
    )
    assert model.feature_name() == list(features.columns)
    return model, features


def _corrupt_batch(
    batch: pd.DataFrame, sample: pd.DataFrame, defect: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if defect == "missing":
        sample = sample.copy()
        sample.loc[sample.index[-1], "row_id"] = "absent"
        return batch, sample
    if defect == "duplicate":
        batch = batch.copy()
        batch.loc[batch.index[-1], "row_id"] = batch.iloc[0]["row_id"]
        sample = batch.loc[:, ["row_id"]].astype("string").assign(target=np.nan)
        return batch, sample
    if defect == "reordered":
        return batch, sample.iloc[::-1]
    raise AssertionError(defect)


def _run_notebook(
    monkeypatch: pytest.MonkeyPatch, source: str, rows: pd.DataFrame, defect: str | None = None
) -> list[pd.DataFrame]:
    import sys
    from types import SimpleNamespace

    predictions: list[pd.DataFrame] = []
    groups = list(rows.groupby(["date_id", "seconds_in_bucket"], sort=True))

    def iter_test():
        for index, (_, batch) in enumerate(groups):
            assert len(predictions) == index
            delivered = batch
            sample = delivered.loc[:, ["row_id"]].astype("string").assign(target=np.nan)
            if defect is not None and index == len(groups) - 1:
                delivered, sample = _corrupt_batch(delivered, sample, defect)
            yield delivered, None, sample

    env = SimpleNamespace(iter_test=iter_test, predict=predictions.append)
    monkeypatch.setitem(sys.modules, "optiver2023", SimpleNamespace(make_env=lambda: env))
    exec(compile(source, "submit.ipynb", "exec"), {})
    return predictions


@pytest.mark.parametrize("method", ["E00", "E01", "E02", "E03", "E04"])
def test_exported_notebook_replays_and_predicts_once_per_batch(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from final import notebook_source

    rows = _clock_rows()
    model, features = _train_clock(method, rows)
    predictions = _run_notebook(monkeypatch, notebook_source(model.model_to_string(), method), rows)
    output = pd.concat(predictions, ignore_index=True)
    assert len(predictions) == rows.groupby(["date_id", "seconds_in_bucket"]).ngroups
    assert output["row_id"].tolist() == rows["row_id"].tolist()
    np.testing.assert_array_equal(output["target"], model.predict(features, num_threads=1))


@pytest.mark.parametrize("method", ["E00", "E01", "E02", "E03", "E04"])
@pytest.mark.parametrize("defect", ["missing", "duplicate", "reordered"])
def test_exported_notebook_rejects_bad_row_ids(
    method: str, defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from final import notebook_source

    rows = _clock_rows()
    model, _features = _train_clock(method, rows)
    source = notebook_source(model.model_to_string(), method)
    with pytest.raises(ValueError, match="Invalid"):
        _run_notebook(monkeypatch, source, rows, defect)


def test_deployment_rewrite_requires_the_same_manifest(tmp_path: Path) -> None:
    payload = b"print('model')\n"
    loader = "print('loader')\n"
    manifest = deployment_manifest("F00", payload, loader, "owner/dataset")
    payload_name = f"optiver-close-{manifest['payload_sha256'][:16]}.py.txt"
    dataset_metadata: dict[str, object] = {
        "id": "owner/dataset",
        "title": "Payload",
        "licenses": [{"name": "unknown"}],
    }
    kernel_metadata: dict[str, object] = {
        "id": "owner/kernel",
        "title": "Close",
        "code_file": "submit.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": ["optiver-trading-at-the-close"],
        "dataset_sources": ["owner/dataset"],
    }

    def place(
        manifest: dict[str, str] = manifest,
        payload_name: str = payload_name,
        payload: bytes = payload,
    ) -> None:
        commit_deployment(
            root,
            manifest=manifest,
            payload_name=payload_name,
            payload=payload,
            dataset_metadata=dataset_metadata,
            loader_source=loader,
            kernel_id="owner/kernel",
            title="Close",
            competition="optiver-trading-at-the-close",
            kernel_metadata=kernel_metadata,
        )

    root = tmp_path / "F00"
    place()
    assert json.loads((root / "deployment-manifest.json").read_text(encoding="utf-8")) == manifest
    place()
    with pytest.raises(FileExistsError, match="置換できません"):
        place(manifest=dict(manifest, dataset_id="owner/other"))
    assert (root / "dataset" / payload_name).read_bytes() == payload
    with pytest.raises(FileExistsError, match="置換できません"):
        place(payload=b"print('other')\n")
    assert (root / "dataset" / payload_name).read_bytes() == payload
    with pytest.raises(FileExistsError, match="置換できません"):
        place(payload_name="optiver-close-aaaaaaaaaaaaaaaa.py.txt")
