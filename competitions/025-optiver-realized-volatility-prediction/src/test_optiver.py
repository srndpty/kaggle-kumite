"""分割・集約・採点の契約を小さなデータで検査する。"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from features import KEYS, add_peers, aggregate, features, rmspe, validate_submission
from run import splits_for

from kumite.checks import check_label_independence


def test_window_and_bucket_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "stock_id=0" / "data.parquet"
    path.parent.mkdir()
    pl.DataFrame(
        {
            "time_id": [2, 1, 1, 1],
            "seconds_in_bucket": [0, 450, 0, 500],
            "bid_price1": [100.0, 2.0, 1.0, 4.0],
            "ask_price1": [100.0, 2.0, 1.0, 4.0],
            "bid_size1": [1, 1, 1, 1],
            "ask_size1": [1, 1, 1, 1],
        }
    ).write_parquet(path)
    rows = aggregate(path, "book").sort("time_id")
    assert rows["book_rv"].to_list() == pytest.approx([np.sqrt(2) * np.log(2), None])
    assert rows["book_rv_150"].to_list() == pytest.approx([np.log(2), None])


def test_peer_median_excludes_self_and_preserves_keys() -> None:
    rows = pl.DataFrame(
        {"stock_id": [2, 1, 3, 2], "time_id": [0, 0, 0, 1], "book_rv": [3.0, 1.0, 7.0, 2.0]}
    )
    assert add_peers(rows)["peer_rv"].to_list() == [4.0, 5.0, 2.0, None]


def test_group_splits_cover_every_row() -> None:
    groups = np.repeat(np.arange(10), 3)
    splits = splits_for(groups)
    for train, valid in splits:
        assert not set(groups[train]) & set(groups[valid])
    np.testing.assert_array_equal(
        np.sort(np.concatenate([valid for _, valid in splits])), np.arange(30)
    )


def test_rmspe_checks_values() -> None:
    y = np.array([1.0, 2.0])
    assert rmspe(y, np.array([2.0, 1.0])) == pytest.approx(np.sqrt(0.625))
    with pytest.raises(ValueError):
        rmspe(y, np.array([1.0, -1.0]))


def test_missing_trades_and_label_independence(tmp_path: Path) -> None:
    book = tmp_path / "book_train.parquet" / "stock_id=0"
    trade = tmp_path / "trade_train.parquet" / "stock_id=0"
    book.mkdir(parents=True)
    trade.mkdir(parents=True)
    pl.DataFrame(
        {
            "time_id": [1, 1, 2, 2],
            "seconds_in_bucket": [0, 1, 0, 1],
            "bid_price1": [1.0, 2.0, 1.0, 3.0],
            "ask_price1": [1.0, 2.0, 1.0, 3.0],
            "bid_size1": [1, 1, 1, 1],
            "ask_size1": [1, 1, 1, 1],
        }
    ).write_parquet(book / "part.parquet")
    pl.DataFrame(
        {"time_id": [1], "seconds_in_bucket": [1], "price": [1.0], "size": [5], "order_count": [2]}
    ).write_parquet(trade / "part.parquet")
    rows = pl.DataFrame({"stock_id": [0, 0], "time_id": [2, 1]})
    summary = features(tmp_path, "train", rows)
    assert summary.select(KEYS).equals(rows)
    assert summary["volume"].to_list() == [0, 5]
    assert summary["orders"].to_list() == [0, 2]
    assert summary["trade_rv"].to_list() == [None, None]

    def build(y: np.ndarray) -> list[np.ndarray]:
        return [
            features(tmp_path, "train", rows.with_columns(target=pl.Series(y)))[
                "book_rv"
            ].to_numpy()
        ]

    check_label_independence(build, np.array([1.0, 2.0]), np.array([1.0, 3.0]))


def test_submission_rejects_wrong_order_and_values(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    ids = pl.Series("row_id", ["0-1", "0-2"])
    path.write_text("row_id,target\n0-1,0.01\n0-2,0.02\n", encoding="utf-8")
    validate_submission(path, ids)
    with pytest.raises(ValueError):
        validate_submission(path, ids.reverse())
    path.write_text("row_id,target\n0-1,0.01\n0-2,-1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_submission(path, ids)
