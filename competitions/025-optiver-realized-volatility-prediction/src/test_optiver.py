"""分割・集約・採点の契約を小さなデータで検査する。"""

import json
import subprocess
import sys
from pathlib import Path

import features as optiver_features
import numpy as np
import polars as pl
import pytest
from features import KEYS, add_peers, aggregate, features, rmspe, validate_submission
from final import notebook_source, write_notebook
from run import splits_for

from kumite.checks import check_label_independence


def book_rows() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time_id": [1, 1, 2, 2, 3, 3],
            "seconds_in_bucket": [0, 599] * 3,
            "bid_price1": [1.0, 1.01, 1.0, 1.02, 1.0, 1.03],
            "ask_price1": [1.0, 1.01, 1.0, 1.02, 1.0, 1.03],
            "bid_size1": [1] * 6,
            "ask_size1": [1] * 6,
        }
    )


@pytest.mark.parametrize(
    "changes",
    [
        (pl.lit(0).alias("bid_size1"), pl.lit(0).alias("ask_size1")),
        (pl.lit(-1.0).alias("bid_price1"),),
        (pl.lit(float("nan")).alias("ask_price1"),),
        (pl.lit(float("inf")).alias("ask_price1"),),
        (pl.lit(None, dtype=pl.Float64).alias("ask_price1"),),
    ],
)
def test_rejects_invalid_book(tmp_path: Path, changes: tuple[pl.Expr, ...]) -> None:
    path = tmp_path / "stock_id=0" / "part.parquet"
    path.parent.mkdir()
    book_rows().with_columns(*changes).write_parquet(path)
    with pytest.raises(ValueError, match="価格または板数量"):
        aggregate(path, "book")


@pytest.mark.parametrize("price", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_invalid_trade(tmp_path: Path, price: float) -> None:
    path = tmp_path / "stock_id=0" / "part.parquet"
    path.parent.mkdir()
    pl.DataFrame(
        {
            "time_id": [1],
            "seconds_in_bucket": [0],
            "price": [price],
            "size": [1],
            "order_count": [1],
        }
    ).write_parquet(path)
    with pytest.raises(ValueError, match="価格または板数量"):
        aggregate(path, "trade")


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_rejects_non_finite_aggregated_rv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: float
) -> None:
    path = tmp_path / "stock_id=0" / "part.parquet"
    path.parent.mkdir()
    book_rows().write_parquet(path)
    monkeypatch.setattr(optiver_features, "rv", lambda _: pl.lit(value))
    with pytest.raises(ValueError, match="実現ボラティリティが非有限"):
        aggregate(path, "book")


def test_generated_notebook_executes_on_three_test_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = tmp_path / "input"
    data = inputs / "competitions" / "optiver"
    work = tmp_path / "working"
    work.mkdir()
    train = pl.DataFrame(
        {"stock_id": [0, 0, 0], "time_id": [1, 2, 3], "target": [0.01, 0.02, 0.03]}
    )
    test = (
        train.select("stock_id", "time_id")
        .reverse()
        .with_columns(row_id=pl.Series(["0-3", "0-2", "0-1"]))
    )
    for split in ("train", "test"):
        book = data / f"book_{split}.parquet" / "stock_id=0"
        trade = data / f"trade_{split}.parquet" / "stock_id=0"
        book.mkdir(parents=True)
        trade.mkdir(parents=True)
        book_rows().write_parquet(book / "part.parquet")
        book_rows().select(
            "time_id",
            "seconds_in_bucket",
            pl.col("bid_price1").alias("price"),
            pl.lit(1).alias("size"),
            pl.lit(1).alias("order_count"),
        ).write_parquet(trade / "part.parquet")
    train.write_csv(data / "train.csv")
    test.write_csv(data / "test.csv")
    (inputs / "train.csv").write_text("unrelated\n", encoding="utf-8")
    write_notebook(tmp_path, "test-owner", notebook_source("E01"))
    notebook = json.loads((tmp_path / "submit.ipynb").read_text(encoding="utf-8"))
    (cell,) = notebook["cells"]
    monkeypatch.setenv("OPTIVER_INPUT_ROOT", str(inputs))
    monkeypatch.setenv("OPTIVER_WORKING_DIR", str(work))
    subprocess.run([sys.executable, "-c", "".join(cell["source"])], check=True)
    validate_submission(work / "submission.csv", test["row_id"])


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
