"""銘柄・時間バケット単位の観測特徴とモデル。"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

KEYS = ["stock_id", "time_id"]
SECOND = "seconds_in_bucket"
METHODS = {
    "E00": ["book_rv"],
    "E01": ["book_rv", "book_rv_300", "book_rv_150"],
    "E02": ["book_rv", "trade_rv", "volume", "orders"],
    "E03": ["book_rv", "peer_rv"],
    "E04": ["book_rv", "book_rv_rank"],
}
PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "seed": 42,
    "verbosity": -1,
    "num_threads": 8,
    "deterministic": True,
    "force_col_wise": True,
}
ROUNDS = 500


def rv(price: pl.Expr) -> pl.Expr:
    """集計窓の内側で、観測された連続価格から実現ボラティリティを求める。"""
    returns = price.cast(pl.Float64).log().diff()
    return pl.when(returns.count() > 0).then(returns.pow(2).sum().sqrt())


def aggregate(path: Path, kind: str) -> pl.DataFrame:
    series = pl.read_parquet(path).sort(["time_id", SECOND])
    stock = int(path.parent.name.removeprefix("stock_id="))
    if kind == "book":
        bid = pl.col("bid_size1").cast(pl.Float64)
        ask = pl.col("ask_size1").cast(pl.Float64)
        wap = (pl.col("bid_price1") * ask + pl.col("ask_price1") * bid) / (bid + ask)
        stats = [
            rv(wap).alias("book_rv"),
            rv(wap.filter(pl.col(SECOND) >= 300)).alias("book_rv_300"),
            rv(wap.filter(pl.col(SECOND) >= 450)).alias("book_rv_150"),
        ]
    else:
        stats = [
            rv(pl.col("price")).alias("trade_rv"),
            pl.col("size").sum().alias("volume"),
            pl.col("order_count").sum().alias("orders"),
        ]
    return (
        series.group_by("time_id")
        .agg(stats)
        .with_columns(
            pl.lit(stock, dtype=pl.Int64).alias("stock_id"),
            pl.col("time_id").cast(pl.Int64),
        )
    )


def input_paths(data: Path, split: str, kind: str) -> list[Path]:
    paths = sorted((data / f"{kind}_{split}.parquet").glob("stock_id=*/*.parquet"))
    if not paths or len({p.parent for p in paths}) != len(paths):
        raise ValueError(f"Parquet パーティションの構成が不正です: {kind}_{split}")
    return paths


def add_peers(frame: pl.DataFrame) -> pl.DataFrame:
    peers = frame.select("time_id", pl.col("stock_id").alias("peer"), "book_rv").drop_nulls(
        "book_rv"
    )
    medians = (
        frame.select(KEYS)
        .join(peers, on="time_id")
        .filter(pl.col("stock_id") != pl.col("peer"))
        .group_by(KEYS)
        .agg(pl.col("book_rv").median().alias("peer_rv"))
    )
    return frame.join(medians, on=KEYS, how="left", validate="1:1")


def load_rows(data: Path, split: str) -> pl.DataFrame:
    rows = pl.read_csv(data / f"{split}.csv")
    if rows.is_empty() or rows.select(KEYS).n_unique() != len(rows):
        raise ValueError("銘柄・時間バケットのキーが重複しています")
    if rows.select(KEYS).null_count().row(0) != (0, 0):
        raise ValueError("銘柄・時間バケットのキーに欠損があります")
    return rows


def features(data: Path, split: str, rows: pl.DataFrame) -> pl.DataFrame:
    frame = rows.select(KEYS)
    for kind in ("book", "trade"):
        summary = pl.concat([aggregate(path, kind) for path in input_paths(data, split, kind)])
        frame = frame.join(summary, on=KEYS, how="left", validate="1:1")
    return add_peers(
        frame.with_columns(
            pl.col("volume", "orders").fill_null(0),
            (pl.col("book_rv").rank("average") / pl.col("book_rv").count())
            .over("time_id")
            .alias("book_rv_rank"),
        )
    )


def predict_submission(data: Path, method: str, output: Path) -> None:
    train = load_rows(data, "train")
    test = load_rows(data, "test")
    y = train["target"].to_numpy()
    rmspe(y, np.zeros_like(y))
    columns = METHODS[method]
    train_features = features(data, "train", train)
    test_features = features(data, "test", test)
    for rows, matrix in ((train, train_features), (test, test_features)):
        if not matrix.select(KEYS).equals(rows.select(KEYS)):
            raise ValueError("集約後の行キーが一致しません")
    prediction = fit_predict(
        train_features.select(columns).to_numpy(), y, test_features.select(columns).to_numpy()
    )
    test.select("row_id").with_columns(target=pl.Series(prediction)).write_csv(output)
    validate_submission(output, test["row_id"])


def validate_submission(path: Path, ids: pl.Series) -> None:
    submission = pl.read_csv(path, schema_overrides={"row_id": pl.String})
    if (
        submission.columns != ["row_id", "target"]
        or ids.null_count()
        or ids.n_unique() != len(ids)
        or not submission["row_id"].equals(ids)
    ):
        raise ValueError("提出列または行 ID が一致しません")
    prediction = submission["target"].to_numpy()
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("提出予測が不正です")


def rmspe(y: np.ndarray, prediction: np.ndarray) -> float:
    if (
        y.shape != prediction.shape
        or not np.isfinite(y).all()
        or not np.isfinite(prediction).all()
        or (y <= 0).any()
        or (prediction < 0).any()
    ):
        raise ValueError("RMSPE の正解または予測が不正です")
    return float(np.sqrt(np.mean(np.square((y - prediction) / y))))


def fit_predict(x: np.ndarray, y: np.ndarray, query: np.ndarray) -> np.ndarray:
    model = lgb.train(
        PARAMS,
        lgb.Dataset(x, label=y, weight=1 / y**2),
        num_boost_round=ROUNDS,
    )
    return np.maximum(np.asarray(model.predict(query), dtype=np.float64), 0)
