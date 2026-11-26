"""締切の順守、ラベル独立性、候補の規則、採点契約、提出形式を検証する。"""

import json
from dataclasses import replace
from pathlib import Path

import final
import numpy as np
import polars as pl
import pytest
from candidates import Rule, Snapshot, candidate_matrix
from data import CATEGORIES, Tables
from evaluation import (
    K,
    assessor,
    candidate_metrics,
    scores,
    split_hash,
    validation_windows,
    window_truth,
)
from features import FEATURES, FeatureBuilder
from final import submission_frame, validate_submission
from model import METHODS, label_candidates, recommend, training_rows

from kumite.experiments import ExperimentRecord

N_ARTICLES = 320
WIDTH = 100
N_CUSTOMERS = 300
N_DAYS = 70


def transactions(seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = 30_000
    return (
        pl.DataFrame(
            {
                "day": rng.integers(0, N_DAYS, n).astype(np.int32),
                # 末尾の 20 人には取引がない
                "customer": rng.integers(0, N_CUSTOMERS - 20, n).astype(np.int32),
                "article": rng.integers(0, N_ARTICLES, n).astype(np.int32),
                "price": rng.random(n),
                "channel": rng.integers(1, 3, n).astype(np.int32),
            }
        )
        .sort("day", "customer", "article", "price", "channel")
        .unique(["day", "customer", "article"], keep="first", maintain_order=True)
    )


@pytest.fixture(scope="module")
def tables() -> Tables:
    ids = np.arange(N_ARTICLES, dtype=np.int32)
    articles = pl.DataFrame({"article": ids, **{c: ids % 7 for c in CATEGORIES}})
    customers = np.arange(N_CUSTOMERS, dtype=np.int32)
    ages = pl.DataFrame({"customer": customers, "age": (customers % 60 + 18).astype(np.float32)})
    return Tables(
        transactions(0),
        articles,
        ages,
        pl.Series("customer_id", [f"c{i:04d}" for i in customers]),
        pl.Series("article_id", [f"0{i:09d}" for i in ids]),
    )


@pytest.mark.parametrize(
    "name", ["E00", "E01", "E02", "E03", "E04", "E06", "E07", "E08", "E09", "E10"]
)
def test_predictions_ignore_transactions_from_cutoff_on(tables: Tables, name: str) -> None:
    """検証週のラベル（締切以降の購入）を差し替えても、予測・候補が変わらない。"""
    cutoff = N_DAYS - 7
    customers = np.arange(N_CUSTOMERS, dtype=np.int32)
    other = pl.concat(
        [
            tables.transactions.filter(pl.col("day") < cutoff),
            transactions(1).filter(pl.col("day") >= cutoff),
        ]
    )
    assert not other.equals(tables.transactions)
    first = recommend(tables, METHODS[name], cutoff, customers)
    second = recommend(replace(tables, transactions=other), METHODS[name], cutoff, customers)
    np.testing.assert_array_equal(first.top, second.top)
    np.testing.assert_array_equal(first.pool, second.pool)


def test_training_features_use_only_days_before_label_week(tables: Tables) -> None:
    method = METHODS["E04"]
    start = N_DAYS - 14
    rows = training_rows(tables, method, start)
    x, y, groups = rows.x, rows.y, rows.groups
    assert x.shape == (groups * WIDTH, len(FEATURES)) and y.shape == (groups * WIDTH,)
    assert y.reshape(-1, WIDTH).any(axis=1).all()
    # ラベル週より後の取引は、候補・特徴・ラベルのどれにも影響しない
    cut = replace(tables, transactions=tables.transactions.filter(pl.col("day") < start + 7))
    again = training_rows(cut, method, start)
    np.testing.assert_array_equal(x, again.x)
    np.testing.assert_array_equal(y, again.y)


def test_training_stats_count_excluded_customers(tables: Tables) -> None:
    start = N_DAYS - 7
    rows = training_rows(tables, METHODS["E04"], start)
    truth = window_truth(tables.transactions, start)
    stats = rows.stats
    assert stats["buyers"] == len(truth.customers) and stats["kept"] == rows.groups
    assert stats["excluded_share"] == pytest.approx(1 - rows.groups / len(truth.customers))
    per_group = rows.y.reshape(rows.groups, WIDTH).sum(axis=1)
    assert stats["positives_per_kept"] == pytest.approx(per_group.mean())
    assert stats["positive_rate"] == pytest.approx(rows.y.mean())
    assert stats["purchases_per_buyer"] == pytest.approx(truth.n_relevant.mean())


def test_label_pool_trains_on_the_same_customers_as_the_narrow_candidates(tables: Tables) -> None:
    """E10 は候補 300 件のまま、E04 と同じ顧客だけを教師にする。"""
    start = N_DAYS - 7
    kept: list[np.ndarray] = []
    pools: list[np.ndarray] = []
    customers = window_truth(tables.transactions, start).customers
    for name in ("E04", "E10", "E08"):
        method = METHODS[name]
        snapshot = Snapshot.at(tables.transactions, tables.articles, start, method.rule)
        frame, _, keep, _ = label_candidates(tables, snapshot, method.label_pool)
        pools.append(candidate_matrix(frame, customers, method.rule.width, N_ARTICLES))
        kept.append(keep)
    np.testing.assert_array_equal(pools[0], pools[1][:, :WIDTH])
    np.testing.assert_array_equal(kept[0], kept[1])
    assert kept[2].sum() >= kept[1].sum()


def test_export_requires_a_fixed_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """選定記録がなければ、データを読む前に失敗し、記録も作らない。"""

    def fail() -> None:
        raise AssertionError("選定記録の前にデータを読みました")

    monkeypatch.setattr(final, "COMP", tmp_path)
    monkeypatch.setattr(final, "load_tables", fail)
    with pytest.raises(FileNotFoundError, match="select"):
        final.export(tmp_path / "plan.json", "S00", "F01")
    assert not list(tmp_path.iterdir())


def test_labels_do_not_change_features_of_the_same_customers(tables: Tables) -> None:
    start = N_DAYS - 7
    customers = np.arange(50, dtype=np.int32)
    other = pl.concat(
        [
            tables.transactions.filter(pl.col("day") < start),
            transactions(2).filter(pl.col("day") >= start),
        ]
    )
    rule = METHODS["E04"].rule
    results: list[np.ndarray] = []
    for rows in (tables.transactions, other):
        snapshot = Snapshot.at(rows, tables.articles, start, rule)
        builder = FeatureBuilder(replace(tables, transactions=rows), snapshot)
        results.append(builder.matrix(snapshot.candidates(customers)))
    np.testing.assert_array_equal(results[0], results[1])


ARTICLES = pl.DataFrame(
    {"article": np.arange(N_ARTICLES), "product_code": np.arange(N_ARTICLES) // 2},
    schema={"article": pl.Int32, "product_code": pl.Int32},
)


def small(rows: list[tuple[int, int, int]]) -> pl.DataFrame:
    """(日, 顧客, 商品)。人気候補が足りるように、顧客 999 が全商品を締切前日に買う。"""
    filler = [(29, 999, a) for a in range(N_ARTICLES)]
    days, customers, articles = zip(*(rows + filler), strict=True)
    return pl.DataFrame(
        {"day": days, "customer": customers, "article": articles},
        schema={"day": pl.Int32, "customer": pl.Int32, "article": pl.Int32},
    )


def test_rule_order_is_own_then_covisit_then_popular() -> None:
    rows = [
        (10, 1, 150),
        (20, 1, 140),
        *((15, c, a) for c in (2, 3, 4) for a in (140, 130)),
        *((28, c, 120) for c in range(10, 20)),
    ]
    snapshot = Snapshot.at(small(rows), ARTICLES, 30, Rule(own=True, half_life=None, covisit=True))
    frame = snapshot.candidates(np.array([1, 5], dtype=np.int32))
    matrix = candidate_matrix(frame, np.array([1, 5]), WIDTH, N_ARTICLES)
    assert matrix[0, :4].tolist() == [140, 150, 130, 120]
    # 履歴のない顧客は人気順だけ
    assert matrix[1, 0] == 120
    assert frame.filter(pl.col("customer") == 5)["own_rank"].null_count() == WIDTH


def test_own_items_outside_the_recent_window_are_ignored() -> None:
    snapshot = Snapshot.at(
        small([(1, 1, 150)]), ARTICLES, 30, Rule(own=True, half_life=None, covisit=False)
    )
    frame = snapshot.candidates(np.array([1], dtype=np.int32))
    assert frame["own_rank"].null_count() == WIDTH


def test_longer_own_history_and_same_product_are_added() -> None:
    rule = Rule(own=True, half_life=None, covisit=False, own_days=365, product=True)
    frame = Snapshot.at(small([(1, 1, 150)]), ARTICLES, 30, rule).candidates(
        np.array([1], dtype=np.int32)
    )
    # 150 と 151 は同じ product_code
    assert frame["article"].to_list()[:2] == [150, 151]
    assert frame["product_rank"].to_list()[:2] == [None, 0]


def test_half_life_prefers_newer_sales() -> None:
    rows = [*((20, c, 150) for c in range(8)), *((29, c, 140) for c in range(4))]
    rule = Rule(own=False, half_life=3.0, covisit=False)
    assert Snapshot.at(small(rows), ARTICLES, 30, rule).popular["article"][0] == 140


def test_scores_and_candidate_metrics_by_hand() -> None:
    rows = pl.DataFrame(
        {"day": [0, 7, 7, 14], "customer": [1, 1, 1, 2], "article": [3, 20, 30, 5]},
        schema={"day": pl.Int32, "customer": pl.Int32, "article": pl.Int32},
    )
    windows = validation_windows(rows, 20)
    assert [w.start for w in windows] == [0, 7, 14]
    oof = np.tile(np.arange(K), (3, 1))
    oof[1, 1] = 20
    fold, total = scores(windows, oof, N_ARTICLES)
    np.testing.assert_allclose(fold, [1 / 4, (1 / 2) / 2, 1 / 6])
    np.testing.assert_allclose(total, np.mean(fold))
    metrics = candidate_metrics(windows[1], oof[1:2], N_ARTICLES)
    assert metrics["recall"] == 0.5 and metrics["map_upper_bound"] == 0.5


def test_assessor_rejects_tampered_outputs(tmp_path: Path) -> None:
    rows = transactions(0)
    windows = validation_windows(rows, N_DAYS - 1)
    total = sum(len(w.customers) for w in windows)
    oof = np.tile(np.arange(K, dtype=np.int32), (total, 1))
    fold, score = scores(windows, oof, N_ARTICLES)
    record = ExperimentRecord(
        competition_id="x",
        run_id="E00",
        metric="map@12",
        greater_is_better=True,
        data_version="v",
        data_hash="d",
        split_version="s",
        split_hash=split_hash(windows),
        seed=42,
        hypothesis="",
        changed_one_thing="",
        model="",
        fold_scores=fold,
        oof_score=score,
        code_version="c",
    )
    assess = assessor(windows, N_ARTICLES, "d")
    (tmp_path / "diagnostics.json").write_text(json.dumps([]), encoding="utf-8")
    np.save(tmp_path / "oof.npy", oof)
    assert assess(lambda name: tmp_path / name, record) == (fold, score)
    with pytest.raises(ValueError, match="ハッシュ"):
        assess(lambda name: tmp_path / name, replace(record, split_hash="other"))
    for bad in (oof[:-1], np.zeros_like(oof), oof + N_ARTICLES):
        np.save(tmp_path / "oof.npy", bad)
        with pytest.raises(ValueError):
            assess(lambda name: tmp_path / name, record)


def test_submission_round_trip_and_rejections(tables: Tables, tmp_path: Path) -> None:
    ranked = np.tile(np.arange(K), (N_CUSTOMERS, 1))
    frame = submission_frame(ranked, tables.customer_ids, tables.article_ids)
    path = tmp_path / "submission.csv"
    frame.write_csv(path)
    assert frame["prediction"][0].split(" ")[:2] == ["0000000000", "0000000001"]
    validate_submission(path, tables.customer_ids, tables.article_ids)
    for bad in (
        frame.reverse(),
        frame.with_columns(prediction=pl.lit(" ".join(["0000000001"] * K))),
        frame.with_columns(prediction=pl.lit("0000000001")),
        frame.with_columns(prediction=pl.col("prediction").str.replace("0000000001", "9999999999")),
    ):
        bad.write_csv(path)
        with pytest.raises(ValueError):
            validate_submission(path, tables.customer_ids, tables.article_ids)
