"""週別指標、分割、特徴量の情報制約の回帰テスト。"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pipeline import (
    AMOUNT,
    CREATED,
    DATE,
    HISTORY,
    ID,
    MAX_AMOUNT,
    fold_weeks,
    history_features,
    load,
    model_inputs,
    stability,
)


@pytest.mark.parametrize(
    ("pred", "score"),
    [
        ([0, 1, 0, 1], 1.0),
        ([1, 0, 0, 1], 0.0),
        ([0, 1, 1, 0], -176.0),
    ],
)
def test_stability_is_week_ordered(pred: list[int], score: float) -> None:
    order = [2, 3, 0, 1]
    result = stability(
        np.array([8, 8, 9, 9])[order], np.array([0, 1, 0, 1])[order], np.array(pred)[order]
    )
    assert result["score"] == pytest.approx(score)


def test_expanding_folds_cover_whole_weeks() -> None:
    weeks = np.repeat(np.arange(50), 2)
    target = np.tile(np.array([0, 1]), 50)
    splits = fold_weeks(weeks, target)
    for (train, valid), start in zip(splits, (11, 24, 37), strict=True):
        assert weeks[train].max() == start - 1
        np.testing.assert_array_equal(np.unique(weeks[valid]), np.arange(start, start + 13))
    with pytest.raises(ValueError, match="検証週"):
        fold_weeks(weeks, np.where(weeks == 49, 0, target))


def test_history_uses_only_dates_strictly_before_decision() -> None:
    decision = date(2024, 1, 3)
    base = pl.LazyFrame({ID: [1], DATE: [decision]})
    history = pl.LazyFrame(
        {
            ID: [1, 1, 1, 1],
            CREATED: [date(2024, 1, 1), date(2024, 1, 2), decision, date(2024, 1, 4)],
            AMOUNT: [30.0, 20.0, 100.0, 200.0],
        }
    )
    observed = history_features(base, history, max_amount=True).collect()
    assert observed.select(*HISTORY, MAX_AMOUNT).row(0) == (2, 1, 30.0)
    modified = history.with_columns(
        pl.when(pl.col(CREATED) >= decision)
        .then(pl.lit(date(2025, 1, 1)))
        .otherwise(pl.col(CREATED))
        .alias(CREATED)
    )
    assert history_features(base, modified, max_amount=True).collect().equals(observed)


def test_unseen_category_is_missing() -> None:
    train, query, categories = model_inputs(
        pl.DataFrame({"cat": ["seen"]}),
        pl.DataFrame({"cat": ["unseen"]}),
        ["cat"],
    )
    assert categories == ["cat"]
    assert train["cat"].iloc[0] == "seen"
    assert pd.isna(query["cat"].iloc[0])


def test_static_join_preserves_base_order_and_rejects_duplicates(tmp_path: Path) -> None:
    directory = tmp_path / "parquet_files" / "test"
    directory.mkdir(parents=True)
    pl.DataFrame({ID: [9, 2, 4], DATE: ["2024-01-01"] * 3}).write_parquet(
        directory / "test_base.parquet"
    )
    for index, case in enumerate([2, 4, 9]):
        pl.DataFrame({ID: [case], "valueA": [index]}).write_parquet(
            directory / f"test_static_0_{index}.parquet"
        )
    bureau = directory / "test_static_cb_0.parquet"
    pl.DataFrame({ID: [2], "valueA": [5]}).write_parquet(bureau)
    for index, case in enumerate([2, 4, 9]):
        pl.DataFrame({ID: [case], CREATED: ["2024-01-01"], AMOUNT: [1000.0]}).write_parquet(
            directory / f"test_applprev_1_{index}.parquet"
        )
    actual = load(tmp_path, "test", history=True, max_amount=True)
    assert actual[HISTORY[0]].to_list() == [0, 0, 0]
    assert actual[MAX_AMOUNT].null_count() == 3
    assert actual[ID].to_list() == [9, 2, 4]
    assert actual["static_0__valueA"].to_list() == [2, 0, 1]
    assert actual["static_cb_0__valueA"].to_list() == [None, 5, None]
    pl.DataFrame({ID: [2, 2], "valueA": [5, 7]}).write_parquet(bureau)
    with pytest.raises(pl.exceptions.ComputeError, match="1:1"):
        load(tmp_path, "test", history=False)


def test_constant_weekly_gini_has_no_penalties() -> None:
    result = stability(np.repeat(np.arange(3), 2), np.tile([0, 1], 3), np.tile([0.2, 0.8], 3))
    assert result["falling_penalty"] == pytest.approx(0, abs=1e-12)
    assert result["residual_std"] == pytest.approx(0, abs=1e-12)
