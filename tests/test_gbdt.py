"""LightGBM の回帰（early stopping と学習し直し）。"""

import numpy as np
import pandas as pd

from kumite.gbdt import lightgbm_regression

PARAMS = {"learning_rate": 0.1, "num_leaves": 7, "seed": 0, "min_data_in_leaf": 5}


def _data(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"a": rng.normal(size=n), "g": rng.integers(0, 3, size=n)})
    y = 2 * x["a"].to_numpy() + x["g"].to_numpy() + rng.normal(scale=0.1, size=n)
    return x, y


def test_early_stopping_records_best_iteration_and_shapes() -> None:
    x_tr, y_tr = _data(300, 0)
    x_va, y_va = _data(100, 1)
    x_te, _ = _data(50, 2)

    out = lightgbm_regression(
        x_tr,
        y_tr,
        x_va,
        y_va,
        x_te,
        {**PARAMS, "num_boost_round": 500, "early_stopping": 20},
        ["g"],
    )

    assert out.valid.shape == (100,) and out.query.shape == (50,)
    assert 1 <= out.info["best_iteration"] <= 500
    assert np.sqrt(np.mean((out.valid - y_va) ** 2)) < 0.5


def test_fixed_rounds_without_validation_rows() -> None:
    x_tr, y_tr = _data(300, 0)
    x_te, _ = _data(50, 2)
    empty = x_tr.iloc[:0]

    out = lightgbm_regression(
        x_tr,
        y_tr,
        empty,
        np.zeros(0),
        x_te,
        {**PARAMS, "num_boost_round": 30, "early_stopping": None},
    )

    assert out.valid.shape == (0,) and out.query.shape == (50,)
    assert out.info == {"rounds": 30}


def test_binary_predictions_and_fixed_rounds() -> None:
    from kumite.checks import validate_probabilities
    from kumite.gbdt import lightgbm_binary

    x_tr, y = _data(300, 0)
    x_va, v = _data(100, 1)
    y_tr, y_va = (y > np.median(y)).astype(int), (v > np.median(v)).astype(int)
    params = {**PARAMS, "num_threads": 2, "num_boost_round": 50, "early_stopping": 5}
    out = lightgbm_binary(x_tr, y_tr, x_va, y_va, x_va, params, ["g"])
    validate_probabilities(np.column_stack((1 - out.valid, out.valid)), 100, 2)
    assert 1 <= out.info["best_iteration"] <= 50
    assert ((out.valid > 0.5) == y_va).mean() > 0.8
    fixed = lightgbm_binary(
        x_tr, y_tr, x_va.iloc[:0], np.zeros(0), x_va, {**params, "early_stopping": None}
    )
    assert fixed.valid.shape == (0,) and fixed.info == {"rounds": 50}
