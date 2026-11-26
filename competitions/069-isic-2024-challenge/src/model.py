"""ISIC のメタデータ特徴と CPU モデル。"""

import lightgbm as lgb
import numpy as np
import pandas as pd

SEED = 42
ROUNDS = 500
NEGATIVE_RATIO = 5
CATEGORICAL = (
    "sex",
    "anatom_site_general",
    "tbp_tile_type",
    "tbp_lv_location",
    "tbp_lv_location_simple",
)
NUMERIC = (
    "age_approx",
    "clin_size_long_diam_mm",
    *tuple(
        f"tbp_lv_{suffix}"
        for suffix in [
            "A",
            "Aext",
            "B",
            "Bext",
            "C",
            "Cext",
            "H",
            "Hext",
            "L",
            "Lext",
            "areaMM2",
            "area_perim_ratio",
            "color_std_mean",
            "deltaA",
            "deltaB",
            "deltaL",
            "deltaLB",
            "deltaLBnorm",
            "eccentricity",
            "minorAxisMM",
            "nevi_confidence",
            "norm_border",
            "norm_color",
            "perimeterMM",
            "radial_color_std_max",
            "stdL",
            "stdLExt",
            "symm_2axis",
            "symm_2axis_angle",
            "x",
            "y",
            "z",
        ]
    ),
)


def features(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    result = frame[list(NUMERIC + CATEGORICAL)].copy()
    for column in NUMERIC:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result[list(NUMERIC)] = result[list(NUMERIC)].replace([np.inf, -np.inf], np.nan)
    for column in CATEGORICAL:
        result[column] = result[column].astype("category")
    if variant == "E01":
        diameter = result["clin_size_long_diam_mm"]
        result["minor_to_long"] = result["tbp_lv_minorAxisMM"] / diameter.where(diameter > 0)
        result["color_distance"] = np.hypot(result["tbp_lv_deltaLB"], result["tbp_lv_deltaL"])
    if variant == "E02":
        if frame["patient_id"].isna().any():
            raise ValueError("患者 ID が欠落しています")
        groups = result.groupby(frame["patient_id"], sort=False)
        result["patient_lesions"] = groups["age_approx"].transform("size")
        for column in ("clin_size_long_diam_mm", "tbp_lv_deltaLB", "tbp_lv_norm_border"):
            result[f"patient_delta_{column}"] = result[column] - groups[column].transform("median")
    return result


def settings(variant: str) -> dict[str, str | int | float | bool]:
    return {
        "objective": "binary",
        "device_type": "cpu",
        "seed": SEED,
        "num_threads": 4,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
        "scale_pos_weight": 5.0 if variant == "E03" else 1.0,
    }


def fit(x: pd.DataFrame, y: np.ndarray, variant: str) -> lgb.Booster:
    if variant == "E04":
        positives = np.flatnonzero(y == 1)
        negatives = np.flatnonzero(y == 0)
        sampled = np.random.default_rng(SEED).choice(
            negatives, size=min(len(negatives), NEGATIVE_RATIO * len(positives)), replace=False
        )
        keep = np.sort(np.concatenate((positives, sampled)))
        x, y = x.iloc[keep], y[keep]
    train = x.copy()
    for column in CATEGORICAL:
        train[column] = train[column].cat.remove_unused_categories()
    return lgb.train(settings(variant), lgb.Dataset(train, label=y), num_boost_round=ROUNDS)


def predict(model: lgb.Booster, x: pd.DataFrame) -> np.ndarray:
    # LightGBM は保存済みの学習カテゴリに照合し、未知値を欠損にする。
    prediction = np.asarray(model.predict(x, num_threads=4), dtype=np.float64)
    if (
        prediction.shape != (len(x),)
        or not (np.isfinite(prediction) & (prediction >= 0) & (prediction <= 1)).all()
    ):
        raise ValueError("病変ごとの予測確率が不正です")
    return prediction
