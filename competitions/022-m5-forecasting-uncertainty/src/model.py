"""階層別の残差から M5 の分位点を推定する。"""

import numpy as np

from kumite.m5 import HORIZON, LEVELS, Hierarchy, Sales, seasonal

QUANTILE_NAMES = ("0.005", "0.025", "0.165", "0.250", "0.500", "0.750", "0.835", "0.975", "0.995")
QUANTILES = np.array([float(value) for value in QUANTILE_NAMES])
EXPERIMENTS = {
    "E01": ("系列別の経験分位点を基準にする", "直近12窓の残差を点予測に加える"),
    "E02": ("極端分位の損失が下がる", "売上規模で尺度化した残差を階層内で共有する"),
    "E03": ("後半週の被覆率が改善する", "残差を予測先の週ごとに分ける"),
    "E04": ("需要変化への追従が速まる", "較正を直近6窓に短縮する"),
    "E05": ("少量販売系列の分位点損失が下がる", "最下層の分位点を最近接整数に丸める"),
}


def recent_mean(data: Sales, hierarchy: Hierarchy, origin: int) -> np.ndarray:
    history = data.actual(origin - HORIZON + 1, origin)
    return np.maximum(hierarchy.aggregate(history.mean(axis=1)), 1)


def forecast(data: Sales, hierarchy: Hierarchy, origin: int, experiment: str) -> np.ndarray:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"未知の実験: {experiment}")
    windows = 6 if experiment == "E04" else 12
    residuals = []
    for past in range(origin - windows * HORIZON, origin, HORIZON):
        actual = data.actual(past + 1, past + HORIZON)
        error = hierarchy.aggregate(actual - seasonal(data, past))
        if experiment == "E02":
            error /= recent_mean(data, hierarchy, past)[:, None]
        residuals.append(error.astype(np.float32))
    errors = np.stack(residuals, axis=1)
    center = hierarchy.aggregate(seasonal(data, origin))
    if experiment == "E02":
        offsets = np.stack(
            [
                np.quantile(errors[hierarchy.levels == level], QUANTILES)
                for level in range(len(LEVELS))
            ]
        )
        correction = recent_mean(data, hierarchy, origin)[:, None] * offsets[hierarchy.levels]
        prediction = center[:, :, None] + correction[:, None, :]
    elif experiment == "E03":
        weeks = errors.reshape(len(center), windows, 4, 7)
        offsets = np.quantile(weeks, QUANTILES, axis=(1, 3)).transpose(1, 2, 0)
        prediction = center[:, :, None] + offsets.repeat(7, axis=1)
    else:
        offsets = np.quantile(errors, QUANTILES, axis=(1, 2)).T
        prediction = center[:, :, None] + offsets[:, None, :]
    if experiment == "E05":
        bottom = hierarchy.levels == len(LEVELS) - 1
        prediction[bottom] = np.rint(prediction[bottom])
    return np.maximum(prediction, 0).astype(np.float32)


def validate_prediction(prediction: np.ndarray, rows: int) -> None:
    if prediction.shape != (rows, HORIZON, len(QUANTILES)):
        raise ValueError(f"予測の形状が不正です: {prediction.shape}")
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("予測に非有限値または負値があります")
    if (prediction[:, :, 1:] < prediction[:, :, :-1]).any():
        raise ValueError("分位点が交差しています")
