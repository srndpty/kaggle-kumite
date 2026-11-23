"""公式サンプルの ID 順序で M5 の分位点予測を書き出す。"""

from pathlib import Path

import numpy as np
import pandas as pd
from model import QUANTILE_NAMES, validate_prediction

from kumite.m5 import HORIZON, LEVELS, Hierarchy

COLUMNS = [f"F{day}" for day in range(1, HORIZON + 1)]


def series_ids(hierarchy: Hierarchy) -> list[str]:
    ids = []

    for name, level in zip(hierarchy.labels["series"], hierarchy.levels, strict=True):
        if LEVELS[level] == ("item_id", "state_id"):
            item, state = name.rsplit("_", 1)
            name = f"{state}_{item}"
        elif len(LEVELS[level]) == 1:
            name = f"{name}_X"
        ids.append(name)
    return ids


def write_submission(
    path: Path,
    sample: Path,
    public: tuple[Hierarchy, np.ndarray],
    private: tuple[Hierarchy, np.ndarray],
) -> None:
    frames = []
    for suffix, (hierarchy, prediction) in zip(
        ("validation", "evaluation"), (public, private), strict=True
    ):
        ids = series_ids(hierarchy)
        validate_prediction(prediction, len(ids))
        for index, quantile in enumerate(QUANTILE_NAMES):
            keys = pd.Index([f"{key}_{quantile}_{suffix}" for key in ids], name="id")
            frames.append(pd.DataFrame(prediction[:, :, index], index=keys, columns=COLUMNS))
    forecasts = pd.concat(frames, verify_integrity=True)
    sample_ids = pd.Index(pd.read_csv(sample, usecols=["id"])["id"])
    order = forecasts.index.get_indexer(sample_ids)
    if len(sample_ids) != len(forecasts) or not sample_ids.is_unique or (order < 0).any():
        raise ValueError("サンプルと予測の ID が一致しません")
    forecasts.iloc[order].to_csv(path)
