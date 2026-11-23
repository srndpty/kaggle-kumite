"""M5 の公式入力を商品×店舗と日番号でそろえる。"""

from pathlib import Path

import numpy as np
import pandas as pd

from kumite.checks import validate_regression
from kumite.m5 import (
    COMMON_FILES as COMMON_FILES,
)
from kumite.m5 import (
    EVALUATION_FILE as EVALUATION_FILE,
)
from kumite.m5 import (
    HORIZON as HORIZON,
)
from kumite.m5 import (
    ORIGINS as ORIGINS,
)
from kumite.m5 import (
    PRIVATE_ORIGIN as PRIVATE_ORIGIN,
)
from kumite.m5 import (
    PUBLIC_ORIGIN as PUBLIC_ORIGIN,
)
from kumite.m5 import (
    VALIDATION_FILE as VALIDATION_FILE,
)
from kumite.m5 import (
    Sales as Sales,
)
from kumite.m5 import (
    load_sales as load_m5_sales,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_sales(*, evaluation: bool = False, data_dir: Path = DATA) -> Sales:
    return load_m5_sales(data_dir, evaluation=evaluation)


def validate_prediction(prediction: np.ndarray, rows: int) -> None:
    if prediction.shape != (rows, HORIZON):
        raise ValueError(f"予測の形状が不正です: {prediction.shape}")
    validate_regression(prediction.ravel(), rows * HORIZON, lower=0)


def submission(
    public: Sales,
    public_pred: np.ndarray,
    private: Sales,
    private_pred: np.ndarray,
    data_dir: Path = DATA,
) -> pd.DataFrame:
    columns = [f"F{day}" for day in range(1, HORIZON + 1)]
    parts = []
    for data, prediction, suffix in (
        (public, public_pred, "validation"),
        (private, private_pred, "evaluation"),
    ):
        validate_prediction(prediction, len(data.series))
        ids = data.series["item_id"] + "_" + data.series["store_id"] + "_" + suffix
        parts.append(pd.DataFrame(prediction, index=pd.Index(ids, name="id"), columns=columns))
    predictions = pd.concat(parts, verify_integrity=True)
    sample = pd.read_csv(data_dir / "sample_submission.csv", usecols=["id"])
    if (
        len(sample) != 2 * len(public.series)
        or set(sample["id"]) != set(predictions.index)
        or sample["id"].duplicated().any()
    ):
        raise ValueError("サンプルと提出 ID が一致しません")
    return predictions.loc[sample["id"]].reset_index()
