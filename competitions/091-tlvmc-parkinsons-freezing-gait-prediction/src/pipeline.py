"""収録単位の特徴量、採点対象、提出 ID を扱う。"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

AXES = ["AccV", "AccML", "AccAP"]
EVENTS = ["StartHesitation", "Turn", "Walking"]
RATES = {"tdcsfog": 128, "defog": 100}
WINDOWS = {"E00": 0, "E01": 0, "E02": 1, "E03": 1, "E04": 3}


def recordings(root: Path, subset: str) -> list[Path]:
    paths = [path for kind in RATES for path in sorted((root / subset / kind).glob("*.csv"))]
    if not paths or (subset == "train" and {path.parent.name for path in paths} != RATES.keys()):
        raise ValueError(f"Incomplete {subset} recordings")
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError(f"Duplicate {subset} IDs")
    return paths


def read_recording(path: Path, *, labeled: bool) -> pd.DataFrame:
    dtype: dict[str, str] = {"Time": "int64", **dict.fromkeys(AXES, "float32")}
    if labeled:
        dtype.update(dict.fromkeys(EVENTS, "uint8"))
        if path.parent.name == "defog":
            dtype.update({"Valid": "bool", "Task": "bool"})
    frame = pd.read_csv(path, usecols=list(dtype), dtype=dtype)
    if frame.empty or not np.all(np.diff(frame["Time"].to_numpy()) > 0):
        raise ValueError(f"Empty recording or unordered Time in {path.name}")
    if not np.isfinite(frame[AXES].to_numpy()).all():
        raise ValueError(f"Non-finite acceleration in {path.name}")
    if labeled and (frame[EVENTS].to_numpy() > 1).any():
        raise ValueError(f"Non-binary event label in {path.name}")
    return frame


def annotated(frame: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "defog":
        return (frame["Valid"] & frame["Task"]).to_numpy()
    return np.ones(len(frame), dtype=bool)


def features(frame: pd.DataFrame, kind: str, trial: str) -> np.ndarray:
    axes = frame[AXES]
    columns = [axes.to_numpy(), np.full((len(frame), 1), kind == "tdcsfog", dtype=np.float32)]
    seconds = WINDOWS[trial]
    if seconds:
        segments = frame["Time"].diff().ne(1).cumsum()
        window = seconds * RATES[kind]
        rolling = axes.groupby(segments, sort=False).rolling(
            window, min_periods=1, center=trial == "E03"
        )
        columns.extend(
            [
                rolling.mean().to_numpy(dtype=np.float32),
                rolling.std(ddof=0).to_numpy(dtype=np.float32),
            ]
        )
    return np.column_stack(columns)


def class_ap(y: np.ndarray, predictions: np.ndarray) -> list[float | None]:
    return [
        float(average_precision_score(label, prob)) if label.any() else None
        for label, prob in zip(y.T, predictions.T, strict=True)
    ]


def score(y: np.ndarray, predictions: np.ndarray) -> float:
    values = [value for value in class_ap(y, predictions) if value is not None]
    if not values:
        raise ValueError("No positive labels in evaluation set")
    return float(np.mean(values))


def submission_frame(ids: np.ndarray, predictions: np.ndarray) -> pd.DataFrame:
    if predictions.shape != (len(ids), len(EVENTS)):
        raise ValueError("Unexpected prediction shape")
    if not np.isfinite(predictions).all() or (predictions < 0).any() or (predictions > 1).any():
        raise ValueError("Invalid event probabilities")
    index = pd.Index(ids, name="Id")
    if index.hasnans or not index.is_unique or index.empty:
        raise ValueError("Invalid test IDs")
    return pd.DataFrame(predictions, index=index, columns=EVENTS).reset_index()


def make_submission(sample: pd.DataFrame, ids: np.ndarray, predictions: np.ndarray) -> pd.DataFrame:
    if list(sample) != ["Id", *EVENTS]:
        raise ValueError("Unexpected sample submission columns")
    ordered = pd.Index(sample["Id"])
    values = submission_frame(ids, predictions).set_index("Id")
    if not ordered.is_unique or len(ordered) != len(values) or not ordered.isin(values.index).all():
        raise ValueError("Test IDs do not match sample submission")
    return values.loc[ordered].rename_axis("Id").reset_index()


def submission_ids(frame: pd.DataFrame, recording: str) -> np.ndarray:
    return (recording + "_" + frame["Time"].astype(str)).to_numpy()


def load_models(values: list[float | str]) -> list[float | lgb.Booster]:
    return [lgb.Booster(model_str=value) if isinstance(value, str) else value for value in values]


def model_groups(
    models: dict[str, list[float | str]],
) -> dict[str, dict[str, list[float | lgb.Booster]]]:
    groups: dict[str, dict[str, list[float | lgb.Booster]]] = {kind: {} for kind in RATES}
    for key, values in models.items():
        fold, _, kind = key.partition(":")
        predictors = load_models(values)
        for target in (kind,) if kind else RATES:
            groups[target][fold] = predictors
    return groups


def predict(models: list[float | lgb.Booster], matrix: np.ndarray) -> np.ndarray:
    if not len(matrix):
        return np.empty((0, len(models)), dtype=np.float32)
    return np.column_stack(
        [
            np.asarray(model.predict(matrix, num_threads=8))
            if isinstance(model, lgb.Booster)
            else np.full(len(matrix), model)
            for model in models
        ]
    ).astype(np.float32)


def predict_recordings(
    paths: list[Path], trial: str, models: dict[str, list[float | str]]
) -> tuple[np.ndarray, np.ndarray]:
    groups = model_groups(models)
    ids = []
    predictions = []
    for path in paths:
        frame = read_recording(path, labeled=False)
        ids.append(submission_ids(frame, path.stem))
        kind = path.parent.name
        matrix = features(frame, kind, trial)
        predictors = groups[kind].values()
        probabilities = np.zeros((len(matrix), len(EVENTS)), dtype=np.float32)
        for fold in predictors:
            probabilities += predict(fold, matrix)
        predictions.append(probabilities / len(predictors))
    return np.concatenate(ids), np.concatenate(predictions)
