"""公式配布語彙による補助教師ラベルと、提出12クラスへの確率集約。"""

import numpy as np
import pandas as pd
from dataset import CLASSES

# 公式配布train/README.mdの30語。検証ラベルやtestから辞書を作らない。
UNKNOWN_WORDS = (
    "bed",
    "bird",
    "cat",
    "dog",
    "eight",
    "five",
    "four",
    "happy",
    "house",
    "marvin",
    "nine",
    "one",
    "seven",
    "sheila",
    "six",
    "three",
    "tree",
    "two",
    "wow",
    "zero",
)
FINE_CLASSES = (*CLASSES[:11], *UNKNOWN_WORDS)


def fine_labels(frame: pd.DataFrame) -> np.ndarray:
    labels = frame.word.map({word: index for index, word in enumerate(FINE_CLASSES)})
    if labels.isna().any():
        raise ValueError("公式語彙にない学習ラベルがあります")
    result = labels.to_numpy(dtype=np.int64)
    if not np.array_equal(np.minimum(result, 11), frame.label.to_numpy()):
        raise ValueError("元の12クラスとの対応が不正です")
    return result


def collapse_probabilities(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.ndim != 2 or probabilities.shape[1] not in (12, 31):
        raise ValueError("12または31クラスの確率が必要です")
    if probabilities.shape[1] == 12:
        return probabilities
    # float32のunknown合算が1+epsになり得る。高精度で合算して全体を再正規化する。
    probabilities = probabilities.astype(np.float64)
    result = np.column_stack((probabilities[:, :11], probabilities[:, 11:].sum(axis=1)))
    return result / result.sum(axis=1, keepdims=True)
