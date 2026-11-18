"""公式CSVを明示的に読み、公開testラベルを隔離する。"""

from pathlib import Path

import numpy as np
import pandas as pd

from kumite.checks import validate_multilabel_probabilities

COMPETITION_DIR = Path(__file__).resolve().parents[1]
DATA_FILES = tuple(
    COMPETITION_DIR / "data" / n for n in ("train.csv", "test.csv", "sample_submission.csv")
)
LABELS = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")


def load_data():
    train, test, sample = (pd.read_csv(f, keep_default_na=False) for f in DATA_FILES)
    if set(train.columns) != {"id", "comment_text", *LABELS} or set(test.columns) != {
        "id",
        "comment_text",
    }:
        raise ValueError("入力列が不正です")
    for frame in (train, test, sample):
        if not frame.id.is_unique or (frame.id == "").any():
            raise ValueError("IDが重複または空です")
    if list(sample.columns) != ["id", *LABELS] or set(sample.id) != set(test.id):
        raise ValueError("提出形式またはIDが不一致です")
    if set(train.id) & set(test.id) or not np.isin(train[list(LABELS)], [0, 1]).all():
        raise ValueError("ID重複またはラベルが不正です")
    return train, test.set_index("id").loc[sample.id].reset_index(), sample


def submission(sample, ids, pred):
    if not np.array_equal(sample.id.to_numpy(), ids):
        raise ValueError("提出IDの順序が不一致です")
    validate_multilabel_probabilities(pred, len(sample), len(LABELS))
    result = sample.copy()
    result[list(LABELS)] = pred
    return result
