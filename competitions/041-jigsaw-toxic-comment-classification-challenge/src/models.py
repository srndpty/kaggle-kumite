"""ラベル別の疎な線形分類器。"""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


def fit_linear(frames, y, weighted=False):
    valid, test = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for i in range(y.shape[1]):
            model = LogisticRegression(
                C=1,
                solver="liblinear",
                max_iter=1000,
                class_weight="balanced" if weighted else None,
                random_state=42,
            )
            model.fit(frames[0], y[:, i])
            valid.append(model.predict_proba(frames[1])[:, 1])
            test.append(model.predict_proba(frames[2])[:, 1])
    return np.column_stack(valid), np.column_stack(test)
