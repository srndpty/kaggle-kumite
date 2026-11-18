"""多ラベル確率検査の契約。"""

import numpy as np
import pytest

from kumite.checks import validate_multilabel_probabilities


def test_independent_probabilities() -> None:
    validate_multilabel_probabilities(np.full((2, 6), 0.9), 2, 6)


@pytest.mark.parametrize(
    "pred", [np.zeros((2, 5)), np.array([[np.nan] * 6] * 2), np.full((2, 6), -1.0)]
)
def test_invalid(pred: np.ndarray) -> None:
    with pytest.raises(ValueError):
        validate_multilabel_probabilities(pred, 2, 6)
