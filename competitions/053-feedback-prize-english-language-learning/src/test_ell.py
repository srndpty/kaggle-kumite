import numpy as np
from model import duplicate_key, fit_model
from train import mcrmse


def test_mcrmse_averages_column_roots() -> None:
    actual = np.zeros((2, 6))
    prediction = np.tile(np.arange(6), (2, 1))
    assert mcrmse(actual, prediction) == 2.5


def test_duplicate_key_normalizes_width_case_and_whitespace() -> None:
    assert duplicate_key(" Ａbc\tDEF\n") == duplicate_key("abc def")


def test_fold_model_only_uses_training_labels() -> None:
    texts = ["fox and cat", "fox and dog", "dog and cat", "unseen word"]
    labels = np.tile(np.arange(4).reshape(-1, 1), (1, 6)).astype(float) + 1
    before = fit_model(texts[:3], labels[:3], "base")(texts[3:])
    labels[3] = 5
    after = fit_model(texts[:3], labels[:3], "base")(texts[3:])
    np.testing.assert_array_equal(before, after)
