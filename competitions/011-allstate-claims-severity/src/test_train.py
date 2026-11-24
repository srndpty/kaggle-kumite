import numpy as np
import pandas as pd
import pytest
from dummy import submission
from train import features, make_splits, restore, target


@pytest.mark.parametrize("variant", ["E00", "E01", "E02", "E03"])
def test_inverse_target(variant):
    values = np.array([0.0, 1.0, 13_981.2])
    np.testing.assert_allclose(restore(target(values, variant), variant, len(values)), values)


def test_unknown_category_uses_missing():
    train, apply = features(
        pd.DataFrame({"cat1": ["NA"]}), pd.DataFrame({"cat1": ["NA", "B"]}), ["cat1"]
    )
    assert train["cat1"].cat.categories.tolist() == ["NA"]
    assert apply["cat1"].isna().tolist() == [False, True]


def test_duplicate_rows_stay_together():
    version, splits = make_splits(pd.DataFrame({"x": [0, 0, 1, 2, 3, 4, 5]}))
    assert version == "duplicate-group-5fold-v1"
    for train, valid in splits:
        assert not ((0 in train and 1 in valid) or (1 in train and 0 in valid))
    np.testing.assert_array_equal(np.sort(np.concatenate([v for _, v in splits])), np.arange(7))


def test_submission_uses_sample_order():
    sample = pd.DataFrame({"id": [5, 3], "loss": [0, 0]})
    result = submission(sample, pd.Series([3, 5]), np.array([22.0, 11.0]))
    assert result.to_dict("list") == {"id": [5, 3], "loss": [11.0, 22.0]}
    with pytest.raises(ValueError):
        submission(sample, pd.Series([3, 5]), np.array([22.0, np.nan]))
