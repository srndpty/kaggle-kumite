import numpy as np
import pandas as pd
import pytest
from train import mae, training_target
from zillow import DATE, ID, TARGET, join_properties, time_splits

from kumite.crossfit import split_roles


def test_time_splits_respect_cutoffs_and_validate_once() -> None:
    dates = pd.Series(pd.date_range("2016-12-01", "2017-09-01", freq="MS"))
    splits = time_splits(dates)
    roles = split_roles(len(dates), splits)
    np.testing.assert_array_equal((roles == 2).sum(axis=0), [0, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    for train, valid in splits:
        assert dates.iloc[train].max() < dates.iloc[valid].min()


def test_property_join_preserves_transactions_and_checks_year_and_key() -> None:
    dates = pd.to_datetime(["2016-01-01", "2016-06-01", "2016-02-01"])
    rows = pd.DataFrame({ID: [10, 10, 20], DATE: dates})
    properties = pd.DataFrame({"yearbuilt": [2000]}, index=[10])
    joined = join_properties(rows, properties, 2016)
    np.testing.assert_array_equal(joined[ID], rows[ID])
    np.testing.assert_allclose(joined["yearbuilt"], [2000, 2000, np.nan])
    with pytest.raises(ValueError, match="取引年"):
        join_properties(rows, properties, 2017)
    with pytest.raises(pd.errors.MergeError):
        join_properties(rows, pd.concat([properties, properties]), 2016)


def test_clipping_changes_only_training_targets() -> None:
    rows = pd.DataFrame({TARGET: [-10.0, 0.0, 10.0, 100.0]})
    train, valid = rows.iloc[:3], rows.iloc[3:]
    np.testing.assert_allclose(training_target(train, "E03"), [-9.8, 0.0, 9.8])
    np.testing.assert_array_equal(rows[TARGET], [-10.0, 0.0, 10.0, 100.0])
    assert mae(valid[TARGET].to_numpy(), np.zeros(1)) == 100.0
