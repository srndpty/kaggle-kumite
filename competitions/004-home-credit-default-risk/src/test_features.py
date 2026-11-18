"""顧客履歴集約・時点制約・結合の不変条件。"""

import numpy as np
import pandas as pd
import pytest
from cv import make_folds
from dataset import KEY, submission
from features import (
    BUREAU_NUMERIC,
    INSTALLMENT_COLUMNS,
    PREVIOUS_NUMERIC,
    application_features,
    bureau_features,
    encode_categories,
    installment_features,
    join_history,
    previous_features,
)

from kumite.checks import check_label_independence, shift_labels


def payments() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [1, 10, 1, 1, -20, -21, 100, 40],
            [1, 10, 1, 1, -20, -18, 100, 50],
            [1, 10, 1, 2, -10, -10, 100, 100],
            [1, 10, 1, 3, -1, 1, 100, 999],  # 将来支払は除く
            [2, 20, 1, 1, 2, -1, 100, 100],  # 今回は期日到来分だけ集計
            [3, 30, 1, 1, 0, 0, 100, 100],
        ],
        columns=INSTALLMENT_COLUMNS,
    )


def test_split_payments_and_time_boundary() -> None:
    result = installment_features(payments()).set_index(KEY)
    assert result.loc[1, "installments_count"] == 2
    assert result.loc[1, "installments_delay_max"] == 2
    assert result.loc[1, "installments_shortfall_max"] == 10
    assert result.loc[1, "installments_late_rate"] == 0.5
    assert result.loc[1, "installments_underpaid_rate"] == 0.5
    assert result.loc[3, "installments_count"] == 1
    assert 2 not in result.index


@pytest.mark.parametrize("column,value", [("AMT_INSTALMENT", 200), ("DAYS_INSTALMENT", -19)])
def test_inconsistent_installment_rejected(column: str, value: int) -> None:
    frame = payments()
    frame.loc[1, column] = value
    with pytest.raises(ValueError, match="一致しません"):
        installment_features(frame)


def test_installment_key_and_customer_checks() -> None:
    frame = payments()
    frame.loc[1, KEY] = 99
    with pytest.raises(ValueError, match="複数顧客"):
        installment_features(frame)
    frame.loc[0, "NUM_INSTALMENT_NUMBER"] = np.nan
    with pytest.raises(ValueError, match="欠損"):
        installment_features(frame)


def test_missing_payment_is_not_zero() -> None:
    frame = payments().iloc[:2].copy()
    frame["AMT_PAYMENT"] = np.nan
    result = installment_features(frame)
    assert result["installments_shortfall_mean"].isna().iloc[0]
    assert result["installments_underpaid_rate"].isna().iloc[0]


def test_bureau_update_cutoff_and_duplicate_key() -> None:
    frame = pd.DataFrame(
        {
            KEY: [1, 1, 2, 3],
            "SK_ID_BUREAU": [1, 2, 3, 4],
            "DAYS_CREDIT": [-10, -20, 1, 0],
            "DAYS_CREDIT_UPDATE": [-1, 1, 0, 0],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active", "Sold"],
            **{c: [10.0, 100.0, 1000.0, np.nan] for c in BUREAU_NUMERIC},
        }
    )
    result = bureau_features(frame).set_index(KEY)
    assert result.loc[1, "bureau_count"] == 1
    assert result.loc[1, "bureau_AMT_CREDIT_SUM_sum"] == 10
    assert result.loc[1, "bureau_status_Active_count"] == 1
    assert result["bureau_AMT_CREDIT_SUM_sum"].isna().loc[3]
    assert 2 not in result.index
    with pytest.raises(ValueError, match="一意"):
        bureau_features(pd.concat([frame, frame.iloc[:1]]))


def test_previous_cutoff() -> None:
    frame = pd.DataFrame(
        {
            KEY: [1, 1, 2],
            "SK_ID_PREV": [1, 2, 3],
            "DAYS_DECISION": [-1, 1, 0],
            "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Canceled"],
            **{c: [10, 100, 20] for c in PREVIOUS_NUMERIC},
        }
    )
    result = previous_features(frame).set_index(KEY)
    assert result.loc[1, "previous_count"] == 1
    assert result.loc[1, "previous_status_Refused_count"] == 0
    assert result.loc[2, "previous_AMT_CREDIT_mean"] == 20


def test_recent_bureau_boundary_and_customer_isolation() -> None:
    frame = pd.DataFrame(
        {
            KEY: [1, 1, 1, 2],
            "SK_ID_BUREAU": [1, 2, 3, 4],
            "DAYS_CREDIT": [-366, -365, 0, -1],
            "DAYS_CREDIT_UPDATE": [-1, -1, 0, -1],
            "CREDIT_ACTIVE": ["Active"] * 4,
            **{c: [100.0, 10.0, 20.0, 1000.0] for c in BUREAU_NUMERIC},
        }
    )
    all_time = bureau_features(frame)
    recent = bureau_features(frame, since=-365)
    joined = join_history(all_time, recent).set_index(KEY)
    assert joined.loc[1, "bureau_count"] == 3
    assert joined.loc[1, "bureau_recent_count"] == 2
    assert joined.loc[1, "bureau_recent_AMT_CREDIT_SUM_sum"] == 30
    frame.loc[frame[KEY] == 2, BUREAU_NUMERIC] = 99999.0
    changed = bureau_features(frame, since=-365)
    pd.testing.assert_frame_equal(
        recent.loc[recent[KEY] == 1],
        changed.loc[changed[KEY] == 1],
    )


def test_join_preserves_ids_and_missing_history() -> None:
    main = pd.DataFrame({KEY: [3, 9, 1], "TARGET": [1, 0, 0]})
    result = join_history(main, installment_features(payments()))
    assert result[KEY].tolist() == [3, 9, 1]
    assert result.loc[1, "installments_count"] == 0
    assert result["installments_delay_mean"].isna().iloc[1]
    for left, right in (
        (pd.concat([main, main.iloc[:1]]), installment_features(payments())),
        (main, pd.DataFrame({KEY: [1, 1], "x_count": [1, 2]})),
    ):
        with pytest.raises(ValueError, match="一意"):
            join_history(left, right)


def test_categories_only_from_training() -> None:
    train = pd.DataFrame({"category": ["a", "b"], "x": [1, 2]})
    valid = pd.DataFrame({"category": ["v"], "x": [3]})
    test = pd.DataFrame({"category": ["t"], "x": [4]})
    frames, cats = encode_categories(train, valid, test)
    assert cats == ["category"]
    assert frames[0]["category"].cat.categories.tolist() == ["a", "b"]
    assert frames[1]["category"].isna().iloc[0]
    assert frames[2]["category"].isna().iloc[0]


def test_fold_group_independence_and_repeatability() -> None:
    frame = pd.DataFrame(
        {
            KEY: np.arange(60),
            "x": np.repeat(np.arange(30), 2),
            "TARGET": np.repeat(np.arange(30) % 2, 2),
        }
    )
    y = frame["TARGET"].to_numpy()
    folds, version = make_folds(frame, y, 42)
    assert version == "stratified_group3-seed42-v1"
    np.testing.assert_array_equal(folds, make_folds(frame, y, 42)[0])
    assert all(folds[k] == folds[k + 1] for k in range(0, 60, 2))
    for k in range(3):
        assert len(np.unique(y[folds == k])) == 2
    with pytest.raises(ValueError, match="一意"):
        make_folds(pd.concat([frame, frame.iloc[:1]]), np.append(y, 0), 42)


def test_features_ignore_validation_targets() -> None:
    frame = pd.DataFrame({KEY: [1, 2, 3, 4], "TARGET": [0, 1, 0, 1], "x": [3.0, 4.0, 5.0, 6.0]})
    y = frame["TARGET"].to_numpy()

    def build(labels: np.ndarray) -> list[np.ndarray]:
        joined = join_history(frame.assign(TARGET=labels), installment_features(payments()))
        return [application_features(joined).to_numpy()]

    # このfixtureは履歴なしの欠損を含むため比較対象を欠損符号へ写す。
    check_label_independence(
        lambda labels: [np.nan_to_num(build(labels)[0], nan=-999)],
        y,
        shift_labels(y, np.array([False, False, True, True]), 2),
    )


def test_submission_identity_and_values() -> None:
    sample = pd.DataFrame({KEY: [3, 1], "TARGET": [0.0, 0.0]})
    assert submission(sample, np.array([3, 1]), np.array([0.2, 0.8]))["TARGET"].tolist() == [
        0.2,
        0.8,
    ]
    for ids, pred in (
        (np.array([1, 3]), np.array([0.2, 0.8])),
        (np.array([3, 1]), np.array([np.nan, 0.8])),
        (np.array([3, 1]), np.array([1.1, 0.8])),
    ):
        with pytest.raises(ValueError):
            submission(sample, ids, pred)
