"""時系列分割・特徴・提出時に守る条件。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dataset import (
    DAY,
    ID,
    TARGET,
    TIME,
    history_counts,
    join_identity,
    make_splits,
    relative_amount,
)
from select_run import selection
from train import run, submission

from kumite.provenance import file_sha256
from kumite.runs import RunStore


def test_identity_join_preserves_transaction_order() -> None:
    trades = pd.DataFrame({ID: [3, 1, 2]})
    identity = pd.DataFrame({ID: [2, 3], "id-12": ["x", "y"]})
    joined = join_identity(trades, identity)
    assert joined[ID].tolist() == [3, 1, 2]
    assert joined["has_identity"].tolist() == [True, False, True]
    assert joined.loc[0, "id_12"] == "y"
    with pytest.raises(ValueError, match="重複"):
        join_identity(trades, pd.concat([identity, identity]))


def test_splits_exclude_gap_and_choose_shorter_windows() -> None:
    train = pd.DataFrame({TIME: np.repeat(np.arange(60) * DAY, 2), TARGET: [0, 1] * 60})
    test = pd.DataFrame({TIME: [65 * DAY]})
    splits, version = make_splits(train, test)
    assert version == "time-v1-14d-gap5d"
    assert len(splits) == 3
    assert all(
        train[TIME].iloc[tr].max() + 5 * DAY < train[TIME].iloc[va].min() for tr, va in splits
    )
    assert all(len(va) == 28 for _, va in splits)


def test_history_uses_strict_past_and_includes_window_left_boundary() -> None:
    past = pd.DataFrame(
        {
            TIME: [0, DAY, DAY, 2 * DAY, 3 * DAY],
            "card1": [1.0, 1.0, 1.0, 1.0, 1.0],
            "card2": [2.0, 2.0, 3.0, 2.0, 2.0],
        }
    )
    query = pd.DataFrame(
        {
            TIME: [2 * DAY, DAY, 2 * DAY],
            "card1": [1.0, 1.0, np.nan],
            "card2": [2.0, 2.0, 2.0],
        },
        index=[8, 6, 7],
    )
    result = history_counts(past, query)
    assert result.to_numpy()[0].tolist() == [1, 2]
    assert result.to_numpy()[1].tolist() == [1, 1]
    assert np.isnan(result.to_numpy()[2]).all()
    expanded = pd.concat([query, query.iloc[:1].set_axis([9])])
    pd.testing.assert_frame_equal(result, history_counts(past, expanded).loc[query.index])


def test_submission_matches_sample_order_and_rejects_nonfinite() -> None:
    sample = pd.DataFrame({ID: [2, 1], TARGET: [0.0, 0.0]})
    ids = pd.Series([1, 2])
    result = submission(sample, ids, np.array([0.2, 0.9]))
    assert result[TARGET].tolist() == [0.9, 0.2]
    with pytest.raises(ValueError, match="確率"):
        submission(sample, ids, np.array([0.2, np.inf]))


def test_run_and_refit_with_synthetic_data(tmp_path: Path) -> None:
    days = np.repeat(np.arange(100), 2)
    labels = np.tile([0, 1], 100)
    train = pd.DataFrame(
        {
            ID: np.arange(len(days)),
            TIME: days * DAY + labels,
            TARGET: labels,
            "card1": 1,
            "card2": 2,
            "amount": labels,
        }
    )
    query = (
        train.iloc[:2]
        .drop(columns=TARGET)
        .assign(
            TransactionID=[300, 301],
            TransactionDT=130 * DAY,
        )
    )
    data = tmp_path / "data"
    data.mkdir()
    train.to_csv(data / "train_transaction.csv", index=False)
    query.to_csv(data / "test_transaction.csv", index=False)
    train[[ID]].iloc[:1].to_csv(data / "train_identity.csv", index=False)
    query[[ID]].to_csv(data / "test_identity.csv", index=False)
    query[[ID]].assign(isFraud=0).iloc[::-1].to_csv(data / "sample_submission.csv", index=False)
    run(tmp_path, "E01", "base", 42)
    store = RunStore(tmp_path)
    selected = selection(store, ["E01"])
    assert selected["selected"] == "E01"
    run(tmp_path, "final-v1", "base", 42, "E01")
    saved = pd.read_csv(tmp_path / "outputs" / "runs" / "final-v1" / "submission.csv")
    assert saved[ID].tolist() == [301, 300]
    assert saved[TARGET].between(0, 1).all()

    log = store.log_path.read_text(encoding="utf-8")
    lines = log.splitlines()
    record = json.loads(lines[0])
    for changes, error in (
        ({"fold_scores": [0.0, 0.0, 0.0]}, "保存済み AUC"),
        ({"oof_score": 0.0}, "保存済み AUC"),
        ({"data_hash": "foreign"}, "データハッシュ"),
        ({"params": {**record["params"], "seed": 43}}, "比較可能"),
    ):
        lines[0] = json.dumps({**record, **changes})
        store.log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=error):
            selection(store, ["E01"])

    store.log_path.write_text(log, encoding="utf-8")
    oof = store.load_array("E01", "oof.npy")
    roles = store.load_array("E01", "split_roles.npy")
    first, second = np.flatnonzero(roles[0] == 2)[:2]
    assert oof[first] != oof[second]
    oof[[first, second]] = oof[[second, first]]
    artifact = store.runs_dir / "E01" / "oof.npy"
    np.save(artifact, oof)
    record["artifacts"]["oof.npy"] = file_sha256(artifact)
    lines[0] = json.dumps(record)
    store.log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="保存済み AUC"):
        selection(store, ["E01"])


def test_relative_amount_uses_only_amounts_before_query_time() -> None:
    past = pd.DataFrame(
        {
            TIME: [0, DAY, DAY, 2 * DAY],
            "card1": [1.0] * 4,
            "card2": [2.0] * 4,
            "TransactionAmt": [10.0, 20.0, 30.0, 1000.0],
        }
    )
    query = past.iloc[[2, 3]].assign(TransactionAmt=[30.0, 60.0])
    result = relative_amount(past, query)
    assert result.tolist() == [3.0, 3.0]
