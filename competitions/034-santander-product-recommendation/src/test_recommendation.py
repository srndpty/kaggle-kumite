"""月境界、順位、提出形式とラベル独立性を検証する。"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from model import ID, average_precision, rank, transitions, validate_ranking
from train import predict, submission_sample, write_submission


def test_missing_previous_month_and_order() -> None:
    panel = pl.DataFrame({ID: [1, 2, 2], "month": [201512, 201601, 201602], "p": [1, 0, 1]})
    current = panel.filter(pl.col("month") == 201602).select(ID, "p")
    current = pl.concat([current, pl.DataFrame({ID: [1], "p": [1]})])
    ids, owned, labels = transitions(panel, current, 201602, ["p"])
    assert ids.tolist() == [2, 1]
    assert owned.tolist() == [[0], [-1]]
    assert labels.tolist() == [[1], [-1]]
    _, held, _ = transitions(panel, current.select(ID), 201602, ["p"])
    np.testing.assert_array_equal(held, owned)


def test_ranking_and_map() -> None:
    train = np.zeros((2, 4), dtype=np.int8)
    owned = np.array([[1, 0, 0, 0], [1, 1, 1, 1]])
    ranked = rank(train, train, owned, "baseline")
    assert ranked.tolist() == [[1, 2, 3, -1], [-1, -1, -1, -1]]
    validate_ranking(ranked, owned)
    np.testing.assert_allclose(
        average_precision(np.array([[0, 1, 0, 1], [0, 0, 0, 0]]), ranked), [(1 + 2 / 3) / 2, 0]
    )
    with pytest.raises(ValueError, match="保有済み"):
        validate_ranking(np.array([[0, 2, 3, -1], [-1, -1, -1, -1]]), owned)


@pytest.mark.parametrize("variant", ["baseline", "recent", "portfolio", "logistic", "seasonal"])
def test_target_month_labels_do_not_change_predictions(variant: str) -> None:
    months = np.array([201512, 201601, 201602, 201603, 201603])
    owned = np.array([[0, 1], [0, 0], [0, 0], [0, 0], [0, 1]])
    labels = np.array([[0, 0], [1, 0], [0, 1], [0, 1], [0, 0]])
    first = predict(owned, labels, months, owned[-2:], 201603, variant)
    labels[-2:] = 1 - labels[-2:]
    np.testing.assert_array_equal(
        first, predict(owned, labels, months, owned[-2:], 201603, variant)
    )


def test_seasonal_gives_previous_year_same_month_double_weight() -> None:
    months = np.array([201503] * 3 + [201504] * 4)
    owned = np.zeros((7, 2), dtype=np.int8)
    labels = np.array([[1, 0]] * 3 + [[0, 1]] * 4, dtype=np.int8)
    assert predict(owned, labels, months, owned[:1], 201603, "baseline").tolist() == [[1, 0]]
    assert predict(owned, labels, months, owned[:1], 201603, "seasonal").tolist() == [[0, 1]]


def test_submission_order_and_duplicate_ids(tmp_path: Path) -> None:
    test, sample, output = (tmp_path / p for p in ("test.csv", "sample.csv", "output.csv"))
    test.write_text("ncodpers\n1\n2\n", encoding="utf-8")
    sample.write_text("ncodpers,added_products\n2,\n1,\n", encoding="utf-8")
    ids = submission_sample(test, sample)[ID].to_numpy()
    write_submission(output, ids, ["a", "b"], np.array([[1, 0], [-1, -1]]))
    assert output.read_text(encoding="utf-8") == "ncodpers,added_products\n2,b a\n1,\n"
    test.write_text("ncodpers\n1\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="不一致"):
        submission_sample(test, sample)
