"""月境界、順位、提出形式とラベル独立性を検証する。"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import train
from model import ID, average_precision, rank, transitions, validate_ranking
from train import EVALUATION_FILE, predict, submission_sample, verify_selection, write_submission

from kumite.experiments import ExperimentRecord, append_record
from kumite.provenance import file_sha256
from kumite.runs import RunStore


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


def test_final_rejects_changed_selection_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore(tmp_path)
    path = store.runs_dir / "E04" / EVALUATION_FILE
    path.parent.mkdir(parents=True)
    np.savez(path, ranked=np.array([[0]]))
    digest = file_sha256(path)
    record = ExperimentRecord(
        competition_id="034",
        run_id="E04",
        metric="map_at_7",
        greater_is_better=True,
        data_version="test",
        data_hash="data",
        split_version="monthly-v2",
        split_hash="split",
        seed=42,
        hypothesis="",
        changed_one_thing="",
        model="logistic",
        params={"variant": "logistic"},
        fold_scores=[1.0],
        oof_score=1.0,
        code_version="test",
        artifacts={EVALUATION_FILE: digest},
    )
    append_record(store.log_path, record)
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "selected_run_id": "E04",
                "variant": "logistic",
                "data_hash": record.data_hash,
                "split_hash": record.split_hash,
                "evaluation_sha256": digest,
                "cv_map_at_7": {"E04": record.oof_score},
            }
        ),
        encoding="utf-8",
    )
    assert verify_selection(store, selection, "logistic") == record
    np.savez(path, ranked=np.array([[1]]))
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--run-id",
            "E06",
            "--variant",
            "logistic",
            "--final",
            "--selection",
            str(selection),
        ],
    )
    with pytest.raises(ValueError, match="評価成果物"):
        train.main()
    assert not (store.runs_dir / "E06").exists()
    monkeypatch.setattr(
        RunStore, "record", lambda _store, _run: replace(record, split_hash="changed")
    )
    with pytest.raises(ValueError, match="採用 run"):
        verify_selection(store, selection, "logistic")


def test_cv_only_excludes_may_and_skips_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    products = [f"ind_{i:02}_ult1" for i in range(24)]
    header = ",".join(["fecha_dato", "ncodpers", *products])
    rows = [
        ",".join([f"2016-{month:02}-28", "1", *([value] * 24)])
        for month, value in ((1, "0"), (2, "0"), (3, "1"), (4, "1"), (5, "MAY"))
    ]
    (data / "train_ver2.csv").write_text("\n".join([header, *rows]), encoding="utf-8")
    for filename in ("test_ver2.csv", "sample_submission.csv"):
        (data / filename).write_bytes(b"invalid for inference")
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "DATA", data)
    monkeypatch.setattr(
        "sys.argv", ["train.py", "--run-id", "E01", "--variant", "baseline", "--cv-only"]
    )
    train.main()
    record = RunStore(tmp_path).record("E01")
    assert set(record.artifacts) == {"oof.npy", "fold_info.json", EVALUATION_FILE}
    assert record.split_version == "monthly-v2"
