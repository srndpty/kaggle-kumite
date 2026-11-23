"""多ラベル分類の分割、予測と提出の回帰テスト。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from model import fit_model, log_loss, predict, write_submission
from train import make_folds


def test_group_folds_cover_all_rows() -> None:
    groups = np.repeat(np.arange(10), 3)
    folds = make_folds(groups)
    assert set(folds) == set(range(5))
    assert all(len(set(folds[groups == group])) == 1 for group in np.unique(groups))
    assert np.array_equal(folds, make_folds(groups))


def test_single_class_uses_smoothed_prior() -> None:
    y = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]])
    frame = pd.DataFrame(
        {"cp_type": ["trt_cp"] * 4, "cp_time": [24, 48] * 2, "cp_dose": ["D1"] * 4}
    )
    result = predict(frame, y, frame.iloc[:2], "E01")
    np.testing.assert_allclose(result[:, :2], [[0.1, 0.9], [0.1, 0.9]])


def test_endpoints_follow_official_clipping() -> None:
    assert log_loss(np.array([[0, 1]]), np.array([[0, 1]])) == pytest.approx(1e-15)
    assert log_loss(np.array([[1]]), np.array([[0]])) == pytest.approx(-np.log(1e-15))


def test_submission_follows_sample_order_and_zeros_controls(tmp_path: Path) -> None:
    train = pd.DataFrame(
        {
            "cp_type": ["trt_cp", "trt_cp", "ctl_vehicle"],
            "cp_time": [24, 48, 24],
            "cp_dose": ["D1", "D2", "D1"],
            "g-0": [0, 1, 0],
            "c-0": [1, 0, 1],
        },
        index=pd.Index(["a", "b", "c"], name="sig_id"),
    )
    labels = pd.DataFrame({"z": [1, 1, 0], "a": [0, 1, 0]}, index=train.index)
    y = labels.to_numpy()
    pred = predict(train, y, train, "E03")
    assert (pred[2] == 0).all()
    changed = y.copy()
    changed[-1] = 1
    np.testing.assert_allclose(predict(train, changed, train, "E03"), pred)
    train.to_csv(tmp_path / "test_features.csv")
    labels.iloc[::-1].to_csv(tmp_path / "sample_submission.csv")
    output = tmp_path / "submission.csv"
    state = fit_model(train, y, "E03")
    state["labels"] = np.asarray(labels.columns, dtype=str)
    write_submission(tmp_path, state, output)
    saved = pd.read_csv(output)
    assert saved.columns.tolist() == ["sig_id", "z", "a"]
    assert saved["sig_id"].tolist() == ["c", "b", "a"]
    np.testing.assert_allclose(saved.iloc[:, 1:].to_numpy(), pred[::-1])


def test_selection_rejects_modified_oof_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    import train

    from kumite.experiments import ExperimentRecord
    from kumite.runs import RunStore

    groups = np.arange(10)
    y = np.zeros((10, 1))
    evaluation = train.Evaluation(
        "lish-moa",
        "mean_columnwise_log_loss",
        False,
        train.log_loss,
        "data",
        train.VERSION,
        "split",
        train.SEED,
    )
    monkeypatch.setattr(train, "evaluation", lambda _: evaluation)
    store = RunStore(tmp_path)
    ids = list(train.HYPOTHESES)

    def write(output: Path) -> None:
        np.save(output / "oof.npy", np.full(y.shape, 0.1))
        (output / "inference.py").write_bytes(b"# source")

    for run_id in ids:
        record = ExperimentRecord(
            competition_id=evaluation.competition_id,
            run_id=run_id,
            metric=evaluation.metric,
            greater_is_better=False,
            data_version="test",
            data_hash="data",
            split_version=train.VERSION,
            split_hash="split",
            seed=train.SEED,
            hypothesis="",
            changed_one_thing="",
            model="",
            params={"trial": run_id},
            fold_scores=[],
            oof_score=train.log_loss(y, np.full(y.shape, 0.1)),
            code_version="test",
        )
        with store.reserve(run_id) as reservation:
            store.publish(reservation, record, write)

    assert train.choose_run(store, groups, y, ids)["run_id"] == "E01"
    original = RunStore.record

    def modified(self: RunStore, run_id: str) -> ExperimentRecord:
        record = original(self, run_id)
        return replace(record, oof_score=0) if run_id == "E02" else record

    with monkeypatch.context() as patch:
        patch.setattr(RunStore, "record", modified)
        with pytest.raises(ValueError, match="OOF"):
            train.choose_run(store, groups, y, ids)
    store.artifact_path("E01", "inference.py").write_bytes(b"# replaced")
    with pytest.raises(ValueError, match="ハッシュ"):
        train.choose_run(store, groups, y, ids)


def test_weights_reject_mismatched_contract_and_payload(tmp_path: Path) -> None:
    import json

    from notebook import verified_weights

    from kumite.provenance import file_sha256

    output = tmp_path / "submit.csv"
    output.write_text("csv")
    weights = output.with_suffix(".npz")
    contract = {"run_id": "E02-r3", "data_hash": "official", "source_sha256": "source"}
    manifest = output.with_suffix(".fit.json")

    def write(contract_in_npz: dict[str, str]) -> None:
        np.savez(weights, contract=np.array(json.dumps(contract_in_npz)), weight=np.zeros(1))
        manifest.write_text(
            json.dumps(
                {
                    "contract": contract,
                    "csv_sha256": file_sha256(output),
                    "weights_sha256": file_sha256(weights),
                }
            )
        )

    write({**contract, "data_hash": "other"})
    with pytest.raises(ValueError, match="学習条件"):
        verified_weights(output, contract)
    write(contract)
    verified_weights(output, contract)
    np.savez(weights, contract=np.array(json.dumps(contract)), weight=np.ones(1))
    with pytest.raises(ValueError, match="生成記録"):
        verified_weights(output, contract)


def test_pca_uses_the_same_columns_after_reordering() -> None:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        rng.normal(size=(70, 66)), columns=[*[f"g-{i}" for i in range(64)], "c-0", "c-1"]
    )
    frame = frame.assign(cp_time=24, cp_dose="D1", cp_type="trt_cp")
    y = np.zeros((len(frame), 1))
    shuffled = frame[frame.columns[::-1]]
    state = fit_model(shuffled, y, "E05")
    assert state["columns"][state["gene_mask"][:-1]].tolist() == [f"g-{i}" for i in range(64)]
    np.testing.assert_allclose(
        predict(frame, y, frame, "E05"), predict(shuffled, y, shuffled, "E05")
    )


def test_validation_labels_do_not_change_predictions() -> None:
    from kumite.checks import check_label_independence

    frame = pd.DataFrame(
        {"cp_type": ["trt_cp"] * 4, "cp_time": [24, 48, 24, 48], "cp_dose": ["D1"] * 4}
    )
    labels = np.array([[0], [1], [0], [1]])
    altered = labels.copy()
    altered[2:] = 1 - altered[2:]

    def build(y: np.ndarray) -> list[np.ndarray]:
        return [predict(frame.iloc[:2], y[:2], frame.iloc[2:], "E03")]

    check_label_independence(build, labels, altered)


@pytest.mark.parametrize("same_source", [True, False])
def test_e03_checks_baseline_source_before_using_oof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same_source: bool
) -> None:
    from types import SimpleNamespace

    import train

    from kumite.runs import RunStore

    model = tmp_path / "model.py"
    model.write_bytes(b"# current source")
    monkeypatch.setattr(train, "MODEL", model)
    monkeypatch.setattr(
        train,
        "evaluation",
        lambda *_: SimpleNamespace(
            data_hash="data", split_hash="split", split_version=train.VERSION
        ),
    )
    monkeypatch.setattr(
        RunStore,
        "record",
        lambda *_: SimpleNamespace(
            params={"trial": "E01"},
            data_hash="data",
            split_hash="split",
            split_version=train.VERSION,
        ),
    )
    monkeypatch.setattr(
        train, "verified_source", lambda *_: model.read_bytes() if same_source else b"# old source"
    )

    def load_oof(*_: object) -> np.ndarray:
        raise RuntimeError("OOF を読み込んだ")

    monkeypatch.setattr(RunStore, "load_array", load_oof)
    error = RuntimeError if same_source else ValueError
    message = "OOF を読み込んだ" if same_source else "E01 を再実行"
    with pytest.raises(error, match=message):
        train.cv(
            RunStore(tmp_path),
            pd.DataFrame(),
            pd.DataFrame(np.zeros((10, 1))),
            np.arange(10),
            "E03",
            "E03-review",
            False,
            "E01-review",
        )
