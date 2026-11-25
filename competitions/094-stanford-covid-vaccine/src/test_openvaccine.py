import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from analyze import require_selection
from data import (
    TARGETS,
    Labels,
    Molecules,
    cluster_splits,
    mcrmse,
    partners,
    sample_rows,
    tokens,
    validate_submission,
    write_submission,
)
from external import private_breakdown
from models import GRUConfig, TrainGroup, batch_loss, fit_predict_gru, window_frame
from run import SOURCE_FILES, Method, load_method, source_spec, to_five, training_weights

from kumite.crossfit import CrossFitResult, split_roles
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.paths import REPO_ROOT
from kumite.provenance import array_sha256
from kumite.quality_gates import SelectionPlan, SourceSpec, dense_oof, reserve_cv, select_runs
from kumite.runs import RunStore


def _molecules(*sequences: str) -> Molecules:
    n = len(sequences)
    return Molecules(
        tuple(f"id_{i}" for i in range(n)),
        sequences,
        tuple("." * len(s) for s in sequences),
        tuple("E" * len(s) for s in sequences),
    )


def test_mcrmse_matches_hand_calculation_and_skips_unscored_molecules() -> None:
    y = np.array([[[1.0, 0.0, 2.0]], [[np.nan, np.nan, np.nan]]])
    predicted = np.array([[[0.0, 0.0, 0.0]], [[9.0, 9.0, 9.0]]])
    assert mcrmse(y, predicted) == pytest.approx((1.0 + 0.0 + 2.0) / 3)
    with pytest.raises(ValueError, match="形"):
        mcrmse(y, predicted[:1])


def test_scored_labels_mask_low_signal_molecules() -> None:
    y = np.arange(2 * 3 * 5, dtype=np.float64).reshape(2, 3, 5)
    labels = Labels(y, np.ones_like(y), np.array([True, False]), np.array([2.0, 0.5]))
    scored = labels.scored()
    assert scored.shape == (2, 3, 3)
    np.testing.assert_array_equal(scored[0], y[0][:, [0, 1, 3]])
    assert np.isnan(scored[1]).all()


def test_partners_and_tokens() -> None:
    np.testing.assert_array_equal(partners("((..))."), [5, 4, -1, -1, 1, 0, -1])
    with pytest.raises(ValueError, match="括弧"):
        partners("(()")
    encoded = tokens(Molecules(("a",), ("AU",), ("()",), ("SS",)), 0)
    assert encoded.shape == (2, 14)
    np.testing.assert_array_equal(encoded.sum(axis=1), [3, 3])


def test_cluster_splits_keep_similar_sequences_together() -> None:
    base = "GGAAACCCUUUAGAGCAUCGAUCGAUGGCAUCGA"
    rng = np.random.default_rng(0)
    others = ["".join(rng.choice(list("ACGU"), len(base))) for _ in range(9)]
    sequences = (base, base[:-1] + "A", *others)
    splits = cluster_splits(sequences)
    covered = np.sort(np.concatenate([valid for _, valid in splits]))
    np.testing.assert_array_equal(covered, np.arange(len(sequences)))
    for train, valid in splits:
        assert not set(train) & set(valid)
        assert (0 in valid) == (1 in valid)


def test_window_frame_pads_outside_the_molecule() -> None:
    feature = np.arange(6, dtype=np.float64).reshape(3, 2)
    frame = window_frame([feature], 2, 1)
    assert frame.shape == (2, 3 * 2 + 2)
    np.testing.assert_array_equal(frame.iloc[0].to_numpy(), [0, 0, 0, 1, 2, 3, 0, 2])
    np.testing.assert_array_equal(frame.iloc[1].to_numpy(), [0, 1, 2, 3, 4, 5, 1, 1])
    assert len(window_frame([feature], None, 1)) == 3


def test_submission_rows_follow_sample_order(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    ids = ["id_1_0", "id_0_0", "id_0_1"]
    pd.DataFrame({"id_seqpos": ids, **{t: 0.0 for t in TARGETS}}).to_csv(sample, index=False)
    molecules = _molecules("AC", "G")
    rows = sample_rows(molecules, [np.full((2, 5), 1.0), np.full((1, 5), 2.0)], sample)
    np.testing.assert_array_equal(rows[:, 0], [2.0, 1.0, 1.0])
    out = tmp_path / "submission.csv"
    write_submission(out, rows, sample)
    validate_submission(out, sample)
    broken = pd.read_csv(out)
    broken.loc[0, "reactivity"] = np.nan
    broken.to_csv(out, index=False)
    with pytest.raises(ValueError, match="NaN"):
        validate_submission(out, sample)
    with pytest.raises(ValueError, match="行 ID"):
        sample_rows(_molecules("A", "G"), [np.zeros((1, 5)), np.zeros((1, 5))], sample)


def test_private_score_uses_filtered_molecules_and_first_91_positions(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for name, keep in (("id_a", 1), ("id_b", 0)):
        values = json.dumps([1.0] * 91 + [100.0] * 11)
        rows.append({"id": name, "test_filter": keep, **dict.fromkeys(TARGETS, values)})
    labels = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(labels, index=False)
    submission = pd.DataFrame(
        {
            "id_seqpos": [f"{m}_{p}" for m in ("id_a", "id_b") for p in range(130)],
            **dict.fromkeys(TARGETS, 0.0),
        }
    )
    path = tmp_path / "submission.csv"
    submission.to_csv(path, index=False)
    parts = private_breakdown(path, labels)
    assert parts == {"all": 1.0, "seen_0_67": 1.0, "extrapolated_68_90": 1.0}


def test_training_and_export_sources_cannot_reach_private_labels() -> None:
    """選定経路のソースは外部評価のモジュールと Private 正解のファイルを参照しない。"""
    for name in SOURCE_FILES:
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert "external" not in text, name
        assert "post_deadline" not in text, name
        assert "private_test_labels" not in text, name


def test_private_evaluation_requires_a_selection_record(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="選定計画 final"):
        require_selection("final", tmp_path)
    (tmp_path / "final.json").write_text("{}", encoding="utf-8")
    require_selection("final", tmp_path)


def test_group_loss_scale_is_not_cancelled_by_batch_normalisation() -> None:
    predicted = torch.tensor([[1.0], [3.0]])
    y = torch.zeros(2, 1)
    weight = torch.ones(2, 1)
    base = float(batch_loss(predicted, y, weight, 1.0))
    # 重みを一律に縮めても、バッチ内の正規化で相殺される（E07 の実装で起きていたこと）
    assert float(batch_loss(predicted, y, 0.5 * weight, 1.0)) == pytest.approx(base)
    # 組の係数は相殺されない
    assert float(batch_loss(predicted, y, weight, 0.5)) == pytest.approx(0.5 * base)
    ratio = float(batch_loss(predicted, y, weight, 0.5)) / float(
        batch_loss(predicted, y, weight, 0.1)
    )
    assert ratio == pytest.approx(5.0)
    with pytest.raises(ValueError, match="loss_scale"):
        TrainGroup(np.zeros((1, 2, 1)), np.zeros((1, 2, 1)), np.ones((1, 2, 1)), loss_scale=0.0)


def test_methods_are_valid_and_source_imports_are_declared() -> None:
    for method_id in ("E00", "E00R", "E01"):
        method = load_method(method_id)
        source_spec(method).snapshot()
    with pytest.raises(ValueError, match="model"):
        Method("", "", "xgboost", "cluster", False, "all", "scored", 0)


def test_training_weights_and_five_columns() -> None:
    error = np.full((2, 1, 5), 0.4)
    labels = Labels(np.zeros_like(error), error, np.array([True, False]), np.zeros(2))
    method = load_method("E00")
    np.testing.assert_array_equal(training_weights(method, labels), np.ones((2, 1, 3)))
    filtered = replace(method, weights="sn_filter")
    np.testing.assert_array_equal(training_weights(filtered, labels)[:, 0, 0], [1.0, 0.0])
    inverse = replace(method, weights="inverse_variance")
    assert training_weights(inverse, labels)[0, 0, 0] == pytest.approx(1 / (0.16 + 0.09))
    five = to_five(np.array([[1.0, 2.0, 3.0]]), method)
    np.testing.assert_array_equal(five, [[1.0, 2.0, 2.0, 3.0, 3.0]])


def test_gru_predicts_every_position_of_longer_queries() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 70, 3))
    y = rng.normal(size=(4, 68, 2))
    config = GRUConfig(8, 1, 0.0, 1, 2, 1e-3, 0.0, 0)
    labeled = TrainGroup(x, y, np.ones_like(y))
    longer_x = rng.normal(size=(1, 90, 3))
    valid, longer = fit_predict_gru([labeled], [x[:2], longer_x], config)
    assert valid.shape == (2, 70, 2)
    assert longer.shape == (1, 90, 2)
    assert np.isfinite(longer).all()
    pseudo = TrainGroup(longer_x, longer, np.full(longer.shape, 0.5))
    (mixed,) = fit_predict_gru([labeled, pseudo], [longer_x], config)
    assert mixed.shape == (1, 90, 2)
    with pytest.raises(ValueError, match="NaN"):
        TrainGroup(x, np.full_like(y, np.nan), np.ones_like(y))


def test_quality_gate_with_masked_mcrmse(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    y = np.array([[[1.0, 1.0, 1.0]], [[np.nan, np.nan, np.nan]], [[2.0, 2.0, 2.0]]])
    splits = [(np.array([0, 1]), np.array([2])), (np.array([1, 2]), np.array([0]))]
    roles = split_roles(len(y), splits)
    evaluation = Evaluation(
        "stanford-covid-vaccine", "mcrmse", False, mcrmse, "synthetic", "v1",
        array_sha256(roles), 42,
    )  # fmt: skip
    root = Path(__file__).parent
    source = SourceSpec(root, ("data.py",), {"x": 1})
    oof = np.array([[[1.0, 1.0, 1.0]], [[np.nan] * 3], [[2.0, 2.0, 2.0]]])
    result = CrossFitResult(oof, np.zeros((1, 5)), [{}, {}])
    with reserve_cv(store, "E00", source, {}) as (reservation, start):
        published = publish_cv_run(
            store, reservation, evaluation, Trial("h", "c", "m", "d"), y, splits, result,
            0.0, lambda _: None, gate=(source, start, y.shape),
        )  # fmt: skip
    assess = dense_oof(y, splits, mcrmse, expected_shape=y.shape, data_hash="synthetic")
    assert select_runs(store, SelectionPlan(("E00",)), assess).scores == {"E00": 0.0}
    tampered = oof.copy()
    tampered[0] = 5.0
    np.save(published / "oof.npy", tampered)
    with pytest.raises(ValueError, match="ハッシュ"):
        select_runs(store, SelectionPlan(("E00",)), assess)
