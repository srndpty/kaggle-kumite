"""被験者分割と、加速度から提出 CSV までの契約を検査する。"""

import numpy as np
import pandas as pd
import pytest
from pipeline import AXES, EVENTS, annotated, features, make_submission, model_groups
from prepare import assign_folds
from select_model import bootstrap_ap
from sklearn.metrics import average_precision_score

from kumite.checks import check_label_independence


def test_subject_folds() -> None:
    records = pd.DataFrame(
        [
            {"subject": f"s{i}", "kind": kind, **dict.fromkeys(EVENTS, 1)}
            for i in range(10)
            for kind in ("defog", "tdcsfog")
        ]
    )
    split = assign_folds(records)
    assert split.groupby("subject")["fold"].nunique().eq(1).all()
    assert split["fold"].nunique() == 5


def test_windows_respect_discontinuities_and_recordings() -> None:
    frame = pd.DataFrame({"Time": [0, 1, 25, 26], **{axis: [2, 4, 10, 14] for axis in AXES}})
    trailing = features(frame, "defog", "E02")
    np.testing.assert_allclose(trailing[:, 4], [2, 3, 10, 12])
    np.testing.assert_allclose(trailing[:, 7], [0, 1, 0, 2])
    centered = features(frame, "defog", "E03")
    np.testing.assert_allclose(centered[:, 4], [3, 3, 12, 12])
    separate = features(frame.iloc[2:], "defog", "E02")
    np.testing.assert_allclose(trailing[2:], separate)


def test_mask_and_label_independence() -> None:
    frame = pd.DataFrame(
        {
            "Time": [0, 1, 2],
            "Valid": [True, True, False],
            "Task": [True, False, True],
            **{axis: [0.5, 0.75, 1] for axis in AXES},
            **{event: [0, 1, 0] for event in EVENTS},
        }
    )
    np.testing.assert_array_equal(annotated(frame, "defog"), [True, False, False])
    np.testing.assert_array_equal(annotated(frame, "tdcsfog"), [True, True, True])

    def build(labels: np.ndarray) -> tuple[np.ndarray]:
        return (features(frame.assign(StartHesitation=labels), "defog", "E03"),)

    check_label_independence(build, frame["StartHesitation"].to_numpy(), np.array([0, 0, 1]))


def test_submission_uses_sample_order_and_checks_alignment() -> None:
    sample = pd.DataFrame({"Id": ["b_0", "a_0"], **dict.fromkeys(EVENTS, 0.0)})
    ids = np.array(["a_0", "b_0"])
    predictions = np.array([[0.3, 0.1, 0.7], [0.1, 0.6, 0.4]])
    result = make_submission(sample, ids, predictions)
    np.testing.assert_allclose(result[EVENTS].to_numpy(), predictions[::-1])
    assert result["Id"].equals(sample["Id"])
    with pytest.raises(ValueError, match="Invalid test IDs"):
        make_submission(sample, np.array(["b_0", "b_0"]), predictions)
    predictions[0, 0] = np.nan
    with pytest.raises(ValueError, match="probabilities"):
        make_submission(sample, ids, predictions)


def test_bootstrap_ap_preserves_score_ties_and_subject_weights() -> None:
    labels = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    scores = np.array([[0.8, 0.5, 0.1], [0.8, 0.2, 0.1], [0.3, 0.5, 0.1], [0.1, 0, 0]])
    subjects = np.array([0, 1, 0, 2])
    samples = np.array([[1, 1, 1], [2, 0, 1]])
    expected = [
        average_precision_score(labels, scores, sample_weight=sample[subjects])
        for sample in samples
    ]
    np.testing.assert_allclose(bootstrap_ap(labels, scores, subjects, samples), expected)


def test_models_are_routed_by_recording_kind() -> None:
    pooled = model_groups({"0": [0.1, 0.2, 0.3]})
    assert pooled["defog"]["0"] == pooled["tdcsfog"]["0"]
    separate = model_groups({"0:defog": [0.1, 0.2, 0.3], "0:tdcsfog": [0.2, 0.3, 0.4]})
    assert separate["defog"]["0"] == [0.1, 0.2, 0.3]
    assert separate["tdcsfog"]["0"] == [0.2, 0.3, 0.4]
