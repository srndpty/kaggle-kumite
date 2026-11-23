from pathlib import Path

import numpy as np
import pytest
from dataset import Sample, split_images
from metric import image_score, match_counts
from model import fuse_views


def test_matching_counts_each_truth_once() -> None:
    truth = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert image_score(truth, truth, np.array([1.0])) == 1.0
    assert image_score(truth, np.repeat(truth, 2, axis=0), np.array([0.5, 1.0])) == 0.5


def test_flip_fusion_weights_overlapping_boxes_and_penalizes_singletons() -> None:
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [1.0, 0.0, 11.0, 10.0], [20.0, 0.0, 30.0, 10.0]])
    fused, scores = fuse_views(boxes, np.array([0.9, 0.6, 0.4]))
    np.testing.assert_allclose(fused, [[0.4, 0.0, 10.4, 10.0]])
    np.testing.assert_allclose(scores, [0.75])


def test_confidence_order_controls_the_matching() -> None:
    truth = np.array([[0.0, 0.0, 10.0, 10.0], [4.0, 0.0, 14.0, 10.0]])
    boxes = np.array([[1.0, 0.0, 11.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    assert image_score(truth, boxes, np.array([0.9, 0.8])) == pytest.approx(1 / 3)
    assert image_score(truth, boxes, np.array([0.8, 0.9])) == pytest.approx(4 / 9)


def test_threshold_and_missing_boxes() -> None:
    truth = np.array([[0.0, 0.0, 10.0, 10.0]])
    boxes = np.array([[0.0, 0.0, 5.0, 10.0]])
    empty = np.empty((0, 4))
    assert image_score(truth, boxes, np.array([1.0])) == 0.0
    assert image_score(truth, empty, np.empty(0)) == 0.0
    assert image_score(empty, boxes, np.array([1.0])) == 0.0
    assert image_score(empty, empty, np.empty(0)) is None


def test_matching_against_scalar_reference() -> None:
    truth = [[0.0, 0.0, 10.0, 10.0], [4.0, 0.0, 14.0, 10.0]]
    predictions = [[1.0, 0.0, 11.0, 10.0], [0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 5.0, 10.0]]
    scores = [0.8, 0.8, 0.3]
    thresholds = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)

    def iou(a: list[float], b: list[float]) -> float:
        width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        intersection = width * height
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
        return intersection / union

    reference = []
    for threshold in thresholds:
        available = truth.copy()
        for index in sorted(range(len(scores)), key=lambda i: -scores[i]):
            if available:
                nearest = max(available, key=lambda box: iou(box, predictions[index]))
                if iou(nearest, predictions[index]) > threshold:
                    available.remove(nearest)
        reference.append(len(truth) - len(available))
    assert (
        match_counts(np.array(truth), np.array(predictions), np.array(scores), thresholds)
        == reference
    )


def test_source_split_keeps_every_source_in_one_fold() -> None:
    samples = [Sample(Path(f"{i}.jpg"), str(i % 7), []) for i in range(21)]
    folds = split_images(samples, "source-v1")
    assert set(folds) == {0, 1, 2}
    for source in {sample.source for sample in samples}:
        assert (
            len(
                {
                    fold
                    for sample, fold in zip(samples, folds, strict=True)
                    if sample.source == source
                }
            )
            == 1
        )
    assert split_images(samples, "random-v1") == split_images(samples, "random-v1")
