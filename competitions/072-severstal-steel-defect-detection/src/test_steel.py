import csv
import json
from pathlib import Path

import numpy as np
import pytest
from dummy import SOURCE
from split import make_folds
from steel import decode_rle, dice, encode_rle, load_labels


def test_rle_uses_one_based_column_order() -> None:
    expected = np.array([[True, False, True], [True, False, False]], dtype=np.bool_)
    np.testing.assert_array_equal(decode_rle("1 2 5 1", (2, 3)), expected)
    assert encode_rle(expected) == "1 2 5 1"
    assert encode_rle(decode_rle("", (2, 3))) == ""


@pytest.mark.parametrize("encoded", ["0 1", "1 0", "6 2", "1 2 2 1", "3"])
def test_invalid_runs_are_rejected(encoded: str) -> None:
    with pytest.raises(ValueError):
        decode_rle(encoded, (2, 3))


def test_dice_scores_empty_pairs_as_one() -> None:
    empty = decode_rle("", (2, 3))
    first = decode_rle("1 2", (2, 3))
    second = decode_rle("2 2", (2, 3))
    assert dice(empty, empty) == 1
    assert dice(empty, first) == 0
    assert dice(first, second) == 0.5


def test_empty_images_and_duplicates_are_kept_together(tmp_path: Path) -> None:
    directory = tmp_path / "train_images"
    directory.mkdir()
    for index, content in enumerate((b"a", b"a", b"b", b"c")):
        (directory / f"{index}.jpg").write_bytes(content)
    (tmp_path / "train.csv").write_text("ImageId,ClassId,EncodedPixels\n", encoding="utf-8")
    labels = load_labels(tmp_path)
    assert all(not masks for masks in labels.values())
    folds = make_folds(tmp_path, labels)
    assert folds["0.jpg"] == folds["1.jpg"]
    assert set(folds.values()) == {0, 1, 2}


def test_dummy_enumerates_hidden_jpegs_in_submission_order(tmp_path: Path) -> None:
    test_dir = tmp_path / "input" / "competition" / "test_images"
    test_dir.mkdir(parents=True)
    (test_dir / "b.jpg").touch()
    (test_dir / "a.jpg").touch()
    output = tmp_path / "submission.csv"
    source = SOURCE.replace("'/kaggle/input'", json.dumps(str(tmp_path / "input")))
    source = source.replace("'/kaggle/working/submission.csv'", json.dumps(str(output)))
    exec(compile(source, "dummy-notebook", "exec"), {})
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows == [
        ["ImageId_ClassId", "EncodedPixels"],
        *[[f"{image}.jpg_{class_id}", ""] for image in ("a", "b") for class_id in range(1, 5)],
    ]
