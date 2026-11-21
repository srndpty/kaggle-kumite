"""現行データと保存済み OOF の監査を検証する。"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from analyze import checked_oof, sensitivity, verify_dataset
from dataset import data_hash
from PIL import Image

from kumite.experiments import ExperimentRecord

RECORD = ExperimentRecord(
    competition_id="070",
    run_id="E00",
    metric="mean_iou_precision",
    greater_is_better=True,
    data_version="kaggle-v1",
    data_hash="",
    split_version="coverage-group-v1",
    split_hash="",
    seed=42,
    hypothesis="",
    changed_one_thing="",
    model="U-Net",
    fold_scores=[1.0] * 5,
    oof_score=1.0,
    code_version="",
)


@pytest.mark.parametrize("directory", ["images", "masks"])
def test_changed_pixel_rejected(tmp_path: Path, directory: str) -> None:
    for name in ("train.csv", "depths.csv", "sample_submission.csv"):
        (tmp_path / name).write_text("id\n", encoding="utf-8")
    path = tmp_path / "competition_data" / "train" / directory / "a.png"
    path.parent.mkdir(parents=True)
    image = Image.new("L", (101, 101))
    image.save(path)
    record = replace(RECORD, data_hash=data_hash(tmp_path))
    image.putpixel((0, 0), 255)
    image.save(path)
    with pytest.raises(ValueError, match="現行データのハッシュ"):
        verify_dataset(tmp_path, record)


def test_oof_rejects_invalid_values_and_scores() -> None:
    masks = np.zeros((5, 2, 2), dtype=bool)
    folds = np.arange(5)
    oof = masks.astype(float)
    np.testing.assert_array_equal(checked_oof(oof, masks, RECORD, folds), np.ones(5))
    for invalid in (oof[:-1], oof.astype(str), np.full_like(oof, np.nan), np.full_like(oof, 2)):
        with pytest.raises(ValueError, match="二値マスク"):
            checked_oof(invalid, masks, RECORD, folds)
    with pytest.raises(ValueError, match="実験ログ"):
        checked_oof(oof, masks, replace(RECORD, fold_scores=[0.0] * 5), folds)


def test_sensitivity_weights_each_group_equally() -> None:
    result = sensitivity(np.zeros(3), np.array([1.0, 1.0, 0.0]), np.array(["a", "a", "b"]))
    assert result["all_rows"][0] == pytest.approx(2 / 3)
    assert result["exclude_duplicates"][0] == 0
    assert result["equal_group_weight"][0] == 0.5
