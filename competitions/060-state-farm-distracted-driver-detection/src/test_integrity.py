import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from cache import expected_meta
from integrity import (
    check_array,
    check_base_run,
    rows_sha256,
    verify_manifest,
    write_manifest,
)

from kumite.experiments import ExperimentRecord


def _frames(order: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {"img": ["a.jpg", "b.jpg", "c.jpg"], "classname": ["c0", "c1", "c2"]}
    ).iloc[order]
    test = pd.DataFrame({"img": ["x.jpg", "y.jpg"]})
    return train, test


def _cache(tmp_path: Path) -> tuple[Path, Path]:
    array = tmp_path / "train.npy"
    np.save(array, np.arange(12, dtype=np.uint8).reshape(3, 4))
    return tmp_path / "manifest.json", array


def test_rows_sha256_depends_on_order() -> None:
    assert rows_sha256(["a", "b"]) != rows_sha256(["b", "a"])


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest, array = _cache(tmp_path)
    digest = write_manifest(manifest, {"size": [4, 3]}, [array])
    assert verify_manifest(manifest, {"size": [4, 3]}, deep=True) == digest


def test_manifest_rejects_reordered_rows(tmp_path: Path) -> None:
    """同じ行数・同じ設定のまま CSV の行順だけが変わると、作り方（meta）の不一致で拒否する。"""
    manifest, array = _cache(tmp_path)
    write_manifest(manifest, expected_meta(*_frames([0, 1, 2])), [array])
    with pytest.raises(ValueError, match="rows_train"):
        verify_manifest(manifest, expected_meta(*_frames([1, 0, 2])))


def test_manifest_rejects_rewritten_file(tmp_path: Path) -> None:
    manifest, array = _cache(tmp_path)
    write_manifest(manifest, {}, [array])
    np.save(array, np.arange(12, dtype=np.uint8).reshape(3, 4)[::-1])  # 行順だけ入れ替えて書き直す
    os.utime(array, ns=(0, array.stat().st_mtime_ns + 10**9))
    with pytest.raises(ValueError, match="変わっています"):
        verify_manifest(manifest, {})


def test_manifest_deep_check_detects_content_with_same_stat(tmp_path: Path) -> None:
    manifest, array = _cache(tmp_path)
    write_manifest(manifest, {}, [array])
    st = array.stat()
    np.save(array, np.arange(12, dtype=np.uint8).reshape(3, 4)[::-1])
    os.utime(array, ns=(st.st_atime_ns, st.st_mtime_ns))  # 大きさ・更新時刻は同じ
    verify_manifest(manifest, {})  # 浅い照合は通る（毎回の読み込みでは中身を読まない）
    with pytest.raises(ValueError, match="内容"):
        verify_manifest(manifest, {}, deep=True)


def test_manifest_missing_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_manifest(tmp_path / "manifest.json", {})


def test_check_array_rejects_shape_and_dtype() -> None:
    check_array(np.zeros((2, 3), np.uint8), (2, 3), "uint8", "x")
    with pytest.raises(ValueError):
        check_array(np.zeros((2, 3), np.uint8), (3, 3), "uint8", "x")
    with pytest.raises(ValueError):
        check_array(np.zeros((2, 3), np.float32), (2, 3), "uint8", "x")


BASE = ExperimentRecord(
    competition_id="060",
    run_id="E02",
    metric="multi_logloss",
    greater_is_better=False,
    data_version="v",
    data_hash="data",
    split_version="driver-v1",
    split_hash="split",
    seed=42,
    hypothesis="",
    changed_one_thing="",
    model="finetune",
    fold_scores=[0.4],
    oof_score=0.4,
    code_version="abc",
    inputs={"cache/manifest": "cache-A"},
)
EXPECTED = {
    "competition_id": "060",
    "data_hash": "data",
    "split_version": "driver-v1",
    "split_hash": "split",
}


def test_check_base_run_accepts_matching_record() -> None:
    check_base_run(BASE, **EXPECTED, inputs={"cache/manifest": "cache-A"})


@pytest.mark.parametrize(
    ("change", "inputs", "field"),
    [
        ({"competition_id": "001"}, {"cache/manifest": "cache-A"}, "competition_id"),
        ({"data_hash": "other"}, {"cache/manifest": "cache-A"}, "data_hash"),
        ({"split_version": "random-v1"}, {"cache/manifest": "cache-A"}, "split_version"),
        ({"split_hash": "other"}, {"cache/manifest": "cache-A"}, "split_hash"),
        # 同じ行数・同じ split 名のまま画像配列の行順が変わる → キャッシュの manifest が変わる
        ({}, {"cache/manifest": "cache-B"}, "inputs"),
        # manifest の導入前の run（入力の記録が無い）は元にできない
        ({"inputs": {}}, {"cache/manifest": "cache-A"}, "inputs"),
    ],
)
def test_check_base_run_rejects_mismatch(
    change: dict[str, Any], inputs: dict[str, str], field: str
) -> None:
    record = replace(BASE, **change)
    with pytest.raises(ValueError, match=field):
        check_base_run(record, **EXPECTED, inputs=inputs)
