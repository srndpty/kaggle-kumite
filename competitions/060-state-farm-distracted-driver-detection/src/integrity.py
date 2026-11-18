"""派生キャッシュ（画像配列・小画像・CNN 特徴）と、後処理の元になる run の真正性の検査。

キャッシュは作るときに manifest（JSON）を書く。manifest には、作り方（meta: 元の行順・前処理の設定・
backbone など）と、作ったファイルの sha256・大きさ・更新時刻を残す。
- 使うとき（`verify_manifest`）は、meta が今の期待と一致し、ファイルの大きさ・更新時刻が manifest と
  一致することを確かめる（毎回 20GB 超を読み直さない）。`deep=True` なら内容の sha256 も照合する
- run には manifest 自身の sha256 を入力として記録する（`inputs`）。manifest が変われば別の入力になる
"""

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from kumite.experiments import ExperimentRecord
from kumite.provenance import file_sha256

MANIFEST_VERSION = 1


def rows_sha256(keys: Iterable[str]) -> str:
    """行の並び（画像のファイル名など）のハッシュ。行順が変われば変わる。"""
    digest = hashlib.sha256()
    for key in keys:
        digest.update(f"{key}\n".encode())
    return digest.hexdigest()


def _stat(path: Path) -> dict[str, int]:
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def write_manifest(path: Path, meta: Mapping[str, Any], files: Sequence[Path]) -> str:
    """files（manifest と同じディレクトリか、その下）の内容ハッシュと meta を書き、manifest の sha256 を返す。"""
    entries = {
        f.relative_to(path.parent).as_posix(): {"sha256": file_sha256(f), **_stat(f)} for f in files
    }
    body = {"version": MANIFEST_VERSION, "meta": dict(meta), "files": entries}
    path.write_text(
        json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
    )
    return file_sha256(path)


def verify_manifest(path: Path, expected_meta: Mapping[str, Any], deep: bool = False) -> str:
    """manifest が今の期待と一致することを確かめ、manifest の sha256 を返す。一致しなければ ValueError。"""
    if not path.exists():
        raise FileNotFoundError(f"{path} がありません。キャッシュを作り直してください")
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("version") != MANIFEST_VERSION:
        raise ValueError(f"{path} の版が違います（{body.get('version')}）")
    meta = body["meta"]
    diff = sorted(k for k in set(meta) | set(expected_meta) if meta.get(k) != expected_meta.get(k))
    if diff:
        raise ValueError(f"{path} の作り方が今の期待と違います（{', '.join(diff)}）")
    for name, entry in body["files"].items():
        f = path.parent / name
        if not f.exists():
            raise ValueError(f"{f} がありません（{path}）")
        if _stat(f) != {"size": entry["size"], "mtime_ns": entry["mtime_ns"]}:
            raise ValueError(f"{f} が manifest の作成後に変わっています（大きさか更新時刻）")
        if deep and file_sha256(f) != entry["sha256"]:
            raise ValueError(f"{f} の内容が manifest と一致しません")
    return file_sha256(path)


def check_array(array: np.ndarray, shape: tuple[int, ...], dtype: str, name: str) -> None:
    if array.shape != shape or array.dtype != np.dtype(dtype):
        raise ValueError(
            f"{name} の形状・型が違います（{array.shape} {array.dtype}、期待 {shape} {dtype}）"
        )


def check_base_run(
    record: ExperimentRecord,
    *,
    competition_id: str,
    data_hash: str,
    split_version: str,
    split_hash: str,
    inputs: Mapping[str, str],
) -> None:
    """後処理の元になる run が、今のデータ・分割・入力と同じ条件で作られたことを確かめる。

    inputs は元の run の記録に同じ値で残っている必要がある入力（画像キャッシュの manifest など）。
    """
    expected = {
        "competition_id": competition_id,
        "data_hash": data_hash,
        "split_version": split_version,
        "split_hash": split_hash,
    }
    diff = [k for k, v in expected.items() if getattr(record, k) != v]
    diff += [f"inputs[{k}]" for k, v in inputs.items() if record.inputs.get(k) != v]
    if diff:
        raise ValueError(
            f"元の run {record.run_id} は今の条件と違います（{', '.join(diff)}）。"
            "同じ条件で作り直した run を元にしてください"
        )
