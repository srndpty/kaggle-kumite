"""公開サンプルの入力証跡を作り、採点時の画像と提出 ID を照合する。"""

from hashlib import sha256
from pathlib import Path

import pandas as pd


def submission_inputs(
    sample_path: Path, expected: object = None, *, allow_hidden: bool = False
) -> tuple[pd.DataFrame, list[Path], dict[str, object]]:
    sample = pd.read_csv(sample_path, dtype={"image_id": str})
    if list(sample.columns) != ["image_id", "label"] or sample.empty:
        raise ValueError("サンプル提出の列または件数が不正です")
    ids: list[str] = sample["image_id"].tolist()
    if (
        sample["image_id"].isna().any()
        or not sample["image_id"].is_unique
        or not all(
            name.isascii() and name.removesuffix(".jpg").isdigit() and name.endswith(".jpg")
            for name in ids
        )
    ):
        raise ValueError("提出画像 ID が不正です")
    directory = sample_path.parent / "test_images"
    paths = [directory / name for name in ids]
    if {path.name for path in directory.iterdir() if path.is_file()} != set(ids):
        raise ValueError("提出画像のファイル集合がサンプル提出と一致しません")
    manifest: dict[str, object] = {
        "sample_sha256": sha256(sample_path.read_bytes()).hexdigest(),
        "images": {path.name: sha256(path.read_bytes()).hexdigest() for path in paths},
    }
    if expected is not None and manifest != expected:
        public_images = expected.get("images") if isinstance(expected, dict) else None
        if (
            not allow_hidden
            or not isinstance(public_images, dict)
            or set(ids) & public_images.keys()
        ):
            raise ValueError("提出入力が公開サンプルの証跡と一致しません")
    return sample, paths, manifest
