"""画像の縮小キャッシュ。生の 640x480 JPEG を毎 epoch 読むのを避けるため、一度だけ縮める。

- `<data>/cache_<W>x<H>/{train,test}/<img>`: 学習・推論用の縮小 JPEG（品質 95）
- `<data>/arrays_<W>x<H>/{train,test}.npy`: 縮小 JPEG をデコードした (枚数, H, W, 3) の uint8 配列。
  行順は `load_train` / `load_test` と同じ。学習時は JPEG をデコードせず、この配列を読む（CPU の
  デコードが律速になり GPU が遊ぶのを避ける）
- `<data>/thumbs_<W>x<H>.npz`: EDA・近傍検査用のグレースケール小画像（train・test、uint8）
- `<data>/cache_manifest_<W>x<H>.json`: 配列と小画像の manifest（作り方・元の行順・内容ハッシュ。`integrity`）。
  `load_data` は使う前に manifest を照合し、run には manifest のハッシュを入力として記録する

実行:
  uv run python competitions/060-state-farm-distracted-driver-detection/src/cache.py build
  uv run python competitions/060-state-farm-distracted-driver-detection/src/cache.py manifest  # 既存のキャッシュを標本で照合して manifest を書く
  uv run python competitions/060-state-farm-distracted-driver-detection/src/cache.py verify    # 内容ハッシュまで照合する
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dataset import DATA_DIR, IMG, load_test, load_train
from integrity import check_array, rows_sha256, verify_manifest, write_manifest
from PIL import Image

SIZE = (320, 240)
THUMB = (32, 24)
CACHE_DIR = DATA_DIR / f"cache_{SIZE[0]}x{SIZE[1]}"
ARRAY_DIR = DATA_DIR / f"arrays_{SIZE[0]}x{SIZE[1]}"
THUMBS_NPZ = DATA_DIR / f"thumbs_{THUMB[0]}x{THUMB[1]}.npz"
MANIFEST = DATA_DIR / f"cache_manifest_{SIZE[0]}x{SIZE[1]}.json"
JPEG_QUALITY = 95
N_SAMPLE_CHECK = 64  # manifest コマンドで、元画像から作り直して照合する行数（split ごと）


def cached_path(split: str, img: str) -> Path:
    return CACHE_DIR / split / img


def array_path(split: str) -> Path:
    return ARRAY_DIR / f"{split}.npy"


def load_array(split: str) -> np.ndarray:
    """デコード済みの画像配列（読み取り専用の memmap）。照合は `verify_cache` で先に行う。"""
    return np.load(array_path(split), mmap_mode="r")


def _keys(split: str, df: pd.DataFrame) -> list[str]:
    """行の識別子。学習はクラスのディレクトリも含める（同名の画像はクラス間で重ならないが、明示する）。"""
    if split == "train":
        return (df["classname"] + "/" + df[IMG]).tolist()
    return df[IMG].tolist()


def expected_meta(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """今の CSV と設定から決まる、キャッシュの作り方。manifest の meta と一致する必要がある。"""
    meta: dict[str, Any] = {
        "kind": "image_cache",
        "size_wh": list(SIZE),
        "thumb_wh": list(THUMB),
        "jpeg_quality": JPEG_QUALITY,
        "resize": "PIL bilinear from original RGB",
        "thumb": "PIL box from original, grayscale (L)",
        "dtype": "uint8",
    }
    for split, df in (("train", train), ("test", test)):
        meta[f"rows_{split}"] = rows_sha256(_keys(split, df))
        meta[f"n_{split}"] = len(df)
    return meta


def verify_cache(train: pd.DataFrame, test: pd.DataFrame, deep: bool = False) -> str:
    """manifest を照合し、配列と小画像の形状・型も確かめて、manifest のハッシュを返す。"""
    digest = verify_manifest(MANIFEST, expected_meta(train, test), deep=deep)
    thumbs = np.load(THUMBS_NPZ)
    for split, df in (("train", train), ("test", test)):
        check_array(load_array(split), (len(df), SIZE[1], SIZE[0], 3), "uint8", f"{split}.npy")
        check_array(thumbs[split], (len(df), THUMB[1], THUMB[0]), "uint8", f"thumbs[{split}]")
    return digest


def _write_manifest(train: pd.DataFrame, test: pd.DataFrame) -> str:
    files = [array_path("train"), array_path("test"), THUMBS_NPZ]
    return write_manifest(MANIFEST, expected_meta(train, test), files)


def _decode(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _write_array(split: str, paths: list[str]) -> None:
    out = np.lib.format.open_memmap(
        array_path(split), mode="w+", dtype=np.uint8, shape=(len(paths), SIZE[1], SIZE[0], 3)
    )
    with ProcessPoolExecutor() as pool:
        for i, img in enumerate(pool.map(_decode, paths, chunksize=256)):
            out[i] = img
    out.flush()
    del out


def _resize(job: tuple[str, str]) -> np.ndarray:
    src, dst = job
    with Image.open(src) as im:
        rgb = im.convert("RGB")
        if not Path(dst).exists():
            rgb.resize(SIZE, Image.Resampling.BILINEAR).save(dst, quality=JPEG_QUALITY)
        thumb = rgb.convert("L").resize(THUMB, Image.Resampling.BOX)
    return np.asarray(thumb, dtype=np.uint8)


def build() -> None:
    thumbs: dict[str, np.ndarray] = {}
    train, test = load_train(), load_test()
    for split, df in (("train", train), ("test", test)):
        (CACHE_DIR / split).mkdir(parents=True, exist_ok=True)
        jobs = [(p, str(cached_path(split, i))) for p, i in zip(df["path"], df[IMG], strict=True)]
        with ProcessPoolExecutor() as pool:
            thumbs[split] = np.stack(list(pool.map(_resize, jobs, chunksize=256)))
        print(f"{split}: {len(jobs)} 枚 → {CACHE_DIR / split}")
        ARRAY_DIR.mkdir(parents=True, exist_ok=True)
        _write_array(split, [dst for _, dst in jobs])
        print(f"{split}: デコード済みの配列 → {array_path(split)}")
    np.savez_compressed(THUMBS_NPZ, train=thumbs["train"], test=thumbs["test"])
    print(f"小画像: {THUMBS_NPZ}")
    print(f"manifest: {MANIFEST}（{_write_manifest(train, test)[:12]}）")


def manifest_existing() -> None:
    """既存のキャッシュ（manifest の導入前に作ったもの）を標本で照合してから manifest を書く。

    各 split の N_SAMPLE_CHECK 行について、元の JPEG から縮小 JPEG・小画像を作り直し、
    配列・小画像の同じ行と一致することを確かめる（行順と内容の対応の確認）。
    """
    train, test = load_train(), load_test()
    thumbs = np.load(THUMBS_NPZ)
    rng = np.random.default_rng(0)
    for split, df in (("train", train), ("test", test)):
        check_array(load_array(split), (len(df), SIZE[1], SIZE[0], 3), "uint8", f"{split}.npy")
        array = load_array(split)
        for i in np.sort(rng.choice(len(df), N_SAMPLE_CHECK, replace=False)):
            if not np.array_equal(array[i], _decode(str(cached_path(split, df[IMG].iloc[i])))):
                raise ValueError(f"{split}.npy の行 {i} が縮小 JPEG と一致しません")
            with Image.open(df["path"].iloc[i]) as im:
                thumb = im.convert("RGB").convert("L").resize(THUMB, Image.Resampling.BOX)
            if not np.array_equal(thumbs[split][i], np.asarray(thumb, dtype=np.uint8)):
                raise ValueError(f"thumbs[{split}] の行 {i} が元画像と一致しません")
        print(f"{split}: {N_SAMPLE_CHECK} 行が元画像・縮小 JPEG と一致")
    print(f"manifest: {MANIFEST}（{_write_manifest(train, test)[:12]}）")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "manifest", "verify"])
    command = parser.parse_args().command
    if command == "build":
        build()
    elif command == "manifest":
        manifest_existing()
    else:
        print(f"一致: {verify_cache(load_train(), load_test(), deep=True)[:12]}")


if __name__ == "__main__":
    main()
