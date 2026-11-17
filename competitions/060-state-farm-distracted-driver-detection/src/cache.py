"""画像の縮小キャッシュ。生の 640x480 JPEG を毎 epoch 読むのを避けるため、一度だけ縮める。

- `<data>/cache_<W>x<H>/{train,test}/<img>`: 学習・推論用の縮小 JPEG（品質 95）
- `<data>/arrays_<W>x<H>/{train,test}.npy`: 縮小 JPEG をデコードした (枚数, H, W, 3) の uint8 配列。
  行順は `load_train` / `load_test` と同じ。学習時は JPEG をデコードせず、この配列を読む（CPU の
  デコードが律速になり GPU が遊ぶのを避ける）
- `<data>/thumbs_<W>x<H>.npz`: EDA・近傍検査用のグレースケール小画像（train・test、uint8）

実行: uv run python competitions/060-state-farm-distracted-driver-detection/src/cache.py
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from dataset import DATA_DIR, IMG, load_test, load_train
from PIL import Image

SIZE = (320, 240)
THUMB = (32, 24)
CACHE_DIR = DATA_DIR / f"cache_{SIZE[0]}x{SIZE[1]}"
ARRAY_DIR = DATA_DIR / f"arrays_{SIZE[0]}x{SIZE[1]}"
THUMBS_NPZ = DATA_DIR / f"thumbs_{THUMB[0]}x{THUMB[1]}.npz"


def cached_path(split: str, img: str) -> Path:
    return CACHE_DIR / split / img


def array_path(split: str) -> Path:
    return ARRAY_DIR / f"{split}.npy"


def load_array(split: str) -> np.ndarray:
    """デコード済みの画像配列（読み取り専用の memmap）。"""
    return np.load(array_path(split), mmap_mode="r")


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
            rgb.resize(SIZE, Image.Resampling.BILINEAR).save(dst, quality=95)
        thumb = rgb.convert("L").resize(THUMB, Image.Resampling.BOX)
    return np.asarray(thumb, dtype=np.uint8)


def build() -> None:
    thumbs: dict[str, np.ndarray] = {}
    for split, df in (("train", load_train()), ("test", load_test())):
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


if __name__ == "__main__":
    build()
