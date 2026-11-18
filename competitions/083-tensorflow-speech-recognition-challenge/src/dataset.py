"""音声の契約、manifest、話者・背景録音単位の分割。"""

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from kumite.provenance import array_sha256, file_sha256

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs"
CLASSES = (
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "silence",
    "unknown",
)
RATE = 16000


def read_wave(path: Path) -> tuple[int, np.ndarray]:
    return decode_wave(path.read_bytes(), str(path))


def decode_wave(content: bytes, source: str) -> tuple[int, np.ndarray]:
    rate, raw = wavfile.read(BytesIO(content))
    if rate <= 0 or raw.size == 0 or raw.ndim not in (1, 2):
        raise ValueError(f"音声の形状または周波数が不正: {source}")
    if np.issubdtype(raw.dtype, np.signedinteger):
        wave = raw.astype(np.float32) / float(-np.iinfo(raw.dtype).min)
    elif raw.dtype == np.uint8:
        wave = (raw.astype(np.float32) - 128) / 128
    elif np.issubdtype(raw.dtype, np.floating):
        wave = raw.astype(np.float32)
    else:
        raise ValueError(f"未対応の音声dtype: {raw.dtype}")
    if not np.isfinite(wave).all() or np.abs(wave).max() > 1.00001:
        raise ValueError(f"音声振幅が不正: {source}")
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    return int(rate), wave


def inspect_audio(path: Path) -> tuple[Path, int, np.ndarray, str]:
    content = path.read_bytes()
    rate, wave = decode_wave(content, str(path))
    return path, rate, wave, hashlib.sha256(content).hexdigest()


def inspect_many(paths: list[Path]):
    # mapの結果順序を入力順に固定し、一度に保持するfuture数を制限する。
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(paths), 512):
            yield from pool.map(inspect_audio, paths[start : start + 512])
            print(f"音声検査 {min(start + 512, len(paths))}/{len(paths)}", flush=True)


def one_second(wave: np.ndarray, rate: int) -> np.ndarray:
    if rate != RATE:
        divisor = math.gcd(rate, RATE)
        wave = resample_poly(wave, RATE // divisor, rate // divisor).astype(np.float32)
    if len(wave) > RATE:
        start = (len(wave) - RATE) // 2
        wave = wave[start : start + RATE]
    return np.pad(wave, (0, max(0, RATE - len(wave)))).astype(np.float32)


def label_index(word: str) -> int:
    return CLASSES.index(word if word in CLASSES[:10] else "unknown")


def make_folds(frame: pd.DataFrame, seed: int, random_split: bool = False) -> np.ndarray:
    y = frame.label.to_numpy(dtype=np.int64)
    folds = np.full(len(frame), -1, dtype=np.int64)
    if random_split:
        splitter = StratifiedKFold(3, shuffle=True, random_state=seed)
        for k, (_, valid) in enumerate(splitter.split(np.zeros(len(y)), y)):
            folds[valid] = k
    else:
        speech = np.flatnonzero(y != CLASSES.index("silence"))
        splitter = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
        for k, (_, valid) in enumerate(
            splitter.split(np.zeros(len(speech)), y[speech], frame.iloc[speech].group)
        ):
            folds[speech[valid]] = k
        background = frame.loc[y == CLASSES.index("silence"), "group"]
        sources = sorted(background.unique())
        if len(sources) < 3:
            raise ValueError("背景録音が3本未満です")
        np.random.default_rng(seed).shuffle(sources)
        allocation = {source: k % 3 for k, source in enumerate(sources)}
        folds[y == CLASSES.index("silence")] = background.map(allocation).to_numpy()
    validate_folds(frame, folds, random_split)
    return folds


def validate_folds(frame: pd.DataFrame, folds: np.ndarray, random_split: bool = False) -> None:
    if folds.shape != (len(frame),) or set(folds) != {0, 1, 2}:
        raise ValueError("分割の被覆が不正です")
    for k in range(3):
        train, valid = frame.loc[folds != k], frame.loc[folds == k]
        if set(train.label) != set(range(12)) or set(valid.label) != set(range(12)):
            raise ValueError(f"fold{k}に全12クラスがありません")
        if not random_split and set(train.group) & set(valid.group):
            raise ValueError("話者・重複・背景録音が分割を跨いでいます")


def identity(frame: pd.DataFrame, test: pd.DataFrame) -> str:
    return hashlib.sha256(
        (frame.to_csv(index=False) + test.to_csv(index=False)).encode()
    ).hexdigest()


def prepare() -> None:
    """全WAVを検査して正規化音声のmmapとmanifestを作る。途中のcacheは利用しない。"""
    out = OUTPUT / "prepared"
    if out.exists():
        raise FileExistsError(f"準備済みデータを上書きしません: {out}")
    out.mkdir(parents=True)
    train_dir = DATA / "train" / "audio"
    test_dir = DATA / "test" / "audio"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError("train.7zとtest.7zをdata/へ展開してください")
    rows: list[dict] = []
    parents: dict[str, str] = {}
    seen: dict[str, str] = {}
    train_audio: list[np.ndarray] = []

    def leader(group: str) -> str:
        parents.setdefault(group, group)
        while parents[group] != group:
            group = parents[group]
        return group

    # 完全重複を同一グループへまとめる。発話ラベル自体は変更しない。
    for path, rate, wave, digest in inspect_many(sorted(train_dir.glob("*/*.wav"))):
        word = path.parent.name
        base = dict(
            path=path.relative_to(DATA).as_posix(),
            sha256=digest,
            rate=rate,
            samples=len(wave),
            rms=float(np.sqrt(np.mean(wave**2))),
            zero_fraction=float(np.mean(wave == 0)),
        )
        if word == "_background_noise_":
            resampled = one_second(wave[:rate], rate)  # 周波数・長さ契約も検査
            if resampled.shape != (RATE,):
                raise ValueError("背景音声が不正です")
            for offset in range(len(wave) // rate):
                train_audio.append(one_second(wave[offset * rate : (offset + 1) * rate], rate))
                rows.append(
                    {
                        **base,
                        "id": f"{base['path']}@{offset}",
                        "offset": offset,
                        "word": "silence",
                        "label": 10,
                        "group": f"noise:{path.name}",
                    }
                )
        else:
            if "_nohash_" not in path.name:
                raise ValueError(f"話者IDを解析できません: {path}")
            group = f"speaker:{path.name.split('_nohash_')[0]}"
            if digest in seen:
                parents[leader(group)] = leader(seen[digest])
            seen[digest] = group
            train_audio.append(one_second(wave, rate))
            rows.append(
                {
                    **base,
                    "id": base["path"],
                    "offset": -1,
                    "word": word,
                    "label": label_index(word),
                    "group": group,
                }
            )
    frame = pd.DataFrame(rows)
    frame["group"] = [leader(g) for g in frame.group]
    if frame.id.duplicated().any():
        raise ValueError("学習IDが重複しています")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    if list(sample.columns) != ["fname", "label"] or sample.fname.duplicated().any():
        raise ValueError("提出テンプレートが不正です")
    test_rows = []
    test_paths = [test_dir / name for name in sample.fname]
    test_audio = np.lib.format.open_memmap(
        out / "test.npy", mode="w+", dtype=np.float32, shape=(len(test_paths), RATE)
    )
    for i, (path, rate, wave, digest) in enumerate(inspect_many(test_paths)):
        name = path.name
        if path.parent != test_dir:
            raise ValueError("test名にディレクトリを含みます")
        test_audio[i] = one_second(wave, rate)
        test_rows.append(
            dict(
                id=name,
                path=path.relative_to(DATA).as_posix(),
                sha256=digest,
                offset=-1,
                rate=rate,
                samples=len(wave),
                rms=float(np.sqrt(np.mean(wave**2))),
            )
        )
    test = pd.DataFrame(test_rows)
    test_audio.flush()
    del test_audio
    cache = np.lib.format.open_memmap(
        out / "train.npy", mode="w+", dtype=np.float32, shape=(len(frame), RATE)
    )
    for start in range(0, len(frame), 512):
        cache[start : start + 512] = np.stack(train_audio[start : start + 512])
    cache.flush()
    del cache, train_audio
    frame.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    folds = make_folds(frame, 42)
    summary = dict(
        n_train=len(frame),
        n_test=len(test),
        class_counts=frame.label.map(dict(enumerate(CLASSES))).value_counts().to_dict(),
        original_word_counts=frame.word.value_counts().to_dict(),
        groups=int(frame.group.nunique()),
        rates=frame.rate.value_counts().to_dict(),
        lengths=frame.samples.describe().to_dict(),
        rms=frame.rms.describe().to_dict(),
        zero_fraction=frame.zero_fraction.describe().to_dict(),
        duplicate_speech_rows=int(frame.loc[frame.offset < 0, "sha256"].duplicated().sum()),
        shared_train_test_hashes=len(set(frame.sha256) & set(test.sha256)),
        fold_counts=pd.crosstab(folds, frame.label).to_dict(),
        data_hash=identity(frame, test),
        split_hash=array_sha256(folds),
        caches={name: file_sha256(out / f"{name}.npy") for name in ("train", "test")},
    )
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def load_prepared(
    *, deep: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """manifestを照合する。deep時は音声cacheの内容も利用前に照合する。"""
    path = OUTPUT / "prepared"
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    frame, test = (pd.read_csv(path / f"{name}.csv") for name in ("train", "test"))
    if identity(frame, test) != summary["data_hash"]:
        raise ValueError("manifestのハッシュが一致しません")
    if deep:
        for name in ("train", "test"):
            if file_sha256(path / f"{name}.npy") != summary["caches"][name]:
                raise ValueError(f"{name}音声cacheのハッシュ不一致")
    arrays = [np.load(path / f"{name}.npy", mmap_mode="r") for name in ("train", "test")]
    for array, manifest in zip(arrays, (frame, test), strict=True):
        if array.dtype != np.float32 or array.shape != (len(manifest), RATE):
            raise ValueError("cacheの形状が不正です")
    return frame, test, arrays[0], arrays[1], summary


def submission(test: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    from kumite.checks import validate_probabilities

    validate_probabilities(probabilities, len(test), len(CLASSES))
    return pd.DataFrame(
        {"fname": test.id, "label": np.asarray(CLASSES)[probabilities.argmax(axis=1)]}
    )


if __name__ == "__main__":
    prepare()
