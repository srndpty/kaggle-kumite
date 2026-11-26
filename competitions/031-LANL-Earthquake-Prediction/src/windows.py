"""公式波形から固定窓の特徴量と破壊周期を作る。"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from kumite.provenance import files_sha256

WINDOW = 150_000
RESET_JUMP = 1.0
DTYPES = {"acoustic_data": "int16", "time_to_failure": "float64"}
ROLLING_WIDTH = 50


def rolling_std_quantile(x: np.ndarray, width: int) -> float:
    sums = np.r_[0.0, np.cumsum(x)]
    squares = np.r_[0.0, np.cumsum(x * x)]
    mean = (sums[width:] - sums[:-width]) / width
    variance = (squares[width:] - squares[:-width]) / width - mean**2
    return float(np.quantile(np.sqrt(np.maximum(variance, 0.0)), 0.20))


def features(signal: np.ndarray) -> np.ndarray:
    x = signal.astype(np.float64)
    q25, q75 = np.quantile(x, [0.25, 0.75])
    base = [x.mean(), x.std(), np.median(x), x.min(), x.max(), q75 - q25]
    peaks = np.quantile(np.abs(x), [0.95, 0.99, 0.999])
    rms = np.sqrt(np.mean(x.reshape(10, -1) ** 2, axis=1))
    power = np.abs(np.fft.rfft(x - x.mean()))[1:] ** 2
    bands = np.array([band.sum() for band in np.array_split(power, 8)])
    if bands.sum() > 0:
        bands /= bands.sum()
    return np.r_[
        base,
        peaks,
        rms.max(),
        np.median(rms),
        rms[-1],
        bands,
        rolling_std_quantile(x, min(ROLLING_WIDTH, len(x))),
    ]


def windows(
    chunks: Iterable[tuple[np.ndarray, np.ndarray]], size: int = WINDOW
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boundaries: list[tuple[int, float, float]] = []
    rows: list[tuple[int, int, int, float]] = []
    values: list[np.ndarray] = []
    carry = np.empty(0, dtype=np.int16)
    offset = 0
    previous = np.nan
    for x, y in chunks:
        if not len(y) or len(x) != len(y) or not np.isfinite(y).all():
            raise ValueError("学習チャンクの長さまたはラベルが不正です")
        jumps = np.flatnonzero(np.diff(y, prepend=previous) > RESET_JUMP)
        start = 0
        for end in [*jumps.tolist(), len(y)]:
            cycle = len(boundaries)
            if cycle:
                joined = np.concatenate((carry, x[start:end]))
                joined_start = offset + start - len(carry)
                for left in range(0, len(joined) - size + 1, size):
                    stop = joined_start + left + size
                    rows.append((joined_start + left, stop, cycle, float(y[stop - offset - 1])))
                    values.append(features(joined[left : left + size]))
                carry = joined[len(joined) // size * size :]
            if end < len(y):
                before = previous if end == 0 else float(y[end - 1])
                boundaries.append((offset + end, before, float(y[end])))
                carry = np.empty(0, dtype=np.int16)
            start = end
        previous = float(y[-1])
        offset += len(y)
    keep = sum(row[2] < len(boundaries) for row in rows)
    if not keep:
        raise ValueError("完全な破壊周期に学習窓がありません")
    return np.asarray(rows[:keep]), np.asarray(values[:keep]), np.asarray(boundaries)


def fold_ids(cycles: np.ndarray, n_folds: int = 5) -> np.ndarray:
    ordered = np.unique(cycles)
    if len(ordered) < n_folds:
        raise ValueError("破壊周期の数が fold 数を下回っています")
    return np.searchsorted([block[-1] for block in np.array_split(ordered, n_folds)], cycles)


def read_ids(data: Path) -> np.ndarray:
    ids = pd.read_csv(data / "sample_submission.csv", dtype={"seg_id": str})["seg_id"]
    if ids.isna().any() or ids.duplicated().any() or ids.empty:
        raise ValueError("提出 ID が不正です")
    result = ids.to_numpy(dtype=str)
    if {path.stem for path in (data / "test").glob("*.csv")} != set(result):
        raise ValueError("テストファイルとサンプルの ID が一致しません")
    return result


def official_hash(data: Path, ids: np.ndarray) -> str:
    return files_sha256(
        [
            data / "train.csv",
            data / "sample_submission.csv",
            *(data / "test" / f"{key}.csv" for key in ids),
        ]
    )


def prepare(data: Path, output: Path) -> None:
    ids = read_ids(data)
    digest = official_hash(data, ids)
    with pd.read_csv(data / "train.csv", dtype=DTYPES, chunksize=1_000_000) as reader:
        rows, train, boundaries = windows(
            (chunk["acoustic_data"].to_numpy(), chunk["time_to_failure"].to_numpy())
            for chunk in reader
        )
    test = []
    for key in ids:
        x = pd.read_csv(data / "test" / f"{key}.csv", dtype=DTYPES)["acoustic_data"].to_numpy()
        if x.shape != (WINDOW,):
            raise ValueError(f"窓長が一致しません: {key}")
        test.append(features(x))
    folds = fold_ids(rows[:, 2])
    temporary = output.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        rows=rows,
        train=train,
        boundaries=boundaries,
        folds=folds,
        test=np.asarray(test),
        ids=ids,
        official_hash=digest,
    )
    temporary.replace(output)
    print("破壊境界（行番号、直前、直後）", boundaries.tolist())
    print("fold 別窓数", np.bincount(folds).tolist())
    print("残り時間分位点", np.quantile(rows[:, 3], [0, 0.25, 0.5, 0.75, 1]).tolist())


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    prepare(root / "data", root / "outputs" / "windows.npz")
