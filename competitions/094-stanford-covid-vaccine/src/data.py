"""OpenVaccine の読み込み・符号化・分割・採点。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupKFold

from kumite.crossfit import Split

COMPETITION = Path(__file__).resolve().parents[1]
DATA = COMPETITION / "data"
TRAIN_JSON = DATA / "train.json"
TEST_JSON = DATA / "test.json"
SAMPLE = DATA / "sample_submission.csv"
BPPS = DATA / "bpps"

TARGETS = ("reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C")
ERRORS = (
    "reactivity_error",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_error_Mg_50C",
    "deg_error_50C",
)
SCORED = ("reactivity", "deg_Mg_pH10", "deg_Mg_50C")
SCORED_INDEX = tuple(TARGETS.index(name) for name in SCORED)
TRAIN_SCORED = 68
VOCAB = {"sequence": "ACGU", "structure": "().", "predicted_loop_type": "BEHIMSX"}
N_TOKENS = sum(len(v) for v in VOCAB.values())

N_FOLDS = 5
KMER = 5
CLUSTER_JACCARD = 0.35


@dataclass(frozen=True)
class Molecules:
    """配列・構造・ループ種別。長さは分子ごとに異なってよい。"""

    ids: tuple[str, ...]
    sequence: tuple[str, ...]
    structure: tuple[str, ...]
    loop: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.ids)

    def length(self, i: int) -> int:
        return len(self.sequence[i])


@dataclass(frozen=True)
class Labels:
    """学習ラベル。`y` と `error` は (分子, 採点位置, 5目的)。"""

    y: np.ndarray
    error: np.ndarray
    sn_filter: np.ndarray
    signal_to_noise: np.ndarray

    def scored(self) -> np.ndarray:
        """採点 3 列。公式の採点対象と同じく `SN_filter == 0` の分子は NaN にする。"""
        y = self.y[:, :, list(SCORED_INDEX)].copy()
        y[~self.sn_filter] = np.nan
        return y


def _molecules(frame: pd.DataFrame) -> Molecules:
    for column, vocab in VOCAB.items():
        unknown = set("".join(frame[column])) - set(vocab)
        if unknown:
            raise ValueError(f"{column} に想定外の文字があります: {sorted(unknown)}")
    if not (frame["sequence"].str.len() == frame["structure"].str.len()).all():
        raise ValueError("配列と構造の長さが一致しません")
    return Molecules(
        tuple(frame["id"]),
        tuple(frame["sequence"]),
        tuple(frame["structure"]),
        tuple(frame["predicted_loop_type"]),
    )


def read_train(path: Path = TRAIN_JSON) -> tuple[Molecules, Labels]:
    frame = pd.read_json(path, lines=True)
    if not (frame["seq_scored"] == TRAIN_SCORED).all():
        raise ValueError("学習データの採点位置数が想定外です")
    y = np.stack([np.stack(frame[c].to_numpy()) for c in TARGETS], axis=-1)
    error = np.stack([np.stack(frame[c].to_numpy()) for c in ERRORS], axis=-1)
    labels = Labels(
        y.astype(np.float64),
        error.astype(np.float64),
        frame["SN_filter"].to_numpy() == 1,
        frame["signal_to_noise"].to_numpy(dtype=np.float64),
    )
    return _molecules(frame), labels


def read_test(path: Path = TEST_JSON) -> Molecules:
    return _molecules(pd.read_json(path, lines=True))


def tokens(molecules: Molecules, i: int) -> np.ndarray:
    """位置ごとの one-hot（塩基 4・構造 3・ループ種別 7）。形は (長さ, 14)。"""
    columns = (molecules.sequence[i], molecules.structure[i], molecules.loop[i])
    parts: list[np.ndarray] = []
    for text, vocab in zip(columns, VOCAB.values(), strict=True):
        codes = np.array([vocab.index(ch) for ch in text])
        parts.append(np.eye(len(vocab))[codes])
    return np.concatenate(parts, axis=1)


def partners(structure: str) -> np.ndarray:
    """ドット括弧表記の対合相手の位置。対合しない位置は -1。"""
    result = np.full(len(structure), -1, dtype=np.int64)
    stack: list[int] = []
    for i, ch in enumerate(structure):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if not stack:
                raise ValueError("括弧が対応していません")
            j = stack.pop()
            result[i], result[j] = j, i
    if stack:
        raise ValueError("括弧が対応していません")
    return result


def bpp(molecule_id: str, root: Path = BPPS) -> np.ndarray:
    return np.load(root / f"{molecule_id}.npy", allow_pickle=False)


def mcrmse(y: np.ndarray, predicted: np.ndarray) -> float:
    """列別 RMSE の平均。最後の軸が目的。`y` が NaN の要素（採点対象外の分子）は除く。"""
    if y.shape != predicted.shape:
        raise ValueError(f"形が一致しません: {y.shape} と {predicted.shape}")
    y2 = y.reshape(-1, y.shape[-1])
    p2 = predicted.reshape(-1, y.shape[-1])
    keep = np.isfinite(y2).all(axis=1)
    if not keep.any():
        raise ValueError("採点対象がありません")
    return float(np.sqrt(((y2[keep] - p2[keep]) ** 2).mean(axis=0)).mean())


def kmer_clusters(sequences: tuple[str, ...]) -> np.ndarray:
    """5-mer 集合の Jaccard 類似度が閾値以上の配列を連結成分にまとめる。"""
    vocab: dict[str, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    for i, seq in enumerate(sequences):
        for kmer in {seq[j : j + KMER] for j in range(len(seq) - KMER + 1)}:
            rows.append(i)
            cols.append(vocab.setdefault(kmer, len(vocab)))
    matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(sequences), len(vocab)))
    inter = (matrix @ matrix.T).toarray()
    sizes = np.asarray(matrix.sum(axis=1)).ravel()
    jaccard = inter / (sizes[:, None] + sizes[None, :] - inter)
    np.fill_diagonal(jaccard, 0.0)
    _, labels = connected_components(csr_matrix(jaccard >= CLUSTER_JACCARD), directed=False)
    return labels.astype(np.int64)


def cluster_splits(sequences: tuple[str, ...]) -> list[Split]:
    """類似配列クラスタ単位の GroupKFold（シャッフルなしで決定的）。"""
    groups = kmer_clusters(sequences)
    folds = GroupKFold(N_FOLDS).split(np.zeros(len(sequences)), groups=groups)
    return [(np.sort(train), np.sort(valid)) for train, valid in folds]


def sample_rows(
    molecules: Molecules, predictions: list[np.ndarray], sample: Path = SAMPLE
) -> np.ndarray:
    """分子ごとの (長さ, 5) 予測を、公式サンプルの `id_seqpos` の順に並べた (行, 5) にする。"""
    if len(predictions) != len(molecules):
        raise ValueError("分子数と予測数が一致しません")
    ids: list[str] = []
    for i, molecule_id in enumerate(molecules.ids):
        if predictions[i].shape != (molecules.length(i), len(TARGETS)):
            raise ValueError(f"{molecule_id}: 予測の形が {predictions[i].shape} です")
        ids.extend(f"{molecule_id}_{p}" for p in range(molecules.length(i)))
    expected = pd.read_csv(sample, usecols=["id_seqpos"])["id_seqpos"]
    if len(ids) != len(expected) or set(ids) != set(expected):
        raise ValueError("予測の行 ID が公式サンプルと一致しません")
    position = pd.Series(np.arange(len(ids)), index=ids)
    return np.concatenate(predictions)[position.loc[expected].to_numpy()]


def write_submission(path: Path, values: np.ndarray, sample: Path = SAMPLE) -> None:
    """公式サンプルの順に並んだ (行, 5) の予測を提出 CSV にする。"""
    frame = pd.read_csv(sample, usecols=["id_seqpos"])
    if values.shape != (len(frame), len(TARGETS)):
        raise ValueError(f"予測の形が {values.shape} です")
    frame[list(TARGETS)] = values
    frame.to_csv(path, index=False)


def validate_submission(path: Path, sample: Path = SAMPLE) -> None:
    """提出 CSV の列・行 ID と順序・有限性を公式サンプルと照合する。"""
    expected = pd.read_csv(sample)
    actual = pd.read_csv(path)
    if list(actual.columns) != list(expected.columns):
        raise ValueError(f"列が公式サンプルと一致しません: {list(actual.columns)}")
    if not actual["id_seqpos"].equals(expected["id_seqpos"]):
        raise ValueError("id_seqpos の集合または順序が公式サンプルと一致しません")
    if not np.isfinite(actual[list(TARGETS)].to_numpy(dtype=np.float64)).all():
        raise ValueError("予測に NaN または Inf があります")
