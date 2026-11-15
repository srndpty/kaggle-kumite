"""特徴量セット。各セットは (train, test) を受け取り (x, x_te) を返す。

- 行ごとに完結する変換（raw / log1p / row_stats）は、CV の外で train/test に適用してよい
- 他の行やラベルを使う特徴（knn_dist / stack_*）は、train 側を必ず out-of-fold で作る
"""

from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd
from cv import N_SPLITS, SPLIT_VERSION, make_folds
from dataset import CLASSES, FEATURES, OUTPUT_DIR, TARGET, encode_target
from runs import load_predictions
from sklearn.neighbors import NearestNeighbors

FeatureSet = Callable[[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]

KNN_NEIGHBORS = (1, 2, 4)  # 各クラスで「近傍 n 件までの距離の和」を作る n
KNN_CACHE = OUTPUT_DIR / "features" / f"knn_dist_{SPLIT_VERSION}.npz"


def raw(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES]


def log1p(df: pd.DataFrame) -> pd.DataFrame:
    # float32 だと predict_proba も float32 になり、確率の和が 1 からずれて logloss が警告を出す。
    return df[FEATURES].astype("float64").transform(np.log1p)


def row_stats(df: pd.DataFrame) -> pd.DataFrame:
    """生カウントに行合計と非ゼロ数を加える。"""
    x = df[FEATURES]
    return x.assign(row_sum=x.sum(axis=1), row_nnz=(x > 0).sum(axis=1))


def _row_wise(fn: Callable[[pd.DataFrame], pd.DataFrame]) -> FeatureSet:
    return lambda train, test: (fn(train), fn(test))


def _class_distances(ref: np.ndarray, ref_y: np.ndarray, query: np.ndarray) -> np.ndarray:
    """クラスごとに、query から ref 内のそのクラスの近傍 n 件までの距離の和を返す（9 × 3 列）。"""
    columns = []
    for k in range(len(CLASSES)):
        nn = NearestNeighbors(n_neighbors=max(KNN_NEIGHBORS), n_jobs=-1).fit(ref[ref_y == k])
        dist, _ = nn.kneighbors(query)
        columns += [dist[:, :n].sum(axis=1) for n in KNN_NEIGHBORS]
    return np.column_stack(columns)


def _compute_knn_dist(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """距離は log1p 空間のユークリッド距離（生カウントのままだと裾の大きい値が距離を支配するため）。

    train: 自分の fold 以外を参照集合にする（自分自身や同じ fold のラベルを使わない）。
    test: 各 fold の参照集合（train の 80%）で計算して平均する。参照集合の大きさを train 側とそろえ、
    test だけ近傍が近くなる（距離が小さく出る）ずれを防ぐ。
    """
    space_tr, space_te = log1p(train).to_numpy(), log1p(test).to_numpy()
    y = encode_target(train[TARGET])
    folds = make_folds(y)
    y_arr = y.to_numpy()
    feats_tr = np.zeros((len(train), len(CLASSES) * len(KNN_NEIGHBORS)))
    feats_te = np.zeros((len(test), feats_tr.shape[1]))
    for fold in range(N_SPLITS):
        ref, query = folds != fold, folds == fold
        feats_tr[query] = _class_distances(space_tr[ref], y_arr[ref], space_tr[query])
        feats_te += _class_distances(space_tr[ref], y_arr[ref], space_te) / N_SPLITS
        print(f"knn_dist: fold {fold} 完了", flush=True)
    return feats_tr, feats_te


def knn_dist(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生カウントに、各クラスの最近傍までの距離特徴（27 列）を加える。計算結果はキャッシュする。"""
    if KNN_CACHE.exists():
        cached = np.load(KNN_CACHE)
        feats_tr, feats_te = cached["train"], cached["test"]
    else:
        feats_tr, feats_te = _compute_knn_dist(train, test)
        KNN_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(KNN_CACHE, train=feats_tr, test=feats_te)
    names = [f"knn_{c}_sum{n}" for c in CLASSES for n in KNN_NEIGHBORS]
    x, x_te = raw(train), raw(test)
    return (
        pd.concat([x, pd.DataFrame(feats_tr, columns=names, index=x.index)], axis=1),
        pd.concat([x_te, pd.DataFrame(feats_te, columns=names, index=x_te.index)], axis=1),
    )


def stack(
    base_run_ids: tuple[str, ...], train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1 段目の run の予測確率を特徴にする。train 側は OOF（同じ分割）、test 側は fold モデル平均。"""
    oofs, tests, names = [], [], []
    for run_id in base_run_ids:
        oof, test_pred = load_predictions(run_id)
        oofs.append(oof)
        tests.append(test_pred)
        names += [f"{run_id}_{c}" for c in CLASSES]
    return (
        pd.DataFrame(np.hstack(oofs), columns=names, index=train.index),
        pd.DataFrame(np.hstack(tests), columns=names, index=test.index),
    )


FEATURE_SETS: dict[str, FeatureSet] = {
    "raw": _row_wise(raw),
    "log1p": _row_wise(log1p),
    "row_stats": _row_wise(row_stats),
    "knn_dist": knn_dist,
    "stack_v1": partial(stack, ("E01", "E03", "E07", "E08")),
}
