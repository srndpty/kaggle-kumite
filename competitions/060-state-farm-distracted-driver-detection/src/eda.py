"""P1 EDA。数値は README に転記する。

実行（先に cache.py で小画像を作る）:
uv run python competitions/060-state-farm-distracted-driver-detection/src/eda.py
"""

import numpy as np
import pandas as pd
from cache import THUMBS_NPZ
from cv import SPLITS
from dataset import CLASSES, DRIVER, load_test, load_train


def _normalized(thumbs: np.ndarray) -> np.ndarray:
    """小画像を平坦にし、画像ごとに平均 0・ノルム 1 にする（明るさの違いを除く）。"""
    x = thumbs.reshape(len(thumbs), -1).astype(np.float32)
    x -= x.mean(axis=1, keepdims=True)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)  # 一様な画像は 0 ベクトル


def nearest(
    query: np.ndarray, ref: np.ndarray, same: bool, batch: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """query の各行に最も近い ref の行（相関が最大）。same=True なら自分自身を除く。"""
    idx = np.empty(len(query), dtype=np.int64)
    sim = np.empty(len(query), dtype=np.float32)
    for s in range(0, len(query), batch):
        block = query[s : s + batch] @ ref.T
        if same:
            block[np.arange(len(block)), np.arange(s, s + len(block))] = -np.inf
        idx[s : s + batch] = block.argmax(axis=1)
        sim[s : s + batch] = block.max(axis=1)
    return idx, sim


def main() -> None:
    train, test = load_train(), load_test()
    print(f"学習 {len(train)} 枚、運転手 {train[DRIVER].nunique()} 人、test {len(test)} 枚")

    # --- 運転手 × クラス ---
    table = pd.crosstab(train[DRIVER], train["classname"])
    per_driver = table.sum(axis=1)
    print(
        f"運転手ごとの枚数: 最少 {per_driver.min()}、中央値 {per_driver.median():.0f}、最多 {per_driver.max()}"
    )
    print(f"10 クラスすべてを撮った運転手: {(table > 0).all(axis=1).sum()} / {len(table)}")
    print(
        f"運転手 × クラスの枚数: 最少 {table.values.min()}、中央値 {np.median(table.values):.0f}、最多 {table.values.max()}"
    )
    share = train["classname"].value_counts(normalize=True).reindex(list(CLASSES))
    print("クラスの割合:", " ".join(f"{c} {v:.3f}" for c, v in share.items()))

    # --- 似た画像（小画像の相関で最近傍） ---
    thumbs = np.load(THUMBS_NPZ)
    tr, te = _normalized(thumbs["train"]), _normalized(thumbs["test"])
    flat = {
        k: int((thumbs[k].reshape(len(thumbs[k]), -1).std(axis=1) == 0).sum())
        for k in ("train", "test")
    }
    print(f"一様な（真っ黒などの）画像: 学習 {flat['train']} 枚、test {flat['test']} 枚")
    idx, sim = nearest(tr, tr, same=True)
    same_driver = train[DRIVER].to_numpy()[idx] == train[DRIVER].to_numpy()
    same_class = train["label"].to_numpy()[idx] == train["label"].to_numpy()
    print(
        f"\n学習画像の最近傍（自分以外の学習画像）: 同じ運転手 {same_driver.mean():.3f}、"
        f"同じクラス {same_class.mean():.3f}、相関の中央値 {np.median(sim):.3f}"
    )
    for name in ("driver", "random"):
        folds = SPLITS[name][1](train)
        hits_driver, hits_class, sims = [], [], []
        for k in range(folds.max() + 1):
            va, trn = np.flatnonzero(folds == k), np.flatnonzero(folds != k)
            j, s = nearest(tr[va], tr[trn], same=False)
            hits_driver.append(train[DRIVER].to_numpy()[trn[j]] == train[DRIVER].to_numpy()[va])
            hits_class.append(train["label"].to_numpy()[trn[j]] == train["label"].to_numpy()[va])
            sims.append(s)
        print(
            f"[{name}] 検証画像の学習側の最近傍: 同じ運転手 {np.concatenate(hits_driver).mean():.3f}、"
            f"同じクラス {np.concatenate(hits_class).mean():.3f}（= 1-NN の正解率）、"
            f"相関の中央値 {np.median(np.concatenate(sims)):.3f}"
        )
    _, sim_te = nearest(te, tr, same=False)
    print(f"test 画像の学習側の最近傍: 相関の中央値 {np.median(sim_te):.3f}")
    # query が ref の先頭の行なので、same=True で自分自身を除ける
    _, sim_tt = nearest(te[:5000], te, same=True)
    print(
        f"test 画像（先頭 5,000 枚）の、自分以外の test 画像の最近傍: 相関の中央値 {np.median(sim_tt):.3f}"
    )


if __name__ == "__main__":
    main()
