"""商品ラベルと同一画像でつながる投稿を同じ fold に割り当てる。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

SEED = 42
SPLIT_VERSION = "split_v1"


def components(rows: pd.DataFrame) -> np.ndarray:
    """商品ラベルを頂点とし、同一画像を共有するラベルを連結する。"""
    if rows[["label_group", "image"]].isna().any().any():
        raise ValueError("商品ラベルと画像 ID は必須です")
    labels, unique_labels = pd.factorize(rows["label_group"])
    parents = list(range(len(unique_labels)))

    def root(label: int) -> int:
        while parents[label] != label:
            parents[label] = parents[parents[label]]
            label = parents[label]
        return label

    image_labels: dict[str, int] = {}
    for image, code in zip(rows["image"], labels, strict=True):
        label = int(code)
        first = image_labels.setdefault(str(image), label)
        parents[root(label)] = root(first)
    return np.array([root(int(label)) for label in labels], dtype=np.int64)


def assign_folds(groups: np.ndarray, n_splits: int) -> np.ndarray:
    folds = np.full(len(groups), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for fold, (_, validation) in enumerate(splitter.split(groups, groups=groups)):
        folds[validation] = fold
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    train = pd.read_csv(args.train, usecols=["posting_id", "label_group", "image"])
    if train["posting_id"].isna().any() or not train["posting_id"].is_unique:
        raise ValueError("posting_id は欠損なしで一意である必要があります")
    groups = components(train)
    result = train[["posting_id"]].assign(component=groups, fold=assign_folds(groups, 5))
    args.output.parent.mkdir(exist_ok=True, parents=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
