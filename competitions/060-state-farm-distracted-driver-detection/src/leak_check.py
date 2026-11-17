"""検証の漏れの検査。

- 運転手: 運転手単位の分割の全 fold で、学習と検証の運転手が重ならない（ランダム分割では重なることも示す）
- ラベル: 分割 0 の検証画像のラベルを別のクラスに変えても、凍結特徴 + ロジスティック回帰（E01）の
  検証・test の予測が変わらない（`kumite.checks.check_label_independence`）
- fine-tune は検証画像のラベルを受け取らない（`models.finetune` の引数に無い。epoch ごとの記録は
  pipeline 側の monitor で計算する）。GPU の学習は再実行で揺らぐので、出力の一致では検査しない

実行（先に E01 を実行して凍結特徴を作っておく）:
uv run python competitions/060-state-farm-distracted-driver-detection/src/leak_check.py
"""

import numpy as np
from cv import SPLITS, check_driver_disjoint
from dataset import N_CLASSES
from models import backbone_features, linear_probe
from pipeline import load_data
from train import EXPERIMENTS

from kumite.checks import check_label_independence, shift_labels
from kumite.crossfit import folds_to_splits


def main() -> None:
    data = load_data()
    check_driver_disjoint(data.train, SPLITS["driver"][1](data.train))
    print("運転手: 運転手単位の分割の全 fold で、学習と検証の運転手が重ならない")
    try:
        check_driver_disjoint(data.train, SPLITS["random"][1](data.train))
    except ValueError:
        print("運転手: ランダム分割では重なる（検査が効いていることの確認）")
    else:
        raise AssertionError("ランダム分割で運転手が重ならないのは想定外です")

    exp = EXPERIMENTS["E01"]
    x = backbone_features(exp.config.backbone, data.train_images, "train")
    x_te = backbone_features(exp.config.backbone, data.test_images, "test")
    train_idx, valid_idx = folds_to_splits(SPLITS["driver"][1](data.train))[0]
    hidden = np.zeros(len(data.y), dtype=bool)
    hidden[valid_idx] = True

    def build(y: np.ndarray) -> list[np.ndarray]:
        valid, test, _ = linear_probe(
            x[train_idx], y[train_idx], x[valid_idx], x_te, exp.params["C"]
        )
        return [valid, test]

    check_label_independence(build, data.y, shift_labels(data.y, hidden, N_CLASSES))
    print(f"ラベル: 分割 0 の検証 {len(valid_idx)} 枚のラベルを変えても、E01 の予測は変わらない")


if __name__ == "__main__":
    main()
