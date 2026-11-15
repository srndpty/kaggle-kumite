"""実験の実行。CV で OOF を作り、テスト予測・提出ファイル・実験ログを出力する。

実行: uv run python competitions/001-otto-group-product-classification-challenge/src/train.py E01
"""

import argparse
import time
from typing import Any

import numpy as np
from cv import N_SPLITS, SEED, make_folds
from dataset import CLASSES, TARGET, encode_target, load_test, load_train
from features import FEATURE_SETS
from models import MODELS
from runs import Experiment, save_run

LGBM_BASE = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "seed": SEED,
    "num_boost_round": 10000,
    "early_stopping": 100,
}
LOGREG_BASE = {"C": 1.0, "max_iter": 2000}

EXPERIMENTS: dict[str, Experiment] = {
    "E00": Experiment(
        model="prior",
        hypothesis="学習 fold のクラス比を返すだけの予測。全実験の下限の目安",
        changed_one_thing="なし（統計的 baseline）",
    ),
    "E01": Experiment(
        model="lightgbm",
        hypothesis="生カウントの 93 特徴量そのままの GBDT。以降の比較の基準",
        changed_one_thing="E00 からモデルを LightGBM に変更（パラメータはほぼ既定値）",
        params=LGBM_BASE,
    ),
    "E02": Experiment(
        model="logistic_regression",
        hypothesis="H1 の比較基準。生カウントの線形モデル",
        changed_one_thing="E00 からモデルを標準化 + 多クラスロジスティック回帰に変更",
        params=LOGREG_BASE,
    ),
    "E03": Experiment(
        model="logistic_regression",
        hypothesis="H1: log1p で裾を潰すと線形モデルの logloss が改善する",
        changed_one_thing="E02 の特徴量を log1p 変換",
        features="log1p",
        params=LOGREG_BASE,
    ),
    "E04": Experiment(
        model="lightgbm",
        hypothesis="H1: 単調変換なので木モデルの結果はほぼ変わらない",
        changed_one_thing="E01 の特徴量を log1p 変換",
        features="log1p",
        params=LGBM_BASE,
    ),
    "E05": Experiment(
        model="lightgbm",
        hypothesis="H3: 行合計・非ゼロ数を加えると木モデルも改善する",
        changed_one_thing="E01 に row_sum, row_nnz の 2 特徴量を追加",
        features="row_stats",
        params=LGBM_BASE,
    ),
    # --- P6: 上位解法の着眼点の再現 ---
    "E07": Experiment(
        model="knn",
        hypothesis="H5 の 1 段目: 近傍法は木・線形と異なる誤り方をする（2 位解法は kNN を重視）",
        changed_one_thing="E03 からモデルを kNN（k=32、距離重み）に変更",
        features="log1p",
        params={"n_neighbors": 32, "uniform_mix": 0.01},
    ),
    "E08": Experiment(
        model="extra_trees",
        hypothesis="H5 の 1 段目: ランダムな分割の木は GBDT と異なる誤り方をする（1 位解法の 1 段目に含まれる）",
        changed_one_thing="E03 からモデルを ExtraTrees（300 本）に変更",
        features="log1p",
        params={"n_estimators": 300, "seed": SEED, "uniform_mix": 0.01},
    ),
    "E09": Experiment(
        model="lightgbm",
        hypothesis="H6: 各クラスの最近傍までの距離を加えると木モデルが改善する（1 位解法の特徴量）",
        changed_one_thing="E01 に kNN 距離特徴 27 列（9 クラス × 近傍 1・2・4 件の距離和）を追加",
        features="knn_dist",
        params=LGBM_BASE,
    ),
    "E10": Experiment(
        model="lightgbm",
        hypothesis="H5: 異なるモデルの OOF 予測を 2 段目で統合すると単一モデルより改善する",
        changed_one_thing="E01 の入力を、E01・E03・E07・E08 の OOF 予測確率 36 列に変更（スタッキング）",
        features="stack_v1",
        params=LGBM_BASE,
    ),
}


def run(run_id: str) -> None:
    exp = EXPERIMENTS[run_id]
    fit_predict = MODELS[exp.model]
    make_features = FEATURE_SETS[exp.features]
    train, test = load_train(), load_test()
    x, x_te = make_features(train, test)
    y = encode_target(train[TARGET])
    folds = make_folds(y)

    started = time.perf_counter()
    oof = np.zeros((len(x), len(CLASSES)))
    test_pred = np.zeros((len(x_te), len(CLASSES)))
    fold_infos: list[dict[str, Any]] = []
    for fold in range(N_SPLITS):
        tr, va = folds != fold, folds == fold
        result = fit_predict(x[tr], y[tr], x[va], y[va], x_te, exp.params)
        oof[va] = result.valid_pred
        test_pred += result.test_pred / N_SPLITS
        fold_infos.append(result.info)
    runtime = time.perf_counter() - started
    save_run(run_id, exp, y, folds, oof, test_pred, test["id"], runtime, fold_infos)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", choices=list(EXPERIMENTS))
    run(parser.parse_args().run_id)


if __name__ == "__main__":
    main()
