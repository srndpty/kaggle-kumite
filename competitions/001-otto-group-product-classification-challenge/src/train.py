"""実験の実行。CV で OOF を作り、テスト予測・提出ファイル・実験ログを出力する。

実行: uv run python competitions/001-otto-group-product-classification-challenge/src/train.py E01
"""

import argparse
import time

from cv import SEED, make_folds
from dataset import TARGET, encode_target, load_test, load_train
from features import KnnDistance
from pipeline import STORE, Experiment, run_cv, save_run
from stacking import Stacking

LGBM_BASE = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "seed": SEED,
    "num_boost_round": 10000,
    "early_stopping": 100,
}
LOGREG_BASE = {"C": 1.0, "max_iter": 2000}
MLP_BASE = {
    "hidden_layer_sizes": (256, 128),
    "alpha": 1e-4,
    "batch_size": 256,
    "learning_rate_init": 1e-3,
    "max_iter": 200,
    "n_iter_no_change": 10,
    "seed": SEED,
}

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
        transform="log1p",
        params=LOGREG_BASE,
    ),
    "E04": Experiment(
        model="lightgbm",
        hypothesis="H1: 単調変換なので木モデルの結果はほぼ変わらない",
        changed_one_thing="E01 の特徴量を log1p 変換",
        transform="log1p",
        params=LGBM_BASE,
    ),
    "E05": Experiment(
        model="lightgbm",
        hypothesis="H3: 行合計・非ゼロ数を加えると木モデルも改善する",
        changed_one_thing="E01 に row_sum, row_nnz の 2 特徴量を追加",
        transform="row_stats",
        params=LGBM_BASE,
    ),
    # --- P6: 上位解法の着眼点の再現 ---
    "E07": Experiment(
        model="knn",
        hypothesis="H5 の 1 段目: 近傍法は木・線形と異なる誤り方をする（2 位解法は kNN を重視）",
        changed_one_thing="E03 からモデルを kNN（k=32、距離重み）に変更",
        transform="log1p",
        params={"n_neighbors": 32, "uniform_mix": 0.01},
    ),
    "E08": Experiment(
        model="extra_trees",
        hypothesis="H5 の 1 段目: ランダムな分割の木は GBDT と異なる誤り方をする（1 位解法の 1 段目に含まれる）",
        changed_one_thing="E03 からモデルを ExtraTrees（300 本）に変更",
        transform="log1p",
        params={"n_estimators": 300, "seed": SEED, "uniform_mix": 0.01},
    ),
    # E09・E10 は旧実装（全行で作った OOF を特徴にする）で、外側の検証ラベルが学習側に入っていた。
    # 記録は experiments.jsonl に残し、コードは nested 版（E11・E12）に置き換えた。
}
# 2 段目は 1 段目の定義を参照するので、辞書を作ってから追加する。
EXPERIMENTS["E11"] = Experiment(
    model="lightgbm",
    hypothesis="H6 の再測定: 外側の検証ラベルを使わずに作った kNN 距離特徴でも、木モデルが改善する",
    changed_one_thing="E01 に kNN 距離特徴 27 列を追加（外側 fold の学習行だけで、内側 4 fold の OOF として作る）",
    supervised=KnnDistance(),
    params=LGBM_BASE,
)
EXPERIMENTS["E13"] = Experiment(
    model="lightgbm",
    hypothesis="E11 が E09 より良かった理由の切り分け: 検証・テスト行の距離を参照集合 4 つの平均にしたことが効いている",
    changed_one_thing="E11 の検証・テスト行の距離特徴を、内側の参照集合 1 つ（fold 0 以外）だけで計算",
    supervised=KnnDistance(average_query=False),
    params=LGBM_BASE,
)
EXPERIMENTS["E14"] = Experiment(
    model="mlp",
    hypothesis="NN の基準: 標準化した生カウントの 2 層 MLP は、線形（E02）より大きく良く、LightGBM（E01）に近づく",
    changed_one_thing="E02 からモデルを MLP（256-128、ReLU、Adam、学習 fold 内 10% で early stopping）に変更",
    params=MLP_BASE,
)
EXPERIMENTS["E15"] = Experiment(
    model="mlp",
    hypothesis="H1 の NN 版: log1p で裾を潰すと MLP の logloss が改善する（線形より効果が大きい）",
    changed_one_thing="E14 の特徴量を log1p 変換",
    transform="log1p",
    params=MLP_BASE,
)
EXPERIMENTS["E12"] = Experiment(
    model="lightgbm",
    hypothesis="H5 の再測定: 1 段目を外側 fold ごとに学習し直す nested stacking でも、単一モデルより改善する",
    changed_one_thing="E01 の入力を、E01・E03・E07・E08 を外側学習行の内側 4 fold で学習した予測確率 36 列に変更",
    transform="none",
    supervised=Stacking(tuple((rid, EXPERIMENTS[rid]) for rid in ("E01", "E03", "E07", "E08"))),
    params=LGBM_BASE,
)
EXPERIMENTS["E16"] = Experiment(
    model="lightgbm",
    hypothesis="NN と GBDT の相補性: MLP を 1 段目に加えると nested stacking がさらに改善する",
    changed_one_thing="E12 の 1 段目に E15（log1p の MLP）を追加（36 列 → 45 列）",
    transform="none",
    supervised=Stacking(
        tuple((rid, EXPERIMENTS[rid]) for rid in ("E01", "E03", "E07", "E08", "E15"))
    ),
    params=LGBM_BASE,
)


def run(run_id: str) -> None:
    exp = EXPERIMENTS[run_id]
    with STORE.reserve(run_id):  # 学習前に run_id を予約し、重複・同時実行を拒否する
        train, test = load_train(), load_test()
        y = encode_target(train[TARGET])
        folds = make_folds(y)
        started = time.perf_counter()
        result = run_cv(exp, train, y, folds, test)
        save_run(run_id, exp, y, folds, result, test["id"], time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", choices=list(EXPERIMENTS))
    run(parser.parse_args().run_id)


if __name__ == "__main__":
    main()
