"""実験の実行。CV で OOF を作り、テスト予測・提出ファイル・実験ログを出力する。

実行: uv run python competitions/019-rossmann-store-sales/src/train.py E01
"""

import argparse
import time
from dataclasses import replace

from cv import SEED
from features import Gaps, RecentStats, StoreStats
from pipeline import STORE, Experiment, load_data, run_cv, save_run

LGBM_BASE = {
    "learning_rate": 0.1,
    "num_leaves": 31,
    "seed": SEED,
    "num_boost_round": 20000,
    "early_stopping": 100,
}

EXPERIMENTS: dict[str, Experiment] = {
    "E00": Experiment(
        model="group_median",
        hypothesis="店舗 × 曜日 × 販促ごとの学習期間の売上中央値。全実験の下限の目安",
        changed_one_thing="なし（統計的 baseline）",
    ),
    "E01": Experiment(
        model="lightgbm",
        hypothesis="生の列 + 日付の分解 + store.csv の GBDT（売上をそのまま RMSE で学習）。以降の比較の基準",
        changed_one_thing="E00 からモデルを LightGBM に変更（パラメータは既定値）",
        params=LGBM_BASE,
    ),
    "E02": Experiment(
        model="lightgbm",
        hypothesis="H1: 目的変数を log1p にして RMSE で学習すると、相対誤差の指標（RMSPE）に合い改善する",
        changed_one_thing="E01 の目的変数を log1p(Sales) に変更（予測は expm1 で戻す）",
        target="log1p",
        params=LGBM_BASE,
    ),
    "E03": Experiment(
        model="lightgbm",
        hypothesis="H4: 前後の休業日・祝日までの日数を加えると、休業前後の売上の山を捉えて改善する",
        changed_one_thing="E02 に休業日・祝日までの日数 4 列を追加",
        gaps=Gaps(),
        target="log1p",
        params=LGBM_BASE,
    ),
    "E04": Experiment(
        model="lightgbm",
        hypothesis="H3: 店舗（× 曜日 × 販促）ごとの過去売上の平均を加えると、店舗 ID をカテゴリで渡すだけより改善する",
        changed_one_thing="E02 に店舗別の log 売上平均 4 列を追加（学習行は自分の期間以外から作る）",
        supervised=StoreStats(),
        target="log1p",
        params=LGBM_BASE,
    ),
    "E05": Experiment(
        model="lightgbm",
        hypothesis="H3 の切り分け: 学習行の統計量に自分の売上を含めると、学習時だけ特徴が強くなり悪化する",
        changed_one_thing="E04 の学習行の統計量を、学習行全体（自分の売上を含む）から作る",
        supervised=StoreStats(out_of_block=False),
        target="log1p",
        params=LGBM_BASE,
    ),
}
EXPERIMENTS["E06"] = replace(
    EXPERIMENTS["E03"],
    hypothesis="H6: 提出は分割ごとのモデルの平均より、全期間で学習し直した 1 モデルの方が良い（CV は E03 と同じ）",
    changed_one_thing="E03 のテスト予測を、全学習行で学習し直した 1 モデル（round 数は best iteration の平均）に変更",
    refit=True,
)
# E03・E06 は CV 0.106 に対して Late が大幅に悪かった。検証窓の行が、窓より先の休業日・祝日の
# 予定まで見ていた（test の末尾では見えない）。検証でも同じ打ち切りを再現して測り直す。
EXPERIMENTS["E07"] = replace(
    EXPERIMENTS["E06"],
    hypothesis="H7: 検証窓でも窓の末尾より先の予定を見せないと、E03 の CV は悪化して Late に近づく",
    changed_one_thing="E06 の休業日・祝日までの日数を、検証窓の末尾までの予定だけから作る（テスト行の入力は E06 と同じ）",
    gaps=Gaps(censor=True),
)
EXPERIMENTS["E08"] = replace(
    EXPERIMENTS["E07"],
    hypothesis="H8: 分からない日数を上限値で埋めると、学習時にほぼ現れない欠損を避けられ改善する",
    changed_one_thing="E07 の分からない日数（NaN）を上限の 30 日で埋める",
    gaps=Gaps(censor=True, fill_unknown=True),
)
EXPERIMENTS["E09"] = replace(
    EXPERIMENTS["E08"],
    hypothesis="H8: 上限を 7 日に下げると、打ち切られた行と普通の行の値の差が小さくなり改善する（効果は前後 3 日以内）",
    changed_one_thing="E08 の日数の上限を 30 日から 7 日に変更",
    gaps=Gaps(cap=7, censor=True, fill_unknown=True),
)
# --- P6: 上位解法の着眼点の再現 ---
EXPERIMENTS["E10"] = replace(
    EXPERIMENTS["E09"],
    hypothesis="H9（1 位解法）: 予測期間ぶん離した直近の売上の平均を加えると、店舗の最近の水準と販促への反応を捉えて改善する",
    changed_one_thing="E09 に、2 か月以上前の直近 3 か月・12 か月の店舗別・店舗 × 販促別の log 売上平均 4 列を追加",
    supervised=RecentStats(),
)
# H2: 同じ実験を、同じ検証行をランダムに分ける分割で測る（run_id に R を付ける）。
for _base in ("E02", "E03", "E04", "E05"):
    EXPERIMENTS[f"{_base}R"] = replace(
        EXPERIMENTS[_base],
        hypothesis=f"H2: ランダム分割は時系列分割より楽観的なスコアを出す（{_base} と同じモデル）",
        changed_one_thing=f"{_base} の検証分割を、同じ検証行のランダム分割に変更",
        split="random",
    )


def run(run_id: str) -> None:
    exp = EXPERIMENTS[run_id]
    with STORE.reserve(run_id) as reservation:  # 学習前に run_id を予約し、重複・同時実行を拒否する
        data = load_data()
        splits = data.splits(exp.split)
        started = time.perf_counter()
        result = run_cv(exp, data, splits)
        save_run(reservation, exp, data, splits, result, time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", choices=list(EXPERIMENTS))
    run(parser.parse_args().run_id)


if __name__ == "__main__":
    main()
