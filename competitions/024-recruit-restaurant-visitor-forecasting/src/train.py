"""実験の実行。CV で OOF を作り、テスト予測・提出ファイル・実験ログを出力する。

実行: uv run python competitions/024-recruit-restaurant-visitor-forecasting/src/train.py E01
"""

import argparse
import time
from dataclasses import replace

from cv import SEED
from features import Holidays, Reservations, StoreStats
from pipeline import STORE_RUNS, Experiment, load_data, run_cv, save_run

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
        hypothesis="店舗 × 曜日ごとの学習期間の来店数の中央値。全実験の下限の目安",
        changed_one_thing="なし（統計的 baseline）",
    ),
    "E01": Experiment(
        model="lightgbm",
        hypothesis="店舗の属性 + 日付の分解の GBDT（来店数をそのまま RMSE で学習）。以降の比較の基準",
        changed_one_thing="E00 からモデルを LightGBM に変更（パラメータは既定値）",
        params=LGBM_BASE,
    ),
    "E02": Experiment(
        model="lightgbm",
        hypothesis="H1: 目的変数を log1p にして RMSE で学習すると、指標（RMSLE）と学習の目的が一致して改善する",
        changed_one_thing="E01 の目的変数を log1p(visitors) に変更（予測は expm1 で戻す）",
        target="log1p",
        params=LGBM_BASE,
    ),
}
EXPERIMENTS["E03"] = replace(
    EXPERIMENTS["E02"],
    hypothesis="H4: 祝日と、前日・翌日が休日かを加えると改善する（予定は予測期間の末尾までしか見せない）",
    changed_one_thing="E02 に祝日・前日休日・翌日休日の 3 列を追加",
    features=(Holidays(),),
)
EXPERIMENTS["E04"] = replace(
    EXPERIMENTS["E02"],
    hypothesis="H5: 店舗・店舗 × 曜日の過去の log 来店数の平均を加えると、店舗 ID のカテゴリだけより改善する",
    changed_one_thing="E02 に店舗別の統計量 4 列を追加（どの行もその日より前の学習行だけから作る）",
    features=(StoreStats(),),
)
EXPERIMENTS["E05"] = replace(
    EXPERIMENTS["E02"],
    hypothesis="H3: 予測の起点より前に入った予約の人数を加えると改善する（改善は窓の 1 週目に集中する）",
    changed_one_thing="E02 に予約人数・件数・起点からの日数の 3 列を追加（学習行にも起点からの日数をランダムに割り当てて打ち切る）",
    features=(Reservations(),),
)
EXPERIMENTS["E05L"] = replace(
    EXPERIMENTS["E05"],
    hypothesis="H3b: 検証行で窓の開始以後に入った予約まで見せると、CV は大きく楽観的になる（test では再現しない）",
    changed_one_thing="E05 の検証行の予約を打ち切らない（テスト行の入力は E05 と同じ）",
    features=(Reservations(valid_rows="all"),),
)
EXPERIMENTS["E05A"] = replace(
    EXPERIMENTS["E05"],
    hypothesis="H3c: 学習行の予約を打ち切らない（来店日までの全予約）と、検証・テスト行との打ち切りの違いを学べず悪化する",
    changed_one_thing="E05 の学習行の予約を、来店日までに入ったすべての予約に変更（起点からの日数は 0）",
    features=(Reservations(train_rows="all"),),
)
EXPERIMENTS["E06"] = replace(
    EXPERIMENTS["E02"],
    hypothesis="H3+4+5: 祝日・店舗別統計量・予約は別の情報源なので、組み合わせると改善がほぼ足し合わさる",
    changed_one_thing="E02 に E03・E04・E05 の特徴をすべて追加（組み合わせの確認。単独の効果は E03〜E05）",
    features=(Holidays(), StoreStats(), Reservations()),
)
EXPERIMENTS["E07"] = replace(
    EXPERIMENTS["E06"],
    hypothesis="H6: 提出は分割ごとのモデルの平均より、全行で学習し直した 1 モデルの方が良い（CV は E06 と同じ）",
    changed_one_thing="E06 のテスト予測を、全学習行で学習し直した 1 モデル（round 数は best iteration の平均）に変更",
    refit=True,
)
# H2: 同じ実験を、同じ検証行をランダムに分ける分割で測る（run_id に R を付ける）。
for _base in ("E02", "E03", "E04"):
    EXPERIMENTS[f"{_base}R"] = replace(
        EXPERIMENTS[_base],
        hypothesis=f"H2: ランダム分割は時系列分割より楽観的なスコアを出す（{_base} と同じモデル）",
        changed_one_thing=f"{_base} の検証分割を、同じ検証行のランダム分割に変更",
        split="random",
    )


def run(run_id: str) -> None:
    exp = EXPERIMENTS[run_id]
    with STORE_RUNS.reserve(run_id) as reservation:  # 学習前に run_id を予約する
        data = load_data()
        splits = data.splits(exp.split)
        started = time.perf_counter()
        result = run_cv(exp, data, splits)
        save_run(reservation, exp, data, splits, result, time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_ids", nargs="+", choices=list(EXPERIMENTS))
    for run_id in parser.parse_args().run_ids:
        run(run_id)


if __name__ == "__main__":
    main()
