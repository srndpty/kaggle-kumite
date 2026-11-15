"""P5: E01 で Late が CV より良い理由（fold モデル平均の効果）を train 内で再現して確かめる。

外側の各 fold を疑似テストとして取り置き、残りで E01 と同じ手順（内側 4 fold CV + 学習 fold の
early stopping）を行う。内側 OOF（単一モデル）・疑似テストの単一モデル予測・4 モデル平均を比べる。
あわせて、内側 OOF で決めた温度 T を平均予測にかけたときの効果（E06 の検証）も測る。

実行: uv run python competitions/001-otto-group-product-classification-challenge/src/averaging_check.py
"""

import numpy as np
import pandas as pd
from calibrate import apply_temperature, fit_temperature
from cv import N_SPLITS, make_folds, make_inner_folds
from dataset import CLASSES, TARGET, encode_target, load_train
from features import raw
from models import lightgbm
from pipeline import multi_logloss
from train import LGBM_BASE


def check_outer_fold(x: pd.DataFrame, y: pd.Series, hold: np.ndarray) -> dict[str, float]:
    x_in, y_in = x[~hold].reset_index(drop=True), y[~hold].reset_index(drop=True)
    x_hold, y_hold = x[hold], y[hold].to_numpy()
    inner_oof = np.zeros((len(x_in), len(CLASSES)))
    hold_preds = []
    inner = make_inner_folds(y_in)
    for k in range(inner.max() + 1):
        tr, va = np.flatnonzero(inner != k), np.flatnonzero(inner == k)
        res = lightgbm(
            x_in.iloc[tr], y_in.iloc[tr], x_in.iloc[va], y_in.iloc[va], x_hold, LGBM_BASE
        )
        inner_oof[va] = res.valid
        hold_preds.append(res.query)
    avg = np.mean(hold_preds, axis=0)
    t = fit_temperature(y_in.to_numpy(), inner_oof)
    return {
        "内側OOF": multi_logloss(y_in, inner_oof),
        "単一(疑似テスト)": float(np.mean([multi_logloss(y_hold, p) for p in hold_preds])),
        "平均(疑似テスト)": multi_logloss(y_hold, avg),
        "T": t,
        "単一+T": float(
            np.mean([multi_logloss(y_hold, apply_temperature(p, t)) for p in hold_preds])
        ),
        "平均+T": multi_logloss(y_hold, apply_temperature(avg, t)),
        # 疑似テストのラベルで決めた、平均予測にとって最適な T（診断用。実運用では使えない）
        "平均に最適なT": fit_temperature(y_hold, avg),
    }


def main() -> None:
    train = load_train()
    x = raw(train)
    y = encode_target(train[TARGET])
    folds = make_folds(y)
    rows = []
    for k in range(N_SPLITS):
        rows.append(check_outer_fold(x, y, folds == k))
        print(f"outer fold {k}: {rows[-1]}", flush=True)
    table = pd.DataFrame(rows)
    table.loc["平均"] = table.mean()
    pd.set_option("display.width", 160)
    print("\n" + table.round(5).to_string())
    m = table.loc["平均"]
    print(f"\n平均の効果（単一 - 平均）: {m['単一(疑似テスト)'] - m['平均(疑似テスト)']:+.5f}")
    print(f"分布の差（単一(疑似テスト) - 内側OOF）: {m['単一(疑似テスト)'] - m['内側OOF']:+.5f}")
    print(f"温度の効果 単一: {m['単一+T'] - m['単一(疑似テスト)']:+.5f}")
    print(f"温度の効果 平均: {m['平均+T'] - m['平均(疑似テスト)']:+.5f}")


if __name__ == "__main__":
    main()
