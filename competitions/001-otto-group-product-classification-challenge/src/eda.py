"""P1 EDA。数値は標準出力、図は outputs/eda/ に出す。

実行: uv run python competitions/001-otto-group-product-classification-challenge/src/eda.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataset import CLASSES, FEATURES, OUTPUT_DIR, TARGET, encode_target, load_test, load_train
from scipy.stats import spearmanr

FIG_DIR = OUTPUT_DIR / "eda"


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def overview(train: pd.DataFrame, test: pd.DataFrame) -> None:
    section("概要")
    print(f"train {train.shape}, test {test.shape}")
    print(f"欠損: train {train.isna().sum().sum()}, test {test.isna().sum().sum()}")
    print(f"特徴量の dtype: {sorted({str(t) for t in train[FEATURES].dtypes})}")
    print(f"最小値: train {train[FEATURES].min().min()}, test {test[FEATURES].min().min()}")
    print(f"最大値: train {train[FEATURES].max().max()}, test {test[FEATURES].max().max()}")
    dup_all = train.duplicated(subset=FEATURES).sum()
    dup_conflict = train.groupby(FEATURES, sort=False)[TARGET].nunique().gt(1).sum()
    print(
        f"特徴量が完全一致する train 行の重複: {dup_all}（うちラベルが食い違う組: {dup_conflict}）"
    )
    for name, df in (("train", train), ("test", test)):
        print(f"全特徴量が 0 の {name} 行: {(df[FEATURES].sum(axis=1) == 0).sum()}")


def target_distribution(train: pd.DataFrame) -> None:
    section("目的変数")
    counts = train[TARGET].value_counts().reindex(CLASSES)
    ratio = counts / counts.sum()
    print(pd.DataFrame({"count": counts, "ratio": ratio.round(4)}).to_string())
    print(f"最多/最少クラス比: {counts.max() / counts.min():.1f}")
    # 事前分布（クラス比）だけで予測したときの multi-class logloss。baseline の下限の目安。
    prior_logloss = -(ratio * np.log(ratio)).sum()
    print(f"一様予測の logloss: {np.log(len(CLASSES)):.4f}")
    print(f"クラス比予測の logloss: {prior_logloss:.4f}")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(CLASSES, counts.to_numpy())
    ax.set_title("target distribution")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "target_distribution.png", dpi=120)
    plt.close(fig)


def id_order(train: pd.DataFrame, test: pd.DataFrame) -> None:
    section("id と目的変数の関係（リーク確認）")
    y = encode_target(train[TARGET])
    rho = spearmanr(train["id"], y).statistic
    print(f"Spearman(id, class): {rho:.4f}")
    is_sorted = bool((y.diff().dropna() >= 0).all())
    print(f"train は class 順に並んでいるか: {is_sorted}")
    print(
        f"train id 範囲: {train['id'].min()}-{train['id'].max()}, test id 範囲: {test['id'].min()}-{test['id'].max()}"
    )


def sparsity(train: pd.DataFrame, test: pd.DataFrame) -> None:
    section("疎性・値の分布")
    x = train[FEATURES]
    nonzero = (x > 0).mean()
    print(
        f"全体の非ゼロ率: train {(x > 0).to_numpy().mean():.4f}, test {(test[FEATURES] > 0).to_numpy().mean():.4f}"
    )
    print(
        f"特徴量ごとの非ゼロ率: min {nonzero.min():.4f} / median {nonzero.median():.4f} / max {nonzero.max():.4f}"
    )
    print(f"非ゼロ率 < 1% の特徴量数: {(nonzero < 0.01).sum()}")
    nnz_per_row = (x > 0).sum(axis=1)
    print(f"行ごとの非ゼロ特徴量数: {nnz_per_row.describe().round(2).to_dict()}")
    values = x.to_numpy().ravel()
    nz = values[values > 0]
    print(f"非ゼロ値の分位点 (50/90/99/99.9%): {np.percentile(nz, [50, 90, 99, 99.9]).tolist()}")
    print(
        f"歪度 上位5特徴量:\n{x.skew().sort_values(ascending=False).head(5).round(1).to_string()}"
    )
    print(
        f"distinct 値が多い上位5特徴量:\n{x.nunique().sort_values(ascending=False).head(5).to_string()}"
    )

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.bar(range(1, len(FEATURES) + 1), nonzero.to_numpy())
    ax.set_xlabel("feature index")
    ax.set_ylabel("nonzero rate")
    ax.set_title("nonzero rate per feature (train)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "nonzero_rate.png", dpi=120)
    plt.close(fig)


def class_profiles(train: pd.DataFrame) -> None:
    section("クラスごとの特徴")
    x = train[FEATURES]
    row_sum = x.sum(axis=1)
    nnz = (x > 0).sum(axis=1)
    profile = pd.DataFrame(
        {
            "row_sum_mean": row_sum.groupby(train[TARGET]).mean(),
            "row_sum_median": row_sum.groupby(train[TARGET]).median(),
            "nnz_mean": nnz.groupby(train[TARGET]).mean(),
        }
    ).round(2)
    print(profile.to_string())

    # クラスごとの非ゼロ率が最も偏る特徴量（その特徴量が出ることがクラスの手がかりになる）
    nz_by_class = (x > 0).groupby(train[TARGET]).mean()
    peak_class = nz_by_class.idxmax()
    concentration = nz_by_class.max() / nz_by_class.mean()
    top = concentration.sort_values(ascending=False).head(8)
    print("\nクラス特異性が高い特徴量（最大クラス非ゼロ率 / 平均）:")
    for feat in top.index:
        print(f"  {feat}: {top[feat]:.1f}倍 → {peak_class[feat]}")

    # クラス平均プロファイル（log1p）の cosine 類似度。高いペアほど分離が難しい見込み。
    means = np.log1p(x).groupby(train[TARGET]).mean().to_numpy()
    unit = means / np.linalg.norm(means, axis=1, keepdims=True)
    sim = unit @ unit.T
    iu = np.triu_indices_from(sim, k=1)
    ranked = sorted(zip(sim[iu], iu[0], iu[1], strict=True), reverse=True)
    print("\nクラス平均プロファイルの cosine 類似度 上位5ペア:")
    for s, i, j in ranked[:5]:
        print(f"  {CLASSES[i]} - {CLASSES[j]}: {s:.3f}")

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(nz_by_class.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("feature index - 1")
    ax.set_title("nonzero rate by class")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "nonzero_rate_by_class.png", dpi=120)
    plt.close(fig)


def correlations(train: pd.DataFrame) -> None:
    section("特徴量間の相関")
    corr = np.log1p(train[FEATURES]).corr().to_numpy()
    iu = np.triu_indices_from(corr, k=1)
    pairs = sorted(zip(corr[iu], iu[0], iu[1], strict=True), key=lambda t: -abs(t[0]))
    print(f"|r| > 0.5 のペア数（log1p 後）: {sum(abs(r) > 0.5 for r, _, _ in pairs)}")
    for r, i, j in pairs[:8]:
        print(f"  {FEATURES[i]} - {FEATURES[j]}: {r:.3f}")


def train_test_shift(train: pd.DataFrame, test: pd.DataFrame) -> None:
    section("train / test の分布差")
    nz_tr = (train[FEATURES] > 0).mean()
    nz_te = (test[FEATURES] > 0).mean()
    mean_tr = np.log1p(train[FEATURES]).mean()
    mean_te = np.log1p(test[FEATURES]).mean()
    diff = pd.DataFrame(
        {"nz_diff": (nz_te - nz_tr).abs(), "log_mean_diff": (mean_te - mean_tr).abs()}
    )
    print(
        f"非ゼロ率の差: max {diff['nz_diff'].max():.4f}（{diff['nz_diff'].idxmax()}）, mean {diff['nz_diff'].mean():.4f}"
    )
    print(
        f"log1p 平均の差: max {diff['log_mean_diff'].max():.4f}（{diff['log_mean_diff'].idxmax()}）"
    )
    rs_tr = train[FEATURES].sum(axis=1).describe().round(2)
    rs_te = test[FEATURES].sum(axis=1).describe().round(2)
    print(pd.DataFrame({"train_row_sum": rs_tr, "test_row_sum": rs_te}).to_string())


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 160)
    train, test = load_train(), load_test()
    overview(train, test)
    target_distribution(train)
    id_order(train, test)
    sparsity(train, test)
    class_profiles(train)
    correlations(train)
    train_test_shift(train, test)
    print(f"\n図: {FIG_DIR}")


if __name__ == "__main__":
    main()
