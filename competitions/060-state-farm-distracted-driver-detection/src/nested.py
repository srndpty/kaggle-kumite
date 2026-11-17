"""P7 入れ子の疑似テスト: 提出と同じ形（fold モデルの平均）の予測を、学習データの中で作る（H4・H5、K001・K002）。

外側の各 fold について、外側の学習行を運転手単位で内側 4 分割し、E02 と同じ設定で 4 モデルを学習する。
- 外側の検証行（疑似テスト）: 4 モデルそれぞれの予測（`pseudo_each.npy`）と、その平均（run の OOF）
- 外側の学習行: 内側の OOF（単一モデルの形。`inner_oof.npy`、内側の fold 番号は `inner_fold.npy`）
外側の検証行のラベルは、どのモデルの学習にも、温度の決定にも使わない。

実行:
  uv run python competitions/060-state-farm-distracted-driver-detection/src/nested.py run
  uv run python competitions/060-state-farm-distracted-driver-detection/src/nested.py report
"""

import argparse
import time

import numpy as np
from cv import driver_folds
from dataset import DRIVER, N_CLASSES
from models import ModelConfig, finetune, logloss, row_logloss
from neighbors import features, knn, smooth
from pipeline import STORE_RUNS, Experiment, load_data, save_run

from kumite.bootstrap import group_bootstrap_diff
from kumite.calibration import apply_temperature, fit_temperature
from kumite.crossfit import CrossFitResult, FoldOutput, cross_fit_splits

RUN_ID = "E06N"
INNER_FOLDS = 4
KNN = {"feature": "resnet", "k": 10}  # E05F と同じ

EXPERIMENT = Experiment(
    model="nested_finetune",
    hypothesis="H5・K001: 提出と同じ形（fold モデルの平均）の予測を学習データの中で作り、"
    "平均の効果・温度・近傍の平均を提出と同じ形で確かめる",
    changed_one_thing="E02 の各外側 fold の予測を、外側の学習行を運転手単位で内側 4 分割して学習した 4 モデルの平均に変更",
    config=ModelConfig(),
    params={"inner_folds": INNER_FOLDS},
)


def run() -> None:
    with STORE_RUNS.reserve(RUN_ID) as reservation:
        data = load_data()
        splits = data.splits(EXPERIMENT.split)
        n, y = len(data.y), data.y
        inner_oof = np.full((len(splits), n, N_CLASSES), np.nan, dtype=np.float32)
        inner_fold = np.full((len(splits), n), -1, dtype=np.int64)
        pseudo_each = np.full((INNER_FOLDS, n, N_CLASSES), np.nan, dtype=np.float32)
        started = time.perf_counter()

        def fit(train_idx: np.ndarray, valid_idx: np.ndarray) -> FoldOutput:
            k = int((inner_fold >= 0).any(axis=1).sum())  # 何番目の外側 fold か
            folds = driver_folds(data.train.iloc[train_idx], INNER_FOLDS)
            inner_fold[k, train_idx] = folds
            tests: list[np.ndarray] = []
            infos: list[float] = []
            for j in range(INNER_FOLDS):
                tr, va = train_idx[folds != j], train_idx[folds == j]
                print(f"  外側 {k} 内側 {j}: 学習 {len(tr)} 枚", flush=True)
                proba, test, _ = finetune(
                    EXPERIMENT.config,
                    data.train_images,
                    tr,
                    y[tr],
                    np.concatenate([va, valid_idx]),
                    data.test_images,
                )
                inner_oof[k, va] = proba[: len(va)]
                pseudo_each[j, valid_idx] = proba[len(va) :]
                tests.append(test)
                infos.append(round(logloss(y[va], proba[: len(va)]), 5))
            valid = pseudo_each[:, valid_idx].astype(np.float64).mean(axis=0)
            return FoldOutput(valid, np.mean(tests, axis=0), {"inner_logloss": infos})

        result: CrossFitResult = cross_fit_splits(n, splits, fit)
        extra = {
            "inner_oof.npy": inner_oof,
            "inner_fold.npy": inner_fold,
            "pseudo_each.npy": pseudo_each,
        }
        save_run(
            reservation,
            EXPERIMENT,
            data,
            splits,
            result,
            time.perf_counter() - started,
            extra,
        )


def report() -> None:
    """外側 fold ごとに、温度を外側の学習行（内側 OOF）だけで決め、疑似テストに適用する。"""
    data = load_data()
    y, drivers = data.y, data.train[DRIVER].to_numpy()
    inner_oof = STORE_RUNS.load_array(RUN_ID, "inner_oof.npy").astype(np.float64)
    inner_fold = STORE_RUNS.load_array(RUN_ID, "inner_fold.npy")
    each = STORE_RUNS.load_array(RUN_ID, "pseudo_each.npy").astype(np.float64)
    single80 = STORE_RUNS.load_array("E02", "oof.npy")
    x = features(str(KNN["feature"]), "train", data.train_images, EXPERIMENT.config.backbone)
    k_nn = int(KNN["k"])

    rows: dict[str, np.ndarray] = {}  # 名前 → 行ごとの logloss（全行）
    temps: list[dict[str, float]] = []

    def put(name: str, idx: np.ndarray, proba: np.ndarray) -> None:
        rows.setdefault(name, np.full(len(y), np.nan))[idx] = row_logloss(y[idx], proba)

    for k, (_, valid_idx) in enumerate(data.splits(EXPERIMENT.split)):
        train_idx = np.flatnonzero(inner_fold[k] >= 0)
        oof_k = inner_oof[k, train_idx]
        # 内側 OOF に近傍の平均をかける（近傍は内側の同じ検証 fold の中だけ。提出の形と同じ）
        oof_k_knn = np.empty_like(oof_k)
        for j in range(INNER_FOLDS):
            m = inner_fold[k, train_idx] == j
            oof_k_knn[m] = smooth(oof_k[m], knn(x[train_idx[m]], k_nn))
        t_single = fit_temperature(y[train_idx], oof_k, logloss)
        t_knn = fit_temperature(y[train_idx], oof_k_knn, logloss)

        singles = each[:, valid_idx]
        avg = singles.mean(axis=0)
        nn = knn(x[valid_idx], k_nn)
        avg_knn = smooth(avg, nn)
        temps.append(
            {
                "T（単一の内側 OOF）": t_single,
                "T（単一 + 近傍の内側 OOF）": t_knn,
                "平均に最適な T（参考）": fit_temperature(y[valid_idx], avg, logloss),
                "平均 + 近傍に最適な T（参考）": fit_temperature(y[valid_idx], avg_knn, logloss),
            }
        )
        for j in range(INNER_FOLDS):
            put(f"単一 {j}", valid_idx, singles[j])
            put(f"単一 {j} + T", valid_idx, apply_temperature(singles[j], t_single))
            put(f"単一 {j} + 近傍", valid_idx, smooth(singles[j], nn))
        put("平均", valid_idx, avg)
        put("平均 + T", valid_idx, apply_temperature(avg, t_single))
        put("平均 + 近傍", valid_idx, avg_knn)
        put("平均 + 近傍 + T", valid_idx, apply_temperature(avg_knn, t_knn))
        put("E02（80% で学習した単一）", valid_idx, single80[valid_idx])

    print("温度（外側 fold ごと）:")
    for k, t in enumerate(temps):
        print(f"  fold {k}: " + "、".join(f"{name} {v:.3f}" for name, v in t.items()))

    def mean_single(suffix: str) -> np.ndarray:
        return np.mean([rows[f"単一 {j}{suffix}"] for j in range(INNER_FOLDS)], axis=0)

    summary = {
        "単一（4 モデルの損失の平均）": mean_single(""),
        "単一 + T": mean_single(" + T"),
        "単一 + 近傍": mean_single(" + 近傍"),
        **{name: rows[name] for name in ("平均", "平均 + T", "平均 + 近傍", "平均 + 近傍 + T")},
        "E02（80% で学習した単一）": rows["E02（80% で学習した単一）"],
    }
    print("\n疑似テスト（全 22,424 行）の logloss:")
    for name, loss in summary.items():
        print(f"  {name}: {loss.mean():.5f}")

    print("\n差（運転手単位 bootstrap、右 - 左。負が改善）:")
    for base, other in [
        ("単一（4 モデルの損失の平均）", "平均"),
        ("単一（4 モデルの損失の平均）", "単一 + T"),
        ("平均", "平均 + T"),
        ("単一（4 モデルの損失の平均）", "単一 + 近傍"),
        ("平均", "平均 + 近傍"),
        ("平均 + 近傍", "平均 + 近傍 + T"),
        ("E02（80% で学習した単一）", "平均"),
    ]:
        d = group_bootstrap_diff(
            summary[base], summary[other], drivers, transform=lambda m: m, n_bootstrap=2000
        )
        print(
            f"  {base} → {other}: {d.point:+.5f}（95% 区間 [{d.low:+.5f}, {d.high:+.5f}]、"
            f"改善の割合 {d.share_negative:.3f}）"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "report"])
    if parser.parse_args().command == "run":
        run()
    else:
        report()


if __name__ == "__main__":
    main()
