"""OOF の誤差と連結成分単位のスコア差を集計する。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from inputs import load_data
from model import PatentRidge, pearson
from split import make_split
from train import ROOT, TRIALS

from kumite.checks import validate_regression
from kumite.runs import RunStore


def paired_interval(
    y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    rng = np.random.default_rng(42)
    group_count = int(groups.max()) + 1
    # 各成分の十分統計を先に計算し、再抽出のたびに全行を複製しない。
    moments = [
        np.vstack(
            [
                np.bincount(groups, weights=value, minlength=group_count)
                for value in (np.ones(len(y)), y, y**2, pred, pred**2, y * pred)
            ]
        )
        for pred in (baseline, candidate)
    ]
    differences = np.empty(2000)
    for index in range(len(differences)):
        weights = np.bincount(rng.integers(group_count, size=group_count), minlength=group_count)
        scores = []
        for matrix in moments:
            n, sy, syy, sp, spp, syp = matrix @ weights
            scores.append((syp - sy * sp / n) / np.sqrt((syy - sy**2 / n) * (spp - sp**2 / n)))
        differences[index] = scores[1] - scores[0]
    return np.quantile(differences, [0.025, 0.975])


def peer_diagnostics(
    train: pd.DataFrame, test: pd.DataFrame, folds: pd.Series, predictions: dict[str, np.ndarray]
) -> None:
    subsets = {"OOF": train, "公開テスト": test}
    subsets.update({f"fold {fold} 学習": train.loc[folds.ne(fold)] for fold in range(5)})
    for label, rows in subsets.items():
        sizes = rows.groupby(["anchor", "context"])["id"].size()
        print(
            f"{label}: {len(sizes)} 群、中央値 {sizes.median():g}、"
            f"範囲 {sizes.min()}–{sizes.max()}、単独 {sizes.eq(1).sum()} 群"
        )

    sizes = train.groupby(["anchor", "context"])["id"].transform("size").to_numpy()
    y = train["score"].to_numpy()
    for label, mask in (
        ("1", sizes == 1),
        ("2–5", (sizes >= 2) & (sizes <= 5)),
        ("6–20", (sizes >= 6) & (sizes <= 20)),
        ("21+", sizes >= 21),
    ):
        mae = [np.abs(y[mask] - predictions[run][mask]).mean() for run in ("E04", "E05")]
        print(f"候補数 {label}: {mask.sum()} 行 / MAE {mae[0]:.5f} → {mae[1]:.5f}")

    rng = np.random.default_rng(42)
    reduced_predictions, full_predictions, targets = [], [], []
    for fold in range(5):
        valid = train.loc[folds.eq(fold)]
        selected = np.sort(
            np.concatenate(
                [
                    rng.choice(indices, size=(len(indices) + 1) // 2, replace=False)
                    for indices in valid.groupby(["anchor", "context"]).indices.values()
                ]
            )
        )
        reduced = valid.iloc[selected]
        model = PatentRidge(TRIALS["E05"][0])
        model.fit(train.loc[folds.ne(fold)])
        reduced_predictions.append(model.predict(reduced))
        full_predictions.append(predictions["E05"][reduced.index.to_numpy()])
        targets.append(reduced["score"].to_numpy())
    reduced_pred, full_pred, actual = map(
        np.concatenate, (reduced_predictions, full_predictions, targets)
    )
    print(
        f"各群の約半数に間引き（{len(actual)} 行）: "
        f"Pearson {pearson(actual, full_pred):.5f} → {pearson(actual, reduced_pred):.5f} / "
        f"予測の平均絶対差 {np.abs(full_pred - reduced_pred).mean():.5f}"
    )


def main(root: Path = ROOT, *, suffix: str = "", diagnose_peers: bool = False) -> None:
    train, test = load_data(root / "data")
    split = make_split(train)
    _, groups = np.unique(split["component"].to_numpy(), return_inverse=True)
    y = train["score"].to_numpy()
    predictions = {}
    store = RunStore(root)
    for trial in TRIALS:
        record = store.record(f"{trial}{suffix}")
        pred = store.load_array(record.run_id, "oof.npy")
        validate_regression(pred, len(train), lower=0)
        if not np.isclose(pearson(y, pred), record.oof_score) or (pred > 1).any():
            raise ValueError(f"OOF と記録が不一致です: {record.run_id}")
        predictions[trial] = pred

    for base, candidate in (("E01", "E02"), ("E02", "E03"), ("E03", "E04"), ("E04", "E05")):
        lo, hi = paired_interval(y, predictions[base], predictions[candidate], groups)
        print(f"{base} → {candidate}: 95% CI [{lo:+.5f}, {hi:+.5f}]")

    pairs = [tuple(sorted((a, b))) for a, b in zip(train["anchor"], train["target"], strict=True)]
    multiple = (
        train.assign(pair=pd.Series(pairs, index=train.index))
        .groupby("pair")["context"]
        .transform("nunique")
        .gt(1)
    )
    # 空白と非英数字を取り除くと同じ綴りになるペアを表記揺れとして集計する。
    anchor_key = train["anchor"].str.replace(r"[\W_]+", "", regex=True)
    target_key = train["target"].str.replace(r"[\W_]+", "", regex=True)
    variants = train["anchor"].ne(train["target"]) & anchor_key.eq(target_key) & anchor_key.ne("")
    unknown = pd.Series(False, index=train.index)
    for fold in range(5):
        valid = split["fold"].eq(fold)
        unknown.loc[valid] = ~train.loc[valid, "context"].isin(train.loc[~valid, "context"])
        overlap = train.loc[valid, "target"].isin(train.loc[~valid, "target"]).mean()
        distribution = train.loc[valid, "score"].value_counts().sort_index().to_dict()
        print(f"fold {fold}: target 重複 {overlap:.2%} / score {distribution}")
    for label, mask in (
        ("表記揺れ", variants),
        ("複数 CPC のペア", multiple),
        ("未知 CPC", unknown),
    ):
        print(label, int(mask.sum()))
        if mask.any():
            for run, pred in predictions.items():
                print(f"  {run}: MAE {np.mean(np.abs(y[mask] - pred[mask])):.5f}")
    if diagnose_peers:
        peer_diagnostics(train, test, split["fold"], predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--diagnose-peers", action="store_true")
    args = parser.parse_args()
    main(suffix=args.run_suffix, diagnose_peers=args.diagnose_peers)
