"""月次の保有状況から新規取得商品の順位を予測する。"""

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

ID = "ncodpers"
DATE = "fecha_dato"
K = 7


def previous_month(month: int) -> int:
    return month - 1 if month % 100 > 1 else month - 89


def transitions(
    panel: pl.DataFrame, current: pl.DataFrame, month: int, products: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prior_columns = [f"{p}_previous" for p in products]
    prior = panel.filter(pl.col("month") == previous_month(month)).select(
        ID, *(pl.col(p).alias(alias) for p, alias in zip(products, prior_columns, strict=True))
    )
    joined = current.join(prior, on=ID, how="left", validate="1:1", maintain_order="left")
    owned = joined.select(prior_columns).fill_null(-1).to_numpy()
    labels = np.full(owned.shape, -1, dtype=np.int8)
    if all(p in current.columns for p in products):
        after = joined.select(products).fill_null(-1).to_numpy()
        known = (owned >= 0) & (after >= 0)
        labels[known] = ((owned == 0) & (after == 1))[known]
    return joined[ID].to_numpy(), owned, labels


def rates(owned: np.ndarray, labels: np.ndarray) -> np.ndarray:
    exposure = ((owned == 0) & (labels >= 0)).sum(axis=0)
    events = (labels == 1).sum(axis=0)
    return (events + 0.5) / (exposure + 1)


def rank(owned: np.ndarray, labels: np.ndarray, query: np.ndarray, variant: str) -> np.ndarray:
    global_rates = rates(owned, labels)
    scores = np.broadcast_to(global_rates, query.shape).copy()
    if variant == "portfolio":
        train_group = np.minimum((owned == 1).sum(axis=1), 2)
        query_group = np.minimum((query == 1).sum(axis=1), 2)
        for group in range(3):
            rows = train_group == group
            if rows.any():
                scores[query_group == group] = rates(owned[rows], labels[rows])
    if variant == "logistic":
        # 同一入力をまとめ、ロジスティック回帰の尤度を件数で重み付けする。
        features, codes = portfolio_patterns(owned)
        query_features, query_codes = portfolio_patterns(query)
        for p in range(owned.shape[1]):
            eligible = (owned[:, p] == 0) & (labels[:, p] >= 0)
            positive = labels[:, p] == 1
            total = np.bincount(codes[eligible], minlength=len(features))
            yes = np.bincount(codes[positive], minlength=len(features))
            no = total - yes
            if yes.any() and no.any():
                x = np.concatenate((features[no > 0], features[yes > 0]))
                y = np.repeat([0, 1], [(no > 0).sum(), (yes > 0).sum()])
                weights = np.concatenate((no[no > 0], yes[yes > 0]))
                model = LogisticRegression(C=1, max_iter=500)
                model.fit(x, y, sample_weight=weights)
                scores[:, p] = model.predict_proba(query_features)[query_codes, 1]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("取得確率が不正です")
    scores[query == 1] = -1
    order = np.argsort(-scores, axis=1, kind="stable")[:, :K]
    return np.where(np.take_along_axis(scores, order, axis=1) >= 0, order, -1).astype(np.int8)


def portfolio_patterns(owned: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # 保有不明と非保有を別の状態として符号化する。
    powers = 3 ** np.arange(owned.shape[1], dtype=np.int64)
    keys, inverse = np.unique((owned + 1) @ powers, return_inverse=True)
    states = (keys[:, None] // powers) % 3 - 1
    return states, inverse


def average_precision(labels: np.ndarray, ranked: np.ndarray) -> np.ndarray:
    valid = ranked >= 0
    hits = np.take_along_axis(labels, np.where(valid, ranked, 0).astype(np.intp), axis=1) == 1
    hits &= valid
    precision = hits.cumsum(axis=1) / np.arange(1, ranked.shape[1] + 1)
    denom = np.maximum(1, np.minimum(K, (labels == 1).sum(axis=1)))
    return (precision * hits).sum(axis=1) / denom


def map_at_7(labels: np.ndarray, ranked: np.ndarray) -> float:
    return float(average_precision(labels, ranked).mean())


def validate_ranking(ranked: np.ndarray, owned: np.ndarray) -> None:
    if ranked.shape != (len(owned), min(K, owned.shape[1])) or not np.isfinite(ranked).all():
        raise ValueError("順位の形状または値が不正です")
    if ((ranked < -1) | (ranked >= owned.shape[1]) | (ranked != np.floor(ranked))).any():
        raise ValueError("商品番号が不正です")
    indices = ranked.astype(np.intp)
    active = indices >= 0
    if (active & (np.take_along_axis(owned, np.maximum(indices, 0), axis=1) == 1)).any():
        raise ValueError("保有済みの商品が含まれます")
    ordered = np.sort(indices, axis=1)
    if ((ordered[:, 1:] >= 0) & (ordered[:, 1:] == ordered[:, :-1])).any():
        raise ValueError("重複した商品が含まれます")
