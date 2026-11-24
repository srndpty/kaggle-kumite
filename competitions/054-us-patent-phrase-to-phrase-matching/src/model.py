"""特許語句ペアの特徴量と Ridge。Notebook とローカルで共用する。"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge


def pearson(actual: np.ndarray, pred: np.ndarray) -> float:
    if np.ptp(actual) == 0 or np.ptp(pred) == 0:
        raise ValueError("定数の Pearson 相関は定義できません")
    return float(np.corrcoef(actual, pred)[0, 1])


class PatentRidge:
    def __init__(self, features: tuple[str, ...]) -> None:
        self.features = features
        self.characters = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
        self.categories = DictVectorizer()
        self.model = Ridge(alpha=1, solver="lsqr")

    def matrix(self, rows: pd.DataFrame, *, fit: bool = False) -> csr_matrix:
        values = []
        categorical: list[dict[str, float]] = []
        for anchor, target, context in rows[["anchor", "target", "context"]].itertuples(
            index=False, name=None
        ):
            left, right = set(anchor.split()), set(target.split())
            jaccard = len(left & right) / len(left | right)
            values.append(
                [
                    len(anchor),
                    len(target),
                    abs(len(anchor) - len(target)),
                    float(anchor == target),
                    jaccard,
                ]
            )
            section = context[0]
            category: dict[str, float] = {}
            if "cpc" in self.features:
                category.update({f"section={section}": 1.0, f"code={context}": 1.0})
            if "interaction" in self.features:
                category.update(
                    {f"jaccard={section}": jaccard, f"exact={section}": float(anchor == target)}
                )
            categorical.append(category)
        parts = [csr_matrix(values)]
        cosine = None
        if "char" in self.features:
            if fit:
                self.characters.fit(pd.concat([rows["anchor"], rows["target"]], ignore_index=True))
            left_chars = self.characters.transform(rows["anchor"])
            right_chars = self.characters.transform(rows["target"])
            cosine = np.asarray(left_chars.multiply(right_chars).sum(axis=1)).ravel()
            parts.append(csr_matrix(cosine[:, None]))
        if "cpc" in self.features:
            if "interaction" in self.features and cosine is not None:
                for category, context, similarity in zip(
                    categorical, rows["context"], cosine, strict=True
                ):
                    category[f"cosine={context[0]}"] = float(similarity)
            categories = (
                self.categories.fit_transform(categorical)
                if fit
                else self.categories.transform(categorical)
            )
            parts.append(csr_matrix(categories))
        if "peer_similarity" in self.features:
            assert cosine is not None
            peers = rows.assign(cosine=cosine).groupby(["anchor", "context"])["cosine"]
            count = peers.transform("size").to_numpy() - 1
            other_sum = peers.transform("sum").to_numpy() - cosine
            other_mean = np.divide(other_sum, count, out=np.zeros_like(cosine), where=count > 0)
            parts.append(csr_matrix(other_mean[:, None]))
        return csr_matrix(hstack(parts, format="csr"))

    def fit(self, rows: pd.DataFrame) -> None:
        self.model.fit(self.matrix(rows, fit=True), rows["score"])

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.clip(np.asarray(self.model.predict(self.matrix(rows))), 0, 1)
