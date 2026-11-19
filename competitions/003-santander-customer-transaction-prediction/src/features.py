"""外側の学習 fold だけで特徴量の変換を fit する。"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler


def log_counts(train: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fit_counts = np.empty(train.shape, dtype=np.float32)
    query_counts = np.zeros(query.shape, dtype=np.float32)
    for col in range(train.shape[1]):
        values, inverse, counts = np.unique(train[:, col], return_inverse=True, return_counts=True)
        fit_counts[:, col] = np.log1p(counts[inverse] - 1)
        locations = np.searchsorted(values, query[:, col])
        safe = np.minimum(locations, len(values) - 1)
        found = values[safe] == query[:, col]
        query_counts[found, col] = np.log1p(counts[safe[found]])
    return fit_counts, query_counts


def transform(
    variant: str, train: np.ndarray, query: np.ndarray
) -> tuple[np.ndarray | csr_matrix, np.ndarray | csr_matrix]:
    if variant == "E03":
        bins = KBinsDiscretizer(
            n_bins=16,
            encode="onehot",
            strategy="quantile",
            quantile_method="averaged_inverted_cdf",
            subsample=200_000,
            random_state=42,
            dtype=np.float32,
        )
        return csr_matrix(bins.fit_transform(train)), csr_matrix(bins.transform(query))
    if variant in ("E02", "E06"):
        fit_counts, query_counts = log_counts(train, query)
        if variant == "E06":
            fit_counts = train * (fit_counts > 0)
            query_counts = query * (query_counts > 0)
        train = np.column_stack((train, fit_counts))
        query = np.column_stack((query, query_counts))
    scaler = StandardScaler()
    return np.asarray(scaler.fit_transform(train)), np.asarray(scaler.transform(query))
