"""語彙とIDFは外側foldの学習文だけから作る。"""

from typing import Any, cast

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


def build_features(train, valid, test, *, bigram=False, chars=False):
    vectorizers = [
        TfidfVectorizer(
            ngram_range=(1, 2 if bigram else 1),
            min_df=3,
            max_features=100_000,
            sublinear_tf=True,
            dtype=cast(Any, np.float32),
        )
    ]
    if chars:
        vectorizers.append(
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=3,
                max_features=150_000,
                sublinear_tf=True,
                dtype=cast(Any, np.float32),
            )
        )
    matrices = [[], [], []]
    for v in vectorizers:
        matrices[0].append(v.fit_transform(train))
        matrices[1].append(v.transform(valid))
        matrices[2].append(v.transform(test))
    return tuple(hstack(parts, format="csr") for parts in matrices), vectorizers
