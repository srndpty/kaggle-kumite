"""同じ質問が学習と検証に分かれないようにグループ化する。"""

from urllib.parse import urlsplit

import numpy as np
import pandas as pd


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def question_groups(data: pd.DataFrame) -> np.ndarray:
    parents = list(range(len(data)))
    first: dict[tuple[str, ...], int] = {}

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    fields = data[["url", "question_title", "question_body"]].fillna("")
    for index, (url, title, body) in enumerate(fields.itertuples(index=False, name=None)):
        parsed = urlsplit(url.strip())
        path = parsed.path.rstrip("/")
        parts = path.split("/")
        if len(parts) > 2 and parts[1] == "questions" and parts[2].isdigit():
            path = f"/questions/{parts[2]}"
        keys = []
        if parsed.hostname and path:
            keys.append(("url", parsed.hostname, path))
        if normalize(title) or normalize(body):
            keys.append(("text", normalize(title), normalize(body)))
        for key in keys:
            parents[root(index)] = root(first.setdefault(key, index))
    return np.asarray([root(index) for index in range(len(data))])
