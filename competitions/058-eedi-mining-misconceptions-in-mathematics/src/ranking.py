"""Eedi の候補検索・再ランキングと提出形式。"""

import re
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from lightgbm import Booster, LGBMRanker
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

KEY = "QuestionId_Answer"
LABEL = "MisconceptionId"
E5_REVISION = "ffb93f3bd4047442299a41ebb6fa998a38507c52"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def read_queries(path: Path, *, labeled: bool) -> pd.DataFrame:
    rows = []
    for question in pd.read_csv(path, keep_default_na=False).to_dict("records"):
        correct = question["CorrectAnswer"]
        answers = {letter: str(question[f"Answer{letter}Text"]) for letter in "ABCD"}
        signature = repr(
            (normalize(question["QuestionText"]), sorted(map(normalize, answers.values())))
        )
        for letter in answers:
            if letter == correct:
                continue
            label = question[f"Misconception{letter}Id"] if labeled else ""
            if labeled and label == "":
                continue
            rows.append(
                {
                    KEY: f"{question['QuestionId']}_{letter}",
                    LABEL: int(float(label)) if labeled else -1,
                    "group": signature,
                    "subject": question["SubjectName"],
                    "construct": question["ConstructName"],
                    "text": (
                        f"{question['SubjectName']}\n{question['QuestionText']}\n"
                        f"Correct answer: {answers[correct]}\nIncorrect answer: {answers[letter]}"
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    if frame[KEY].duplicated().any():
        raise ValueError("問題と選択肢の ID が重複しています")
    return frame


def load_mapping(path: Path) -> tuple[np.ndarray, list[str]]:
    frame = pd.read_csv(path).sort_values(LABEL)
    if frame[LABEL].duplicated().any() or frame["MisconceptionName"].isna().any():
        raise ValueError("誤概念マッピングに重複または欠損があります")
    return frame[LABEL].to_numpy(dtype=int), frame["MisconceptionName"].tolist()


def embed(model_dir: Path, texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(texts), 32):
            batch = tokenizer(
                texts[start : start + 32],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            result.append(torch.nn.functional.normalize(pooled, dim=1).numpy())
    return np.concatenate(result)


def token_overlap(query: str, name: str) -> float:
    left, right = (set(re.findall(r"\w+", text.casefold())) for text in (query, name))
    return len(left & right) / max(1, len(left | right))


def retrieve(
    queries: pd.DataFrame,
    names: list[str],
    trial: str,
    model_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    texts: list[str] = queries["text"].tolist()
    if trial == "E04":
        texts = [
            f"{construct}\n{text}"
            for construct, text in zip(queries["construct"], texts, strict=True)
        ]
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), dtype=cast("type[np.float64]", np.float32)
    )
    corpus = cast(csr_matrix, vectorizer.fit_transform(names))
    sparse = (cast(csr_matrix, vectorizer.transform(texts)) @ corpus.T).toarray()
    if trial == "E00":
        ranking = np.argsort(-sparse, axis=1, kind="stable")[:, :100]
        return ranking, np.take_along_axis(sparse, ranking, 1)[..., None]

    vectors = embed(
        model_dir, [f"passage: {name}" for name in names] + [f"query: {query}" for query in texts]
    )
    dense = vectors[len(names) :] @ vectors[: len(names)].T
    dense_order = np.argsort(-dense, axis=1, kind="stable")[:, :100]
    if trial == "E01":
        return dense_order, np.take_along_axis(dense, dense_order, 1)[..., None]

    sparse_order = np.argsort(-sparse, axis=1, kind="stable")[:, :100]
    ranks = []
    for order in (sparse_order, dense_order):
        rank = np.zeros_like(sparse, dtype=np.int16)
        np.put_along_axis(rank, order, np.arange(1, order.shape[1] + 1)[None, :], axis=1)
        ranks.append(rank)
    fused = sum(np.where(rank > 0, 1.0 / (60 + rank), 0.0) for rank in ranks)
    ranking = np.argsort(-fused, axis=1, kind="stable")[:, :100]
    features = [np.take_along_axis(x, ranking, 1) for x in (sparse, dense, *ranks, fused)]
    if trial == "E03":
        features.append(
            np.asarray(
                [
                    [token_overlap(query, names[i]) for i in candidates]
                    for query, candidates in zip(texts, ranking, strict=True)
                ]
            )
        )
    return ranking, np.stack(features, axis=-1)


def train_ranker(candidates: np.ndarray, features: np.ndarray, y: np.ndarray) -> Booster:
    relevant = candidates == y[:, None]
    keep = relevant.any(axis=1)
    if not keep.any():
        raise ValueError("学習候補に正解がありません")
    width = candidates.shape[1]
    ranker = LGBMRanker(
        objective="lambdarank",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        random_state=42,
        n_jobs=4,
        verbosity=-1,
    )
    ranker.fit(
        features[keep].reshape(-1, features.shape[-1]),
        relevant[keep].ravel().astype(int),
        group=np.full(int(keep.sum()), width),
    )
    return ranker.booster_


def rerank(ranker: Booster, candidates: np.ndarray, features: np.ndarray) -> np.ndarray:
    scores = np.asarray(ranker.predict(features.reshape(-1, features.shape[-1])))
    if not np.isfinite(scores).all():
        raise ValueError("再ランキングのスコアに非有限値があります")
    return np.take_along_axis(
        candidates, np.argsort(-scores.reshape(candidates.shape), axis=1, kind="stable"), axis=1
    )


def validate_ranking(pred: np.ndarray, n_rows: int, ids: np.ndarray, width: int) -> None:
    if (
        pred.shape != (n_rows, width)
        or not np.issubdtype(pred.dtype, np.integer)
        or not np.isin(pred, ids).all()
        or np.any(np.diff(np.sort(pred, axis=1), axis=1) == 0)
    ):
        raise ValueError("予測の形状、誤概念 ID、または重複が不正です")


def submission(
    sample: pd.DataFrame, queries: pd.DataFrame, pred: np.ndarray, ids: np.ndarray
) -> pd.DataFrame:
    validate_ranking(pred, len(queries), ids, 25)
    by_key = pd.Series([" ".join(map(str, row)) for row in pred], index=queries[KEY])
    if (
        not sample[KEY].is_unique
        or not by_key.index.is_unique
        or set(sample[KEY]) != set(by_key.index)
    ):
        raise ValueError("サンプル提出と推論対象の ID が一致しません")
    return sample[[KEY]].assign(**{LABEL: sample[KEY].map(by_key)})
