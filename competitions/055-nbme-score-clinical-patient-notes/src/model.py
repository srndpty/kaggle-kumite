"""採点項目を条件とするノート側トークンの二値分類。"""

import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from data import SEED, Pair
from spans import Span, characters, normalize
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedTokenizerFast

MODEL = "distilbert/distilbert-base-uncased"
MAX_LENGTH = 512
STRIDE = 128
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5


@dataclass
class Window:
    pair_index: int
    ids: list[int]
    spans: list[Span | None]
    labels: list[float]


def tokenizer() -> PreTrainedTokenizerFast:
    result = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if not isinstance(result, PreTrainedTokenizerFast):
        msg = "文字オフセットを返す高速トークナイザーが必要です"
        raise TypeError(msg)
    return result


def encode(pairs: list[Pair], tok: PreTrainedTokenizerFast, trial: str) -> list[Window]:
    windows: list[Window] = []
    for index, pair in enumerate(pairs):
        feature = pair.feature_id if trial == "E01" else pair.feature
        encoded = tok(
            feature,
            pair.text,
            truncation="only_second",
            max_length=MAX_LENGTH,
            stride=STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        truth = characters(pair.spans, len(pair.text))
        for window_index, ids in enumerate(encoded["input_ids"]):
            spans = [
                (start, end) if side == 1 and end > start else None
                for (start, end), side in zip(
                    encoded["offset_mapping"][window_index],
                    encoded.sequence_ids(window_index),
                    strict=True,
                )
            ]
            labels: list[float] = []
            for span in spans:
                if span is None:
                    labels.append(-1)
                    continue
                start, end = span
                overlap = sum(offset in truth for offset in range(start, end))
                labels.append(float(2 * overlap >= end - start if trial == "E02" else overlap > 0))
            windows.append(Window(index, ids, spans, labels))
    return windows


class Tagger(nn.Module):
    def __init__(self, config_dir: str | None = None) -> None:
        super().__init__()
        self.encoder = (
            AutoModel.from_pretrained(MODEL)
            if config_dir is None
            else AutoModel.from_config(
                AutoConfig.from_pretrained(config_dir, local_files_only=True)
            )
        )
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, ids: Tensor, mask: Tensor) -> Tensor:
        encoded = self.encoder(input_ids=ids, attention_mask=mask)
        return self.head(encoded.last_hidden_state).squeeze(-1)


def batches(windows: list[Window], *, shuffle: bool) -> Iterator[list[Window]]:
    order = torch.randperm(len(windows)).tolist() if shuffle else range(len(windows))
    for start in range(0, len(windows), BATCH_SIZE):
        yield [windows[index] for index in order[start : start + BATCH_SIZE]]


def batch_tensors(
    windows: list[Window], pad: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    length = max(len(window.ids) for window in windows)
    ids = [window.ids + [pad] * (length - len(window.ids)) for window in windows]
    masks = [[1] * len(window.ids) + [0] * (length - len(window.ids)) for window in windows]
    labels = [window.labels + [-1] * (length - len(window.ids)) for window in windows]
    return (
        torch.tensor(ids, device=device),
        torch.tensor(masks, device=device),
        torch.tensor(labels, dtype=torch.float32, device=device),
    )


def fit(windows: list[Window], pad: int, device: torch.device) -> Tagger:
    torch.manual_seed(SEED)
    model = Tagger().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS):
        for batch in batches(windows, shuffle=True):
            ids, mask, labels = batch_tensors(batch, pad, device)
            supervised = labels >= 0
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids, mask)
            loss_function(logits[supervised], labels[supervised]).backward()
            optimizer.step()
    return model


@torch.inference_mode()
def predict(
    model: Tagger,
    windows: list[Window],
    count: int,
    pad: int,
    device: torch.device,
) -> list[dict[Span, float]]:
    model.eval()
    scores: list[dict[Span, list[float]]] = [defaultdict(list) for _ in range(count)]
    for batch in batches(windows, shuffle=False):
        ids, mask, _ = batch_tensors(batch, pad, device)
        probabilities = model(ids, mask).sigmoid().cpu().tolist()
        for window, values in zip(batch, probabilities, strict=True):
            for span, value in zip(window.spans, values[: len(window.spans)], strict=True):
                if span is not None:
                    if not 0 <= value <= 1:
                        msg = "トークン確率が範囲外です"
                        raise ValueError(msg)
                    scores[window.pair_index][span].append(value)
    return [{span: sum(values) / len(values) for span, values in row.items()} for row in scores]


def decode(text: str, scores: dict[Span, float], *, join_whitespace: bool) -> list[Span]:
    positive = normalize((span for span, score in scores.items() if score >= 0.5), len(text))
    if not join_whitespace:
        return positive
    merged: list[Span] = []
    for start, end in positive:
        if merged and text[merged[-1][1] : start].isspace():
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def decode_pair(pair: Pair, scores: dict[Span, float], trial: str) -> list[Span]:
    positive = decode(pair.text, scores, join_whitespace=trial == "E03")
    if trial != "E04" or pair.feature_id != "309":
        return positive
    mismatches = [
        match.span()
        for match in re.finditer(r"\b(?:2\s+weeks?|years?)\b", pair.text, re.IGNORECASE)
    ]
    return [
        (start, end)
        for start, end in positive
        if all(end <= left or start >= right for left, right in mismatches)
    ]
