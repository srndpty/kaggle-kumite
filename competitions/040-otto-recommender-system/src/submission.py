"""提出テンプレートとの照合と上位商品リストの検査。"""

import csv
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

COLUMNS = ["session_type", "labels"]


def template_keys(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        if next(reader, None) != COLUMNS:
            raise ValueError(f"公式テンプレートの列が一致しません: {path}")
        for key, _ in reader:
            yield key


def write_submission(
    path: Path, template: Path, predictions: Iterable[tuple[str, Sequence[int]]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(COLUMNS)
        for expected, (key, aids) in zip(template_keys(template), predictions, strict=True):
            if expected != key:
                raise ValueError(f"予測と公式テンプレートのキーが一致しません: {expected}, {key}")
            writer.writerow((key, " ".join(map(str, aids))))


def validate_submission(path: Path, template: Path) -> None:
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        if next(reader, None) != COLUMNS:
            raise ValueError("提出ファイルの列が一致しません")
        for expected, (key, labels) in zip(template_keys(template), reader, strict=True):
            if expected != key:
                raise ValueError(f"提出ファイルのキー順が一致しません: {expected}, {key}")
            aids = [int(aid) for aid in labels.split()]
            if len(aids) > 20 or len(aids) != len(set(aids)) or any(aid < 0 for aid in aids):
                raise ValueError(f"無効な提出商品リスト: {key}")
