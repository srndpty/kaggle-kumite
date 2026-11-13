"""100コンペのカタログ (CSV) の読み込み。"""

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class Competition:
    number: int
    category: str
    title: str
    slug: str
    url: str
    learning_goal: str
    difficulty: int
    compute: str

    @property
    def dir_name(self) -> str:
        """`competitions/` 配下でのディレクトリ名。"""
        return f"{self.number:03d}-{self.slug}"


def load_catalog(csv_path: Path) -> list[Competition]:
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        return [
            Competition(
                number=int(row["number"]),
                category=row["category"],
                title=row["title"],
                slug=row["slug"],
                url=row["url"],
                learning_goal=row["learning_goal"],
                difficulty=int(row["difficulty"]),
                compute=row["compute"],
            )
            for row in csv.DictReader(f)
        ]


def find_competition(catalog: list[Competition], number: int) -> Competition:
    for competition in catalog:
        if competition.number == number:
            return competition
    raise LookupError(f"カタログに #{number} はありません")
