"""`python -m kumite` のエントリポイント。通常は `dev` コマンド経由で呼ぶ。"""

import argparse
import subprocess
from collections.abc import Sequence

from kumite.catalog import find_competition, load_catalog
from kumite.kaggle_data import download_data
from kumite.paths import CATALOG_CSV, COMPETITIONS_DIR, TEMPLATE_DIR
from kumite.scaffold import create_competition_dir


def _list() -> int:
    for c in load_catalog(CATALOG_CSV):
        mark = "*" if (COMPETITIONS_DIR / c.dir_name).is_dir() else " "
        print(
            f"{mark} {c.number:03d} [{c.category}] 難度{c.difficulty} 負荷{c.compute:<2} {c.title}"
        )
    print("\n* = ディレクトリ作成済み")
    return 0


def _new(number: int) -> int:
    competition = find_competition(load_catalog(CATALOG_CSV), number)
    target = create_competition_dir(competition, COMPETITIONS_DIR, TEMPLATE_DIR)
    print(f"作成しました: {target}")
    return 0


def _download(number: int) -> int:
    competition = find_competition(load_catalog(CATALOG_CSV), number)
    try:
        data_dir = download_data(competition, COMPETITIONS_DIR)
    except subprocess.CalledProcessError as e:
        print(
            f"ダウンロードに失敗しました。Rules の承諾を確認してください: {competition.url}/rules"
        )
        return e.returncode
    print(f"取得しました: {data_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kumite", description="Kaggle 100本ノックの補助コマンド")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="カタログ一覧と着手状況")
    for name, help_text in (("new", "コンペ用ディレクトリを生成"), ("download", "データを取得")):
        sub.add_parser(name, help=help_text).add_argument("number", type=int, help="カタログ番号")

    args = parser.parse_args(argv)
    command: str = args.command
    if command == "list":
        return _list()
    number: int = args.number
    return _new(number) if command == "new" else _download(number)
