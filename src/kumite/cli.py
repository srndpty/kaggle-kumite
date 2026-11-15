"""`python -m kumite` のエントリポイント。通常は `dev` コマンド経由で呼ぶ。"""

import argparse
import subprocess
from collections.abc import Sequence
from pathlib import Path

from kumite.catalog import find_competition, load_catalog
from kumite.experiments import LOG_FILE_NAME, compare_runs, find_record, read_records
from kumite.kaggle_data import download_data, show_submissions, submit
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


def _submit(number: int, file: Path, message: str) -> int:
    competition = find_competition(load_catalog(CATALOG_CSV), number)
    submit(competition, file, message)
    show_submissions(competition)
    return 0


def _submissions(number: int) -> int:
    show_submissions(find_competition(load_catalog(CATALOG_CSV), number))
    return 0


def _compare(number: int, base_run_id: str, run_ids: list[str]) -> int:
    competition = find_competition(load_catalog(CATALOG_CSV), number)
    records = read_records(COMPETITIONS_DIR / competition.dir_name / LOG_FILE_NAME)
    base = find_record(records, base_run_id)
    direction = "大きいほど良い" if base.greater_is_better else "小さいほど良い"
    folds = " ".join(f"{s:.5f}" for s in base.fold_scores)
    print(f"指標 {base.metric}（{direction}）")
    print(f"基準 {base.run_id}: OOF {base.oof_score:.5f}  folds [{folds}]")
    print("差は (実験 - 基準)。fold ごとの差・平均・改善/悪化した fold 数")
    for run_id in run_ids:
        other = find_record(records, run_id)
        c = compare_runs(base, other)
        diffs = " ".join(f"{d:+.5f}" for d in c.diffs)
        print(
            f"{run_id}: OOF {other.oof_score:.5f}  差 [{diffs}]  平均 {c.mean_diff:+.5f}"
            f"  改善 {c.n_improved} / 悪化 {c.n_worsened}"
        )
    print("注: fold どうしは学習データが重なるので、改善 fold 数は独立な再現回数ではない")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kumite", description="Kaggle 100本ノックの補助コマンド")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="カタログ一覧と着手状況")
    for name, help_text in (
        ("new", "コンペ用ディレクトリを生成"),
        ("download", "データを取得"),
        ("submissions", "提出履歴とスコアを表示"),
    ):
        sub.add_parser(name, help=help_text).add_argument("number", type=int, help="カタログ番号")
    submit_parser = sub.add_parser("submit", help="ファイルを提出し、提出履歴を表示")
    submit_parser.add_argument("number", type=int, help="カタログ番号")
    submit_parser.add_argument("file", type=Path, help="提出ファイル")
    submit_parser.add_argument("-m", "--message", required=True, help="提出メッセージ")
    compare_parser = sub.add_parser("compare", help="実験を同じ fold どうしで比較")
    compare_parser.add_argument("number", type=int, help="カタログ番号")
    compare_parser.add_argument("base", help="基準の run_id")
    compare_parser.add_argument("run_ids", nargs="+", help="比較する run_id")

    args = parser.parse_args(argv)
    command: str = args.command
    if command == "list":
        return _list()
    number: int = args.number
    match command:
        case "new":
            return _new(number)
        case "download":
            return _download(number)
        case "submissions":
            return _submissions(number)
        case "compare":
            base_run_id: str = args.base
            run_ids: list[str] = args.run_ids
            return _compare(number, base_run_id, run_ids)
        case _:
            file: Path = args.file
            message: str = args.message
            return _submit(number, file, message)
