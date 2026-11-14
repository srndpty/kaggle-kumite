"""kaggle CLI によるデータ取得と提出。"""

import subprocess
import zipfile
from pathlib import Path

from kumite.catalog import Competition


def download_data(competition: Competition, competitions_dir: Path) -> Path:
    """データを `data/` に取得して展開する。zip は展開後に削除する。"""
    competition_dir = competitions_dir / competition.dir_name
    if not competition_dir.is_dir():
        raise FileNotFoundError(
            f"{competition_dir} がありません。先に `dev new {competition.number}` を実行してください"
        )
    data_dir = competition_dir / "data"
    data_dir.mkdir(exist_ok=True)

    subprocess.run(
        ["kaggle", "competitions", "download", "-c", competition.slug, "-p", str(data_dir)],
        check=True,
    )
    for archive in data_dir.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(data_dir)
        archive.unlink()
    return data_dir


def submit(competition: Competition, file: Path, message: str) -> None:
    subprocess.run(
        [
            "kaggle",
            "competitions",
            "submit",
            "-c",
            competition.slug,
            "-f",
            str(file),
            "-m",
            message,
        ],
        check=True,
    )


def show_submissions(competition: Competition) -> None:
    subprocess.run(["kaggle", "competitions", "submissions", "-c", competition.slug], check=True)
