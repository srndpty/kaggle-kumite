"""リポジトリ内の基準パス。"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_CSV = REPO_ROOT / "catalog" / "kaggle100_catalog.csv"
COMPETITIONS_DIR = REPO_ROOT / "competitions"
TEMPLATE_DIR = REPO_ROOT / "templates" / "competition"
