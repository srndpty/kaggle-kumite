"""CV と提出コードが照合する入力。"""

from hashlib import sha256
from pathlib import Path

from kumite.paths import REPO_ROOT
from kumite.provenance import files_sha256

ROOT = Path(__file__).resolve().parents[1]
SPLIT_VERSION = "split_v1"
SEED = 0


def data_digest(data: Path) -> str:
    return files_sha256(data / name for name in ("train.csv", "test.csv", "sample_submission.csv"))


def inference_sources(root: Path) -> dict[str, bytes]:
    paths = (REPO_ROOT / "src/kumite/checks.py", root / "src/inputs.py", root / "src/model.py")
    return {path.name: path.read_bytes() for path in paths}


def source_digests(sources: dict[str, bytes]) -> dict[str, str]:
    return {name: sha256(content).hexdigest() for name, content in sources.items()}
