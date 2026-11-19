"""公式入力を実験記録と照合する。"""

import hashlib
from pathlib import Path

import pandas as pd

from kumite.provenance import file_sha256


def official_data_hash(root: Path) -> str:
    hashes = "".join(
        file_sha256(root / "data" / name)
        for name in ("train.csv", "test.csv", "sample_submission.csv")
    )
    return hashlib.sha256(hashes.encode()).hexdigest()


def read_verified_train(root: Path, expected_hash: str) -> pd.DataFrame:
    if official_data_hash(root) != expected_hash:
        raise ValueError("公式データのハッシュが実験記録と一致しません")
    return pd.read_csv(root / "data" / "train.csv")
