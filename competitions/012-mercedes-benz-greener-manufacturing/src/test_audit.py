"""最終監査の異常系を、学習を使わずに確認する。"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score
from train import SEED, SPLIT_VERSION, make_splits, validate_binary_features

from kumite.crossfit import split_roles
from kumite.experiments import ExperimentRecord
from kumite.provenance import array_sha256, files_sha256
from kumite.runs import RunStore


@pytest.fixture
def audit_root(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    for name in ("analyze.py", "train.py"):
        shutil.copyfile(Path(__file__).with_name(name), src / name)
    data = tmp_path / "data"
    data.mkdir()
    rows = np.arange(10)
    y = rows + 10.0
    x = pd.DataFrame({"X5": [f"c{i}" for i in rows], "binary": rows % 2})
    x.assign(ID=rows, y=y).to_csv(data / "train.csv", index=False)
    x.iloc[:2].assign(ID=[10, 11]).to_csv(data / "test.csv", index=False)
    sample = pd.DataFrame({"ID": [10, 11], "y": [10.0, 11.0]})
    sample.to_csv(data / "sample_submission.csv", index=False)
    _, splits = make_splits(x)
    data_hash = files_sha256(
        data / name for name in ("train.csv", "test.csv", "sample_submission.csv")
    )
    split_hash = array_sha256(split_roles(len(x), splits))
    store = RunStore(tmp_path)

    def write_outputs(out: Path) -> None:
        np.save(out / "oof.npy", y + 1.0)
        np.save(out / "test.npy", sample["y"].to_numpy())
        sample.to_csv(out / "submission.csv", index=False)
        (out / "fold_info.json").write_text("[]", encoding="utf-8")

    for i, variant in enumerate(("baseline", "rare", "strong", "categorical", "interaction")):
        run_id = f"E{i:02}"
        record = ExperimentRecord(
            competition_id=tmp_path.name,
            run_id=run_id,
            metric="r2",
            greater_is_better=True,
            split_version=SPLIT_VERSION,
            split_hash=split_hash,
            seed=SEED,
            data_version="fixture",
            data_hash=data_hash,
            hypothesis="",
            changed_one_thing="",
            model="fixture",
            params={"variant": variant},
            fold_scores=[],
            oof_score=float(r2_score(y, y + 1.0)),
            code_version="fixture",
        )
        with store.reserve(run_id) as reservation:
            store.publish(reservation, record, write_outputs)
    return tmp_path


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("artifact", "ハッシュがログと一致しません"),
        ("score", "OOF R² が記録と一致しません"),
        ("metric", "評価条件が一致しません"),
        ("variant", "variant は baseline が必要です"),
        ("missing_hash", "成果物の記録が不正です"),
        ("none", "採用: E00"),
    ],
)
def test_optimized_audit(audit_root: Path, fault: str, message: str) -> None:
    if fault == "artifact":
        path = audit_root / "outputs" / "runs" / "E00" / "oof.npy"
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
    elif fault != "none":
        log = audit_root / "experiments.jsonl"
        lines = log.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[0])
        if fault == "score":
            record["oof_score"] = 0.5
        elif fault == "metric":
            record["metric"] = "mae"
        elif fault == "variant":
            record["params"]["variant"] = "rare"
        else:
            del record["artifacts"]["oof.npy"]
        lines[0] = json.dumps(record)
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-O", str(audit_root / "src" / "analyze.py")],
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    selection = audit_root / "outputs" / "selection.json"
    if fault == "none":
        assert result.returncode == 0, result.stderr
        assert message in result.stdout
        evidence = json.loads(selection.read_text(encoding="utf-8"))
        assert evidence["selected_run_id"] == "E00"
        assert evidence["comparisons"]["E01"]["high"] == 0
    else:
        assert result.returncode != 0
        assert message in result.stderr
        assert "採用:" not in result.stdout
        assert not selection.exists()


@pytest.mark.parametrize(("frame_index", "value"), [(0, -1), (1, 0.5)])
def test_non_binary_numeric_features_are_rejected(frame_index: int, value: float) -> None:
    frames = [
        pd.DataFrame({"category": ["a", "b", "c"], "numeric": [0.0, 1.0, 0.0]}) for _ in range(2)
    ]
    frames[frame_index].loc[2, "numeric"] = value
    with pytest.raises(ValueError, match="0/1 以外"):
        validate_binary_features(*frames)
