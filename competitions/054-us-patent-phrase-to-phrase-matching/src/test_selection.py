"""確定した選定記録と提出ソースの照合を検査する。"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from evidence import ROOT, SEED, SPLIT_VERSION, data_digest, inference_sources, source_digests
from export import verified_source
from inputs import load_data
from model import ALPHA, pearson
from select_model import freeze_selection, select
from split import make_split
from train import TRIALS

from kumite.experiments import ExperimentRecord
from kumite.provenance import file_sha256
from kumite.runs import RunStore


@pytest.fixture
def evidence_root(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.csv").write_text(
        "id,anchor,target,context,score\n"
        + "".join(f"{i},anchor{i},target,A01,{i / 8}\n" for i in range(9)),
        encoding="utf-8",
    )
    (data / "test.csv").write_text(
        "id,anchor,target,context\nx,unknown,example,A01\n", encoding="utf-8"
    )
    (data / "sample_submission.csv").write_text("id,score\nx,0\n", encoding="utf-8")
    src = tmp_path / "src"
    src.mkdir()
    for name in ("model.py", "inputs.py"):
        shutil.copyfile(ROOT / "src" / name, src / name)
    store = RunStore(tmp_path)
    store.runs_dir.mkdir(parents=True)
    split_path = tmp_path / "outputs" / f"{SPLIT_VERSION}.csv"
    train, _ = load_data(data)
    make_split(train).to_csv(split_path, index=False)
    y = train["score"].to_numpy()
    for trial, (features, _, _) in TRIALS.items():
        prediction = y if trial == "E05" else np.roll(y, 1)
        record = ExperimentRecord(
            competition_id="054",
            run_id=trial,
            metric="pearson",
            greater_is_better=True,
            data_version="official",
            data_hash=data_digest(data),
            split_version=SPLIT_VERSION,
            split_hash=file_sha256(split_path),
            seed=SEED,
            hypothesis="",
            changed_one_thing="",
            model="Ridge",
            fold_scores=[],
            oof_score=pearson(y, prediction),
            code_version="fixture",
            params={
                "alpha": ALPHA,
                "features": features,
                "source_sha256": source_digests(inference_sources(tmp_path)),
            },
        )
        with store.reserve(trial) as reservation:
            store.publish(
                reservation, record, lambda path, pred=prediction: np.save(path / "oof.npy", pred)
            )
    return tmp_path


def test_frozen_selection_is_required_and_cannot_be_overwritten(evidence_root: Path) -> None:
    path = evidence_root / "selection.json"
    with pytest.raises(FileNotFoundError):
        verified_source(evidence_root, path, "", baseline=False)
    selection = select(evidence_root, "", inference_sources(evidence_root))
    assert selection["selected"] == "E05"
    freeze_selection(path, selection)
    with pytest.raises(FileExistsError):
        freeze_selection(path, selection)
    assert verified_source(evidence_root, path, "", baseline=False)[0] == "E05"
    path.write_text(json.dumps({**selection, "selected": "E04"}), encoding="utf-8")
    with pytest.raises(ValueError, match="選定記録"):
        verified_source(evidence_root, path, "", baseline=False)


def test_rejects_altered_oof_and_recomputes_pearson(evidence_root: Path) -> None:
    sources = inference_sources(evidence_root)
    store = RunStore(evidence_root)
    path = store.runs_dir / "E05" / "oof.npy"
    np.save(path, np.linspace(1, 0, 9))
    with pytest.raises(ValueError, match="ハッシュ"):
        select(evidence_root, "", sources)
    # ハッシュだけ差し替えても、保存済みスコアの不一致を検出する。
    records = [json.loads(line) for line in store.log_path.read_text(encoding="utf-8").splitlines()]
    records[-1]["artifacts"]["oof.npy"] = file_sha256(path)
    store.log_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="OOF Pearson"):
        select(evidence_root, "", sources)


@pytest.mark.parametrize(
    ("name", "message"),
    [("src/model.py", "推論ソース SHA256"), ("data/test.csv", "データまたは分割のハッシュ")],
)
def test_export_rejects_changed_inputs(evidence_root: Path, name: str, message: str) -> None:
    path = evidence_root / "selection.json"
    freeze_selection(path, select(evidence_root, "", inference_sources(evidence_root)))
    changed = evidence_root / name
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(ValueError, match=message):
        verified_source(evidence_root, path, "", baseline=False)
