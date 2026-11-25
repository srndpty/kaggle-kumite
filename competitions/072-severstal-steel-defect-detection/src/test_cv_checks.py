import csv
from dataclasses import replace
from pathlib import Path

import pytest
from cv_checks import SOURCE_FILES, read_sources, source_hashes, verify_sources
from export import verify_full_baseline
from select_model import select

from kumite.experiments import ExperimentRecord
from kumite.runs import RunStore


def record(run_id: str, score: float = 0.5) -> ExperimentRecord:
    return ExperimentRecord(
        competition_id="072",
        run_id=run_id,
        metric="mean_dice",
        greater_is_better=True,
        data_version="test",
        data_hash="data",
        split_version="v1",
        split_hash="split",
        seed=42,
        hypothesis="test",
        changed_one_thing="test",
        model="test",
        fold_scores=[score] * 3,
        oof_score=score,
        code_version="test",
    )


def publish_oof(store: RunStore, entry: ExperimentRecord, value: float, *, offset: int = 0) -> Path:
    def write(output: Path) -> None:
        with (output / "oof.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("image_id", "class_id", "fold", "dice"))
            for fold in range(3):
                for klass in range(1, 5):
                    writer.writerow((f"{fold}.jpg", klass, (fold + offset) % 3, value))

    with store.reserve(entry.run_id) as reservation:
        return store.publish(reservation, entry, write) / "oof.csv"


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("tamper", "ハッシュ"),
        ("nan", "Dice が不正"),
        ("range", "Dice が不正"),
        ("fold", "fold が揃っていません"),
        ("score", "実験ログと一致しません"),
        ("split", "評価条件が揃っていません"),
        ("seed", "評価条件が揃っていません"),
        ("direction", "評価条件が揃っていません"),
    ],
)
def test_rejects_invalid_candidate_even_when_it_cannot_win(
    tmp_path: Path, fault: str, message: str
) -> None:
    store = RunStore(tmp_path)
    publish_oof(store, record("base"), 0.5)
    publish_oof(store, record("winner", 0.9), 0.9)
    candidate = replace(
        record("candidate", 0.3),
        split_version="v2" if fault == "split" else "v1",
        seed=7 if fault == "seed" else 42,
        greater_is_better=fault != "direction",
    )
    value = {"nan": float("nan"), "range": 1.01, "score": 0.4}.get(fault, 0.3)
    path = publish_oof(store, candidate, value, offset=int(fault == "fold"))
    if fault == "tamper":
        path.write_text("changed", encoding="utf-8")
    output = tmp_path / "selection.json"
    with pytest.raises(ValueError, match=message):
        select(store, "base", ["winner", "candidate"], output)
    assert not output.exists()


def test_selects_using_verified_oof(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    publish_oof(store, record("base"), 0.5)
    publish_oof(store, record("winner", 0.9), 0.9)
    assert select(store, "base", ["winner"], tmp_path / "selection.json") == "winner"


@pytest.mark.parametrize("fault", [None, "wrong-baseline", "fold-1", "full-oof"])
def test_full_baseline_lineage(tmp_path: Path, fault: str | None) -> None:
    store = RunStore(tmp_path)
    hashes = source_hashes(read_sources())
    for name in ("base", "other"):
        with store.reserve(name) as reservation:

            def write(output: Path, identity: str = name) -> None:
                for artifact in ("oof.csv", *(f"fold-{i}.pt" for i in range(3))):
                    (output / artifact).write_text(f"{identity}/{artifact}", encoding="utf-8")

            store.publish(
                reservation,
                replace(record(name), params={"variant": "baseline"}, inputs=hashes),
                write,
            )
    inputs = store.input_hashes("base", *(f"fold-{i}.pt" for i in range(3)))
    if fault == "fold-1":
        inputs["base/fold-1.pt"] = "different"
    full_source = "other" if fault == "wrong-baseline" else "base"
    full = replace(
        record("full"),
        params={"variant": "baseline", "score_source": "source_cv", "source_run": full_source},
        inputs={**hashes, **store.input_hashes(full_source, "oof.csv")},
    )
    if fault == "full-oof":
        full.inputs["base/oof.csv"] = "different"
    cv = replace(record("gate"), inputs=inputs)
    if fault is None:
        verify_full_baseline(store, cv, full, hashes)
    else:
        with pytest.raises(ValueError, match="一致しません"):
            verify_full_baseline(store, cv, full, hashes)


def test_rejects_model_source_change_and_missing_manifest(tmp_path: Path) -> None:
    for name in SOURCE_FILES:
        (tmp_path / name).write_text(name, encoding="utf-8")
    hashes = source_hashes(read_sources(tmp_path))
    with pytest.raises(ValueError, match="未記録"):
        verify_sources(record("historic"), hashes)
    cv = replace(record("cv"), inputs=hashes)
    verify_sources(cv, hashes)
    (tmp_path / "model.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="変更"):
        verify_sources(cv, source_hashes(read_sources(tmp_path)))
