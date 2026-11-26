"""画像単位の CV、再採点、モデル選定。"""

import argparse
import json
import tomllib
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import cast

import torch
from data import REPO, ROOT, SEED, Image, dataset
from masks import disjoint, image_score, ious
from model import Settings, fit, predict, rankings, settings

from kumite.experiments import ExperimentRecord
from kumite.provenance import code_version
from kumite.quality_gates import (
    SelectionPlan,
    SourceSpec,
    publish_gated_cv,
    reserve_cv,
    select_runs,
    verify_cv_run,
    write_selection,
)
from kumite.runs import RunStore

TRIALS = {
    "E00": "Mask R-CNN の基準",
    "E01": "入力解像度",
    "E02": "水平反転",
    "E03": "重複除去後の面積比",
    "S00": "P4 勝者の再学習",
    "S01": "マスク内平均確率による候補の再順位付け",
}
METRIC = "image mean AP IoU"
MODEL = "Mask R-CNN ResNet-50 FPN"
PLAN = ROOT / "selection.toml"
RECEIPT = ROOT / "outputs" / "selection.json"
PREDICTIONS = "oof.json"
RAW = "raw_oof.json"
WEIGHTED_RAW = "weighted_raw_oof.json"


def plan(path: Path) -> SelectionPlan:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    return SelectionPlan(tuple(config["candidates"]), float(config["tolerance"]))


def run_settings(trial: str, path: Path) -> Settings:
    if trial not in ("S00", "S01"):
        return settings(trial)
    base = tomllib.loads(path.read_text(encoding="utf-8"))["base_trial"]
    if base not in ("E00", "E01", "E02", "E03"):
        raise ValueError("P6 の比較元が不正です")
    result = settings(base)
    if trial == "S01":
        result["rank_by_mask"] = True
    return result


def source(trial: str, path: Path) -> SourceSpec:
    competition = ROOT.relative_to(REPO).as_posix()
    files = {
        "uv.lock",
        f"{competition}/quality-gates.toml",
        path.relative_to(REPO).as_posix(),
    }
    files.update(file.relative_to(REPO).as_posix() for file in (REPO / "src/kumite").glob("*.py"))
    files.update(
        file.relative_to(REPO).as_posix()
        for file in (ROOT / "src").glob("*.py")
        if not file.name.startswith(("test_", "dummy"))
    )
    return SourceSpec(
        REPO,
        tuple(sorted(files)),
        dict(run_settings(trial, path)),
        import_roots=("src", f"{competition}/src"),
    )


def score(
    rows: list[Image], folds: list[int], predictions: list[list[str]]
) -> tuple[list[float], float]:
    values = [
        image_score(ious(row.masks, masks, row.size))
        for row, masks in zip(rows, predictions, strict=True)
    ]
    scores = [
        [value for value, index in zip(values, folds, strict=True) if index == fold]
        for fold in range(3)
    ]
    return [sum(group) / len(group) for group in scores], sum(values) / len(values)


def read_rows(path: Path, rows: list[Image], folds: list[int]) -> list[object]:
    saved = json.loads(path.read_text(encoding="utf-8"))
    values: list[object] = []
    for row, fold, item in zip(rows, folds, saved, strict=True):
        if item["id"] != row.id or item["fold"] != fold or not isinstance(item["masks"], list):
            raise ValueError(f"OOF の ID、fold、マスクが不正です: {row.id}")
        values.append(item["masks"])
    return values


def assess(artifact: Callable[[str], Path], record: ExperimentRecord) -> tuple[list[float], float]:
    rows, folds, data_hash, split_hash = dataset()
    if (record.data_hash, record.split_hash) != (data_hash, split_hash):
        raise ValueError("データまたは分割が CV 記録と一致しません")
    predictions = cast("list[list[str]]", read_rows(artifact(PREDICTIONS), rows, folds))
    return score(rows, folds, predictions)


def run(trial: str, run_id: str, baseline: str, path: Path) -> Path:
    store = RunStore(ROOT)
    spec = source(trial, path)
    config = run_settings(trial, path)
    dependency = {"E03": (RAW, "E00"), "S01": (WEIGHTED_RAW, "S00")}.get(trial)
    inputs = store.input_hashes(baseline, dependency[0]) if dependency else {}
    with reserve_cv(store, run_id, spec, inputs, plan(path)) as (reservation, start):
        started = perf_counter()
        rows, folds, data_hash, split_hash = dataset()
        candidates: list[list[tuple[float, str]]]
        weighted: list[list[tuple[float, str]]] = [[] for _ in rows]
        if dependency:
            artifact, parent_trial = dependency
            parent = verify_cv_run(store, baseline, assess)
            if parent.params != dict(run_settings(parent_trial, path)) or (
                parent.data_hash,
                parent.split_hash,
            ) != (data_hash, split_hash):
                raise ValueError("依存 run の設定、データまたは分割が一致しません")
            candidates = cast(
                "list[list[tuple[float, str]]]",
                read_rows(store.artifact_path(baseline, artifact), rows, folds),
            )
        else:
            candidates = [[] for _ in rows]
            device = torch.device("cuda")
            for fold in range(3):
                print(f"fold {fold} 学習開始", flush=True)
                model = fit(
                    [row for row, index in zip(rows, folds, strict=True) if index != fold],
                    config,
                    device,
                )
                for index, row in enumerate(rows):
                    if folds[index] == fold:
                        if trial == "S00":
                            candidates[index], weighted[index] = rankings(
                                model, row.path, config, device, weighted=True
                            )
                        else:
                            candidates[index] = predict(model, row.path, config, device)
                del model
        predictions = [
            disjoint(masks, row.size, config["retain"])
            for row, masks in zip(rows, candidates, strict=True)
        ]
        fold_scores, oof_score = score(rows, folds, predictions)
        record = ExperimentRecord(
            competition_id="077",
            run_id=run_id,
            metric=METRIC,
            greater_is_better=True,
            data_version="official-v1",
            data_hash=data_hash,
            split_version="sample-group-v1",
            split_hash=split_hash,
            seed=SEED,
            hypothesis=TRIALS[trial],
            changed_one_thing=TRIALS[trial],
            model=MODEL,
            params=dict(config),
            fold_scores=fold_scores,
            oof_score=oof_score,
            runtime_sec=perf_counter() - started,
            code_version=code_version(REPO),
            inputs=inputs,
        )

        def write(directory: Path) -> None:
            files: dict[str, list[object]] = {PREDICTIONS: list(predictions)}
            if trial == "E00":
                files[RAW] = list(candidates)
            if trial == "S00":
                files[WEIGHTED_RAW] = list(weighted)
            for name, values in files.items():
                payload = [
                    {"id": row.id, "fold": fold, "masks": masks}
                    for row, fold, masks in zip(rows, folds, values, strict=True)
                ]
                (directory / name).write_text(
                    json.dumps(payload, allow_nan=False), encoding="utf-8"
                )

        return publish_gated_cv(
            store, reservation, record, write, source=spec, before_training=start, assess=assess
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", choices=(*TRIALS, "smoke", "select"))
    parser.add_argument("--run-id")
    parser.add_argument("--baseline")
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    args = parser.parse_args()
    path = args.plan.resolve()
    if args.trial == "smoke":
        rows, folds, _, _ = dataset()
        train = sorted(
            (row for row, fold in zip(rows, folds, strict=True) if fold != 0),
            key=lambda row: len(row.masks),
            reverse=True,
        )[:2]
        valid = next(row for row, fold in zip(rows, folds, strict=True) if fold == 0)
        device = torch.device("cuda")
        config = settings("E01")
        model = fit(train, config, device, smoke=True)
        masks = disjoint(predict(model, valid.path, config, device), valid.size, config["retain"])
        print("smoke", image_score(ious(valid.masks, masks, valid.size)))
    elif args.trial == "select":
        selection = select_runs(RunStore(ROOT), plan(path), assess)
        write_selection(args.receipt, selection)
        print(selection)
    else:
        baseline = args.baseline or ("S00" if args.trial == "S01" else "E00")
        print(run(args.trial, args.run_id or args.trial, baseline, path))


if __name__ == "__main__":
    main()
