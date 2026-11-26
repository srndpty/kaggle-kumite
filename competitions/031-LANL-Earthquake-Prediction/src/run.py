"""LANL の固定 CV、モデル選定、提出ファイルの公開。"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TypedDict

import numpy as np
from lightgbm import Booster, LGBMRegressor
from p0 import COLUMNS, ROOT
from windows import (
    CACHE_ARRAYS,
    CACHE_NAME,
    MANIFEST_NAME,
    WINDOW,
    fold_ids,
    official_hash,
    read_ids,
    verify_manifest,
)

from kumite.checks import validate_regression
from kumite.crossfit import FoldOutput, cross_fit_splits, folds_to_splits, split_roles
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.experiments import ExperimentRecord
from kumite.paths import REPO_ROOT
from kumite.provenance import array_sha256, file_sha256
from kumite.quality_gates import (
    SelectionPlan,
    SourceSpec,
    dense_oof,
    publish_export,
    reserve_cv,
    select_runs,
    write_selection,
)
from kumite.runs import RunStore


class ModelParams(TypedDict):
    objective: str
    n_estimators: int
    learning_rate: float
    num_leaves: int
    random_state: int
    deterministic: bool
    force_col_wise: bool
    n_jobs: int
    verbosity: int


PARAMS = ModelParams(
    objective="regression_l1",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=15,
    random_state=42,
    deterministic=True,
    force_col_wise=True,
    n_jobs=4,
    verbosity=-1,
)
FEATURES = {
    "B00": (),
    "E01": (6, 7, 8),
    "E02": (9, 10, 11),
    "E03": tuple(range(12, 20)),
    "E04": (6, 7, 8, 20),
}
HYPOTHESES = {
    "B00": "窓全体の振幅統計で残り時間を推定できる",
    "E01": "大振幅の分位点が破壊直前の誤差を下げる",
    "E02": "局所 RMS が窓内の短い変動を捉える",
    "E03": "帯域別パワーが振幅の近い窓を識別する",
    "E04": "静かな区間の局所標準偏差が大振幅の分位点を補う",
}


@dataclass(frozen=True)
class Dataset:
    rows: np.ndarray
    train: np.ndarray
    test: np.ndarray
    ids: np.ndarray
    folds: np.ndarray
    digest: str

    @property
    def y(self) -> np.ndarray:
        return self.rows[:, 3]


def load() -> Dataset:
    data = ROOT / "data"
    cached = ROOT / "outputs" / CACHE_NAME
    ids = read_ids(data)
    raw = official_hash(data, ids)
    with np.load(cached, allow_pickle=False) as cache:
        arrays = {key: cache[key] for key in CACHE_ARRAYS}
    verify_manifest(ROOT / MANIFEST_NAME, raw, arrays)
    if not np.array_equal(arrays["ids"], ids):
        raise ValueError("公式データと抽出済み窓の ID が一致しません")
    rows, train, test, folds = (arrays[key] for key in ("rows", "train", "test", "folds"))
    if (
        rows.shape != (len(train), 4)
        or train.shape != (len(rows), 21)
        or test.shape != (len(ids), 21)
        or (rows[:, 1] - rows[:, 0] != WINDOW).any()
        or (rows[1:, 0] < rows[:-1, 1]).any()
        or not np.array_equal(folds, fold_ids(rows[:, 2]))
        or not all(np.isfinite(x).all() for x in (rows, train, test))
    ):
        raise ValueError("抽出済み窓の形状、順序、分割または値が不正です")
    return Dataset(rows, train, test, ids, folds, f"{raw}:{file_sha256(cached)}")


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def inputs(dataset: Dataset, variant: str) -> tuple[np.ndarray, np.ndarray]:
    columns = [*range(6), *FEATURES[variant]]
    return dataset.train[:, columns], dataset.test[:, columns]


def fit(x: np.ndarray, y: np.ndarray) -> LGBMRegressor:
    return LGBMRegressor(**PARAMS).fit(x, y)


def predict(model: Booster, x: np.ndarray) -> np.ndarray:
    pred = np.asarray(model.predict(x, num_threads=4))
    validate_regression(pred, len(x))
    return np.maximum(pred, 0)


def fold_prediction(
    x: np.ndarray, y: np.ndarray, train: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    return predict(fit(x[train], y[train]).booster_, x[valid])


def source(variant: str) -> SourceSpec:
    local = (ROOT / "src").relative_to(REPO_ROOT).as_posix()
    shared = REPO_ROOT / "src" / "kumite"
    files = ["uv.lock", "pyproject.toml"]
    files.append((ROOT / MANIFEST_NAME).relative_to(REPO_ROOT).as_posix())
    files.extend(f"{local}/{name}" for name in ("run.py", "windows.py", "p0.py"))
    files.extend(item.relative_to(REPO_ROOT).as_posix() for item in sorted(shared.rglob("*.py")))
    return SourceSpec(
        REPO_ROOT, tuple(files), {"variant": variant, "model": PARAMS}, ("src", local)
    )


def variant_for(run_id: str) -> str:
    return run_id.split("-", 1)[0]


def train_run(store: RunStore, dataset: Dataset, plan: SelectionPlan, run_id: str) -> None:
    variant = variant_for(run_id)
    x, _ = inputs(dataset, variant)
    splits = folds_to_splits(dataset.folds)
    y = dataset.y
    evaluation = Evaluation(
        "031",
        "MAE",
        False,
        mae,
        dataset.digest,
        "cycle-block-v1",
        array_sha256(split_roles(len(y), splits)),
        42,
    )
    trial = Trial(
        HYPOTHESES[variant],
        {
            "B00": "窓全体の振幅統計",
            "E01": "B00 に絶対振幅の上位分位点を追加",
            "E02": "B00 に区間 RMS 統計を追加",
            "E03": "B00 に帯域別 FFT パワーを追加",
            "E04": "E01 に移動標準偏差の下位分位点を追加",
        }[variant],
        "LightGBM L1",
        "official-lanl-windows-v2",
        {"variant": variant, **PARAMS},
    )
    spec = source(variant)

    def fit_fold(train: np.ndarray, valid: np.ndarray) -> FoldOutput:
        median = np.full(len(valid), np.median(y[train]))
        return FoldOutput(
            fold_prediction(x, y, train, valid),
            np.empty(0),
            {"median_mae": mae(y[valid], median), "validation_windows": len(valid)},
        )

    with reserve_cv(store, run_id, spec, {}, plan) as (reservation, start):
        before = perf_counter()
        result = cross_fit_splits(len(y), splits, fit_fold)
        publish_cv_run(
            store,
            reservation,
            evaluation,
            trial,
            y,
            splits,
            result,
            perf_counter() - before,
            lambda _: None,
            save_query=False,
            gate=(spec, start, y.shape),
        )


def validate_submission(path: Path, ids: np.ndarray) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != COLUMNS:
            raise ValueError("提出列が一致しません")
        rows = list(reader)
        if [row["seg_id"] for row in rows] != ids.tolist():
            raise ValueError("提出 ID、順序、または件数が一致しません")
        if any(set(row) != set(COLUMNS) for row in rows):
            raise ValueError("提出行の列数が一致しません")
    predictions = np.asarray([float(row["time_to_failure"]) for row in rows])
    validate_regression(predictions, len(ids), lower=0)
    return predictions


def validate_export_outputs(path: Path, ids: np.ndarray, test: np.ndarray) -> None:
    submitted = validate_submission(path / "submission.csv", ids)
    saved = Booster(model_file=str(path / "model.txt"))
    if not np.allclose(predict(saved, test), submitted, rtol=0, atol=1e-12):
        raise ValueError("保存モデルの再推論結果が提出予測と一致しません")


def diagnostics(dataset: Dataset, prediction: np.ndarray) -> dict[str, object]:
    y = dataset.y
    return {
        "cycle_mae": {
            str(int(cycle)): mae(y[mask], prediction[mask])
            for cycle in np.unique(dataset.rows[:, 2])
            if (mask := dataset.rows[:, 2] == cycle).any()
        },
        "time_band_mae": {
            f"{left}-{right}": {"mae": mae(y[mask], prediction[mask]), "windows": int(mask.sum())}
            for left, right in [(0, 2), (2, 5), (5, 10), (10, np.inf)]
            if (mask := (y >= left) & (y < right)).any()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "select", "export"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--export-id", default="F00")
    args = parser.parse_args()
    plan_file = args.plan
    with plan_file.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    plan = SelectionPlan(tuple(raw["candidates"]), raw["tolerance"])
    receipt = plan_file.with_name(plan_file.stem.replace("plan-", "selection-") + ".json")
    dataset = load()
    store = RunStore(ROOT)
    if args.action == "train":
        run_ids = args.runs or plan.candidates
        if not set(run_ids) <= set(plan.candidates):
            parser.error("候補計画にない run ID が指定されました")
        for run_id in run_ids:
            train_run(store, dataset, plan, run_id)
        return
    assess = dense_oof(
        dataset.y,
        folds_to_splits(dataset.folds),
        mae,
        expected_shape=dataset.y.shape,
        data_hash=dataset.digest,
    )
    if args.action == "select":
        selected = select_runs(store, plan, assess)
        write_selection(receipt, selected)
        print(selected)
        print(json.dumps(diagnostics(dataset, store.load_array(selected.winner, "oof.npy"))))
        return
    winner = json.loads(receipt.read_text(encoding="utf-8"))["winner"]
    variant = variant_for(winner)
    x, test = inputs(dataset, variant)

    def write(path: Path, record: ExperimentRecord) -> None:
        if record.run_id != winner:
            raise ValueError("選定記録の勝者が変わりました")
        model = fit(x, dataset.y)
        model.booster_.save_model(str(path / "model.txt"))
        pred = predict(model.booster_, test)
        with (path / "submission.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(COLUMNS)
            writer.writerows(zip(dataset.ids.tolist(), pred.tolist(), strict=True))

    out = publish_export(
        store,
        args.export_id,
        plan=plan,
        receipt=receipt,
        assess=assess,
        source=source(variant),
        write=write,
        required_outputs=("model.txt", "submission.csv"),
        validate_outputs=lambda path: validate_export_outputs(path, dataset.ids, test),
    )
    print(out)


if __name__ == "__main__":
    main()
