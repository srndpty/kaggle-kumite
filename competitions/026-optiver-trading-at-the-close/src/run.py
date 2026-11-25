"""共通 Quality Gate を通して取引日順 CV を公開する。"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from evaluation import mae, splits_for
from features import BASE, KEYS, PARAMS, ROUNDS, feature_matrix, fit_model

from kumite.crossfit import FoldOutput, cross_fit_splits, split_roles
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.paths import REPO_ROOT
from kumite.provenance import array_sha256, files_sha256
from kumite.quality_gates import SelectionPlan, SourceSpec, reserve_cv
from kumite.runs import RunStore

COMP = Path(__file__).resolve().parents[1]
DATA = COMP / "data" / "train.csv"
HYPOTHESES = {
    "E00": "現バッチの板とオークション入力で MAE を下げる",
    "E01": "スプレッドと需給の比率が生の値を補う",
    "E02": "銘柄ごとの直近の変化が予測に寄与する",
    "E03": "同時刻の他銘柄との相対位置が需給の偏りを捉える",
    "E04": "引けまでの節目を示すフラグが MAE を下げる",
}


def load_plan(path: Path) -> SelectionPlan:
    value = json.loads(path.read_text(encoding="utf-8"))
    return SelectionPlan(tuple(value["candidates"]), value["tolerance"])


def load_rows() -> pd.DataFrame:
    rows = pd.read_csv(DATA, usecols=[*dict.fromkeys([*KEYS, *BASE]), "row_id", "target"])
    if rows["row_id"].isna().any() or not rows["row_id"].is_unique or rows.duplicated(KEYS).any():
        raise ValueError("学習行のキーが不正です")
    return rows


def source_for(method: str) -> SourceSpec:
    relative = COMP.relative_to(REPO_ROOT).as_posix()
    files = [
        path.relative_to(REPO_ROOT).as_posix()
        for folder in (COMP / "src", REPO_ROOT / "src/kumite")
        for path in folder.glob("*.py")
        if not path.name.startswith("test_")
    ]
    files.extend(("uv.lock", "pyproject.toml", f"{relative}/quality-gates.toml"))
    return SourceSpec(
        REPO_ROOT,
        tuple(sorted(files)),
        {"method": method, "model": PARAMS, "rounds": ROUNDS},
        import_roots=("src", f"{relative}/src"),
    )


def run(methods: list[str]) -> None:
    rows = load_rows()
    y = rows["target"].to_numpy()
    splits = splits_for(rows["date_id"].to_numpy(), y)
    roles = split_roles(len(y), splits)
    validation = np.concatenate([valid for _, valid in splits])
    print(f"定数ゼロ OOF MAE {mae(y[validation], np.zeros(len(validation))):.8f}", flush=True)
    evaluation = Evaluation(
        COMP.name[4:],
        "mae",
        False,
        mae,
        files_sha256([DATA]),
        "date-forward-v1",
        array_sha256(roles),
        42,
    )
    store = RunStore(COMP)
    for run_id in methods:
        method = run_id.split("-")[0]
        source = source_for(method)
        started = perf_counter()
        with reserve_cv(store, run_id, source, {}) as (reservation, start):
            x = feature_matrix(rows, method)

            def fit(
                train: np.ndarray, valid: np.ndarray, x: pd.DataFrame = x, run_id: str = run_id
            ) -> FoldOutput:
                model = fit_model(x.iloc[train], y[train])
                prediction = np.asarray(model.predict(x.iloc[valid]))
                print(f"{run_id} 検証開始日 {rows.iloc[valid[0]]['date_id']} 終了", flush=True)
                return FoldOutput(prediction, np.empty(0), {})

            def write(path: Path) -> None:
                np.save(path / "split.npy", roles)

            result = cross_fit_splits(len(y), splits, fit)
            publish_cv_run(
                store,
                reservation,
                evaluation,
                Trial(HYPOTHESES[method], method, "LightGBM", "official-v1", PARAMS),
                y,
                splits,
                result,
                perf_counter() - started,
                write,
                save_query=False,
                gate=(source, start, y.shape),
            )
            print(f"{run_id} 公開完了", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=COMP / "selection-p4.json")
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if any(run_id not in plan.candidates for run_id in args.runs):
        parser.error("選定計画に登録済みの run ID を指定してください")
    run(args.runs)
