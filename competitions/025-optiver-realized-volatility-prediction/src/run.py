"""固定分割で実現ボラティリティの特徴を比較する。"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from features import (
    KEYS,
    METHODS,
    PARAMS,
    ROUNDS,
    features,
    fit_predict,
    input_paths,
    load_rows,
    rmspe,
)
from sklearn.model_selection import GroupKFold

from kumite.crossfit import FoldOutput, Split, cross_fit_splits, split_roles
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.paths import REPO_ROOT
from kumite.provenance import array_sha256, files_sha256
from kumite.quality_gates import SelectionPlan, SourceSpec, reserve_cv
from kumite.runs import RunStore

COMP = Path(__file__).resolve().parents[1]
DATA = COMP / "data"
PLAN = SelectionPlan(("E00-r2", "E01-r2", "E02-r2", "E03-r2", "E04"))
SPLIT_VERSION = "time-group-v1"
HYPOTHESES = {
    "E00": "観測窓の板 RV で次の窓を予測する",
    "E01": "直近の板の値動きが持続する",
    "E02": "約定活動が板だけでは捉えられない情報を持つ",
    "E04": "同時刻の板 RV 順位が相対的な変動を捉える",
    "E03": "同時刻の他銘柄が市場共通の変動を捉える",
}


def source_for(method: str) -> SourceSpec:
    relative = COMP.relative_to(REPO_ROOT).as_posix()
    files = [
        p.relative_to(REPO_ROOT).as_posix()
        for p in (COMP / "src").glob("*.py")
        if not p.name.startswith("test_")
    ]
    files.extend(
        p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "src/kumite").glob("*.py")
    )
    files.extend(("uv.lock", "pyproject.toml", f"{relative}/quality-gates.toml"))
    return SourceSpec(
        REPO_ROOT,
        tuple(sorted(files)),
        {"features": METHODS[method], "model": PARAMS, "rounds": ROUNDS},
        import_roots=("src", f"{relative}/src"),
    )


def splits_for(groups: np.ndarray) -> list[Split]:
    return list(GroupKFold(n_splits=5).split(groups, groups=groups))


def data_hash() -> str:
    return files_sha256(
        [
            DATA / "train.csv",
            *input_paths(DATA, "train", "book"),
            *input_paths(DATA, "train", "trade"),
        ]
    )


def run(methods: list[str]) -> None:
    rows = load_rows(DATA, "train")
    y = rows["target"].to_numpy()
    rmspe(y, np.zeros_like(y))
    splits = splits_for(rows["time_id"].to_numpy())
    roles = split_roles(len(y), splits)
    if not (np.sum(roles == 2, axis=0) == 1).all():
        raise ValueError("各行を1回ずつ検証する必要があります")
    evaluation = Evaluation(
        COMP.name[4:],
        "rmspe",
        False,
        rmspe,
        data_hash(),
        SPLIT_VERSION,
        array_sha256(roles),
        42,
    )
    joined = features(DATA, "train", rows)
    if not joined.select(KEYS).equals(rows.select(KEYS)):
        raise ValueError("集約後の行キーが一致しません")
    store = RunStore(COMP)
    for run_id in methods:
        method = run_id.split("-")[0]
        source = source_for(method)
        x = joined.select(METHODS[method]).to_numpy()
        started = perf_counter()
        with reserve_cv(store, run_id, source, {}) as (reservation, start):

            def fit(train: np.ndarray, valid: np.ndarray, x: np.ndarray = x) -> FoldOutput:
                return FoldOutput(fit_predict(x[train], y[train], x[valid]), np.empty(0), {})

            def write(path: Path) -> None:
                np.save(path / "split.npy", roles)

            result = cross_fit_splits(len(y), splits, fit)
            publish_cv_run(
                store,
                reservation,
                evaluation,
                Trial(
                    HYPOTHESES[method],
                    ", ".join(METHODS[method]),
                    "LightGBM",
                    "official-v1",
                    PARAMS,
                ),
                y,
                splits,
                result,
                perf_counter() - started,
                write,
                save_query=False,
                gate=(source, start, y.shape),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("methods", nargs="+", choices=PLAN.candidates)
    run(parser.parse_args().methods)
