"""共通 Quality Gate を通して、7日窓の時系列 CV を公開する。"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
from data import COMP, FILES, LAST_DAY, load_tables
from evaluation import (
    assessor,
    candidate_metrics,
    scores,
    split_hash,
    validation_windows,
    window_ap,
)
from model import METHODS, PARAMS, recommend, settings

from kumite.experiments import ExperimentRecord
from kumite.paths import REPO_ROOT
from kumite.provenance import array_sha256, code_version, files_sha256, peak_memory_mb
from kumite.quality_gates import SelectionPlan, SourceSpec, publish_gated_cv, reserve_cv
from kumite.runs import RunStore

HYPOTHESES = {
    "E00": ("直近7日の人気上位を全員に薦める", "Baseline"),
    "E01": ("本人の直近28日の購入を先頭に置くと MAP@12 が上がる", "本人の購入履歴を追加"),
    "E02": ("人気を半減期3日で減衰させると MAP@12 が小幅に上がる", "人気の集計方法"),
    "E03": (
        "直近購入と一緒に買われた商品を足すと候補 Recall は上がるが MAP@12 は横ばい",
        "共購入の候補を追加",
    ),
    "E04": ("候補を固定して LambdaRank で並べ替えると MAP@12 が大きく上がる", "並べ替えモデル"),
    "E05": ("学習週を1週から4週に増やすと MAP@12 が上がる", "学習週の数"),
    "E06": ("本人の購入を365日さかのぼると、直近に購入のない顧客で改善する", "本人の購入の期間"),
    "E07": ("購入した商品と同じ製品の別商品を候補に足すと改善する", "同じ製品の候補を追加"),
    "E08": ("候補を100件から300件に増やすと Recall が上がり MAP@12 も小幅に上がる", "候補数"),
    "E10": (
        "候補300件のまま教師の顧客を E04 と揃えると、E08 の悪化の一部が戻る",
        "教師の顧客（候補の先頭100件に正解がある顧客）",
    ),
    "E09": ("E06・E07・E08 の組み合わせが各単独より高い", "E06・E07・E08 の組み合わせ"),
}


def load_plan(path: Path) -> SelectionPlan:
    value = json.loads(path.read_text(encoding="utf-8"))
    return SelectionPlan(tuple(value["candidates"]), value["tolerance"])


def method_name(run_id: str) -> str:
    return run_id.split("-")[0]


def source_for(name: str) -> SourceSpec:
    relative = COMP.relative_to(REPO_ROOT).as_posix()
    own = ("data", "evaluation", "candidates", "features", "model", "run", "final")
    shared = (
        "__init__",
        "crossfit",
        "experiments",
        "paths",
        "provenance",
        "quality_gates",
        "ranking",
        "runs",
        "source_audit",
    )
    return SourceSpec(
        REPO_ROOT,
        (
            *(f"{relative}/src/{module}.py" for module in own),
            *(f"src/kumite/{module}.py" for module in shared),
            f"{relative}/quality-gates.toml",
            "pyproject.toml",
            "uv.lock",
        ),
        settings(name),
        import_roots=("src", f"{relative}/src"),
    )


def run(run_ids: list[str], plan: SelectionPlan) -> None:
    tables = load_tables()
    windows = validation_windows(tables.transactions, LAST_DAY)
    data_hash = files_sha256(FILES)
    assess = assessor(windows, tables.n_articles, data_hash)
    store = RunStore(COMP)
    for run_id in run_ids:
        name = method_name(run_id)
        source = source_for(name)
        started = perf_counter()
        with reserve_cv(store, run_id, source, {}, plan) as (reservation, start):
            ranked: list[np.ndarray] = []
            diagnostics: list[dict[str, object]] = []
            for window in windows:
                result = recommend(tables, METHODS[name], window.start, window.customers)
                ranked.append(result.top.astype(np.int32))
                gain = {} if result.model is None else _gain(result.model)
                diagnostics.append(
                    {
                        "start": window.start,
                        "customers": len(window.customers),
                        "candidates_sha256": array_sha256(result.pool),
                        **candidate_metrics(window, result.pool, tables.n_articles),
                        "training": result.training,
                        "gain": gain,
                    }
                )
                print(
                    f"{run_id} 開始日 {window.start} MAP@12 "
                    f"{window_ap(window, result.top, tables.n_articles).mean():.5f} {diagnostics[-1]}",
                    flush=True,
                )
            oof = np.concatenate(ranked)
            fold_scores, oof_score = scores(windows, oof, tables.n_articles)
            hypothesis, change = HYPOTHESES[name]
            record = ExperimentRecord(
                competition_id=COMP.name[4:],
                run_id=run_id,
                metric="map@12",
                greater_is_better=True,
                data_version="official-v1",
                data_hash=data_hash,
                split_version="weekly-v1",
                split_hash=split_hash(windows),
                seed=PARAMS["seed"],
                hypothesis=hypothesis,
                changed_one_thing=change,
                model="LightGBM LambdaRank" if METHODS[name].label_weeks else "rule",
                params=settings(name),
                fold_scores=fold_scores,
                oof_score=oof_score,
                runtime_sec=perf_counter() - started,
                peak_memory_mb=peak_memory_mb(),
                code_version=code_version(REPO_ROOT),
            )

            def write(
                path: Path, oof: np.ndarray = oof, items: list[dict[str, object]] = diagnostics
            ) -> None:
                np.save(path / "oof.npy", oof)
                (path / "diagnostics.json").write_text(json.dumps(items), encoding="utf-8")

            publish_gated_cv(
                store,
                reservation,
                record,
                write,
                source=source,
                before_training=start,
                assess=assess,
            )
        print(f"{run_id} 公開完了 OOF {oof_score:.6f}", flush=True)


def _gain(model: lgb.Booster) -> dict[str, float]:
    values = model.feature_importance("gain")
    names = model.feature_name()
    return {n: float(v) for n, v in sorted(zip(names, values, strict=True), key=lambda x: -x[1])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=COMP / "selection-p4.json")
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if any(run_id not in plan.candidates for run_id in args.runs):
        parser.error("選定計画に登録済みの run ID を指定してください")
    run(args.runs, plan)
