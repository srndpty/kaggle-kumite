"""CV・予測検査・証跡を一体として保存する。"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from cv import make_folds
from dataset import COMPETITION_DIR, DATA_FILES, feature_rows, load_data, submission
from features import build_features
from metrics import normalized_gini, scores

from kumite.checks import validate_probabilities
from kumite.crossfit import FoldOutput, cross_fit, folds_to_splits
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.gbdt import lightgbm_binary
from kumite.provenance import array_sha256, files_sha256
from kumite.runs import RunStore

STORE = RunStore(COMPETITION_DIR)


@dataclass(frozen=True)
class Experiment:
    hypothesis: str
    parent: str
    change: str
    prior: bool = False
    missing_nan: bool = False
    missing_count: bool = False
    categorical: bool = True
    weighted: bool = False
    drop_calc: bool = False


EXPERIMENTS = {
    "E00": Experiment("陽性率の定数予測を下限とする", "", "統計的 baseline", prior=True),
    "E01": Experiment("LightGBM は定数より順位性能が良い", "E00", "LightGBM に変更"),
    "E02": Experiment(
        "数値欠損を大小関係から分離すると改善する", "E01", "数値 -1 を NaN に変更", missing_nan=True
    ),
    "E03": Experiment(
        "欠損パターンは予測に寄与する", "E02", "欠損数を追加", missing_nan=True, missing_count=True
    ),
    "E04": Experiment(
        "カテゴリへの順序制約は順位性能を悪化させる",
        "E01",
        "カテゴリを数値として使用",
        categorical=False,
    ),
    "E05": Experiment(
        "重み付けは LogLoss を悪化させても順位を改善する可能性がある",
        "E01",
        "陽性クラス重みを追加",
        weighted=True,
    ),
}


# P4 完了後に追加。1位本人の転載writeupに基づく特徴群のアブレーション。
EXPERIMENTS["E06"] = Experiment(
    "ps_calc_* を除くとノイズ特徴による過学習が減り Gini が改善する",
    "E04",
    "ps_calc_* のみ除外",
    categorical=False,
    drop_calc=True,
)


def run(run_id: str, experiment: str, seed: int = 42) -> None:
    exp = EXPERIMENTS[experiment]
    with STORE.reserve(run_id) as reservation:
        start = time.perf_counter()
        train, test, sample = load_data()
        x, xt = feature_rows(train), feature_rows(test)
        y = train["target"].to_numpy(dtype=np.int64)
        folds, version = make_folds(x, y, seed)
        params = {
            "learning_rate": 0.05,
            "num_leaves": 31,
            "num_boost_round": 5000,
            "early_stopping": 100,
            "seed": seed,
            "num_threads": 8,
            "deterministic": True,
            "force_col_wise": True,
        }

        def fit(tr: np.ndarray, va: np.ndarray) -> FoldOutput:
            if exp.prior:
                p = float(y[tr].mean())
                return FoldOutput(np.full(len(va), p), np.full(len(xt), p))
            frames, cats = build_features(
                x.iloc[tr],
                x.iloc[va],
                xt,
                missing_nan=exp.missing_nan,
                missing_count=exp.missing_count,
                categorical=exp.categorical,
                drop_calc=exp.drop_calc,
            )
            fold_params = dict(params)
            if exp.weighted:
                fold_params["scale_pos_weight"] = float((y[tr] == 0).sum() / (y[tr] == 1).sum())
            out = lightgbm_binary(frames[0], y[tr], frames[1], y[va], frames[2], fold_params, cats)
            print(f"fold {int(folds[va[0]])}: {scores(y[va], out.valid)}", flush=True)
            return FoldOutput(out.valid, out.query, {**out.info, "params": fold_params})

        result = cross_fit(folds, fit)
        for pred in (result.oof, result.query):
            validate_probabilities(np.column_stack((1 - pred, pred)), len(pred), 2)
        splits = folds_to_splits(folds)
        fold_metrics = [scores(y[va], result.oof[va]) for _, va in splits]
        report = {
            "oof": scores(y, result.oof),
            "folds": fold_metrics,
            "mean_fold_gini": float(np.mean([s["normalized_gini"] for s in fold_metrics])),
        }

        def write(out: Path) -> None:
            submission(sample, test["id"].to_numpy(), result.query).to_csv(
                out / "submission.csv", index=False
            )
            pd.DataFrame({"id": train["id"], "fold": folds}).to_csv(out / "folds.csv", index=False)
            (out / "metrics.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        publish_cv_run(
            STORE,
            reservation,
            Evaluation(
                "002",
                "normalized_gini",
                True,
                normalized_gini,
                files_sha256(DATA_FILES),
                version,
                array_sha256(folds),
                seed,
            ),
            Trial(
                exp.hypothesis,
                exp.change,
                "prior" if exp.prior else "lightgbm_binary",
                f"nan={exp.missing_nan}|count={exp.missing_count}|cat={exp.categorical}|drop_calc={exp.drop_calc}",
                {
                    **params,
                    "experiment": experiment,
                    "parent": exp.parent,
                    "weighted": exp.weighted,
                },
            ),
            y,
            splits,
            result,
            time.perf_counter() - start,
            write,
        )
