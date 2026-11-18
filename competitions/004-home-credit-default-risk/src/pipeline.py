"""顧客内集約・CV・予測検査・実験保存を一体で実行する。"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from cv import make_folds
from dataset import COMPETITION_DIR, DATA_DIR, DATA_FILES, KEY, load_data, submission
from features import (
    BUREAU_NUMERIC,
    INSTALLMENT_COLUMNS,
    PREVIOUS_NUMERIC,
    application_features,
    bureau_features,
    encode_categories,
    installment_features,
    join_history,
    previous_features,
)
from sklearn.metrics import log_loss, roc_auc_score

from kumite.checks import validate_probabilities
from kumite.crossfit import FoldOutput, cross_fit_splits, folds_to_splits
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.gbdt import lightgbm_binary
from kumite.provenance import array_sha256, files_sha256
from kumite.runs import RunStore

STORE = RunStore(COMPETITION_DIR)


@dataclass(frozen=True)
class Experiment:
    parent: str
    hypothesis: str
    change: str
    history: str = ""
    missing_count: bool = False
    prior: bool = False


EXPERIMENTS = {
    "E00": Experiment("", "陽性率を基準に評価経路を検査する", "学習foldの陽性率", prior=True),
    "E01": Experiment("E00", "申込情報だけで順位を予測できる", "主表のLightGBM"),
    "E02": Experiment("E01", "他社借入量と延滞履歴が有効", "bureau集約を追加", history="bureau"),
    "E03": Experiment(
        "E01", "過去申込の頻度と承認状況が有効", "previous集約を追加", history="previous"
    ),
    "E04": Experiment(
        "E01", "返済遅延と不足額が有効", "installments集約を追加", history="installments"
    ),
    "E05": Experiment("E01", "欠損パターンが有効", "主表の欠損数を追加", missing_count=True),
}

# 自力E02〜E05完了後、2位本人公開READMEの期間別集約に基づいて事前登録。
EXPERIMENTS["E06"] = Experiment(
    "E02",
    "直近の借入状況は全履歴集約に追加情報を持つ",
    "直近365日のbureau集約を追加",
    history="bureau_recent",
)


def parameters(seed: int) -> dict[str, object]:
    config = json.loads((COMPETITION_DIR / "training_config.json").read_text(encoding="utf-8"))
    return {**config["params"], "seed": seed}


def load_history(kind: str) -> pd.DataFrame:
    if kind in ("bureau", "bureau_recent"):
        columns = [
            KEY,
            "SK_ID_BUREAU",
            "DAYS_CREDIT",
            "DAYS_CREDIT_UPDATE",
            "CREDIT_ACTIVE",
            *BUREAU_NUMERIC,
        ]
        raw = pd.read_csv(DATA_DIR / "bureau.csv", usecols=columns)
        history = bureau_features(raw)
        if kind == "bureau_recent":
            history = join_history(history, bureau_features(raw, since=-365))
    elif kind == "previous":
        columns = [
            KEY,
            "SK_ID_PREV",
            "DAYS_DECISION",
            "NAME_CONTRACT_STATUS",
            *PREVIOUS_NUMERIC,
        ]
        history = previous_features(
            pd.read_csv(DATA_DIR / "previous_application.csv", usecols=columns)
        )
    elif kind == "installments":
        history = installment_features(
            pd.read_csv(DATA_DIR / "installments_payments.csv", usecols=INSTALLMENT_COLUMNS)
        )
    else:
        raise ValueError(f"未登録の履歴: {kind}")
    return history


def build_matrix(
    train: pd.DataFrame, test: pd.DataFrame, exp: Experiment
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if exp.history:
        history = load_history(exp.history)
        train, test = join_history(train, history), join_history(test, history)
    x, xt = application_features(train), application_features(test)
    if exp.missing_count:
        for frame in (x, xt):
            frame["application_missing_count"] = frame.isna().sum(axis=1)
    return x, xt


def auc(y: np.ndarray, pred: np.ndarray) -> float:
    return float(roc_auc_score(y, pred))


def run(run_id: str, experiment: str, seed: int = 42, benchmark: bool = False) -> Path:
    exp = EXPERIMENTS[experiment]
    with STORE.reserve(run_id) as reservation:
        start = time.perf_counter()
        train, test, sample = load_data()
        y = train["TARGET"].to_numpy(dtype=np.int64)
        folds, version = make_folds(train, y, seed)
        splits = folds_to_splits(folds)
        if benchmark:
            splits = splits[:1]
            version = "benchmark-fold0-" + version
        x, xt = build_matrix(train, test, exp)
        params = parameters(seed)

        def fit(tr: np.ndarray, va: np.ndarray) -> FoldOutput:
            if exp.prior:
                p = float(y[tr].mean())
                return FoldOutput(np.full(len(va), p), np.full(len(xt), p))
            frames, cats = encode_categories(x.iloc[tr], x.iloc[va], xt)
            out = lightgbm_binary(frames[0], y[tr], frames[1], y[va], frames[2], params, cats)
            print(
                f"fold {int(folds[va[0]])}: AUC={auc(y[va], out.valid):.6f} {out.info}", flush=True
            )
            return out

        result = cross_fit_splits(len(y), splits, fit)
        validated = np.sort(np.concatenate([va for _, va in splits]))
        for pred in (result.oof[validated], result.query):
            validate_probabilities(np.column_stack((1 - pred, pred)), len(pred), 2)
        fold_scores = [auc(y[va], result.oof[va]) for _, va in splits]
        metrics = {
            "mean_fold_auc": float(np.mean(fold_scores)),
            "oof_auc": auc(y[validated], result.oof[validated]),
            "oof_logloss": float(log_loss(y[validated], result.oof[validated])),
            "fold_logloss": [float(log_loss(y[va], result.oof[va])) for _, va in splits],
        }

        def write(out: Path) -> None:
            submission(sample, test[KEY].to_numpy(), result.query).to_csv(
                out / "submission.csv", index=False
            )
            pd.DataFrame({KEY: train[KEY], "fold": folds}).to_csv(out / "folds.csv", index=False)
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            schema = {
                "features": list(x.columns),
                "definition": asdict(exp),
                "lightgbm": lgb.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            }
            (out / "features.json").write_text(
                json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        return publish_cv_run(
            STORE,
            reservation,
            Evaluation(
                "004",
                "roc_auc",
                True,
                auc,
                files_sha256(DATA_FILES),
                version,
                array_sha256(np.stack([np.isin(np.arange(len(y)), va) for _, va in splits])),
                seed,
            ),
            Trial(
                exp.hypothesis,
                exp.change,
                "prior" if exp.prior else "lightgbm_binary",
                f"application-v1|history={exp.history}|missing_count={exp.missing_count}",
                {
                    **params,
                    "experiment": experiment,
                    "definition": asdict(exp),
                    "benchmark": benchmark,
                },
            ),
            y,
            splits,
            result,
            time.perf_counter() - start,
            write,
        )
