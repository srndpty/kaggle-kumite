"""固定した分割で抽出器を比較し、OOF と提出ファイルを保存する。"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from spans import (
    FILES,
    ID,
    SEED,
    TARGET,
    decode,
    fit,
    jaccard,
    load,
    make_folds,
    score_tokens,
    write_submission,
)

from kumite.experiments import ExperimentRecord
from kumite.provenance import array_sha256, code_version, files_sha256
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
TRIALS = {
    "E00": ("採点経路を確認する", "全文"),
    "E01": ("非中立の行で全文より Jaccard が上がる", "語と感情で範囲を抽出"),
    "E02": ("綴りの変形や強調表記を含む行で改善する", "文字 n-gram"),
    "E03": ("否定や強調を含む行で改善する", "前後の語"),
    "E04": ("中立の行で不要な切り詰めが減る", "中立は全文"),
    "E05": ("単語集合の重なりを考慮した区間選択で Jaccard が上がる", "Jaccard を近似する区間選択"),
}


def run(trial: str, run_id: str) -> Path:
    store = RunStore(ROOT)
    data = ROOT / "data"
    with store.reserve(run_id) as reservation:
        start = perf_counter()
        train, test, sample = load(data)
        folds = make_folds(train)
        oof = train["text"].tolist()
        predictions = test["text"].tolist()
        if trial != "E00":
            query_scores = []
            for fold in range(5):
                valid = train.loc[folds == fold]
                trained = fit(train.loc[folds != fold], trial)
                values = decode(valid, score_tokens(valid, trained, trial), trial)
                for i, value in zip(np.flatnonzero(folds == fold), values, strict=True):
                    oof[i] = value
                query_scores.append(score_tokens(test, trained, trial))
            predictions = decode(test, np.mean(query_scores, axis=0), trial)
        scores = np.array([jaccard(a, b) for a, b in zip(train[TARGET], oof, strict=True)])
        hypothesis, change = TRIALS[trial]
        record = ExperimentRecord(
            competition_id="048",
            run_id=run_id,
            metric="word Jaccard",
            greater_is_better=True,
            data_version="official-v1",
            data_hash=files_sha256(data / name for name in FILES),
            split_version="text-group-v1-seed42",
            split_hash=array_sha256(folds),
            seed=SEED,
            hypothesis=hypothesis,
            changed_one_thing=change,
            model="全文" if trial == "E00" else "L2 ロジスティック回帰",
            params={"trial": trial, "solver": "liblinear", "max_iter": 200},
            fold_scores=[float(scores[folds == i].mean()) for i in range(5)],
            oof_score=float(scores.mean()),
            runtime_sec=perf_counter() - start,
            code_version=code_version(ROOT.parents[1]),
        )

        def write(out: Path) -> None:
            frame = pd.DataFrame({ID: train[ID], "fold": folds, "prediction": oof, "score": scores})
            frame.to_csv(out / "oof.csv", index=False)
            write_submission(test, sample, predictions, out / "submission.csv")

        return store.publish(reservation, record, write)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", choices=TRIALS)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(run(args.trial, args.run_id or args.trial))
