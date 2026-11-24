"""固定分割で Ridge の比較を実行し、実験ログに追記する。"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from evidence import ROOT, SEED, SPLIT_VERSION, data_digest, inference_sources, source_digests
from inputs import load_data, submission
from model import ALPHA, PatentRidge, pearson
from split import check_split, make_split

from kumite.crossfit import FoldOutput, cross_fit, folds_to_splits
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.provenance import file_sha256
from kumite.runs import RunStore

TRIALS: dict[str, tuple[tuple[str, ...], str, str]] = {
    "E01": ((), "語句の長さと一致率で類似度を予測できる", "Ridge Baseline"),
    "E02": (("char",), "表記揺れの誤差が減る", "文字 TF-IDF コサイン"),
    "E03": (("char", "cpc"), "技術分野でスコア分布が変わる", "CPC の one-hot"),
    "E04": (
        ("char", "cpc", "interaction"),
        "語句一致の効果が技術分野で変わる",
        "CPC と類似度の交互作用",
    ),
    "E05": (
        ("char", "cpc", "interaction", "peer_similarity"),
        "同じ anchor と CPC の他候補から候補集合の類似度傾向を捉えられる",
        "他候補の文字 TF-IDF コサインの平均",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", choices=TRIALS)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    features, hypothesis, change = TRIALS[args.trial]
    data = ROOT / "data"
    data_hash = data_digest(data)
    source_hashes = source_digests(inference_sources(ROOT))
    train, test = load_data(data)
    split_path = ROOT / "outputs" / f"{SPLIT_VERSION}.csv"
    split_path.parent.mkdir(exist_ok=True)
    split = make_split(train)
    if split_path.exists():
        check_split(split_path, split)
    else:
        split.to_csv(split_path, index=False, lineterminator="\n")
    split_hash = file_sha256(split_path)
    folds = split["fold"].to_numpy()
    y = train["score"].to_numpy()
    store = RunStore(ROOT)

    def fit_fold(learn: np.ndarray, valid: np.ndarray) -> FoldOutput:
        model = PatentRidge(features)
        model.fit(train.iloc[learn])
        return FoldOutput(model.predict(train.iloc[valid]), model.predict(test))

    with store.reserve(args.run_id or args.trial) as reservation:
        started = perf_counter()
        result = cross_fit(folds, fit_fold)
        submission(train, result.oof)
        test_submission = submission(test, result.query)
        if (
            data_digest(data) != data_hash
            or file_sha256(split_path) != split_hash
            or source_digests(inference_sources(ROOT)) != source_hashes
        ):
            raise ValueError("CV の実行中に入力または推論ソースが変わりました")

        def write_artifacts(directory: Path) -> None:
            test_submission.to_csv(directory / "submission.csv", index=False)

        publish_cv_run(
            store,
            reservation,
            Evaluation(
                competition_id="054",
                metric="pearson",
                greater_is_better=True,
                data_hash=data_hash,
                split_version=SPLIT_VERSION,
                split_hash=split_hash,
                seed=SEED,
                score=pearson,
            ),
            Trial(
                hypothesis,
                change,
                "Ridge",
                "official",
                {"alpha": ALPHA, "features": features, "source_sha256": source_hashes},
            ),
            y,
            folds_to_splits(folds),
            result,
            perf_counter() - started,
            write_artifacts,
        )


if __name__ == "__main__":
    main()
