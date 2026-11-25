"""OpenVaccine の CV run を、共通 Quality Gate を通して実行・公開する。"""

import argparse
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from data import (
    BPPS,
    CLUSTER_JACCARD,
    COMPETITION,
    KMER,
    N_FOLDS,
    SCORED_INDEX,
    TARGETS,
    TEST_JSON,
    TRAIN_JSON,
    TRAIN_SCORED,
    Labels,
    Molecules,
    cluster_splits,
    mcrmse,
    read_test,
    read_train,
    sample_rows,
)
from models import (
    GRUConfig,
    TrainGroup,
    fit_predict_gru,
    fit_predict_lightgbm,
    position_features,
    window_frame,
)
from sklearn.model_selection import KFold

from kumite.crossfit import FitFold, FoldOutput, Split, cross_fit_splits, split_roles
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.paths import REPO_ROOT
from kumite.provenance import array_sha256, files_sha256
from kumite.quality_gates import SourceSpec, reserve_cv
from kumite.runs import RunStore

COMPETITION_ID = "stanford-covid-vaccine"
METHODS_FILE = Path(__file__).with_name("methods.toml")
SRC = COMPETITION.relative_to(REPO_ROOT).as_posix() + "/src"
SOURCE_FILES = (
    f"{SRC}/data.py",
    f"{SRC}/models.py",
    f"{SRC}/run.py",
    f"{SRC}/final.py",
    "src/kumite/__init__.py",
    "src/kumite/crossfit.py",
    "src/kumite/cv_run.py",
    "src/kumite/experiments.py",
    "src/kumite/gbdt.py",
    "src/kumite/paths.py",
    "src/kumite/provenance.py",
    "src/kumite/quality_gates.py",
    "src/kumite/runs.py",
    "src/kumite/source_audit.py",
    "pyproject.toml",
    "uv.lock",
)
IMPORT_ROOTS = ("src", SRC)

WINDOW = 4
TAU = 0.3  # 逆分散重みの下限側の残差。実験前に固定（調整しない）
# 疑似ラベルのバッチの損失に掛ける係数（正規化後の損失の 0.5 倍）。E07 では要素重みに掛けたため
# バッチ内の正規化で相殺され、実効的に 1 だった（README の P6 を参照）。
PSEUDO_WEIGHT = 0.5
LIGHTGBM = {
    "objective": "regression",
    "num_boost_round": 600,
    "early_stopping": None,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.5,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 16,
}
GRU = {
    "hidden": 128,
    "layers": 2,
    "dropout": 0.2,
    "epochs": 60,
    "batch_size": 64,
    "learning_rate": 2e-3,
    "weight_decay": 1e-4,
}


@dataclass(frozen=True)
class Method:
    hypothesis: str
    changed_one_thing: str
    model: str
    split: str
    use_bpp: bool
    weights: str
    targets: str
    seed: int
    pseudo_label: bool = False

    def __post_init__(self) -> None:
        allowed = {
            "model": ("lightgbm", "gru"),
            "split": ("cluster", "random"),
            "weights": ("all", "sn_filter", "inverse_variance"),
            "targets": ("scored", "all"),
        }
        for name, values in allowed.items():
            if getattr(self, name) not in values:
                raise ValueError(f"{name} は {values} のいずれかです: {getattr(self, name)}")
        if self.pseudo_label and self.model != "gru":
            raise ValueError("疑似ラベルは GRU だけで使えます")

    def settings(self) -> dict[str, Any]:
        """CV と export で共有する実効設定。"""
        model = {"lightgbm": {**LIGHTGBM, "seed": self.seed, "window": WINDOW}}.get(
            self.model, {**GRU, "seed": self.seed}
        )
        return {
            "method": asdict(self),
            "model_params": model,
            "split": {"n_folds": N_FOLDS, "kmer": KMER, "cluster_jaccard": CLUSTER_JACCARD},
            "tau": TAU,
            "pseudo_weight": PSEUDO_WEIGHT,
        }

    def split_version(self) -> str:
        if self.split == "cluster":
            return f"kmer{KMER}-jaccard{CLUSTER_JACCARD}-groupkfold{N_FOLDS}-v1"
        return f"random-kfold{N_FOLDS}-seed42-v1"

    def target_index(self) -> list[int]:
        return list(SCORED_INDEX) if self.targets == "scored" else list(range(len(TARGETS)))


def load_method(method_id: str, path: Path = METHODS_FILE) -> Method:
    methods = tomllib.loads(path.read_text(encoding="utf-8"))
    if method_id not in methods:
        raise ValueError(f"未定義の手法です: {method_id}")
    return Method(**methods[method_id])


def source_spec(method: Method) -> SourceSpec:
    return SourceSpec(REPO_ROOT, SOURCE_FILES, method.settings(), IMPORT_ROOTS)


def data_hash() -> str:
    return files_sha256([TRAIN_JSON, TEST_JSON, *sorted(BPPS.glob("*.npy"))])


def make_splits(method: Method, molecules: Molecules) -> list[Split]:
    if method.split == "cluster":
        return cluster_splits(molecules.sequence)
    folds = KFold(N_FOLDS, shuffle=True, random_state=42).split(np.zeros(len(molecules)))
    return [(np.sort(train), np.sort(valid)) for train, valid in folds]


def training_weights(method: Method, labels: Labels) -> np.ndarray:
    """(分子, 採点位置, 学習する目的) の重み。"""
    error = labels.error[:, :, method.target_index()]
    if method.weights == "all":
        return np.ones_like(error)
    if method.weights == "sn_filter":
        return np.broadcast_to(labels.sn_filter[:, None, None], error.shape).astype(np.float64)
    return 1.0 / (error**2 + TAU**2)


def to_five(predicted: np.ndarray, method: Method) -> np.ndarray:
    """提出用の 5 列。採点 3 列だけ学習した場合、採点外の 2 列は同条件の Mg ありの予測で埋める。"""
    if method.targets == "all":
        return predicted
    reactivity, mg_ph10, mg_50c = np.moveaxis(predicted, -1, 0)
    return np.stack([reactivity, mg_ph10, mg_ph10, mg_50c, mg_50c], axis=-1)


def scored_of(predicted: np.ndarray, method: Method) -> np.ndarray:
    return predicted if method.targets == "scored" else predicted[..., list(SCORED_INDEX)]


def fold_fitter(
    method: Method,
    molecules: Molecules,
    labels: Labels,
    test: Molecules,
) -> FitFold:
    """fold ごとに学習し、検証 (分子, 68, 3) と、サンプル順の test (行, 5) を返す関数を作る。"""
    features = position_features(molecules, method.use_bpp)
    test_features = position_features(test, method.use_bpp)
    y = labels.y[:, :, method.target_index()]
    weight = training_weights(method, labels)
    settings = method.settings()["model_params"]

    def per_molecule(flat: np.ndarray, lengths: list[int]) -> list[np.ndarray]:
        return np.split(flat, np.cumsum(lengths)[:-1])

    def fit(train: np.ndarray, valid: np.ndarray) -> FoldOutput:
        if method.model == "lightgbm":
            params = {k: v for k, v in settings.items() if k != "window"}
            x_train = window_frame([features[i] for i in train], TRAIN_SCORED, WINDOW)
            x_valid = window_frame([features[i] for i in valid], TRAIN_SCORED, WINDOW)
            x_test = window_frame(test_features, None, WINDOW)
            valid_flat, test_flat = fit_predict_lightgbm(
                x_train,
                y[train].reshape(-1, y.shape[2]),
                weight[train].reshape(-1, y.shape[2]),
                (x_valid, x_test),
                params,
            )
            valid_pred = valid_flat.reshape(len(valid), TRAIN_SCORED, -1)
            test_pred = per_molecule(test_flat, [test.length(i) for i in range(len(test))])
        else:
            config = GRUConfig(**settings)
            lengths = sorted({test.length(i) for i in range(len(test))})
            groups = [[i for i in range(len(test)) if test.length(i) == n] for n in lengths]
            test_x = [np.stack([test_features[i] for i in g]) for g in groups]
            labeled = TrainGroup(np.stack([features[i] for i in train]), y[train], weight[train])
            training = [labeled]
            if method.pseudo_label:
                # 教師はこの fold の学習行だけで学習する（検証ラベルを疑似ラベルに混ぜない）
                teacher = fit_predict_gru([labeled], test_x, config)
                training += [
                    TrainGroup(x, p, np.ones(p.shape), loss_scale=PSEUDO_WEIGHT)
                    for x, p in zip(test_x, teacher, strict=True)
                ]
            valid_out, *test_out = fit_predict_gru(
                training, [np.stack([features[i] for i in valid]), *test_x], config
            )
            valid_pred = valid_out[:, :TRAIN_SCORED]
            test_pred: list[np.ndarray] = [np.empty(0)] * len(test)
            for g, out in zip(groups, test_out, strict=True):
                for row, i in zip(out, g, strict=True):
                    test_pred[i] = row
        scored = scored_of(valid_pred, method)
        query = sample_rows(test, [to_five(p, method) for p in test_pred])
        truth = labels.scored()[valid]
        print(f"fold MCRMSE {mcrmse(truth, scored):.5f}", flush=True)
        return FoldOutput(scored, query, {"n_train": len(train), "n_valid": len(valid)})

    return fit


def run(method_id: str, run_id: str) -> None:
    method = load_method(method_id)
    store = RunStore(COMPETITION)
    source = source_spec(method)
    with reserve_cv(store, run_id, source, {}) as (reservation, start):
        started = perf_counter()
        digest = data_hash()
        molecules, labels = read_train()
        test = read_test()
        splits = make_splits(method, molecules)
        roles = split_roles(len(molecules), splits)
        y = labels.scored()
        result = cross_fit_splits(
            len(molecules), splits, fold_fitter(method, molecules, labels, test)
        )
        evaluation = Evaluation(
            COMPETITION_ID,
            "mcrmse",
            False,
            mcrmse,
            digest,
            method.split_version(),
            array_sha256(roles),
            method.seed,
        )
        trial = Trial(
            method.hypothesis,
            method.changed_one_thing,
            method.model,
            "bpp" if method.use_bpp else "tokens",
            method.settings(),
        )

        def write(path: Path) -> None:
            np.save(path / "split.npy", roles)

        publish_cv_run(
            store,
            reservation,
            evaluation,
            trial,
            y,
            splits,
            result,
            perf_counter() - started,
            write,
            gate=(source, start, y.shape),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run(args.method, args.run_id or args.method)
