"""CV の結果を採点し、成果物と実験ログを保存する（コンペに依らない部分）。

コンペ側は、評価条件（`Evaluation`）、何を試したか（`Trial`）、提出ファイルの書き方だけを渡す。
予測の検査（`kumite.checks`）は、呼ぶ前にコンペ側で済ませる。
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kumite.crossfit import CrossFitResult, Split
from kumite.experiments import ExperimentRecord
from kumite.paths import REPO_ROOT
from kumite.provenance import code_version, peak_memory_mb
from kumite.quality_gates import CVStart, SourceSpec, dense_oof, publish_gated_cv
from kumite.runs import Reservation, RunStore


@dataclass(frozen=True)
class Evaluation:
    """何で評価するか。同じコンペ・同じ分割の run で共通。"""

    competition_id: str
    metric: str
    greater_is_better: bool
    score: Callable[[np.ndarray, np.ndarray], float]  # (目的変数, 予測) -> スコア
    data_hash: str
    split_version: str
    split_hash: str
    seed: int


@dataclass(frozen=True)
class Trial:
    """何を試したか。"""

    hypothesis: str
    changed_one_thing: str
    model: str
    data_version: str
    params: dict[str, Any] = field(default_factory=dict[str, Any])
    inputs: dict[str, str] = field(default_factory=dict[str, str])  # 使った他 run の成果物


def publish_cv_run(
    store: RunStore,
    reservation: Reservation,
    evaluation: Evaluation,
    trial: Trial,
    y: np.ndarray,
    splits: Sequence[Split],
    result: CrossFitResult,
    runtime_sec: float,
    write_artifacts: Callable[[Path], None],
    *,
    save_query: bool = True,
    gate: tuple[SourceSpec, CVStart, tuple[int, ...]] | None = None,
) -> Path:
    """分割ごと・検証行全体を採点し、`oof.npy`・`fold_info.json` と追加成果物を保存する。

    全体のスコアは、いずれかの分割で検証された行だけで計算する。
    `test.npy` はクエリ予測を生成した実験で保存する。
    """
    fold_scores = [evaluation.score(y[valid], result.oof[valid]) for _, valid in splits]
    validated = np.sort(np.concatenate([valid for _, valid in splits]))
    oof_score = evaluation.score(y[validated], result.oof[validated])
    for k, (score, info) in enumerate(zip(fold_scores, result.infos, strict=True)):
        print(f"fold {k}: {score:.5f} {info}")
    print(
        f"OOF {evaluation.metric}: {oof_score:.5f}（fold 標準偏差 {np.std(fold_scores):.5f}、"
        f"検証 {len(validated)} / {len(y)} 行）"
    )

    record = ExperimentRecord(
        competition_id=evaluation.competition_id,
        run_id=reservation.run_id,
        metric=evaluation.metric,
        greater_is_better=evaluation.greater_is_better,
        data_version=trial.data_version,
        data_hash=evaluation.data_hash,
        split_version=evaluation.split_version,
        split_hash=evaluation.split_hash,
        seed=evaluation.seed,
        hypothesis=trial.hypothesis,
        changed_one_thing=trial.changed_one_thing,
        model=trial.model,
        params=trial.params,
        fold_scores=fold_scores,
        oof_score=oof_score,
        runtime_sec=runtime_sec,
        peak_memory_mb=peak_memory_mb(),
        code_version=code_version(REPO_ROOT),
        inputs=trial.inputs,
    )

    def write(out_dir: Path) -> None:
        np.save(out_dir / "oof.npy", result.oof)
        if save_query:
            np.save(out_dir / "test.npy", result.query)
        (out_dir / "fold_info.json").write_text(json.dumps(result.infos), encoding="utf-8")
        write_artifacts(out_dir)

    if gate is None:
        final = store.publish(reservation, record, write)
    else:
        source, before_training, expected_oof_shape = gate
        assess = dense_oof(
            y,
            splits,
            evaluation.score,
            expected_shape=expected_oof_shape,
            data_hash=evaluation.data_hash,
        )
        if save_query and not np.isfinite(result.query).all():
            raise ValueError("クエリ予測に NaN または Inf があります")
        final = publish_gated_cv(
            store,
            reservation,
            record,
            write,
            source=source,
            before_training=before_training,
            assess=assess,
        )
    print(f"出力: {final}")
    return final
