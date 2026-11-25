"""イベント採点と欠番処理の回帰テスト。"""

import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import polars as pl
import pytest
from metric import TOLERANCES, score
from pipeline import EVENTS, bins, candidates, limit_candidates, matrix, submission
from polars.testing import assert_frame_equal
from train import split


def fixture() -> tuple[pl.DataFrame, pl.DataFrame]:
    truth = pl.DataFrame(
        {
            "series_id": ["a"] * 5,
            "step": [100, 130, 500, 550, None],
            "event": ["onset", "onset", "wakeup", "wakeup", "wakeup"],
        }
    )
    predicted = pl.DataFrame(
        {
            "series_id": ["a"] * 5,
            "step": [100, 101, 142, 500, 750],
            "event": ["onset", "onset", "onset", "wakeup", "wakeup"],
            "score": [0.9, 0.8, 0.8, 0.9, 0.7],
        }
    )
    return truth, predicted


def test_boundary_duplicate_tie_and_empty() -> None:
    truth, predicted = fixture()
    detail = score(truth, predicted)
    assert detail["onset/12"] == pytest.approx(0.5)
    assert detail["onset/36"] == pytest.approx(0.5 + 1 / 3)
    assert detail["wakeup/12"] == pytest.approx(0.5)
    assert set(score(truth, predicted.clear()).values()) == {0}


def load_official_metric(reference: Path) -> ModuleType:
    if not reference.is_file():
        message = "Kaggle 公式メトリックの照合用コードが未取得"
        if os.environ.get("KUMITE_REQUIRE_OFFICIAL_METRIC") == "1":
            pytest.fail(message)
        pytest.skip(message)
    official = ModuleType("official_event_metric")
    # pandas 3 は apply にグループキー列を渡さない。
    source = reference.read_text(encoding="utf-8").replace(
        "class_counts[group[event_column_name].iat[0]]", "class_counts[group.name[0]]"
    )
    exec(compile(source, str(reference), "exec"), official.__dict__)
    return official


def test_official_metric_parity() -> None:
    reference = Path(__file__).resolve().parents[1] / "outputs/reference/metric.txt"
    official = load_official_metric(reference)
    truth, predicted = fixture()
    expected = official.score(
        pd.DataFrame(truth.drop_nulls("step").to_dict(as_series=False)),
        pd.DataFrame(predicted.to_dict(as_series=False)),
        {event: list(TOLERANCES) for event in EVENTS},
        "series_id",
        "step",
        "event",
        "score",
    )
    assert np.mean(list(score(truth, predicted).values())) == pytest.approx(expected)


def test_gap_and_submission(tmp_path: Path) -> None:
    path = tmp_path / "series.parquet"
    pl.DataFrame(
        {
            "series_id": ["a"] * 4,
            "step": [0, 1, 12, 13],
            "anglez": [0.0, 0.0, 100.0, 100.0],
            "enmo": [0.0] * 4,
        }
    ).write_parquet(path)
    frame = bins(path)
    assert frame["segment"].to_list() == [1, 2]
    np.testing.assert_array_equal(matrix(frame, False)[:, 2], [0, 100])
    probabilities = frame.with_columns(*(pl.lit(0.5).alias(event) for event in EVENTS))
    result = candidates(probabilities, 0)
    assert submission(result, path)["row_id"].to_list() == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="Unknown"):
        submission(result.with_columns(pl.lit(99).alias("step")), path)


def test_split_assigns_each_series_once() -> None:
    ids = ["a", "b", "c", "d", "e", "f"]
    frame = pl.DataFrame({"series_id": ids * 3})
    assignments = split(frame)
    assert assignments["series_id"].to_list() == ids
    assert assignments["fold"].value_counts()["count"].to_list() == [2, 2, 2]
    assert assignments.equals(split(frame.reverse()))


def test_candidate_limit_keeps_best_per_event() -> None:
    events = pl.DataFrame(
        {
            "series_id": ["a"] * 20,
            "step": list(range(10)) * 2,
            "event": ["onset"] * 10 + ["wakeup"] * 10,
            "score": [i / 10 for i in range(10)] * 2,
        }
    )
    result = limit_candidates(events, events)
    assert result.height == 16
    assert result["step"].min() == 2
    assert result["step"].max() == 9


def test_shuffled_input_and_duplicate_steps(tmp_path: Path) -> None:
    path = tmp_path / "series.parquet"
    raw = pl.DataFrame(
        {
            "series_id": ["a"] * 24 + ["b"] * 24,
            "step": list(range(24)) * 2,
            "anglez": list(range(48)),
            "enmo": [0.1] * 48,
            "timestamp": ["2018-08-14T22:26:00-0400"] * 48,
        }
    )
    raw.write_parquet(path)
    ordered = bins(path, clock=True)
    raw.sample(fraction=1, shuffle=True, seed=42).write_parquet(path)
    shuffled = bins(path, clock=True)
    # polars の group_by はグループ内の行順を保証せず、平均の足し算の順で末尾ビットが変わる
    # （0.1 の平均が 0.09999999999999999 と 0.10000000000000002 になるなど）。
    # 推論ソースは提出済みの inference.sha256 と照合するため本体は変えず、丸め誤差の範囲で比べる。
    # 整数・文字列の列（ID・区切り・時刻）は完全一致を要求する。
    assert_frame_equal(ordered, shuffled, rel_tol=1e-12, abs_tol=0.0)
    exact = [c for c, dtype in ordered.schema.items() if not dtype.is_float()]
    assert ordered.select(exact).equals(shuffled.select(exact))
    # 一定値の窓の標準偏差では、1e-17 の差が平方根で 1e-9 程度に増幅され、片側が 0 になる。
    # 特徴量の尺度（enmo 約 0.1、anglez 数十）に比べて十分小さい絶対許容差で比べる。
    np.testing.assert_allclose(
        matrix(ordered, True, True), matrix(shuffled, True, True), rtol=1e-12, atol=1e-8
    )

    def detect(frame: pl.DataFrame) -> pl.DataFrame:
        return candidates(
            frame.with_columns(*((pl.col("step") / 24).alias(event) for event in EVENTS)), 15
        )

    assert detect(ordered).equals(detect(shuffled))
    pl.concat([raw, raw.head(1)]).write_parquet(path)
    with pytest.raises(ValueError, match="Duplicate step"):
        bins(path)


def test_candidate_limit_counts_elapsed_span() -> None:
    frame = pl.DataFrame({"series_id": ["a", "a"], "step": [0, 17280 * 4]})
    events = pl.DataFrame(
        {
            "series_id": ["a"] * 50,
            "step": list(range(50)),
            "event": ["onset"] * 50,
            "score": [i / 50 for i in range(50)],
        }
    )
    result = limit_candidates(events, frame)
    assert result.height == 40
    assert result["step"].min() == 10


def test_required_official_metric_rejects_missing_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KUMITE_REQUIRE_OFFICIAL_METRIC", "1")
    with pytest.raises(pytest.fail.Exception, match="Kaggle"):
        load_official_metric(tmp_path / "missing.txt")
