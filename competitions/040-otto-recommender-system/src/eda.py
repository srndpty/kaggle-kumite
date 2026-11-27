"""固定標本で疑似切断後の正解と過去の人気・露出を確認する。"""

import json
from collections import Counter
from pathlib import Path
from typing import cast

import polars as pl
from evaluation import ACTIONS, Event, split_session, validation_windows

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_EVERY = 100


def summarize(train: Path) -> list[dict[str, object]]:
    data = pl.scan_parquet(train / "*.parquet")
    last_timestamp = int(data.select(pl.col("end").max()).collect().item())
    results: list[dict[str, object]] = []
    for window in validation_windows(last_timestamp):
        history = window.past_week()
        counts = (
            data.filter((pl.col("start") >= history.start) & (pl.col("end") < history.end))
            .select("events")
            .explode("events")
            .unnest("events")
            .group_by("type", "aid")
            .len()
            .collect(engine="streaming")
        )
        known = set(counts["aid"].to_list())
        popular = {
            action: set(
                counts.filter(pl.col("type") == action)
                .sort(["len", "aid"], descending=[True, False])
                .head(20)["aid"]
                .to_list()
            )
            for action in ACTIONS
        }
        sample = data.filter(
            (pl.col("start") >= window.start)
            & (pl.col("end") < window.end)
            & (pl.col("events").list.len() >= 2)
            & (pl.col("session") % SAMPLE_EVERY == 0)
        ).collect(engine="streaming")
        total: Counter[str] = Counter()
        repeat: Counter[str] = Counter()
        unseen: Counter[str] = Counter()
        top20: Counter[str] = Counter()
        for session, events in sample.select("session", "events").iter_rows():
            prefix, truth = split_session(session, cast(list[Event], events))
            seen = {event["aid"] for event in prefix}
            for action in ACTIONS:
                targets = truth[action]
                total[action] += len(targets)
                repeat[action] += len(targets & seen)
                unseen[action] += len(targets - known)
                top20[action] += len(targets & popular[action])
        lengths = sample["events"].list.len()
        results.append(
            {
                "start_ms": window.start,
                "end_ms_exclusive": window.end,
                "sessions": sample.height,
                "median_events": lengths.median(),
                "p90_events": lengths.quantile(0.9),
                "targets": dict(total),
                "prefix_repeat": dict(repeat),
                "absent_from_previous_week": dict(unseen),
                "in_previous_week_top20": dict(top20),
            }
        )
    return results


if __name__ == "__main__":
    print(json.dumps(summarize(ROOT / "data" / "parquet" / "train"), indent=2))
