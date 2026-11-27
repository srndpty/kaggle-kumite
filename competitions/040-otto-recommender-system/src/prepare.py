"""公式 JSONL をセッション単位の Parquet に変換する。"""

import io
import json
from itertools import islice
from pathlib import Path
from uuid import uuid4

import polars as pl

from kumite.provenance import file_sha256

ROOT = Path(__file__).resolve().parents[1]
BATCH_SESSIONS = 50_000
SCHEMA = {
    "session": pl.UInt32,
    "events": pl.List(pl.Struct({"aid": pl.UInt32, "ts": pl.Int64, "type": pl.String})),
}


def convert(source: Path, destination: Path) -> None:
    """ファイルを分割して変換し、全件成功後に公開する。"""
    digest = file_sha256(source)
    if destination.exists():
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if manifest["source_sha256"] != digest:
            raise ValueError(f"既存 Parquet と正本のハッシュが一致しません: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}-{uuid4().hex}"
    staging.mkdir()
    try:
        session_count = 0
        event_count = 0
        with source.open("rb") as handle:
            for part, lines in enumerate(iter(lambda: list(islice(handle, BATCH_SESSIONS)), [])):
                frame = pl.read_ndjson(io.BytesIO(b"".join(lines)), schema=SCHEMA)
                frame = frame.with_columns(
                    start=pl.col("events").list.first().struct.field("ts"),
                    end=pl.col("events").list.last().struct.field("ts"),
                )
                if frame["session"].null_count() or frame["start"].null_count():
                    raise ValueError(f"セッション ID またはイベントが空です: {source}, part {part}")
                frame.write_parquet(staging / f"{part:05}.parquet")
                session_count += frame.height
                event_count += int(frame["events"].list.len().sum())
        if not session_count:
            raise ValueError(f"セッションがありません: {source}")
        if file_sha256(source) != digest:
            raise ValueError(f"変換中に正本が変わりました: {source}")
        (staging / "manifest.json").write_text(
            json.dumps(
                {
                    "source_sha256": digest,
                    "sessions": session_count,
                    "events": event_count,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        staging.rename(destination)
    finally:
        if staging.exists():
            for part_path in staging.iterdir():
                part_path.unlink()
            staging.rmdir()


if __name__ == "__main__":
    for split in ("train", "test"):
        convert(ROOT / "data" / f"{split}.jsonl", ROOT / "data" / "parquet" / split)
