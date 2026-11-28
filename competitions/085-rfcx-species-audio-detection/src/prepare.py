"""配布 FLAC を FFmpeg で統一した PCM WAV に変換する。"""

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import soundfile as sf
from dataset import RATE

from kumite.provenance import file_sha256

SETTINGS = {"version": 1, "sample_rate": RATE, "channels": 1, "codec": "pcm_s16le"}


def verified_pcm(data: Path, originals: list[Path]) -> tuple[list[Path], str]:
    """PCM と元 FLAC のハッシュを照合し、来歴の指紋を返す。"""
    manifest = json.loads((data / "pcm/manifest.json").read_text(encoding="utf-8"))
    if any(manifest.get(key) != value for key, value in SETTINGS.items()) or not manifest.get(
        "decoder"
    ):
        raise ValueError("PCM 変換設定が不正です")
    entries = {}
    paths = []
    for original in originals:
        key = original.relative_to(data).as_posix()
        output = data / "pcm" / Path(key).with_suffix(".wav")
        info = sf.info(output)
        actual = {
            "source_sha256": file_sha256(original),
            "output_sha256": file_sha256(output),
            "output_frames": info.frames,
        }
        if (
            info.samplerate != RATE
            or info.channels != 1
            or info.subtype != "PCM_16"
            or actual != manifest["files"].get(key)
        ):
            raise ValueError(f"PCM 来歴が一致しません: {key}")
        entries[key] = actual
        paths.append(output)
    payload = {**SETTINGS, "decoder": manifest["decoder"], "files": entries}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return paths, digest


def prepare(data: Path) -> None:
    originals = [
        path for split in ("train", "test") for path in sorted((data / split).glob("*.flac"))
    ]
    if not originals:
        raise ValueError("配布 FLAC が見つかりません")
    version = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
    (data / "pcm").mkdir(exist_ok=True)
    for split in ("train", "test"):
        (data / "pcm" / split).mkdir(exist_ok=True)

    def convert(path: Path, staging: Path) -> tuple[str, dict[str, str | int]]:
        output = data / "pcm" / path.parent.name / f"{path.stem}.wav"
        expected = sf.info(path)
        target_frames = round(expected.frames * RATE / expected.samplerate)
        temp = staging / f"{path.parent.name}-{path.stem}.wav"
        original_hash = file_sha256(path)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-xerror",
                "-y",
                "-i",
                str(path),
                "-ac",
                "1",
                "-ar",
                str(RATE),
                "-c:a",
                "pcm_s16le",
                str(temp),
            ],
            check=True,
        )
        result = sf.info(temp)
        if (
            result.samplerate != RATE
            or result.channels != 1
            or result.subtype != "PCM_16"
            or abs(result.frames - target_frames) > 1
        ):
            raise ValueError(f"PCM の形式または長さが不正です: {path}")
        if file_sha256(path) != original_hash:
            raise ValueError(f"変換中に FLAC が変更されました: {path}")
        entry = {
            "source_sha256": original_hash,
            "output_sha256": file_sha256(temp),
            "output_frames": result.frames,
        }
        temp.replace(output)
        return path.relative_to(data).as_posix(), entry

    with TemporaryDirectory(dir=data / "pcm") as tmp:
        staging = Path(tmp)
        entries = {}
        with ThreadPoolExecutor(max_workers=8) as workers:
            for done, (key, entry) in enumerate(
                workers.map(lambda path: convert(path, staging), originals), 1
            ):
                entries[key] = entry
                if done % 500 == 0 or done == len(originals):
                    print(f"{done}/{len(originals)} recordings", flush=True)
        receipt = staging / "manifest.json"
        receipt.write_text(
            json.dumps({**SETTINGS, "decoder": version, "files": entries}, sort_keys=True),
            encoding="utf-8",
        )
        receipt.replace(data / "pcm/manifest.json")


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parents[1] / "data")
