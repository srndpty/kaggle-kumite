"""配布 FLAC を FFmpeg で統一した PCM WAV に変換する。"""

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf
from dataset import RATE


def main() -> None:
    data = Path(__file__).resolve().parents[1] / "data"
    originals = [
        path for split in ("train", "test") for path in sorted((data / split).glob("*.flac"))
    ]
    if not originals:
        raise ValueError("配布 FLAC が見つかりません")
    version = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
    (data / "pcm").mkdir(exist_ok=True)
    (data / "pcm" / "decoder.txt").write_text(version + "\n", encoding="utf-8")
    for split in ("train", "test"):
        (data / "pcm" / split).mkdir(exist_ok=True)

    def convert(path: Path) -> None:
        output = data / "pcm" / path.parent.name / f"{path.stem}.wav"
        expected = sf.info(path)
        target_frames = round(expected.frames * RATE / expected.samplerate)
        temp = output.with_suffix(".partial.wav")
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
            or abs(result.frames - target_frames) > 1
        ):
            raise ValueError(f"PCM の形式または長さが不正です: {path}")
        temp.replace(output)

    with ThreadPoolExecutor(max_workers=8) as workers:
        for done, _ in enumerate(workers.map(convert, originals), 1):
            if done % 500 == 0 or done == len(originals):
                print(f"{done}/{len(originals)} recordings", flush=True)


if __name__ == "__main__":
    main()
