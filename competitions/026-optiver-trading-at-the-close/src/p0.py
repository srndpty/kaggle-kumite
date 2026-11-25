"""逐次推論 API の採点疎通用 Notebook を作る。"""

import argparse
import json
from pathlib import Path

from kumite.kaggle_notebook import write_cpu_notebook

CPU_IMAGE = (
    "gcr.io/kaggle-images/python@sha256:"
    "287c4e0e224e592dc6113940a6cf3d099b814c7bff0c1e8da57f8e6bad123ac5"
)
SLUG = "optiver-trading-at-the-close"
SOURCE = """import optiver2023

env = optiver2023.make_env()
for _test, _revealed_targets, sample_prediction in env.iter_test():
    sample_prediction['target'] = 0.0
    env.predict(sample_prediction)
"""


def write_notebook(output: Path, owner: str, source: str, *, stage: str) -> None:
    write_cpu_notebook(
        output,
        source,
        kernel_id=f"{owner}/kumite-optiver-close-{stage}",
        title=f"Kumite Optiver Close {stage.upper()}",
        competition=SLUG,
    )
    metadata = output / "kernel-metadata.json"
    settings = json.loads(metadata.read_text(encoding="utf-8"))
    settings["docker_image"] = CPU_IMAGE
    metadata.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_notebook(args.output, args.owner, SOURCE, stage="p0")


if __name__ == "__main__":
    main()
