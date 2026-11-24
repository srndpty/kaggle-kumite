"""凍結 ResNet-18 で公式画像から L2 正規化済みの埋め込みを作る。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


class Images(Dataset[torch.Tensor]):
    def __init__(self, directory: Path, names: list[str]) -> None:
        self.directory = directory
        self.names = names
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.directory / self.names[index]) as source:
            return self.transform(source.convert("RGB"))


def embed(directory: Path, names: list[str], weights: Path) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = resnet18()
    backbone.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model = torch.nn.Sequential(*list(backbone.children())[:-1], torch.nn.Flatten())
    model = model.eval().to(device)
    loader = DataLoader(
        Images(directory, names), batch_size=96, num_workers=min(6, len(names) // 96)
    )
    with torch.inference_mode():
        return np.concatenate(
            [
                torch.nn.functional.normalize(model(batch.to(device)), dim=1).cpu().numpy()
                for batch in loader
            ]
        )


def main() -> None:
    from kumite.provenance import file_sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source_hash = file_sha256(Path(__file__))
    names = sorted(pd.read_csv(args.csv, usecols=["image"])["image"].unique().tolist())
    values = embed(args.images, names, args.weights)
    if file_sha256(Path(__file__)) != source_hash:
        raise ValueError("埋め込み生成中に画像ソースが変更されました")
    np.savez(args.output, names=np.array(names), values=values)
    manifest = {
        "weights_sha256": file_sha256(args.weights),
        "csv_sha256": file_sha256(args.csv),
        "embeddings_sha256": file_sha256(args.output),
        "images_source_sha256": source_hash,
        "preprocessing": repr(ResNet18_Weights.IMAGENET1K_V1.transforms()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
