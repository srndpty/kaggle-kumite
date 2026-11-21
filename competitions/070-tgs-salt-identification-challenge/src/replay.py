"""保存済み重みから OOF・投票・提出 CSV を再現する。"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from analyze import verify_dataset, verify_split
from models import load_checkpoint, segment
from segmentation import write_submission
from train import ROOT, VARIANTS

from kumite.crossfit import FoldOutput, cross_fit
from kumite.provenance import file_sha256
from kumite.runs import RunStore


def replay(store: RunStore, run_id: str) -> None:
    record = store.record(run_id)
    data, groups, folds = verify_dataset(store.competition_dir / "data", record)
    verify_split(store, run_id, groups, folds)
    infos = json.loads(store.artifact_path(run_id, "fold_info.json").read_text(encoding="utf-8"))
    variant = record.params["variant"]
    if variant not in VARIANTS or record.params["cuda_dtype"] != "bfloat16":
        raise ValueError("記録された推論設定に対応していません")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU 学習時の推論を再現するには CUDA が必要です")
    device = torch.device("cuda")
    torch.backends.cudnn.deterministic = True

    def predict_fold(_train: np.ndarray, valid: np.ndarray) -> FoldOutput:
        fold = int(folds[valid[0]])
        model = load_checkpoint(store.artifact_path(run_id, f"fold-{fold}.pt"), device)
        threshold = float(infos[fold]["threshold"])
        if not 0 <= threshold <= 1:
            raise ValueError(f"fold {fold}: 閾値が不正です")
        return FoldOutput(
            segment(model, data.train[valid], device, threshold, variant),
            segment(model, data.test, device, threshold, variant),
        )

    result = cross_fit(folds, predict_fold)
    for name, actual in (("oof.npy", result.oof), ("test.npy", result.query)):
        expected = np.load(store.artifact_path(run_id, name), mmap_mode="r")
        if not np.array_equal(actual, expected):
            raise ValueError(f"{run_id}: 再推論した {name} が保存値と一致しません")

    with TemporaryDirectory() as temp:
        path = Path(temp) / "submission.csv"
        write_submission(path, data.test_ids, result.query > 0.5)
        expected_csv = store.artifact_path(run_id, "submission.csv")
        if file_sha256(path) != file_sha256(expected_csv):
            raise ValueError(f"{run_id}: 再生成した提出 CSV が一致しません")
    print(f"{run_id}: OOF・テスト投票・提出 CSV が再現しました")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    replay(RunStore(ROOT), parser.parse_args().run_id)
