"""同一 JPEG をまとめた固定3 fold を生成する。"""

from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

SEED = 42
SPLIT_VERSION = "v1"


def make_folds(data: Path, labels: dict[str, dict[int, str]]) -> dict[str, int]:
    groups: dict[str, list[str]] = defaultdict(list)
    for image_id in sorted(labels):
        content = (data / "train_images" / image_id).read_bytes()
        groups[sha256(content).hexdigest()].append(image_id)
    image_groups = list(groups.values())
    strata = [
        tuple(sorted({class_id for image_id in group for class_id in labels[image_id]}))
        for group in image_groups
    ]
    counts = Counter(strata)
    keys = [str(classes) if counts[classes] >= 3 else "rare" for classes in strata]
    if 0 < keys.count("rare") < 3:
        common_positive = str(max((key for key in counts if key), key=counts.__getitem__))
        keys = [common_positive if key == "rare" else key for key in keys]
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    return {
        image_id: fold
        for fold, (_, validation) in enumerate(splitter.split(image_groups, keys))
        for index in validation
        for image_id in image_groups[index]
    }
