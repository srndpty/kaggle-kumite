"""P1 用に、ラベル分布と取引キーの出現状況を確認する。"""

from pathlib import Path

import pandas as pd
from dataset import CARD, DAY, ID, TARGET, TIME

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data = ROOT / "data"
    train = pd.read_csv(data / "train_transaction.csv", usecols=[ID, TIME, TARGET, *CARD])
    test = pd.read_csv(data / "test_transaction.csv", usecols=[ID, TIME, *CARD])
    for name, rows in (("train", train), ("test", test)):
        identity = pd.read_csv(data / f"{name}_identity.csv", usecols=[ID])[ID]
        has_key = rows[CARD].notna().all(axis=1)
        duplicates = rows.loc[has_key, CARD].duplicated(keep=False)
        print(
            name,
            "rows",
            len(rows),
            "unique_ids",
            rows[ID].is_unique,
            "day_range",
            (int(rows[TIME].min() // DAY), int(rows[TIME].max() // DAY)),
            "identity_rate",
            rows[ID].isin(identity).mean(),
            "card_key_rate",
            has_key.mean(),
            "repeated_among_complete",
            duplicates.mean(),
        )
    train["period"] = train[TIME] // (28 * DAY)
    print(train.groupby("period")[TARGET].agg(["size", "mean"]).to_string())


if __name__ == "__main__":
    main()
