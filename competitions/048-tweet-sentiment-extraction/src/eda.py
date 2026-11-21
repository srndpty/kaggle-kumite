"""感情別の抽出長と教師データの位置ずれを集計する。"""

import pandas as pd
from spans import FILES, TARGET, group_key, load, position, words
from train import ROOT


def main() -> None:
    data = ROOT / "data"
    raw = pd.read_csv(data / FILES[0], dtype=str, keep_default_na=False)
    train, test, _ = load(data)
    print(f"train={len(raw)}, test={len(test)}, 除外={len(raw) - len(train)}")
    print(
        f"空本文={raw['text'].str.strip().eq('').sum()}, 空正解={raw[TARGET].str.strip().eq('').sum()}"
    )
    print(f"重複本文グループ={train['text'].map(group_key).value_counts().gt(1).sum()}")
    for sentiment, frame in train.groupby("sentiment"):
        answers = frame[TARGET].str.strip()
        texts = frame["text"]
        pairs = list(zip(texts, answers, strict=True))
        missing = sum(answer not in text for text, answer in pairs)
        ambiguous = sum(answer in text and position(text, answer) is None for text, answer in pairs)
        inside_word = 0
        for text, answer in pairs:
            interval = position(text, answer)
            if interval is not None:
                start, end = interval
                tokens = words(text)
                inside_word += int(
                    start not in {token.start() for token in tokens}
                    or end not in {token.end() for token in tokens}
                )
        print(
            sentiment,
            len(frame),
            {
                "全文率": float(texts.str.strip().eq(answers).mean()),
                "抽出語数の中央値": float(
                    pd.Series([len(answer.split()) for answer in answers]).median()
                ),
                "原文不一致": missing,
                "複数一致": ambiguous,
                "語の途中に境界": inside_word,
            },
        )


if __name__ == "__main__":
    main()
