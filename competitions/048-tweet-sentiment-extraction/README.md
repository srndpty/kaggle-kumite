# #048 Tweet Sentiment Extraction

深さは **Standard**。感情ラベルが与えられたツイートから、その感情を表す原文の連続区間を抽出する。公式指標は、小文字化して空白で分割した単語集合の Jaccard 類似度を行ごとに求めた平均。 [Kaggle 公式仕様](https://www.kaggle.com/competitions/tweet-sentiment-extraction)

## 進捗

- [x] P0 採点確認。公式データを取得し、非公開の CPU Notebook の全文予測 E00 が Public 0.59451、Private 0.58762 を返すことを確認。
- [x] P1 問題設定・EDA。学習 27,481 行のうち本文と正解が共に空の1行を除外。テスト 3,534 行。重複本文は正規化後に51グループ。
- [x] P2 検証分割。正規化した本文をグループとし、感情を層化する5分割（seed 42）。語彙と抽出器は fold 内で学習する。正解位置が複数ある113行は位置教師から外し、CV の採点に使う。原文不一致は0行。
- [x] P3 ベースライン。E01 の OOF Jaccard は 0.63641。Late Submission は Public 0.63778、Private 0.64024（固定分割版）。
- [x] P4 自力の比較実験。E02〜E04 を実行。
- [x] P5 上位解法。入賞者2件の本人資料を調査。
- [x] P6 アブレーション。E05 は区間の選択式だけを変更。
- [x] P7 最終評価。OOF で E02 を選定し、固定分割の Late Submission で Public 0.64928、Private 0.64991。
- [x] P8 振り返り・知見を記録。 [振り返り](report.md) · [知見カード K016](../../knowledge/K016-character-features-in-text.md)

| 感情 | 行数 | 正解が全文 | 正解の語数中央値 |
|---|---:|---:|---:|
| negative | 7,781 | 14.66% | 2 |
| neutral | 11,117 | 89.99% | 11 |
| positive | 8,582 | 13.02% | 2 |

## 実験前の仮説

- E01。語と感情で推定した連続区間は、positive・negative の全文予測を上回る。
- E02。語内の文字 3〜5-gram を E01 に追加すると、繰り返し文字などの表記変形を含む行で改善する。
- E03。前後の語を E01 に追加すると、否定や強調を含む行で改善する。
- E04。E01 の neutral だけを全文に固定すると、neutral の Jaccard が改善する。

P4 の3件を完了してから上位解法を調査した。E05 は学習を E01 に固定し、区間選択に単語集合の Jaccard の近似値を使う。複数の語を含む範囲の選択が改善すると予想した。

## 実行

リポジトリルートで実行する。

```powershell
dev download 48
$d = "competitions/048-tweet-sentiment-extraction"
uv run python "$d/src/eda.py"
foreach ($trial in "E00", "E01", "E02", "E03", "E04", "E05") {
    uv run python "$d/src/train.py" $trial --run-id "$trial-r2"
}
uv run python "$d/src/select_model.py" "$d/outputs/selection-replay.json" --run-suffix=-r2
```

`train.py` の引数で試行を切り替える。再実行には `--run-id` で未使用の ID を指定し、全試行で同じ接尾辞を使う。Notebook はローカルの抽出コードから組み立て、Kaggle で共有コードを実行する。E01 以降はローカルで作った fold 割り当てを Notebook に入れ、Kaggle 側で学習行の ID と順序を検証する。

```powershell
$env:PYTHONUTF8 = "1"
$owner = "your-kaggle-username"
uv run python "$d/src/notebook.py" E00 $owner "$d/outputs/kaggle/E00"
kaggle kernels push -p "$d/outputs/kaggle/E00"
kaggle kernels status $owner/kumite-tweet-sentiment-e00
kaggle competitions submit tweet-sentiment-extraction -k $owner/kumite-tweet-sentiment-e00 -v 1 -f submission.csv -m "Kumite E00"
```

`-v` には実行が完了した Notebook のバージョンを指定する。
