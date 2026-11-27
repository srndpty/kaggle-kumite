# #053 Feedback Prize – English Language Learning

英文エッセイから cohesion、syntax、vocabulary、phraseology、grammar、conventions の６軸を予測する。深さは Standard。CPU で学習し、６軸別 RMSE の平均である MCRMSE を最小化する。

[公式概要](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/overview)

## P0–P2　採点と検証

訓練平均のダミー Notebook は Late Submission で採点された。Public 0.618673、Private 0.644705。疎通の確認に使用した。

訓練 3,911 件。本文の語数は最小 14、中央値 402、最大 1,260。空本文と正規化後の重複は０件。６軸とも得点範囲は 1–5 で、軸間の Pearson 相関は 0.639–0.735。

本文を NFKC・小文字・空白により正規化して同一本文のリークを防ぐ。６軸平均の等頻度５区間を層とし、StratifiedGroupKFold の seed 42 で５分割する。検証件数は 782 / 782 / 783 / 782 / 782。語彙、IDF、スケーラーは fold の学習行で適合する。

## P3–P4　事前仮説

| Run | 構成・変更 | 予想 |
|---|---|---|
| E00 | 単語 unigram TF-IDF と Ridge | 語彙分布から６軸の得点を推定できる |
| E01 | E00 に単語 bigram を追加 | phraseology と syntax の RMSE が低下 |
| E02 | E00 に文字 3–5 gram を追加 | conventions と grammar の RMSE が低下 |
| E03 | E00 に語数・文字数・文数・平均文長・異なり語率・句読点率を追加 | cohesion の RMSE が低下 |

候補の優先順は E00、E03、E01、E02。実質同点は MCRMSE 0.001 以内。P4 の完了後に上位解法を確認する。
