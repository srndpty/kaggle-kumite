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

## P5–P6　上位解法と追加仮説

- １位チームの公開実装は Transformer を多数学習し、過去データの擬似ラベルも使った。 [公開コード](https://github.com/rohitsingh02/kaggle-feedback-english-language-learning-1st-place-solution)
- ５位の本人解説では、複数のモデル・プーリングに加え、評価軸ごとのアンサンブル重みを検証した。 [本人の解法解説](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/writeups/psi-5th-place-solution)
- 効率部門17位チームは、擬似ラベルで小型モデルを学習し、教師側では埋め込みに RBF Kernel Ridge などを使った。 [本人の解法解説](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/writeups/pdsg-efficiency-17th-place-solution)

E04 は効率部門の教師モデルから非線形回帰の着眼点を取り出し、E00 と同じ単語 unigram TF-IDF に RBF Kernel Ridge（alpha 1、gamma 1）を適用する。非線形な語彙類似度によって MCRMSE が下がると予想する。最終選定では４件を再実行し、同じ計画ハッシュの E04 と比較する。
