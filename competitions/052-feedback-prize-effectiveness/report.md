# #052 Feedback Prize – Predicting Effective Arguments 振り返り

## 自力の比較

エッセイを束ねた固定５分割で、同じ論証行の LogLoss を比較した。95%区間は固定 OOF 予測をエッセイグループ単位に対応づけて再標本化したもの（2,000回、seed 42）。分割や再学習の変動は含まない。

| Run | OOF LogLoss | E00との差 | 95%区間 |
|---|---:|---:|---|
| E00 単語 unigram | 0.805329 | — | — |
| E01 論証種別 | 0.785609 | −0.019720 | −0.021814〜−0.017821 |
| E02 文字 n-gram | 0.797261 | −0.008069 | −0.010645〜−0.005433 |
| E03 他の論証文脈 | 0.751580 | −0.053750 | −0.061125〜−0.046781 |

E01 は全論証種別で改善した。結論文（−0.03364）と Evidence（−0.03235）の差が大きく、Position は −0.00259 だった。E02 は全体で改善した一方、Rebuttal では +0.00557 悪化した。

E03 は16語以下（10,033件）で −0.04700、17語以上で −0.05628。短い論証も改善したが、効果は短い群に集中しなかった。論証種別では Claim の改善が最大（−0.06818）。正解クラス別では Ineffective −0.06740、Adequate −0.01280、Effective −0.13639 だった。文脈が論証の具体的な不足を補ったのか、同一エッセイの全体的な品質や題材を捉えたのかは分離していない。 [P4 選定記録](selection-p4.json)

## P5–P6　上位解法と再現

- ７位チームはエッセイ全体または論証の連結を入力とする Transformer に、MLM 事前学習と旧コンペのソフト疑似ラベルを組み合わせた。 [７位の本人解説](https://www.kaggle.com/competitions/feedback-prize-effectiveness/writeups/now-you-see-me-now-you-see-me-7th-place-solution)
- 11位チームは論証の開始・終了を別のマーカーで示し、その位置から分類した。 [11位の本人解説](https://www.kaggle.com/competitions/feedback-prize-effectiveness/writeups/tifo-11th-solution-feature-design-and-knowledge-di)
- 12位チームは旧データの疑似ラベルをクラス別に選別し、モデル予測と論証種別を２段目の分類器に入力した。 [12位の本人解説](https://www.kaggle.com/competitions/feedback-prize-effectiveness/writeups/team-animals-12th-place-solution)

E04 は11位の境界の着眼点を CPU 向けに適用する。冒頭と末尾の各８トークン（通常の TF-IDF と同様に２文字以上）に別の接頭辞を付け、独立した TF-IDF（min_df=2）を E00 に追加する。段落の入り方や締め方を区別すると LogLoss が下がると予想する。
