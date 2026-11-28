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

E04-final の OOF LogLoss は 0.821570。ベースラインとの差は +0.016241、95%区間 +0.013639〜+0.018830 で、全論証種別で悪化した。この比較で検証したのは境界語彙の TF-IDF 追加。元の解法の Transformer 境界トークンに関する結論は出せない。

## 最終選定

最終計画の５候補を保存 OOF から再採点し、E00〜E03 は初回と同じスコアを再現した。最良の E03-final と次点の E01-final の差は 0.034029 で、事前の同点許容差 0.001 を超えた。E03-final を採用した。 [最終選定記録](selection-final.json)

文字 n-gram は [K016](../../knowledge/K016-character-features-in-text.md)、他の論証の文脈は [K060](../../knowledge/K060-other-discourses-as-context.md) に記録した。

## Kaggle 外部評価

各構成を非公開 CPU Notebook 内で全訓練行に再適合し、提出サンプルの順で予測を生成した。

| CV run | Export ID | OOF LogLoss | Late Public | Late Private |
|---|---|---:|---:|---:|
| E00-external | baseline | 0.805329 | 0.80192 | 0.79778 |
| E03-final | final | 0.751580 | 0.74395 | 0.74798 |
| 最終 − ベースライン | | −0.053750 | −0.05797 | −0.04980 |

同じエッセイの他の論証を用いた構成は CV・Public・Private で改善した。外部評価の学習行数と評価対象は CV と異なるため、改善幅の差の要因は分離していない。学習済みモデルの外部スコアは最終選定確定後に確認した。 [ベースライン選定記録](selection-baseline.json)
