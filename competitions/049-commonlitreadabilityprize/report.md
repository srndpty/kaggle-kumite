# #049 振り返り

## モデルと外部評価

固定した5分割の OOF RMSE で E04 を選定し、全学習行で再学習した。単語 unigram の E01 に語数・文数・平均語長・平均文長を追加した構成。E02 に対する差は −0.00626。グループ単位のペア bootstrap 2,000回の95%区間は [−0.01433, +0.00202] で、文字 n-gram の E02 との差には不確実性が残る。

E04 の比較で測ったのは、標準化した4特徴を L2 正規化済み TF-IDF にまとめて加えたときの効果。個々の特徴の寄与や、TF-IDF に対する数値特徴の相対スケールを変えた場合の効果は未検証。

| Run | OOF RMSE | Late Public | Late Private |
|---|---:|---:|---:|
| E00 定数 | 1.03348 | 1.048 | 1.028 |
| E01 単語 unigram | 0.72984 | 0.713 | 0.725 |
| E04 文章構造を追加 | 0.69913 | 0.674 | 0.705 |

E04 は E01 に比べて OOF で 0.03071、Public で 0.039、Private で 0.020 改善し、改善の向きが揃った。Kaggle の非公開 Notebook は E00・E01・E04 ともに version 1 を提出した。 [Kaggle 提出履歴](https://www.kaggle.com/competitions/commonlitreadabilityprize/submissions)

## 誤差分析

E04 の OOF を各指標の四分位で分けた。順番は値が小さい区間から大きい区間。目的値の両端、特に読みやすさの低い抜粋で誤差が大きい。語数による誤差の差は、目的値や学習語彙カバー率ほど明瞭でなかった。

| 区分 | 第1区間 | 第2区間 | 第3区間 | 第4区間 |
|---|---:|---:|---:|---:|
| 語数 | 0.68512 | 0.72046 | 0.70737 | 0.68352 |
| 目的値 | 0.87079 | 0.51560 | 0.52921 | 0.80645 |
| 学習語彙カバー率 | 0.77562 | 0.68335 | 0.66737 | 0.66363 |

文字 n-gram の E02 は E01 に対し OOF RMSE を 0.02445 改善した。グループ単位のペア bootstrap 95%区間は [−0.02987, −0.01947]。学習語彙カバー率の低い区間（約89.5%以下、712件）では 0.80305 → 0.76922 で、4区間中もっとも改善が大きかった。最高カバー率の区間でも 0.70500 → 0.67761 と改善した。単語 bigram は全体では E01 を改善したが、頻度別の効果は未検証。文字 n-gram と文章構造の組み合わせも今回の比較範囲外。 [K016](../../knowledge/K016-character-features-in-text.md)

## 上位解法からの検証

- 1位の本人解説と公開コードでは、外部文を教師モデルで疑似ラベル化し、文埋め込みモデルの追加学習に使った。外部データと事前学習モデルを必要とするため、公式入力だけで行う今回の比較から外した。 [本人解説](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/257844) · [公開コード](https://github.com/mathislucka/kaggle_clrp_1st_place_solution)
- 2位は複数の Transformer を組み合わせ、一部で同じ RoBERTa 系の表現に SVR または Ridge を適用した。公開スコアで調整したアンサンブルと後処理も報告している。E05 では回帰器を比較する着眼点を取り出し、同じ unigram TF-IDF の下で Ridge を線形 SVR（C=1、ε=0.1）に置き換えた。OOF RMSE は 0.72984 → 0.74228。この設定の SVR はベースラインを上回らなかった。 [本人解説](https://www.kaggle.com/competitions/commonlitreadabilityprize/writeups/rist-takoi-2nd-place-solution) · [公開コード](https://github.com/TakoiHirokazu/kaggle_commonLit_readability_prize)
