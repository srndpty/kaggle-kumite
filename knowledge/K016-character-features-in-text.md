# K016: 単語特徴に文字 n-gram を追加すると短文の予測が改善する

- 状態: 再現
- 分類: 特徴量 / モデル

## 仮説

単語単位の語彙だけでなく文字n-gramを使うと、単語内の部分的な一致も予測に使える。

## 観測

#041 Jigsawの単語unigram TF-IDF+ラベル別LRにchar_wbの3〜5gramを追加すると、seed42のfold平均macro AUCは0.979929→0.985127（+0.005198）。3/3foldで改善し、ラベル平均OOF LogLossも0.054970→0.048473へ改善した。一方、単語bigramの追加はAUC 0.978583へ悪化した。

seed43でもfold平均macro AUCは0.979964→0.985175（+0.005211）。2つの分割seedで改善方向と大きさが揃った。

#048 の範囲抽出でも、語と感情に基づくロジスティック回帰に語内の文字 3〜5-gram を追加すると、OOF Jaccard が 0.63641 から 0.64348 に上がった。negative は +0.01627、positive は +0.00798、neutral は −0.00008 だった。英字が3回以上連続する2,157行では +0.02650。

#042 の英語コメントでは、単語 unigram に文字 3〜5-gram を追加すると、全体 AUC は 0.91659 から 0.92935、属性別 AUC を含む合成スコアは 0.85835 から 0.87788 に上がった。選定後にローカル採点した公式 Private データでも、合成スコアは 0.85897 から 0.87896 に上がった。単語 bigram を個別の TF-IDF 特徴群として足した E02 は、OOF 合成スコアが 0.82938 に下がった。

#049 CommonLit の読みやすさ回帰では、単語 unigram TF-IDF に語内の文字 3〜5-gram を追加すると、OOF RMSE が 0.72984 から 0.70539 に下がった。学習語彙のカバー率が最も低い四分位の抜粋では 0.80305 → 0.76922、最も高い四分位でも 0.70500 → 0.67761 だった。

#046 Quora の質問分類では、単語 unigram の E01 に文字 3〜5-gram を加えた E02 の OOF F1 が 0.60346 → 0.62319 に上がった。単語語彙のカバー率が 100% の行でも誤検出は 38,324 → 34,364 件に減った。

## 適用条件

英語のWikipediaコメント・6ラベル分類。文面重複をグループ化した多ラベル層化3分割。語彙・IDFは学習foldのみでfit。min_df=3、単語最大10万・文字最大15万、sublinear_tf、各行列のL2正規化、LRのC=1。

#048 は英語ツイートの範囲抽出で、本文の重複を束ねた5分割。語と感情の二値特徴に語内 n-gram の二値特徴を追加した。

#049 は英語の文学作品の抜粋2,834件を、本文の類似と出典 URL でグループ化した5分割。単語と文字はそれぞれ L2 正規化し、Ridge の正則化係数を固定した。

#046 は陽性率 6.19% の英語の質問。外側を同一文面グループで分け、各学習側の保留データで F1 閾値を選んだ。単語・文字ブロックはそれぞれ L2 正規化し、ロジスティック回帰の `C=1` を固定した。

## 未検証

綴りの変形を加えた対照データでの頑健性、他言語への一般化、正則化Cや特徴数上限を揃え直した場合の差。#041 の E03 は単語・文字を別々にL2正規化して結合し、結合後は再正規化していないので、効果を語彙のカバー範囲だけに帰属できない。

## 次の検証機会

別の短文分類で、文字特徴の追加と単語語彙の拡張を分けて比較する。

## 証跡

| コンペ | run | 結果 |
|---|---|---|
| [041 Jigsaw](../competitions/041-jigsaw-toxic-comment-classification-challenge/README.md) | E01→E03 | seed42: fold平均macro AUC +0.005198。単語 bigram の E02 は −0.001346 |
| 同上 | E01-s43→E03-s43 | seed43: +0.005211 |
| [#048 Tweet Sentiment Extraction](../competitions/048-tweet-sentiment-extraction/report.md) | E01 → E02 | OOF Jaccard +0.00707、本文グループ単位の95%区間 [0.00502, 0.00898] |
| [#042 Jigsaw Unintended Bias](../competitions/042-jigsaw-unintended-bias-in-toxicity-classification/report.md) | E01 → E03-r2 | 全体 OOF AUC +0.01276、属性別 AUC を含む合成スコア +0.01954 |
| [#049 CommonLit Readability Prize](../competitions/049-commonlitreadabilityprize/report.md) | E01 → E02 | OOF RMSE −0.02445、グループ単位の95%区間 [−0.02987, −0.01947] |
| [#046 Quora Insincere Questions](../competitions/046-quora-insincere-questions-classification/report.md) | E01 → E02 | OOF F1 +0.01973、文面グループ単位の 95% 区間 [+0.01813, +0.02132] |
| [#047 Google QUEST](../competitions/047-google-quest-challenge/report.md) | B00 → E02 | 結合した質問・回答テキストへの文字 3–5 gram 追加で OOF 平均 Spearman 0.29164 → 0.29677 |
