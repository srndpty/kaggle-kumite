# #041 Jigsaw Toxic Comment Classification Challenge 振り返り

Standard。Wikipediaのコメントから6種類の毒性を予測する多ラベル分類。表形式・時系列・画像に続き、初めてのNLPとして単語・文字の疎な表現と軽量Transformerを比較した。詳細な条件・証跡は[README](README.md)とexperiments.jsonlを参照。

## 結果

主評価はfold平均macro AUC（大きいほど良い）。文面重複をまとめた多ラベル層化3分割で比較した。

| run | 内容 | fold平均macro AUC | ラベル平均OOF LogLoss |
|---|---|---:|---:|
| E01 | 単語TF-IDF + LR | 0.979929 | 0.054970 |
| E02 | 単語bigram追加 | 0.978583 | 0.058175 |
| E03 | 文字3〜5gram追加 | 0.985127 | 0.048473 |
| E04 | balanced重み | 0.981131 | 0.106705 |
| E05 | DistilBERT | 0.990429 | 0.037216 |
| E06 | E05とE03の1:1確率平均 | 0.990703 | 0.038606 |

最終モデルは**seed42のE06**。E05との差はseed42 +0.000274、seed43 +0.000142と両方で改善し、事前規則で採用した。確認用seed43の方が点推定は高いが、提出は事前に決めたseed42を維持した。提出前に選定結果と提出ファイルのハッシュを保存した。最終の外部評価は採点待ち。E01の初回Late SubmissionはPublic 0.97489 / Private 0.97462。外部評価値はモデル選定に使用していない。

## 自力の仮説で効いたこと・効かなかったこと

文字特徴は3/3foldで改善し、別の分割seedでも改善量が揃った（E01→E03: seed42 +0.005198、seed43 +0.005211）。一方、単語bigramは全foldで悪化した。特徴数上限10万・正則化C=1を固定した条件での結果で、bigram一般が有害だとは言えない。

クラス重み付けはAUCを改善しても、6ラベルすべてのLogLossを悪化させた。Porto Seguroでは重み付けが順位・確率とも悪化したのに対し、今回は順位が改善した。補助指標だけを見て主評価の採否を変えないこと、順位と確率の評価を併記することを再確認した。

DistilBERTは全ラベルでbaselineのAUCを上回った。モデル方式・事前学習・共有表現をまとめて変えた比較なので、「文脈を拾うことだけ」の効果を示したものではない。単語LRの約80秒、文字追加の約12分に対し、DistilBERTは取得・tokenizeを含み約18分だった。

## 上位解法から学び、再現したこと

自力の3比較（E02〜E04）完了後に調査した。E05はそれ以前に設定を固定し実行を開始していた。

- [2位本人writeup](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/writeups/neongen-computer-says-no-2nd-place-solution-overvi): 多様な埋め込み・モデルの平均、往復翻訳によるaugmentation。本文は検索サービスの索引経由で確認した。
- [27位本人の公開資料](https://github.com/zake7749/DeepToxic/blob/master/Readme.md): 単体では弱い文字モデルも単語モデルと平均すると改善した。今回のE06の直接の着眼点。
- [34位本人の公開コード](https://github.com/peterhurford/kaggle-toxic_comment): TF-IDF・文面の統計量・ニューラルモデルを多段で統合。特徴量設計とモデルの多様性という別のアプローチ。

E06のseed42での改善はthreatの寄与が大きい（OOF AUC +0.002641、AP +0.039581）。3ラベルでAUCは悪化し、6ラベルすべてでLogLossが悪化した。macro AUCだけの改善を、全ラベル・全指標の改善とは解釈しない。

E06は「異なる表現を平均する」原理の小規模な再現で、当時のRNN/CNNや解法全体の忠実再現ではない。係数は実行前に1:1で固定し、OOFラベルで調整していない。元runの分割・データ・ラベル列順・成果物ハッシュを検査した。翻訳、疑似ラベル、多段スタッキングは今回行っていない。

## エラー分析と限界

1,000文字超のOOF AUCはE05 0.985282→E06 0.989256。全文を使う文字モデルが、先頭256tokenに制限したTransformerを補う可能性があるが、切り捨ての影響を単独で検証したわけではない。短文でも0.987919→0.988551へ改善した。

希少なthreatのAPはE05でも0.467程度で、AUCが高いことと陽性を正確に拾えることは同じではない。損失の大きい行のIDと区分別・ラベル別指標はoutputs/analysisに保存した。

正規化文面以外の近似重複、潜在的な投稿者・記事依存は除去できていない。OOFは1foldモデル、提出は3fold平均であり評価する予測の形も異なる。bf16のGPU計算のビット一致は保証しない。終了後公開のtestラベルは読んでいない。

## 保存した知見

[K016](../../knowledge/K016-character-features-in-text.md)を文字特徴の知見として作成し、[K005](../../knowledge/K005-stacking-diversity.md)に固定平均での関連観測を追記した。[K014](../../knowledge/K014-ranking-vs-probabilities.md)を順位と確率の変化が分かれた別条件の証跡で更新した。

検証: dev check（lint・整形・型検査・全200テスト）成功。未確定runなし。初回P0の予約記録不足は、RunStoreで予約したP0-r2で作り直して採点を再確認した。
