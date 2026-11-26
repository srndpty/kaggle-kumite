# K014: 不均衡分類では順位と確率の評価を分ける

- 状態: 再現
- 分類: 検証 / モデル

## 仮説

順位指標の良さから確率の良さは判断できない。不均衡に対する重み付けも、主評価で検証して採否を決める。

## 観測

Porto Seguroで陽性重みを学習foldの陰性数÷陽性数（約26.44）にすると、OOF Gini 0.262451→0.256030、LogLoss 0.152456→0.588285。今回は順位にも効果がなかった。

別の診断として、保存済みE06の予測pを0.1+0.8*pにすると、Giniは0.280191のまま、LogLossは0.151923→0.205791。これは厳密単調変換なので順位を保存する。新たな学習runや外部提出ではない。

#041 Jigsawでも順位と確率の変化が分かれた。単語TF-IDF+LRでbalanced重みにすると、seed42のfold平均macro AUCは0.979929→0.981131へ改善した一方、ラベル平均OOF LogLossは0.054970→0.106705へ悪化した。Porto Seguroと異なり、順位は改善した。重みの効果はモデル・データ・評価指標ごとに検証する必要がある。

## 適用条件

重み付けの観測はPorto SeguroのLightGBM（固定の層化5分割）とJigsawの単語TF-IDF+LR（文面グループによる多ラベル層化3分割）、いずれもseed42。主指標は同点平均順位の2*AUC-1、補助指標はLogLossとAverage Precision。

## 未検証

他の重み・サンプリング・モデルでの効果、重み付き予測の較正は未検証。定数に近い予測では同点処理によって公式Giniと差が出うる。一般に重み付けが無効だとは言えない。

## 次の検証機会

信用リスクや不正検知で、順位評価と確率評価を併記し、重み付けと較正を別々に検証する。

## 証跡

| コンペ | run | 結果 |
|---|---|---|
| [002 Porto Seguro](../competitions/002-porto-seguro-safe-driver-prediction/README.md) | E01/E05 | 重み付けで順位・確率とも悪化 |
| 同上 | E06の保存済みOOF診断 | 単調変換でGini不変、LogLoss悪化。outputs/analysis/E01-E04-E06.json |
| [041 Jigsaw](../competitions/041-jigsaw-toxic-comment-classification-challenge/README.md) | E01/E04 | balanced重みでmacro AUCは改善、LogLossは約2倍へ悪化 |
| [069 ISIC 2024](../competitions/069-isic-2024-challenge/report.md) | B00/E03 | 陽性重み5倍で pAUC 0.020184 → 0.022024、log loss 0.062310 → 0.180435 |
