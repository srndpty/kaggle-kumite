# K022: 保有商品の組み合わせで新規取得率を推定する

- 状態: 仮説
- 分類: 推薦 / 順位付け

## 仮説

保有済み商品を推薦から外した後も、保有商品の組み合わせは未保有商品の取得を予測する手がかりになる。

## 観測

#034 の3月・4月 CV では、全体の商品取得率 E01 の OOF MAP@7 は 0.01947178。保有数を3群に分ける E03 は 0.02033894、24商品の保有状態を入力にする商品別 L2 ロジスティック回帰 E04 は 0.02362319。どちらも両月で E01 を上回った。すべての run で同じ保有済み除外を適用した。

E04 の手法を5月までのデータで再学習した E06 の6月評価は、E01 比 Public −0.00005、Private +0.00003。3月・4月の改善幅は6月では再現しなかった。

## 適用条件

直前の暦月の保有状態を使える顧客商品パネル。取得率および学習対象は、前月の非保有と当月の状態を確認できる顧客・商品に限定する。検証は対象月より前に確定した取得情報だけを使う。

## 未検証

顧客の属性や取引履歴が使える場合の相対効果、保有状態を観測できない顧客での効果、保有済み除外単独の寄与、他の推薦課題への一般化。

## 次の検証機会

候補の既利用状態が観測できる推薦課題で、人気順と利用履歴に条件付けた順位を同じ時系列分割で比較する。

## 証跡

| コンペ | run | 結果 |
|---|---|---|
| [#034 Santander Product Recommendation](../competitions/034-santander-product-recommendation/experiments.jsonl) | E01 → E03 | OOF MAP@7 0.01947178 → 0.02033894 |
| [#034 Santander Product Recommendation](../competitions/034-santander-product-recommendation/experiments.jsonl) | E01 → E04 | OOF MAP@7 0.01947178 → 0.02362319 |
| [#034 Santander Product Recommendation](../competitions/034-santander-product-recommendation/report.md) | E01 → E06 | Private MAP@7 0.02388 → 0.02391 |
