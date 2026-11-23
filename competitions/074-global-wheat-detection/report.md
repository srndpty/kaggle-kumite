# #074 Global Wheat Detection 振り返り

固定矩形の Late Submission で採点経路を確認した。Public / Private はともに 0.0004。学習データの取得と検証分割の根拠は [README](README.md) に記録した。

## ベースライン

撮影元を分けた 3-fold の OOF は **0.445057**（`E00-r2`）。fold 別では 0.412320 / 0.420016 / 0.499349 だった。全学習画像で再学習したモデルの Late Submission は **Public 0.5459 / Private 0.4791**。 [Kaggle 提出履歴](https://www.kaggle.com/competitions/global-wheat-detection/submissions)

撮影元別の OOF は 0.380109（`arvalis_2`）から 0.592181（`rres_1`）まで開いた。IoU 0.50 で誤検出は 35,325 件、見逃しは 30,868 件。穂が51本以上ある画像群では誤検出 13,785 件に対して見逃し 19,680 件だった。

## 分割方式の診断

画像単位ランダム分割の OOF は **0.612980**（`E01`）。同じ3,373枚を採点した撮影元分割より **+0.167923** 高かった。撮影元の共有に加えて各 fold の学習画像数と構成も変わる。未知の撮影元への汎化を評価する主分割には、事前に決めた撮影元分割を使う。

## 解像度

768 px（`E02`）の OOF は **0.486631**（基準から **+0.041575**）。改善は 3 fold 中2つ。正解矩形の面積中央値が 5,500 px² 未満の1,261枚では、IoU 0.50 の見逃しが 18,346 → 17,401 件、誤検出が 12,875 → 13,545 件となり、画像平均スコアは 0.440778 → 0.437172 だった。

矩形面積中央値が 5,500 px² 以上の2,112枚では、画像平均スコアが 0.447612 → 0.516161 に改善した。小さい穂が多い画像では見逃しが減った一方で画像平均スコアは下がり、この画像群が全体改善の主因という仮説は支持されなかった。

## 色の拡張と NMS

- 明度・コントラストを拡張した `E03` は **0.462752**（+0.017696）。7撮影元中6つで改善した。照明の正解ラベルがないため、どの照明条件で改善したかは未検証。
- NMS の IoU を 0.65 に上げた `E04` は **0.406032**（−0.039024）。穂が51本以上の画像で IoU 0.50 の見逃しは19,680 → 19,042件、誤検出は13,785 → 21,372件。見逃し減少より誤検出増加が大きかった。

## 上位解法

- 2位の本人 writeup。EfficientDet の学習で CutMix・MixUp を試し、元画像の再構成や擬似ラベルも検討した。推論では変換を使った複数の検出結果を Weighted Boxes Fusion で統合した。 [2位の解法](https://www.kaggle.com/competitions/global-wheat-detection/writeups/overfeat-2nd-place-solution-with-code-mit-complian)
- 9位の本人 writeup。DetectoRS と UniverseNet を組み合わせ、高信頼のテスト画像に擬似ラベルを付けた。検出器によって信頼度の分布が異なるため、順位に基づく変換を施してから矩形を融合した。 [9位の解法](https://www.kaggle.com/competitions/global-wheat-detection/writeups/katz-kashani-amir-update-the-9th-place-solution-wi)

`E05` では2位解法から複数変換の推論と矩形融合を切り出した。固定した E00 の重みで元画像と水平反転を推論し、座標を戻して IoU 0.55 を超える矩形を信頼度で加重平均する。融合後の信頼度を閾値 0.25 で切る。学習データと重みを固定し、擬似ラベルや複数モデルは使わない。

OOF は **0.463676**（+0.018619）で、3 fold とも改善した。IoU 0.50 の誤検出は 35,325 → 30,728 件、見逃しは 30,868 → 30,164 件。水平反転だけの効果、融合だけの効果、高解像度と組み合わせた場合の効果は未検証。

## 最終選定

撮影元分割で採点した5構成を [選定記録](selection.json) に固定した。全3,373画像の平均が最も高かった **E02（768 px、0.486631）** を採用した。撮影元別では7つのうち6つで改善し、`ethz_1` で 0.400733 → 0.356782 に低下した。51本以上の密集画像での改善は +0.002850 で、20本以下の +0.084182 より小さかった。

公開テスト10枚の画像 ID は、ローカル GPU と Kaggle CPU で一致した。矩形の座標と順序が一致したのは8枚で、矩形数は GPU 263件、CPU 262件だった。外部評価には CPU Notebook 版1の出力を使ったため、ベースラインとの外部比較には推論環境の差も含まれる。 [最終 Notebook](https://www.kaggle.com/code/shion5/kumite-wheat-e02-cpu)

## 最終外部評価

E02 の Late Submission は **Public 0.6054 / Private 0.5343**。ベースラインとの差は Public **+0.0595**、Private **+0.0552** で、撮影元分割の OOF **+0.041575** と同じ方向だった。外部評価では全学習画像での再学習、撮影元の構成、CPU と GPU の推論差が含まれる。テストの正解は非公開のため、撮影元別や密度別の改善は OOF で確認した。 [Kaggle 提出履歴](https://www.kaggle.com/competitions/global-wheat-detection/submissions)

撮影元分割の差を [K011](../../knowledge/K011-group-split-vs-random-split.md)、水平反転推論と矩形融合を [K023](../../knowledge/K023-horizontal-flips-for-segmentation.md)、NMS の見逃しと誤検出の関係を [K034](../../knowledge/K034-nms-threshold-for-dense-detection.md) に記録した。
