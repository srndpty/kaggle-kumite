# 知見カード

100コンペを通して「何がわかったか」を、手法・原理ごとに1枚のカードにまとめる。
実験ログ（各コンペの `experiments.jsonl`）が「何を実行したか」の証跡なのに対し、カードは
「どの条件で何が有効だったか、どこまで一般化できるか」を書く。

## 運用

- ファイル名は `K###-<slug>.md`。番号は作成順
- 同じ知見を別コンペで再確認したら、新しいカードを作らず「証跡」に行を足す。反例も同じカードに書き、「適用条件」を更新する
- 実測していない一般化は「未検証」に書き、「適用条件」には書かない
- 状態: `仮説`（1件で観測）/ `再現`（2件以上で同じ向き）/ `条件付き`（反例あり、条件を特定済み）

## 書式

```markdown
# K###: <手法・原理の名前>

- 状態: 仮説 / 再現 / 条件付き
- 分類: 検証 / 特徴量 / モデル / アンサンブル / 較正 / 前処理 など

## 仮説
## 観測
## 適用条件
## 未検証
## 次の検証機会
## 証跡

| コンペ | run | 結果 |
|---|---|---|
```

## 一覧

| カード | 状態 | 要旨 |
|---|---|---|
| [K001](K001-oof-vs-fold-average.md) | 条件付き | OOF（単一モデル）と提出（fold 平均）は別物。Otto では平均で約 -0.02、NN の State Farm では -0.13（学習し直しの 1 本は Private +0.073）。時系列分割では平均が逆効果で、学習し直しが良い（Rossmann 約 0.004、Recruit 0.038） |
| [K002](K002-calibrate-on-submission-form.md) | 再現 | 較正は提出と同じ形の予測で決める。単一モデルの OOF で決めると平均予測には効きすぎ、逆効果になる（Otto +0.0009、State Farm +0.028） |
| [K003](K003-monotone-transform-by-model.md) | 条件付き | log1p は線形モデルには効くが、木モデルと標準化した MLP には効かなかった |
| [K004](K004-cross-fold-label-leak.md) | 仮説 | 全行で作った OOF 特徴を後段の CV に使うと、外側の検証ラベルが学習側に入る |
| [K005](K005-stacking-diversity.md) | 仮説 | 単体で弱くても誤り方の違うモデル（kNN・ExtraTrees・MLP）はスタッキングで効く |
| [K006](K006-class-wise-knn-distance.md) | 仮説 | クラス別の最近傍距離は、意味の伏せられた特徴でも強い手がかりになる |
| [K007](K007-time-split-vs-random-split.md) | 再現 | 未来を予測する問題では、ランダム分割は 0.025〜0.045 楽観的。特徴量の採否が変わることもある（Rossmann は逆転、Recruit は逆転せず） |
| [K008](K008-reproduce-information-cutoff-in-validation.md) | 再現 | 検証でも「その時点で分かる情報の範囲」を test と同じにする（予定・予約の打ち切り）。予約は学習行にも打ち切りを入れる。ラベル由来の統計量は学習行では直前まで使う |
| [K009](K009-log-target-for-relative-error.md) | 再現 | 相対誤差・対数誤差の指標では、目的変数を log にして学習する（RMSPE -0.009、RMSLE -0.032） |
| [K010](K010-map-rare-days-to-common-days.md) | 仮説 | 祝日のような少数の特殊な日は、土曜などの普通の日に対応付けると学べる |
| [K011](K011-group-split-vs-random-split.md) | 条件付き | State Farmのランダム分割は19〜30倍楽観的。SpeechではAccuracyがグループ0.8563・ランダム0.7154と逆方向で、差の大きさ・方向は条件依存 |
| [K012](K012-test-neighbour-averaging.md) | 仮説 | test の中の近傍と予測を平均すると logloss が下がる（State Farm Private -0.050）。OOF では効果を 2 倍に見積もる |
| [K013](K013-categorical-splits-vs-numeric.md) | 仮説 | 名義カテゴリの数値扱いがLightGBMのカテゴリ分割より良かった（Porto Seguro、3 seed平均Gini +0.016054） |
| [K014](K014-ranking-vs-probabilities.md) | 再現 | 順位と確率を分けて評価する。Porto Seguroの重み付けは両方悪化、JigsawではAUC改善でもLogLossは約2倍へ悪化 |
| [K015](K015-feature-group-ablation.md) | 条件付き | 特徴群の除外で Porto Seguro の Gini と MoA の Log Loss は改善。Mercedes-Benz では R² が悪化 |
| [K016](K016-character-features-in-text.md) | 再現 | 文字3〜5-gram 追加で #041 の macro AUC が約 +0.0052、#048 の範囲抽出の Jaccard が +0.00707、#042 の属性別 AUC を含む合成スコアが +0.01954 |
| [K017](K017-customer-history-aggregation.md) | 再現 | #004 の履歴集約で AUC 約 +0.0056〜+0.0057、#005 の過去申込件数・経過日数で平均安定性スコア +0.00512 |
| [K018](K018-recent-history-aggregation.md) | 仮説 | Home Creditで全期間借入集約に直近365日集約を追加し、2 seedでAUC +0.000788 / +0.001562 |
| [K019](K019-fine-grained-supervision-cv-vs-external.md) | 仮説 | Speechの31クラス学習は2seedのCV Accuracyが約+0.04だがPrivateは-0.00529。内部改善が外部へ再現しない例 |
| [K020](K020-quantile-bins-for-additive-effects.md) | 仮説 | Santander では数値特徴を分位点ビンに置換すると、ロジスティック回帰の OOF AUC が +0.02516 |
| [K021](K021-value-frequency-in-numeric-features.md) | 仮説 | Santander では値の出現回数追加で OOF AUC +0.00554、反復値を残した列の追加で +0.00473 |
| [K022](K022-product-holdings-for-ranking.md) | 仮説 | #034 で前月保有商品に条件付けて順位付けすると MAP@7 が +0.00415。#033 では購入履歴を先頭に置くと MAP@12 +0.01714、共購入で +0.00179 |
| [K023](K023-horizontal-flips-for-segmentation.md) | 再現 | 左右反転は #070 のセグメンテーションと #059 の画像分類で OOF を改善。#059 は学習時 +0.02879、推論時 +0.01860 |
| [K024](K024-extreme-value-mixture-regression.md) | 仮説 | Elo でまれな極端値が二乗誤差の約78%を占めた。同じ特徴での群別回帰の確率加重は OOF RMSE +0.00217 |
| [K025](K025-identity-weighting-bias-auc.md) | 仮説 | #042 の属性言及ありの陰性を2倍に重み付けすると、OOF BPSN は +0.05288、BNSP は −0.04093 |
| [K026](K026-image-summary-with-tabular.md) | 仮説 | #016 の掲載表に幅・高さ・縦横比・明度などの画像統計を追加すると OOF QWK +0.01143 |
| [K027](K027-qwk-threshold-calibration.md) | 仮説 | #016 の内側 OOF 閾値調整で QWK +0.11453、2段階以上の誤分類率は15.87%から24.35% |
| [K028](K028-evaluate-all-hierarchy-levels.md) | 再現 | #021 の点予測と #022 の分位点予測で、商品×店舗の損失と12階層平均の順位が逆転した |
| [K029](K029-round-count-quantiles.md) | 仮説 | #022 の商品×店舗で分位点を丸めると WSPL が 0.30587 → 0.29824、50%区間の被覆率が 58.13% → 73.22% |
| [K030](K030-missing-value-ablation.md) | 仮説 | #010 の数値欠損処理と #023 の気象欠測フラグで明確な改善なし |
| [K031](K031-absolute-loss-for-mae.md) | 仮説 | #014 の時系列 CV で LightGBM の学習損失を二乗誤差から絶対誤差に変えると MAE が 0.07284 → 0.06913 |
| [K032](K032-class-weighting-and-per-class-recall.md) | 仮説 | #059 でクラス頻度の逆平方根を損失に掛けると Accuracy +0.02383。改善への寄与は最多クラスが最大で、クラス別再現率の変化は混在 |
| [K033](K033-shrink-probabilities-to-label-priors.md) | 仮説 | MoA で個別確率をラベルの出現率へ 10% 縮小すると、希少ラベル帯の OOF Log Loss が 0.005034 → 0.004564 |
| [K034](K034-nms-threshold-for-dense-detection.md) | 再現 | 抑制幅を変えると誤検出と見逃しが逆方向に変わった。#074 は空間、#090 は時間方向で確認 |
| [K035](K035-staged-noisy-supervision.md) | 仮説 | #084 の弱ラベル使用量を揃えると、noisy の後に curated で学習した構成が混合学習より OOF lwlrap +0.01083 |

| [K036](K036-periodic-local-time.md) | 仮説 | #090 で現地時刻を1日周期で符号化すると、系列単位3-fold の OOF AP が +0.049922 |
| [K037](K037-summary-source-overlap.md) | 仮説 | #050 で原文と要約の語句一致率を加えると、課題単位 OOF MCRMSE が 0.85492 → 0.75594 |
| [K038](K038-retrieval-vs-reranking.md) | 条件付き | 候補固定の再ランキングで #058 の MAP@25 +0.01279、#037 の平均 F1 +0.07892、#033 の MAP@12 +0.00467。候補 Recall の改善は最終指標に対応しないことがある（K050） |
| [K039](K039-sensor-event-window-length.md) | 仮説 | #091 の過去3秒窓は過去1秒窓より Turn AP +0.03172、StartHesitation AP −0.00870 |
| [K040](K040-l1-for-original-scale-mae.md) | 仮説 | #011 で元尺度 L1 は L2 より全体 MAE が 29.02 改善し、実測上位 1% の MAE は悪化 |
| [K041](K041-patent-peer-similarity.md) | 仮説 | #054 で同じ anchor・CPC の他候補の類似度を加えると OOF Pearson +0.00118 |
| [K042](K042-empty-vs-positive-mask-dice.md) | 仮説 | #072 で非空 Dice が +0.09924 でも、空マスクの誤検出が増えて全体 Dice は −0.28264 |
| [K043](K043-weather-by-meter.md) | 仮説 | #023 の気象追加は全体 RMSLE −0.068107。冷水・蒸気・温水は改善し、電力は +0.03759 |
| [K044](K044-recency-weighting-and-weekly-gini.md) | 仮説 | #005 の新しさ重視の学習重みは3窓平均安定性スコア −0.04699。最初の窓で週別 Gini の低下が大きくなった |
| [K045](K045-measurement-error-weighting.md) | 仮説 | #094 で荒れたラベルの除外は CV MCRMSE −0.01871、逆分散重みは −0.02467。Private でも改善したが優劣は逆転 |
| [K046](K046-base-pair-probability-features.md) | 仮説 | #094 で対合確率による相手位置の集約を GRU に足すと CV −0.00814、Private −0.00839 |
| [K047](K047-fold-local-pseudo-labels.md) | 仮説 | #094 で fold 内教師の疑似ラベルは CV −0.00565、Private −0.00226 |
| [K048](K048-collection-round-shift-vs-length.md) | 仮説 | #094 は CV ≈ Public だが Private は約 0.12 悪い。長さ外挿の位置はむしろ誤差が小さく、別の収集ラウンドが主因 |
| [K049](K049-short-window-realized-volatility.md) | 仮説 | #025 で直近300秒・150秒の板 RV を10分 RV に加えると OOF RMSPE −0.01134 |
| [K050](K050-candidate-count-vs-ranker.md) | 仮説 | #033 で候補を100→300件にすると Recall は約2倍だが MAP@12 −0.00182。教師の顧客を揃えると −0.00011 で、低下の大半は教師の変化に対応 |
| [K051](K051-band-limit-waveform-classification.md) | 仮説 | #092 の 1D CNN で 25〜500 Hz の帯域制限を加えると OOF AUC 0.50067 → 0.72396 |
| [K052](K052-multi-detector-waveform-input.md) | 仮説 | #092 の帯域制限・白色化した波形で、1 検出器から 3 検出器に増やすと OOF AUC 0.73009 → 0.78815 |
