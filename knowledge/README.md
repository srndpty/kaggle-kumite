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
| [K011](K011-group-split-vs-random-split.md) | 仮説 | グループにほぼ重複した行が集まるデータでは、ランダム分割は 19〜30 倍楽観的で、augmentation の採否も逆転する |
| [K012](K012-test-neighbour-averaging.md) | 仮説 | test の中の近傍と予測を平均すると logloss が下がる（State Farm Private -0.050）。OOF では効果を 2 倍に見積もる |
| [K013](K013-categorical-splits-vs-numeric.md) | 仮説 | 名義カテゴリの数値扱いがLightGBMのカテゴリ分割より良かった（Porto Seguro、3 seed平均Gini +0.016054） |
| [K014](K014-ranking-vs-probabilities.md) | 仮説 | 順位と確率を分けて評価する。Porto Seguroの陽性重み付けはGini・LogLossとも悪化、単調変換はGini不変でもLogLoss悪化 |
| [K015](K015-feature-group-ablation.md) | 仮説 | 特徴群を除く比較実験も有効。Porto Seguroのcalc除外で3 seed平均Gini +0.002654、原因は未特定 |
