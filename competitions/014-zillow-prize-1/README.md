# #014 Zillow Prize: Zillow’s Home Value Prediction

深さは **Standard**。Zestimate と成約価格の対数差 `logerror` を物件属性と対象月から予測する。
公式指標は MAE。提出は `ParcelId,201610,201611,201612,201710,201711,201712` の CSV。
[公式評価仕様](https://www.kaggle.com/competitions/zillow-prize-1/overview) · [公式データ仕様](https://www.kaggle.com/c/zillow-prize-1/data)

## 進捗

- [x] P0 公式データ取得とゼロ予測の Late Submission 採点
- [x] P1 EDA と事前仮説
- [x] P2 情報締切・年別結合・検証分割の固定
- [x] P3 二乗誤差ベースラインの計測と提出
- [x] P4 自力の仮説 E02〜E04 の検証
- [x] P5 上位2チームの本人資料の調査
- [x] P6 郵便番号ごとの平均評価額のアブレーション
- [x] P7 CV 選定、最終採点と誤差分析
- [x] P8 レポートと知見カード

CV で E04 を選定した。OOF MAE は 0.06909、Kaggle Public は 0.06507、Private は 0.07543。

## 事前仮説

| Run | 比較元 | 変更と予想 |
|---|---|---|
| E02 | E01 | 絶対誤差で学習する。裾に引かれにくくなり全体 MAE が下がる。 |
| E03 | E01 | 学習ラベルを学習行の1・99パーセンタイルでクリップする。二乗誤差のままでも全体 MAE が下がる。 |
| E04 | E02 | 締切までの日数に半減期365日の重みを付ける。後半の検証窓で MAE が下がる。 |
| E05 | E01 | 学習取引から郵便番号別の平均評価額を作り、1特徴加える。地域の水準を捉えて全体 MAE が下がる。 |

## 検証と再現

`zillow-time-v1` は、年別の物件スナップショットを利用できる条件で取引ラベルを時系列検証する評価契約。
検証期間は2017年1–2月、3–4月、5–6月、7–8月の4窓。
各窓の開始日より前の取引で学習し、4窓を合わせた72,812件の MAE で選定する。
各取引は同年の物件スナップショットと結合する。カテゴリ辞書は学習行だけから作り、物件 ID は照合に使う。
2017年の物件スナップショットは10月公開のため、取引ラベルの締切は再現できるが、各 CV 時点での物件属性の可用性には制約がある。
感度分析では、税額と建物・土地・合計評価額の4列を除いた同じ窓で E02 と E04 を再計測した。
既存の CV と外部採点を採用判断の記録とする。

最終予測は各年10月15日より前の公式取引で学習し、その年の物件スナップショットから10〜12月分を生成する。
提出雛形の物件 ID が年別スナップショットに存在しない場合は、対象月以外の属性を欠損として予測し、年ごとの件数を表示する。
選定・誤差分析・最終提出では公式 CSV、取引行順、分割、学習設定と OOF を照合し、OOF から MAE を再計算する。
最終提出では全候補から選定記録を再構成し、一致した選定 JSON を成果物に保存する。

```powershell
dev download 14
uv run python competitions/014-zillow-prize-1/src/dummy.py
dev submit 14 competitions/014-zillow-prize-1/outputs/p0-dummy.csv -m "P0 zero baseline"

$train = "competitions/014-zillow-prize-1/src/train.py"
uv run python $train eda
uv run python $train cv median E01 E02 E03 E04 E05
uv run python $train final baseline-v1 --source E01
dev submit 14 competitions/014-zillow-prize-1/outputs/runs/baseline-v1/submission.csv -m "E01 squared error baseline"
uv run python $train select median E01 E02 E03 E04 E05
uv run python $train analyze E02 E04
uv run python $train analyze E01 E04
uv run python $train cv E02-no-tax E04-no-tax
uv run python $train analyze E02-no-tax E04-no-tax
uv run python $train final final-v1
dev submit 14 competitions/014-zillow-prize-1/outputs/runs/final-v1/submission.csv -m "E04 CV-selected final"
```

別 ID で CV から提出までを再現するには、次のように実行する。

```powershell
$selection = "competitions/014-zillow-prize-1/outputs/selection-replay.json"
uv run python $train cv E04 --run-id E04-r2
uv run python $train select median E01 E02 E03 E04-r2 E05 --output $selection
uv run python $train final final-r2 --selection $selection
```

選定には候補 run のローカル成果物が必要。別環境では候補を再計測し、
実際に生成した run ID を `select` に渡して新しい選定 JSON を作る。

[レポート](report.md) · [実験ログ](experiments.jsonl) · [CV 選定](selection.json)
