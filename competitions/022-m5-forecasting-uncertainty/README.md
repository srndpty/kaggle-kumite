# #022 M5 Forecasting - Uncertainty

深さは Standard、計算負荷は L。#021 と同じ Walmart の販売データを使い、12階層・42,840系列の28日先までの9分位点を予測する。主評価は WSPL。点予測は過去4週の同曜日平均で固定し、予測残差の扱いを比較する。[公式評価仕様](https://www.kaggle.com/competitions/m5-forecasting-uncertainty/overview)

## 事前仮説

| Run | E01 への変更 | 予想 |
|---|---|---|
| E01 | 直近12回の連続した28日窓の予測残差から、系列別の経験分位点を求める | 同曜日平均を中心とする区間の基準になる |
| E02 | 過去28日平均売上（下限1）で尺度化した残差を階層内で共有する | 残差の標本が不足する系列で極端分位の損失が下がる |
| E03 | 残差を予測先の週ごとに分ける | 後半週の区間被覆率が改善する |
| E04 | 較正を直近6窓に短縮する | 需要変化への追従が速まり、裾は不安定になりやすい |

## 検証設計

`m5-uncertainty-28d-v1` は `d_1829`、`d_1857`、`d_1885` の翌日から28日を検証する。較正窓の実績、採点尺度と重みの計算は起点当日までで打ち切る。階層ごとに実績と点予測を集計してから分位点を推定し、分位点は非負に丸める。3窓の平均 WSPL を最小にした実験を選び、同値なら実験番号の若い方を採用する。

## P0 と EDA

ゼロ予測の提出は採点完了。Public WSPL は 3.66597、Private は 3.63393。公式サンプルとは全771,120行の ID と順序を照合した。

初回 CV 起点までの直近336日で、系列別ゼロ日率の中央値は全体集計で 0%、商品で 4.17%、商品×州で 28.87%、商品×店舗で 65.18%。日販平均の系列間中央値は商品で 6.19 個、商品×州で 1.90 個、商品×店舗で 0.55 個。最下層ではゼロが多く、区間の被覆率は離散性の影響を受ける。

## 上位解法の着眼点

E02〜E04 の保存後に、参加者本人の資料を2件確認した。

- **2位 GoodsForecast** は、過去需要のヒストグラム、特異スペクトル分析によるトレンド、機械学習による外部要因の予測を組み合わせ、分位点に丸めのヒューリスティックを適用した。[本人論文の要旨](https://ideas.repec.org/a/eee/intfor/v38y2022i4p1434-1441.html) · [公開実装の再現手順](https://github.com/Mcompetitions/M5-methods/tree/master/Code%20of%20Winning%20Methods/U2)
- **6位 de Rezende ら** は、状態空間モデルと負の二項分布で過分散の離散需要を表し、モンテカルロで将来の軌跡を生成した。商品×店舗では最高スコアだった。[本人論文の要旨](https://www.sciencedirect.com/science/article/pii/S0169207021001801)

**E05 の事前仮説**は、商品×店舗の予測だけを最近接整数に丸めると、少量販売系列の分位点損失が下がること。2位の丸めの着眼点を E01 に適用した単独比較とし、丸め規則はこの実験用に固定した。

## 再現コマンド

リポジトリのルートで実行する。以下は初回の run ID を使う。

```powershell
dev download 22
$c = "competitions/022-m5-forecasting-uncertainty"
uv run python "$c/src/dummy.py"
dev submit 22 "$c/outputs/dummy.csv" -m kumite-p0-zero
uv run python "$c/src/eda.py"
uv run python "$c/src/train.py" cv E01
uv run python "$c/src/train.py" baseline E01
dev submit 22 "$c/outputs/runs/baseline-v1/submission.csv" -m kumite-baseline-E01
uv run python "$c/src/train.py" cv E02 E03 E04
uv run python "$c/src/train.py" cv E05
uv run python "$c/src/train.py" select E01 E02 E03 E04 E05
uv run python "$c/src/train.py" final final-v1
uv run python "$c/src/analyze.py" E01 E02 E03 E04 E05
dev submit 22 "$c/outputs/runs/final-v1/submission.csv" -m kumite-final-v1-E05
dev submissions 22
```

再計測する場合は `cv <実験名> --run-id <新しいID>` を実験ごとに実行し、`select` に新しい5件の ID と `--output <新しい選定ファイル>` を渡す。`final <新しいID> --selection <新しい選定ファイル>` で CSV を再生成できる。[実験ログ](experiments.jsonl) · [選定記録](selection.json) · [実験レポート](report.md)

## 提出後の選定監査

初回提出時の記録は [selection.json](selection.json) に保存している。レビュー対応では、全候補の保存済み予測と尺度・重みから3窓の WSPL を再計算し、実験ログと照合した。照合した各候補のスコア・予測の SHA-256、元の選定記録の SHA-256（改行は LF）、照合時刻は [selection-audit.json](selection-audit.json) に記録した。

同じ監査を再実行する場合は、空いている出力名を指定する。

```powershell
$c = "competitions/022-m5-forecasting-uncertainty"
uv run python "$c/src/train.py" select E01 E02 E03 E04 E05 --audit-existing "$c/selection.json" --output "$c/selection-audit-r2.json"
```

提出当時の CV 成果物で最終 CSV を再生成する場合は、`final <新しいID> --selection "$c/selection-audit.json"` を使う。新しい CV で作る選定記録には、選定時刻と成果物のハッシュが入る。

## 進捗

- [x] P0. 公式データの取得、ダミー提出と採点確認
- [x] P1. EDA と事前仮説の記録
- [x] P2. 情報締切の検証と CV 窓の固定
- [x] P3. E01 の CV と Late Submission
- [x] P4. E02〜E04 の単独比較
- [x] P5. 上位入賞者2組の解法を確認
- [x] P6. 分位点の丸めを E05 として比較
- [x] P7. E05 の最終提出と区間別の誤差分析
- [x] P8. 実験レポートと知見の保存

採用した E05 の平均 CV WSPL は 0.20305。Kaggle は Public 0.20150、Private 0.20434。[実験レポート](report.md) · [分位点の丸めと被覆率](../../knowledge/K029-round-count-quantiles.md)
