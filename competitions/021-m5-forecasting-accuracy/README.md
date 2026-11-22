# #021 M5 Forecasting - Accuracy

- 深さは Standard、計算負荷は L。
- Walmart の商品×店舗の日次販売数を28日先まで予測する。暦、州別 SNAP、週次価格、一日ごとの販売履歴を利用できる。
- 30,490 本の最下層系列を12階層に集計し、42,840 系列の WRMSSE を最小化する。提出は `id,F1,...,F28` の CSV で、Public と Private を合わせて60,980行。
- 終了後の評価は Kaggle のファイル提出を利用する。[公式データ](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) · [主催者の評価ガイド](https://storage.googleapis.com/kaggle-forum-message-attachments/772349/15032/M5-Competitors-Guide-Final-10-March-2020.pdf)

## 進捗

- [x] P0. データ取得、ダミー提出、採点確認
- [x] P1. EDA と予測仮説の記録
- [x] P2. 情報締切の検査と分割の固定
- [x] P3. Baseline の計測
- [x] P4. 自力の仮説の比較
- [x] P5. 上位解法2件の調査
- [x] P6. 上位解法の着眼点を1件再現
- [x] P7. CV 選定、最終提出、誤差分析
- [x] P8. レポートと知見の保存

暦・州別 SNAP を追加した E02 を採用した。平均 CV WRMSSE は 0.69883、Kaggle Public は 0.72569、Private は 0.61599。[実験レポート](report.md)

## 事前仮説

| run | E01 への変更 | 予想 |
|---|---|---|
| E02 | イベントの有無と州別 SNAP を追加 | 暦要因のある日の誤差が下がる |
| E03 | 予定価格と締切前28日の平均価格に対する比を追加 | 価格変更のある系列の誤差が下がる |
| E04 | 最終非ゼロ売上からの経過日数と直近28日間の非ゼロ日率を追加 | 間欠需要系列の誤差が下がる |
| E05 | 1位の公開前処理から対象日基準の売上ラグ28〜42日を追加 | 曜日パターンとその前後の売上を特徴に持つことで 12階層平均 WRMSSE が下がる |

## 検証と再現

`m5-28d-v1` は `d_1829`、`d_1857`、`d_1885` の直後の28日間を順に検証する。売上履歴、RMSSE の尺度、売上金額による重みは起点までで打ち切る。対象日の暦と予定価格は予測窓まで参照する。学習例は `d_1096` から7日間隔で作り、予測期間28日分の実績が外側の締切までにそろう起点を採用する。

E00 は過去4週の同曜日平均。E01 は系列カテゴリ、曜日・月、何日先か、締切時点の直近7・28日平均を使う全系列共通 LightGBM。CPU 8スレッド、seed 42、300ラウンドで固定する。seed 42 で全系列を16群に分け、各学習起点で1群ずつ持ち回り、その群の28日分を学習に使う。各 CV 窓の学習に全系列が登場し、採点は全系列で行う。3窓の平均 WRMSSE で選定し、同値なら実験番号が小さい方を優先する。選定後に `d_1913` と `d_1941` を締切として再学習する。

初回実行時のコマンドを以下に示す。

```powershell
dev download 21
uv run python competitions/021-m5-forecasting-accuracy/src/dummy.py
uv run python competitions/021-m5-forecasting-accuracy/src/eda.py
uv run python competitions/021-m5-forecasting-accuracy/src/train.py cv E00
uv run python competitions/021-m5-forecasting-accuracy/src/train.py cv E01 --run-id E01-r2
uv run python competitions/021-m5-forecasting-accuracy/src/train.py cv E03 --run-id E03-r2
uv run python competitions/021-m5-forecasting-accuracy/src/train.py cv E02 E04 E05
uv run python competitions/021-m5-forecasting-accuracy/src/analyze.py E01-r2 E02 E03-r2 E04
uv run python competitions/021-m5-forecasting-accuracy/src/train.py select E00 E01-r2 E02 E03-r2 E04 E05
uv run python competitions/021-m5-forecasting-accuracy/src/train.py final final-v1
```

実験値と成果物のハッシュは [実験ログ](experiments.jsonl)、選定時のスコアは [選定記録](selection.json) に保存した。予測と提出 CSV は `outputs/runs/` に保存する。再計測では各実験を `cv <実験名> --run-id <新しいID>` で実行し、選定にその6件の ID と `--output <新しい選定ファイル>` を渡す。続けて `final <新しいID> --selection <新しい選定ファイル>` で提出を再生成できる。
