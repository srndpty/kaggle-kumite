# #023 ASHRAE - Great Energy Predictor III

- URL: https://www.kaggle.com/competitions/ashrae-energy-prediction
- カテゴリ: B / 難度: 3 / 計算負荷: L
- 今回の比較実験テーマ: 建物×気象・欠損・時差とリーク
- 深さ: Standard

## コンペ概要

建物情報と時間別の気象から、建物・メーター・時刻ごとのエネルギー計測値を予測する。省エネ改修の効果を評価するには、改修しなかった場合の使用量を推定する必要がある。

- 分野は表形式・時系列回帰。
- 主催は ASHRAE。開催は2019年10月15日〜12月19日、参加は3,614チーム。
- メーターは電力、冷水、蒸気、温水。入力には建物属性、メーター区分、計測時刻、サイトごとの気象を使う。

[公式概要](https://www.kaggle.com/competitions/ashrae-energy-prediction/overview) · [公式データ](https://www.kaggle.com/competitions/ashrae-energy-prediction/data)

## 基本情報（P0 で記入）

- 評価指標: RMSLE（小さいほど良い）
- 提出形式（File / Code / 推論API）: File（`row_id,meter_reading`）
- Late Submission 可否: 2026-11-25 に定数 0 で採点確認。提出 ID 58880404、Public 4.699、Private 4.406。
- データ概要: 公式 CSV 6本。建物とメーターの時間別計測、建物属性、サイト別気象を使う。

## 進捗

- [x] P0. 参加・評価環境の確認（Rules 承諾、データ取得、ダミー提出で採点されるか）
- [x] P1. 問題設定・EDA（予測仮説を3件以上。予想する結果も先に書く）
- [x] P2. 検証分割の設計（リーク経路の洗い出し、fold の確定）
- [x] P3. Baseline の固定（seed・fold・指標・前処理。可能なら初回の Late Submission）
- [x] P4. 自力の仮説検証（上位解法を読む前に 3〜6 実験）
- [x] P5. 上位解法の調査（アプローチの異なる上位解法を 2〜3 件。出典の種類を記録）
- [x] P6. 上位解法の着眼点の再現（1〜3 件をアブレーション）
- [x] P7. 最終評価・エラー分析（事前に決めたモデルの外部評価と CV との整合性）
- [x] P8. 知見の保存（`report.md`、`knowledge/` のカード）

## 仮説

| # | 仮説 | 根拠（EDA） | 出所（自力 / 上位解法） | 結果 |
|---|---|---|---|---|
| H1 | 同時刻の気象値の追加で冷暖房メーターの RMSLE が改善する | 気象と空調需要の関連 | 自力（E01） | 冷水・蒸気・温水は改善、電力は悪化 |
| H2 | 欠測率が高いサイトで気象欠測フラグが効く | サイト別の気象欠測率に差がある | 自力（E02） | 全体で実質同等。サイト1も同等 |
| H3 | 気温ピークから推定した時刻補正は時差が大きいサイトで効く | サイト別の推定時差が −8〜+1 時間 | 自力（E03） | サイト間で結果が分かれ、全体 RMSLE は +0.009790 |

## 実験ログ

詳細は `experiments.jsonl`（`kumite.runs.RunStore` で追記。記録は書き換えず、やり直しは別の run_id）。主要な結果のみ抜粋する。

| run | 変更点（1つだけ） | OOF | Late | 判定 |
|---|---|---|---|---|
| E00 | 建物・メーター・カレンダー | 1.069567 | — | 基準 |
| E01 | 同時刻の気象値 | 1.001460254 | — | E00 から −0.068107 |
| E02 | E01 + 気象欠測フラグ | 1.001460238 | — | E01 と実質同等 |
| E03 | E01 + カレンダー時刻補正 | 1.011251 | — | E01 から +0.009790 |
| E04 | E02 + 24時間前の気温 | 1.002898 | — | E02 から +0.001437 |

## 検証設計

`expanding-28d-v1` は学習期間末尾の連続する28日窓を3つ使い、各窓より前の行で学習する。主指標は全検証行を結合した RMSLE。学習とテストの年度差は、この分割で再現できる範囲を超える。気象はサイトと配布時刻で結合し、時刻補正は学習期間内の気象観測から推定してカレンダー特徴だけに適用する。学習窓の外の目的値を特徴やハイパーパラメータの推定に使わない。

E00 は建物・メーター・カレンダーによる LightGBM 回帰。全実験で seed 42、500ラウンド、`log1p(meter_reading)` を目的値とする。E01 は同時刻の気象値、E02 は E01 に気象欠測フラグ、E03 は E01 に時刻補正を追加する。

追加検証では、より早い月を含むローリング起点を別の `split_version` で比較する余地がある。2016年のデータのみで年度をまたぐ予測を直接検証することはできず、最初の窓の誤差には学習行数と検証時期の両方が影響しうる。

## EDA の観測

学習データは2016年の20,216,100行。サイト別のゼロ率は0.13〜33.63%、メーター別は4.40〜26.93%だった。サイト1の降水深は全行で欠測し、雲量も80.64%が欠測した。公式の建物 CSV には空白付きの面積・階数があり、読み込み時に空白を取り除いて整数化する。

3つの検証窓の開始日は10月9日、11月6日、12月4日。最初の窓の学習期間で推定したサイト別時差は −8〜+1 時間だった。これは気温ピークから再現した値であり、正式なタイムゾーンを表す値としては使わない。

## 再現手順

リポジトリルートで実行する。既に記録済みの条件を再実行する場合は `--run-id` で未使用の ID を指定する。

```powershell
dev download 23
uv run python competitions/023-ashrae-energy-prediction/src/p0.py
uv run python competitions/023-ashrae-energy-prediction/src/analyze.py
foreach ($method in 'E00', 'E01', 'E02', 'E03') {
    uv run python competitions/023-ashrae-energy-prediction/src/run.py $method
    if ($LASTEXITCODE -ne 0) { break }
}
```

## 上位解法と追試

- 1位は外れ計測の除去、気温のラグや平滑化、メーター・サイト・建物別モデルの組み合わせを用いた。 [1位チーム本人の writeup](https://www.kaggle.com/competitions/ashrae-energy-prediction/writeups/isamu-matt-1st-place-solution-team-isamu-matt)
- 2位は特徴を絞り、計測を手作業で点検したうえでサイト×メーターごとに調整した複数モデルを組み合わせた。 [2位チーム本人の writeup](https://www.kaggle.com/competitions/ashrae-energy-prediction/writeups/cha0s-2nd-place-solution)

E04 では1位の気温ラグに着目し、P4 の OOF 最小値を記録した E02 に、同じサイトのちょうど24時間前の気温を1列追加する。建物の熱慣性により冷暖房メーターで改善を予想する。該当時刻の観測がなければ欠測を保持する。検証境界より前の観測気温は利用でき、未来の観測や目的変数を使わない。E04 を将来提出する場合は、学習期間の気象ファイルも参照してテスト初日のラグを計算する。

E04 は E02 より全体で +0.001437。最後の窓では改善したが、前2窓の悪化が上回った。F00 当時の選定条件は OOF RMSLE の実測最小値、厳密な同点では特徴追加の少なさ、次いで run ID。最小値の E02 を全学習行で再学習した F00 を提出した。E01との差は約 0.000000017 で、実質同等と解釈する。選定時は全候補の OOF と分割のハッシュを照合し、元データで RMSLE を再計算する。

```powershell
uv run python competitions/023-ashrae-energy-prediction/src/run.py E04
uv run python competitions/023-ashrae-energy-prediction/src/final.py F00-repro E00 E01 E02 E03 E04 --rmsle-tolerance 0
uv run python competitions/023-ashrae-energy-prediction/src/analyze.py --run-id E02
```

## 今後の選定規則

次回の比較実験では、全検証行の OOF RMSLE に対する絶対許容差を 0.00001 とする。約470万行をまとめた RMSLE は小数第5位までを実用上の比較単位とし、最小スコアとの差が許容差以内の候補から、特徴追加の少なさ、run ID の順で選ぶ。許容差は設定ファイル `selection.json` に保存する。上の再現コマンドは F00 当時の選定規則を使う。

## 最終提出と知見

F00 は全41,697,600件を推論し、Kaggle で採点された。提出 ID は 58882246、**Public RMSLE 1.365、Private RMSLE 1.442**。選定に使った E02 の OOF RMSLE は 1.001460238。提出は `outputs/runs/F00/submission.csv` に保存される。

[振り返りと誤差分析](report.md) · [K030 欠測処理](../../knowledge/K030-missing-value-ablation.md) · [K043 気象特徴のメーター別評価](../../knowledge/K043-weather-by-meter.md)
