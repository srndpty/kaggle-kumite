# #031 LANL Earthquake Prediction

- カテゴリ B / 難度 4 / 計算負荷 L / 深さ Standard
- 比較テーマは破壊周期を隔離した検証と、振幅・局所変動・周波数の寄与

実験室で収録した音響信号の 150,000 点の窓から、窓の末尾から次の破壊までの秒数を予測する。評価指標は窓単位の MAE。提出は `seg_id,time_to_failure` の CSV。

[公式概要](https://www.kaggle.com/competitions/LANL-Earthquake-Prediction) · [公式データ仕様](https://www.kaggle.com/competitions/LANL-Earthquake-Prediction/data)

## 進捗

- [x] P0. 公式データ取得とダミー採点
- [x] P1. 問題設定・EDA と仮説の事前登録
- [x] P2. 破壊周期単位の固定分割
- [x] P3. Baseline の固定
- [x] P4. 自力の比較実験
- [x] P5. 上位解法の調査
- [x] P6. 着眼点を一つ再現
- [x] P7. 最終評価・エラー分析
- [x] P8. レポートと知見の保存

## P0 の結果

2026-11-26 に公式データを取得し、テスト 2,624 件の ID を照合した。Kaggle のダミー採点は Public MAE 4.01773、Private MAE 6.66993。定数提出は選定候補から除外する。

## P0 の再現手順

リポジトリルートから実行する。ダミー作成時に `train.csv` の存在と、サンプルの ID 集合が `test/` の CSV ファイル名と一致することを確認する。出力はサンプルと同じ順序で全件を `0.0` とする。

```powershell
dev download 31
uv run python competitions/031-LANL-Earthquake-Prediction/src/p0.py
dev submit 31 competitions/031-LANL-Earthquake-Prediction/outputs/p0-dummy.csv -m "P0 constant dummy"
dev submissions 31
```

Kaggle がスコアを返したら P1 に進む。CV 公開・選定・最終 export は [Quality Gate 仕様](../../docs/agent/experiment-quality-gates.md) に従う。

## 事前仮説と比較条件

破壊境界はラベルが直前の行から 1 秒を超えて増加する場所として検出する。記録の両端の周期と周期末尾の端数を除外する。周期内で 150,000 点の非重複窓を作り、窓末尾のラベルを用いる。周期を時系列順に五つの連続ブロックに分け、分割名を `cycle-block-v1` とする。モデル入力は窓内の音響信号のみ。固定済み分割で、外側検証ラベルを変えても予測が変わらないことを検査する。

| run | Baseline からの変更 | 予想 |
|---|---|---|
| B00 | 平均、標準偏差、中央値、最小値、最大値、四分位範囲 | L1 回帰で学習側ラベル中央値の定数予測を上回る |
| E01 | 絶対振幅の 95・99・99.9 パーセンタイル | 大振幅の情報で破壊直前の MAE が下がる |
| E02 | 10 等分した区間の RMS の最大値・中央値・末尾値 | 短い変動を残すと全体 MAE が下がる |
| E03 | 平均除去後の FFT の等幅 8 帯域の相対パワー | 振幅が近い窓でも残り時間を識別できる |

CPU LightGBM の L1 回帰を固定する。seed 42、300 木、学習率 0.05、15 葉、早期終了なし。主評価は全窓の OOF MAE。完全同点は事前登録順で決める。P4 の候補は [事前登録ファイル](plan-p4.json) に記録した。

公式データで 16 境界を確認した。境界直前はすべて 0.0011 秒未満、リセット直後は 7.06〜16.11 秒だった。完全な 15 周期から 4,100 窓を取得した。fold 別の窓数は 886 / 712 / 862 / 801 / 839。窓末尾の残り時間は 0.0069〜16.0691 秒で、中央値は 5.3301 秒。境界一覧、窓の行位置、周期、fold は `outputs/windows.npz` に保存する。

## 上位解法と追試の事前登録

- 1 位 Team Zoo は移動標準偏差・ピーク・MFCC を使い、学習とテストの特徴分布の差を調べて LightGBM とニューラルネットを組み合わせた。[本人の解法記事](https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94390)
- 2 位 Jun Koda は大振幅ピークを除いた信号の標準偏差を重視し、学習とテストの分布差を検討した。[本人の解法記事](https://www.kaggle.com/competitions/LANL-Earthquake-Prediction/writeups/jun-koda-2nd-place-solution) · [主催者・参加者共著論文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7865129/)
- 9 位 mykper は残り秒数を周期内の進行度に近い尺度に変換して学習した。[本人の解法記事](https://www.kaggle.com/competitions/LANL-Earthquake-Prediction/discussion/94609)

P4 の勝者 E01 に、1 位の着眼点である 50 点の移動標準偏差の 20 パーセンタイルを 1 列加える。E04 では公式の生波形を使い、静かな区間の変動量が振幅の上位分位点を補い、全体 MAE を下げると予想する。固定窓と分割を維持し、最終計画には新しい run ID で P4 の全候補も登録する。

## 最終候補の再現

以下は公式データ取得後にリポジトリルートで実行する。既存の実験ログと選定記録を保持するため、[最終候補計画](plan-final.json) と同じ順序・許容差で run ID に再現用の接尾辞を付ける。再実行時は未使用の接尾辞を使う。

```powershell
$base = "competitions/031-LANL-Earthquake-Prediction"
uv run python "$base/src/windows.py"
$plan = "$base/outputs/plan-repro.json"
$settings = Get-Content "$base/plan-final.json" -Raw | ConvertFrom-Json
$settings.candidates = @($settings.candidates | ForEach-Object { "$_-repro" })
$settings | ConvertTo-Json | Set-Content -Encoding utf8 $plan
uv run python "$base/src/run.py" train --plan $plan
uv run python "$base/src/run.py" select --plan $plan
uv run python "$base/src/run.py" export --plan $plan --export-id F00-repro
```

学習窓 ID（開始・終了行と周期）の SHA-256 は `26053198dba7af73c568138a08a8b0526ad64d236098013004f0587bb529dfdc`。P4 と最終比較で、窓 ID、境界、ラベル、fold、既存の 20 特徴列が一致することを確認した。

## 結果

最終採用は E04。全窓 OOF MAE は 2.24701、F00 の Late Public MAE は 1.52295、Late Private MAE は 2.62490。候補比較と時間帯別の誤差は [レポート](report.md) を参照。
