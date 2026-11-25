# #026 Optiver - Trading at the Close

- URL: https://www.kaggle.com/competitions/optiver-trading-at-the-close
- カテゴリ: B / 難度: 5 / 計算負荷: L
- 今回の比較実験テーマ: オンライン推論API・時系列検証
- 深さ: Standard

## コンペ概要

NASDAQ の引け前10分間に観測される板とオークション情報から、銘柄ごとの次の60秒の相対価格変化を予測する。

## 基本情報（P0）

- 評価指標は行単位の MAE。Kaggle の非公開 CPU Notebook から公式逐次推論 API を呼ぶ。
- 公式データを取得済み。学習データは 5,237,980 行、481 取引日、200 銘柄。銘柄・取引日・秒位置の組と `row_id` はそれぞれ一意。目的値の欠損は88行。
- 2026-11-26 に定数ゼロの非公開 Notebook v6 を Late Submission として提出し、Private MAE 5.6036 を得た。ダミーは選定対象から除く。
- 公式 API は Python 3.10 向けのバイナリで配布されている。提出用イメージを指定できる Kaggle CLI 2.2.5 で CPU イメージを固定する。

[公式評価・提出条件](https://www.kaggle.com/competitions/optiver-trading-at-the-close/overview) · [ダミー Notebook](https://www.kaggle.com/code/shion5/kumite-optiver-close-p0)

## 進捗

CV run の公開には共通 Quality Gate が必須。評価単位・分割・採点関数を決めたら、
[Quality Gate 仕様](../../docs/agent/experiment-quality-gates.md) に沿って学習・選定・最終 export の入口を実装する。

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
| E01 | 板スプレッドと板・オークションの数量比が MAE を下げる | 生の数量は銘柄間の規模差を含むため、比率が需給を比較しやすくする | 自力 | −0.01003 |
| E02 | 同一銘柄の10秒・60秒差分が MAE を下げる | 価格と不均衡の直近の変化が次の動きに先行しうる | 自力 | −0.00644 |
| E03 | 現バッチ内での中央値差と順位が MAE を下げる | 相対的な需給の偏りを同時刻の他銘柄と比較できる | 自力 | +0.00016 |
| E04 | 引けまでの節目を示す時刻フラグが MAE を下げる | 秒位置を木が直接使う構成に、取引所の挙動が変わる節目を明示する | 6位解法 | 0.00000 |

目的値の標準偏差は0秒で11.60、270秒で13.34、540秒で10.17。板の価格と WAP はそれぞれ220行が欠損し、far_price と near_price は半数を超える行で欠損する。LightGBM には欠損のまま渡す。

## 検証設計

末尾90取引日を30日ずつ3区間に分ける。各 fold の学習は検証区間の開始より前の取引日をすべて使う。`date-forward-v1` の主指標は全検証行の OOF MAE。目的値欠損行の説明変数は同時刻の集約とラグに含める。現バッチと同日の過去バッチだけから特徴量を計算し、翌日には履歴を引き継がない。

[選定計画](selection-p4.json) の候補順は E00、E01、E02、E03。同点許容差は0。定数ゼロの OOF MAE は 6.02155。E00 は現時点の板・オークション入力を使う CPU LightGBM L1 回帰で、seed は42、学習は1,000ラウンドに固定する。

## 上位解法と再現

- 1位は CatBoost・GRU・Transformer を組み合わせ、時系列のホールドアウトでブレンドを選んだ。推論期間中に公開されたラベルを使う追加学習も行った。 [本人の解法記事](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution)
- 6位はニューラルネットの学習途中の評価変動を指数移動平均で抑え、特徴量には引けに向けた特定の秒位置を示すフラグを使った。E04 ではフラグ4列だけを E00 に加える。 [本人の解法記事](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/486040)
- 最終候補は E00-r2、E01-r2、E02-r2、E03-r2、E04 の順で登録した。同点許容差は0。提出 ID の型正規化と E04 の追加後に、初回4候補を同じモデル設定で再実行し、保存 OOF がそれぞれ初回と一致することを確認した。 [最終選定計画](selection-p7.json)

## 実験ログ

詳細は `experiments.jsonl`（`kumite.runs.RunStore` で追記。記録は書き換えず、やり直しは別の run_id）。主要な結果のみ抜粋する。

| run | 変更点（1つだけ） | OOF | Late | 判定 |
|---|---|---|---|---|
| E00 | Baseline | 5.94366 | 未提出 | 基準 |
| E01 | 板とオークションの比率・スプレッド | 5.93363 | 未提出 | 改善 |
| E02 | WAP と不均衡サイズの10秒・60秒差分 | 5.93722 | 未提出 | 改善 |
| E03 | 現バッチの中央値差・百分位順位 | 5.94382 | 未提出 | 改善なし |
| E04 | 240、300、420、480秒の到達フラグ | 5.94366 | 未提出 | E00-r2 と同一予測 |
| F00 | E01-r2 の設定で全量学習 | 5.93363（参照値） | 5.5239 | Late Private 採点済み |

## 再現手順

リポジトリルートから実行する。P0 の Notebook は Kaggle CLI 2.2.5 で push し、完了したバージョンを Code Competition として提出する。最終 export は `outputs/exports/outputs/runs/F00/` に保存される。Kaggle のソース上限に合わせ、export を照合した上で `outputs/deployment/F00/` の Dataset と読み込み用 Notebook に梱包する。すでに公開済みの run ID を再実行するときは、新しい ID と事前選定計画を用意する。

```powershell
$comp = 'competitions/026-optiver-trading-at-the-close'
dev download 26
uv run python "$comp/src/p0.py" <Kaggleユーザー名> "$comp/outputs/p0"
uv run python "$comp/src/run.py" E00 E01 E02 E03
uv run python "$comp/src/run.py" --plan "$comp/selection-p7.json" E00-r2 E01-r2 E02-r2 E03-r2 E04
uv run python "$comp/src/final.py" <Kaggleユーザー名> "$comp/selection-p7.json"
uv run python "$comp/src/deployment/package_kaggle.py" F00
```
