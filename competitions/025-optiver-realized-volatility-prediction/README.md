# #025 Optiver Realized Volatility Prediction

- URL: https://www.kaggle.com/competitions/optiver-realized-volatility-prediction
- カテゴリ: B / 難度: 4 / 計算負荷: L
- 今回の比較実験テーマ: 板情報・秒単位の特徴量・RV
- 深さ: Standard

## コンペ概要

注文板と約定の最初の10分間から、同じ銘柄の次の10分間の実現ボラティリティを予測する。評価単位は銘柄と時間バケットの組。

## 基本情報（P0）

- 評価指標: RMSPE（小さいほど良い）。
- 提出形式: ネット無効の Notebook が `row_id,target` の `submission.csv` を生成する。
- 公式データ: 銘柄別の注文板・約定 Parquet と行キーを記載した CSV。ダウンロード版にはテストの冒頭行だけが含まれる。
- Late Submission: 2026-11-26 に定数 0.003 を出す非公開 Notebook v1 を提出し、Late Private RMSPE 1.04387 で採点された。Kaggle の表示上の Public 値は 0.00000。ダミーはモデル選定の対象外。
- 公式データ取得済み。学習 428,932 行はすべて銘柄・時間バケットの一意な組で、時間バケットは 3,830 個。目的値の最小は 0.000105263。

[公式評価・提出条件](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/overview/evaluation) · [公式データ仕様](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/data) · [ダミー Notebook](https://www.kaggle.com/code/shion5/kumite-optiver-p0)

リポジトリルートからダミー Notebook を再生成する。

```powershell
$comp = 'competitions/025-optiver-realized-volatility-prediction'
uv run python "$comp/src/p0.py" <Kaggleユーザー名> "$comp/outputs/p0"
```

## 進捗

CV run の公開には共通 Quality Gate が必須。評価単位・分割・採点関数を決めたら、
[Quality Gate 仕様](../../docs/agent/experiment-quality-gates.md) に沿って学習・選定・最終 export の入口を実装する。

- [x] P0. 参加・評価環境の確認（Rules 承諾、データ取得、ダミー提出で採点されるか）
- [x] P1. 問題設定・EDA（予測仮説を3件以上。予想する結果も先に書く）
- [x] P2. 検証分割の設計（リーク経路の洗い出し、fold の確定）
- [x] P3. Baseline の固定（seed・fold・指標・前処理）
- [x] P4. 自力の仮説検証（上位解法を読む前に 3〜6 実験）
- [x] P5. 上位解法の調査（アプローチの異なる上位解法を 2〜3 件。出典の種類を記録）
- [x] P6. 上位解法の着眼点の再現（1〜3 件をアブレーション）
- [x] P7. 最終評価・エラー分析（事前に決めたモデルの外部評価と CV との整合性）
- [x] P8. 知見の保存（`report.md`、`knowledge/` のカード）

## 仮説

| # | 仮説 | 根拠（EDA） | 出所（自力 / 上位解法） | 結果 |
|---|---|---|---|---|
| E01 | 直近300秒・150秒の板 RV を加えると RMSPE が下がる | 予測対象に近い時間の変動が持続すると予想 | 自力 | −0.01134 |
| E02 | 約定 RV・出来高・注文数を加えると RMSPE が下がる | 約定の頻度と規模が板 RV にない流動性の情報を持つと予想 | 自力 | −0.01098 |
| E03 | 他銘柄の同時刻の板 RV 中央値を加えると RMSPE が下がる | 市場全体の変動が複数銘柄に現れると予想 | 自力 | −0.00248 |
| E04 | 同時刻の板 RV 順位を加えると RMSPE が下がる | 市場内で相対的に変動が大きい銘柄を識別できると予想 | 1位解法から着想 | −0.00307 |

比較は時間バケット単位の 5-fold `time-group-v1` で行う。時系列順は仮定しない。全行 OOF RMSPE を主指標とし、P4 の候補順を E00・E01・E02・E03、同点許容差を 0 とした。

## データと分割

- 板と約定は各112銘柄に分割され、観測の秒位置は両方とも 0–599。約定側には 428,913 組がある。目的値の四分位点は 0.002026、0.003048、0.004738。
- 同一バケットの銘柄を同じ fold に置く。銘柄間の統計は推論時にも観測できる同一バケットの入力だけで計算する。ラベルは学習と採点に使用し、特徴量には渡さない。

## 上位解法と再現

- 1位は近傍集約を LightGBM・MLP・CNN のブレンドに使い、変化の大きい取引活動指標を時間バケット内で順位化した。E04 では順位化の発想を板 RV に適用し、E00 への特徴量追加として検証した。 [本人の解法記事](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970)
- 7位は正規化価格から推定した実価格を使って時間バケット間に有向グラフを張り、経路探索で時系列順を復元した。今回は公開入力から直接計算できる特徴と固定分割に絞った。 [本人のコードと説明](https://github.com/michaelpoluektov/orvp)
- P4 実行後、E04 と最終 Notebook のソースを含む証跡をそろえるため、P4 の4候補を同じ設定で `-r2` として再実行した。最終候補順は E00-r2、E01-r2、E02-r2、E03-r2、E04。許容差は 0。比較前のコミットは作成していない。

## 再現手順

既存の CV 証跡と F00 のソースは、レビュー修正前のコミット `6860db5` に対応する。当時と同じ条件で再生成する場合は、そのコミットの別作業ツリーを使う。次のコマンドは実験ログと出力が空の状態から実行する。

```powershell
$comp = 'competitions/025-optiver-realized-volatility-prediction'
./dev.ps1 download 25
uv run python "$comp/src/run.py" E00-r2 E01-r2 E02-r2 E03-r2 E04
New-Item -ItemType Directory -Force "$comp/outputs/selections" | Out-Null
uv run python "$comp/src/final.py" <Kaggleユーザー名>
```

現行コードには板・約定価格、板数量、集計 RV の検証と、生成 Notebook を Kaggle 相当の入力構成で実行するテストを追加した。入力のないバケットや価格観測が1件だけの窓では RV の欠損を許容する。実行テストでは `OPTIVER_INPUT_ROOT` と `OPTIVER_WORKING_DIR` を一時ディレクトリに向ける。

## 結果

E01-r2 を OOF RMSPE 0.25466 で選定した。全量学習用 Notebook F00 の Late Private RMSPE は 0.25143。 [実験ログ](experiments.jsonl) · [結果とエラー分析](report.md)
