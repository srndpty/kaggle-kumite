# #007 IEEE-CIS Fraud Detection

深さは **Standard**。CPU で取引時の不正確率を予測する。公式指標は ROC AUC、提出列は `TransactionID,isFraud`。 [Kaggle 公式概要](https://www.kaggle.com/competitions/ieee-fraud-detection/overview)

## 進捗

- [x] P0 公式データ取得、定数 0.5 の E00 で Public・Private ともに 0.500000。
- [x] P1 問題設定・EDA と仮説の事前記録。
- [x] P2 検証分割と履歴の情報 cutoff を固定。
- [x] P3 E01 を固定し、Public 0.921238、Private 0.897297 を確認。
- [x] P4 本人確認列・短期履歴・時刻を個別比較。
- [x] P5 1位・2位・10位の手法を調査。
- [x] P6 過去平均取引額に対する比率を E05 で検証。
- [x] P7 E05 を固定して最終提出。Public 0.923145、Private 0.895779。OOF を集団別に分析。
- [x] P8 レポートと知見カードを更新。 [振り返り](report.md)

## EDA・検証設計

学習は 590,540 取引、テストは 506,691 取引。本人確認情報の結合率はそれぞれ 24.42% と 28.01%。学習ではカードキーを構成できる行の 99.41% が同じキーを持つ別の行を含む。28 日単位の不正率は 2.40%～4.17%。`card1,card2` は複数取引をまとめるための近似キーとして扱う。

経過日の学習範囲は 1～182 日、テストは 213～395 日なので、学習末尾から 30 日空けた 28 日の検証窓を3つ作る。検証窓は 99～126 日、127～154 日、155～182 日。主評価は窓別 ROC AUC の単純平均。モデル入力から取引 ID と生の経過秒を除外し、カテゴリ語彙を各 fold の学習行で確定する。検証窓より未来の取引、同時刻の取引、検証ラベルは履歴特徴に使わない。

| Run | 事前仮説 |
|---|---|
| E01 | 取引列だけでベースラインを固定する |
| E02 | 本人確認列の追加で本人確認ありの取引の AUC が上がる |
| E03 | 短期の件数は再出現カードで有効。本分割では学習履歴を凍結するため、30 日の空白で評価時点には件数信号が残らないと予想する |
| E04 | 24 時間周期の特徴で複数の検証窓の AUC が上がる |

## 再現手順

```powershell
./dev.ps1 download 7
uv run python competitions/007-ieee-fraud-detection/src/eda.py
uv run python competitions/007-ieee-fraud-detection/src/dummy.py
./dev.ps1 submit 7 competitions/007-ieee-fraud-detection/outputs/runs/E00/submission.csv -m "Kaggle Kumite P0 E00"
uv run python competitions/007-ieee-fraud-detection/src/train.py E01 base
uv run python competitions/007-ieee-fraud-detection/src/train.py baseline-v1 base --source E01
./dev.ps1 submit 7 competitions/007-ieee-fraud-detection/outputs/runs/baseline-v1/submission.csv -m "Kaggle Kumite E01 baseline-v1"
uv run python competitions/007-ieee-fraud-detection/src/train.py E02 identity
uv run python competitions/007-ieee-fraud-detection/src/train.py E03 history
uv run python competitions/007-ieee-fraud-detection/src/train.py E04 clock
uv run python competitions/007-ieee-fraud-detection/src/train.py E01-s43 base --seed 43
uv run python competitions/007-ieee-fraud-detection/src/train.py E02-s43 identity --seed 43
uv run python competitions/007-ieee-fraud-detection/src/train.py E03-s43 history --seed 43
uv run python competitions/007-ieee-fraud-detection/src/train.py E04-s43 clock --seed 43
uv run python competitions/007-ieee-fraud-detection/src/train.py E05 relative_amount
uv run python competitions/007-ieee-fraud-detection/src/select_run.py E01 E02 E03 E04 E05 E01-s43 E02-s43 E03-s43 E04-s43 --output competitions/007-ieee-fraud-detection/outputs/selection-replay.json
uv run python competitions/007-ieee-fraud-detection/src/train.py final-v1 --selection competitions/007-ieee-fraud-detection/outputs/selection-replay.json
./dev.ps1 submit 7 competitions/007-ieee-fraud-detection/outputs/runs/final-v1/submission.csv -m "Kaggle Kumite E05 final-v1"
./dev.ps1 submissions 7
```
