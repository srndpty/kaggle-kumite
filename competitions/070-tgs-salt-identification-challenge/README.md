# #070 TGS Salt Identification Challenge

- 深さ: Standard
- 課題: 地震画像から塩の領域を画素単位で予測する。
- 評価: 画像単位で IoU が 0.50〜0.95 の10閾値を厳密に超える割合を平均する。両方空なら1点。
- 提出: `id,rle_mask` の CSV。列優先・1始まり。空マスクは空欄。
- [Kaggle 公式概要](https://www.kaggle.com/competitions/tgs-salt-identification-challenge/overview)

## 進捗

- [x] P0. 全空マスクの Late Submission が採点された。Public 0.38000、Private 0.38978。
- [x] P1. データ概要・深度と塩被覆率の関係を確認。
- [x] P2. 被覆率で層化し、同一・左右反転で一致する画像をグループ化した5分割を確認。
- [x] P3. E00 の5-fold OOF と Late Submission を確認。
- [x] P4. 自力の仮説3件を検証。
- [x] P5. 上位解法3件を調査。
- [x] P6. 1位解法の推論時左右反転を E04 で検証。
- [x] P7. E03 を固定して Late Submission を採点し、被覆率別にエラー分析。
- [x] P8. レポートを作成し、知見カード K002 と K023 に記録。

## データと検証

公式データは学習4,000枚、テスト18,000枚の101×101グレースケール画像。学習の空マスクは1,562枚（39.05%）、平均被覆率は24.80%。被覆率の区分 `0`、`(0, 0.1]`、`(0.1, 0.5]`、`(0.5, 1]` の件数は順に1,562、576、931、931枚。

| 深度順の四分位（各1,000枚） | 深度範囲 | 空マスク率 | 平均被覆率 |
|---|---:|---:|---:|
| 1 | 51–351 | 56.1% | 15.22% |
| 2 | 351–514 | 29.3% | 31.05% |
| 3 | 514–669 | 30.0% | 29.27% |
| 4 | 669–959 | 40.8% | 23.64% |

画素が同一、または左右反転で一致する学習画像は80枚の1グループで、全件が空マスクだった。グループ全体を fold 0 に配置した。検証は各 fold 800枚。テストの430枚が同じ画像グループに該当した。地震断面の由来 ID は使用できないため、近接パッチ由来の類似は残り得る。

## 学習前の仮説

| Run | 変更 | 予想 |
|---|---|---|
| E00 | U-Net、BCE、閾値 0.50 | 比較の基準 |
| E01 | 外側学習側の内側検証だけで画素閾値を決める | 空画像の誤検出と小領域の見逃しのバランスが改善する |
| E02 | BCE と soft Dice の等重み平均を使う | 小面積のマスクの IoU が改善する |
| E03 | 学習時に画像とマスクを同時に左右反転する | 少量データへの過学習が減り、OOF が改善する |

## 実験結果

| Run | OOF | Late Public | Late Private |
|---|---:|---:|---:|
| E00 | 0.69978 | 0.75086 | 0.77971 |
| E01 | 0.70010 |  |  |
| E02 | 0.64660 |  |  |
| **E03** | **0.73200** | **0.74718** | **0.78082** |
| E04 | 0.71370 |  |  |

最終構成は E03 の5モデル多数決に固定した。全 run の OOF 比較で首位で、E00 との差 +0.03223 のペア bootstrap 95% 区間は +0.00724〜+0.07415 だった。

## 再実行

リポジトリのルートで `dev download 70` を実行し、以下を実行する。

```powershell
$src = "competitions/070-tgs-salt-identification-challenge/src"
uv run python "$src/train.py" --run-id E00 --variant baseline
uv run python "$src/train.py" --run-id E01 --variant threshold --baseline E00
uv run python "$src/train.py" --run-id E02 --variant dice
uv run python "$src/train.py" --run-id E03 --variant flip
uv run python "$src/train.py" --run-id E04 --variant flip-tta --baseline E00
dev compare 70 E00 E01 E02 E03 E04
```

分割の seed は42。実験ログは `experiments.jsonl`、モデルと予測は `outputs/runs/` に保存する。最終候補は OOF 首位とベースラインを比較し、画像別スコア差の重複グループ単位 bootstrap 95% 区間の下限が正なら採用する。比較は `uv run python "$src/analyze.py" E00 E01 E02 E03 E04` で再現できる。

外部評価と誤りの内訳は [report.md](report.md) を参照。

## 選定と提出の監査

先に保存済みの run を用意し、次のコマンドをリポジトリのルートで実行する。

```powershell
uv run python "$src/analyze.py" E00 E01 E02 E03 E04 --selection "$src/../outputs/selection.json"
uv run python "$src/replay.py" E03
```

`analyze.py` は現行データと保存済みの分割・OOF を照合し、再採点結果から選定する。指定先に選定証跡を新規作成する。[selection.json](selection.json) は Late Submission 完了後に作成した監査記録で、作成時刻を UTC で保持する。`replay.py` は学習時と同じ CUDA 推論設定で保存済み重みを読み、OOF・テスト投票・提出 CSV を照合する。
