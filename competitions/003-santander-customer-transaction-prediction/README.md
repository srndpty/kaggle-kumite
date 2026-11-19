# #003 Santander Customer Transaction Prediction

匿名化された数値特徴から顧客が特定の取引を行うか予測する。深さは **Standard**。CPU で、出現頻度・非線形な主効果・交互作用を比較する。

- 主催は Banco Santander。2019 年開催、賞金総額 $65,000。評価は ROC AUC。提出 CSV は `ID_code,target` の 2 列で、`target` に陽性確率を入れる。 [Kaggle 公式概要](https://www.kaggle.com/c/santander-customer-transaction-prediction/overview)
- 公式ページに Late Submission の表示がある。テストには採点対象外の行が含まれる。 [Kaggle データ仕様](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data)

## 進捗

- [x] P0. 2026-11-19 に規約承諾後、公式データの取得とダミー採点を確認した。定数 0.5 の予測で Public・Private ROC AUC ともに 0.50000。
- [x] P1. 実データ EDA と事前仮説の記録。
- [x] P2. seed 42 の層化 5-fold を固定し、200,000 行を 1 回ずつ検証。
- [x] P3. E01 の OOF AUC 0.85949、Late Submission の Public 0.86096、Private 0.86065。
- [x] P4. 自力で立てた仮説を E02〜E05 で検証。
- [x] P5. 1 位・2 位・17 位の本人 writeup を調査。出典と比較は [レポート](report.md#上位解法とアブレーション)。
- [x] P6. 値が反復する場合だけ生値を残す追加特徴を E01 に加えて検証。E06 の OOF AUC は 0.86422。
- [x] P7. OOF で E05 を固定。Public 0.89250、Private 0.88914。OOF の順位帯別エラーを集計。
- [x] P8. 実測結果を [レポート](report.md) と知見カード K020・K021 に保存。

## 検証設計

学習・テストはそれぞれ 200,000 行、特徴量は 200 列。学習データの陽性率は 10.049%、ID 順の十分位別では 9.83〜10.40%。両データに欠損・Inf はなく、学習データに定数列・完全重複行はなかった。学習データで特徴値が一度だけ出現する行の割合は、列によって 0.018〜71.839% と幅があった。

既定は seed 42 のシャッフル付き層化 5-fold。特徴量が完全一致する行がある場合は層化 Group K-fold で同じ fold に配置する。ID は EDA と提出照合に使う。頻度・標準化・ビン境界の推定は学習 fold に限定する。テストの特徴量は推論に使い、提出順は雛形の ID に揃える。

## 事前仮説

| Run | 比較基準 | 変更と予想 |
|---|---|---|
| E01 | — | 生の数値特徴を fold 内で標準化し、L2 ロジスティック回帰で予測する。 |
| E02 | E01 | 列別の出現回数の `log1p` を追加する。値の反復に情報があれば AUC が上がる。学習行自身は集計から除く。 |
| E03 | E01 | 生値を 16 分位点ビンの疎な one-hot 表現に置き換える。主効果が非線形なら AUC が上がる。 |
| E04 | E01 | 生値を使う深さ 1 の LightGBM に切り替える。非線形な主効果が効くなら E03 と同じ方向の改善を予想する。 |
| E05 | E04 | 木の最大深さを 3 にする。主効果が中心なら交互作用を許しても AUC の改善は小さい。 |

E04 と E05 は 600 本、学習率 0.1 を固定した。モデルの設定と分割・データのハッシュは [実験ログ](experiments.jsonl) に、実測値は [レポート](report.md) に記録した。

## 再現手順

規約承諾とデータアクセスを確認したうえで、リポジトリのルートから実行する。
初回の再現では記録済みの E01〜E06 と衝突しないよう、接尾辞 `r2` を使う。再度実行するときは、`$suffix` を `r3` など全実験で未使用の値に変更する。

```powershell
$d = 'competitions/003-santander-customer-transaction-prediction'
$suffix = 'r2'
$baseline = "E01-$suffix"
dev download 3
uv run python "$d/src/train.py" --dummy "$d/outputs/dummy.csv"
dev submit 3 "$d/outputs/dummy.csv" -m 'P0 dummy'
dev submissions 3
uv run python "$d/src/train.py" --eda
uv run python "$d/src/train.py" --run-id $baseline --variant E01
dev submit 3 "$d/outputs/runs/$baseline/submission.csv" -m "$baseline baseline"
foreach ($variant in 'E02', 'E03', 'E04', 'E05') {
    uv run python "$d/src/train.py" --run-id "$variant-$suffix" --variant $variant
}
dev compare 3 $baseline "E02-$suffix" "E03-$suffix" "E04-$suffix" "E05-$suffix"
dev compare 3 "E04-$suffix" "E05-$suffix"
uv run python "$d/src/train.py" --run-id "E06-$suffix" --variant E06
dev compare 3 $baseline "E06-$suffix"
```

P6 まで実施したら OOF AUC で候補を選ぶ。上位 2 件の差が 0.001 未満の場合は、それぞれ `--seed 43` と別 run ID で再実行し、両 seed の平均 AUC で決める。同点は seed 42 の fold 別 `predict_sec` の合計が短い方を選ぶ。提出には選んだ手法の seed 42 の成果物を使う。最終候補を確定した後に提出し、取得可能な外部スコアを比較する。

実際に選んだ E05 は次点 E03 を 0.00387 上回った。最終外部評価には `outputs/runs/E05/submission.csv` を使った。

[selection.json](selection.json) は Late Submission 採点後に既存の実験ログから作成した監査記録。候補、選定規則、提出 SHA256 を収録している。

実験番号を使う run ID は `--variant` と揃える。`E05-s43` のような接尾辞付きの ID も使える。ロジスティック回帰で収束警告が出た場合は、run の保存前に学習を停止する。

`uv run python "$d/src/analyze.py" --run-id <採用run>` で、予測順位の上位・下位 10% にある陰性数・陽性数を、学習 fold で未出現だった値の列割合別に集計する。所見が既存カードと一致する場合は証跡を追記する。独立した新しい知見は新規カードにまとめる。
