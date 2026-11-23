# #093 Mechanisms of Action Prediction

深さは **Standard**。化合物処理後の測定値から複数の作用ラベルの確率を推定する。公式指標は列ごとの二値 Log Loss の平均。対照群は全ラベルを 0 とする。 [公式概要](https://www.kaggle.com/competitions/lish-moa/overview) · [データ仕様](https://www.kaggle.com/competitions/lish-moa/data)

## 進捗

- [x] P0 公式データ取得・ダミー採点
- [x] P1 EDA・事前仮説
- [x] P2 化合物単位の検証分割
- [x] P3 ベースライン・初回外部採点
- [x] P4 自力比較 3 件
- [x] P5 上位解法 2 件の調査
- [x] P6 PCA の単独比較
- [x] P7 CV でモデルを選定・外部採点・ラベル頻度帯別の分析
- [x] P8 レポート・知見カード

## P0〜P2

公式 CSV を取得し、処理群を一律 0.01、対照群を 0 とした非公開 CPU Notebook v1 の採点を確認した。Public **0.02654**、Private **0.02515**。 [E00 Notebook](https://www.kaggle.com/code/shion5/kumite-moa-e00)

学習は 23,814 行（処理群 21,948 行）、3,289 化合物 ID、206 ラベル。ラベル別の陽性行数の中央値は 38.5、陽性を含む化合物数の中央値は 6.5。最少はどちらも 1。対照群 1,866 行の陽性は 0 件。

`moa-drug-group-v1` は `sig_id` 順の `GroupKFold` 5 分割。fold サイズは 4,763 行が 4 つと 4,762 行が 1 つ。化合物 ID は分割に使い、各 fold の処理群だけで標準化と学習を行う。検証 fold に陽性がない列もそのまま集計する。

## 事前仮説

| 実験 | 比較元 | 変更点と予想 |
|---|---|---|
| E01 | | 遺伝子発現・細胞生存率・時間・用量によるラベル別 L2 ロジスティック回帰。単一クラス列は平滑化出現率を使用する。 |
| E02 | E01 | 発現情報を外すと Log Loss が悪化する。 |
| E03 | E01 | 確率を処理群の平滑化出現率へ 10% 縮小すると希少ラベルの過信による損失が減る。 |
| E04 | E01 | 共有隠れ層の MLP に置き換えると希少ラベルの損失が減る。 |
| E01-row | E01 | 同じモデルを行シャッフル 5 分割で検証すると、化合物を共有するため Log Loss が下がる。 |

## 上位解法と E05 の事前仮説

- 1 位 Hungry for Gold は、複数のニューラルネット、段階的なスタッキング、DeepInsight CNN を合成した。2 入力の残差ネットでは遺伝子発現と細胞生存率の PCA 成分を特徴に使った。 [チーム本人の writeup](https://www.kaggle.com/competitions/lish-moa/writeups/hungry-for-gold-1st-place-winning-solution-hungry-)
- 2 位は、表形式特徴を学習可能な変換で並べ替える 1D-CNN を中心に、TabNet と DNN を合成した。一部の非採点ラベルを事前学習に使った。 [本人の writeup](https://www.kaggle.com/competitions/lish-moa/discussion/202256)
- E05 は 1 位の次元圧縮に着想を得て、E01 の標準化済み遺伝子発現ブロックを PCA 64 成分に置き換える。細胞生存率・時間・用量は E01 と同じ。PCA は fold の学習側だけで fit し、成分を再標準化しない。発現情報を圧縮すると独立回帰の損失が下がると予想する。

## 実験結果

| run | 変更点 | OOF Log Loss |
|---|---|---:|
| E00 | P0 と同じ定数確率 | 0.025040 |
| E01-r3 | 独立ロジスティック回帰 | 0.035158 |
| **E02-r3** | **遺伝子発現を除く** | **0.020693** |
| E03-r3 | 出現率へ 10% 縮小 | 0.023966 |
| E04-r3 | 共有隠れ層 MLP | 0.021743 |
| E05-r3 | 遺伝子発現を PCA 64 成分に圧縮 | 0.023067 |
| E01-row-r3 | 診断用の行シャッフル分割 | 0.026924 |

モデル選定は固定した化合物分割の E01〜E05 の最小 OOF Log Loss に従い、E02-r3 を採用した。E00 と行分割は診断用。選定記録と詳細は [selection.json](selection.json) と [実験ログ](experiments.jsonl) に保存した。

固定係数版の Late Submission は E01-r3 が Public **0.03425**、Private **0.03053**。E02-r3 は Public **0.02147**、Private **0.01907**。実験ログの `late_score` には Private を記録した。 [ベースライン Notebook](https://www.kaggle.com/code/shion5/kumite-moa-e01-r3) · [最終 Notebook](https://www.kaggle.com/code/shion5/kumite-moa-e02-r3) · [提出履歴](https://www.kaggle.com/competitions/lish-moa/submissions)

## 再現

ラベル別 L2 回帰は切片を除く係数に L2 ペナルティを掛け、L-BFGS の行列演算で同時に最適化する。L2 の強さは C=1 相当、最大 200 反復。E04 は ReLU 256 ユニット、AdamW（学習率 0.001、重み減衰 0.01）、バッチ 256、20 epoch。乱数 seed は 42。

以下は別の run ID を使う再実行例。ローカルの回帰学習は CUDA が利用できる環境で CUDA を使う。

```powershell
$d = 'competitions/093-lish-moa'
dev download 93
uv run python "$d/src/train.py" prepare
foreach ($trial in 'E00', 'E01', 'E02', 'E03', 'E04', 'E05') {
    $extra = @()
    if ($trial -eq 'E03') { $extra = @('--baseline', 'E01-repro') }
    uv run python "$d/src/train.py" cv $trial --run-id "$trial-repro" @extra
    if ($LASTEXITCODE -ne 0) { throw "実験 $trial に失敗" }
}
uv run python "$d/src/train.py" cv E01 --row --run-id E01-row-repro
uv run python "$d/src/train.py" select E01-repro E02-repro E03-repro E04-repro E05-repro --output "$d/outputs/selection-repro.json"
uv run python "$d/src/train.py" final "$d/outputs/selection-repro.json" "$d/outputs/final-repro.csv"
$owner = 'your-kaggle-username'
uv run python "$d/src/notebook.py" $owner "$d/outputs/final-repro" --selection "$d/outputs/selection-repro.json" --expected "$d/outputs/final-repro.csv"
```

`final` は提出 CSV と同名の NPZ に学習済み係数・前処理を保存する。Notebook 生成時は NPZ をコードセルに同梱し、公開入力でローカルの CSV と比較する。公開入力と係数、Notebook のハッシュは `evidence.json` に残る。Kaggle 上では学習 CSV のハッシュを確認し、同梱係数で推論する。生成した非公開 Notebook は `kaggle kernels push -p <Notebookディレクトリ>` で登録し、完走後に `kaggle competitions submit lish-moa -k <Notebook ID> -v <バージョン> -f submission.csv -m <メッセージ>` で採点する。

ダミー Notebook は `notebook.py` にユーザー名と出力先だけを指定して生成する。ベースラインの再提出には `train.py baseline <E01のrun ID> <出力CSV>` で係数を保存し、`notebook.py` に `--baseline <E01のrun ID> --expected <出力CSV>` を付ける。

[振り返り](report.md) · [K011 分割](../../knowledge/K011-group-split-vs-random-split.md) · [K015 特徴群の除外](../../knowledge/K015-feature-group-ablation.md) · [K033 出現率への縮小](../../knowledge/K033-shrink-probabilities-to-label-priors.md)
