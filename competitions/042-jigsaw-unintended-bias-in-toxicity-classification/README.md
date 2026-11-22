# #042 Jigsaw Unintended Bias in Toxicity Classification

英語コメント本文から有害性を予測する。深さは **Standard**、CPU で実施する。全体 AUC と、属性に言及するコメントに関する3種類の AUC を比較する。提出列は `id,prediction`。 [Kaggle 公式概要・指標](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification)

## 進捗

- [x] P0. データ取得と公式採点の確認
- [x] P1. 問題設定・EDA と仮説の事前記録
- [x] P2. リーク経路の確認と分割の固定
- [x] P3. ベースラインの CV 固定
- [x] P4. 自力の仮説検証
- [x] P5. 上位解法の調査
- [x] P6. 連続ラベル学習のアブレーション
- [x] P7. 最終モデルの固定、公式公開データでの Public・Private ローカル採点、エラー分析
- [x] P8. 振り返りと知見の記録

採用モデル E03-r2 の公式 Late Submission は、非公開 Kaggle Notebook v1 で採点された。Private は **0.87895**。詳細は [振り返り](report.md) に記録した。 [E03-r2 Notebook](https://www.kaggle.com/code/shion5/kumite-042-e03-r2)

## 評価環境と EDA

公式データの学習行は 1,804,874 件で、有害スコアが 0.5 以上の行は 144,334 件。小文字化と空白正規化で重複する本文は 40,884 行あり、そのうち 808 グループで二値ラベルが食い違った。公式対象9属性の注釈はそれぞれ 77.55% が欠損している。属性に言及する陽性行は属性によって 1,030〜7,311 件と差がある。

検証は正規化本文をグループとする層化3分割、seed 42。記事や投稿時期を単位とした外挿は未検証。属性ラベルには 0.5 の閾値を使い、欠損は指標計算上の非所属に含める。予測入力は本文のみ。学習時の語彙・IDF と E04 の重みは学習 fold 内で作る。OOF では全体 AUC と Subgroup・BPSN・BNSP の指数 −5 のべき平均を等重みで集約する。

3 fold は各 601,624〜601,625 行で、陽性は各 48,111〜48,112 行。E01 の OOF は 0.85835。全体 AUC は 0.91659、BPSN のべき平均は 0.75487 だった。

E00 の定数 0.5 は Notebook 経由で採点された。Private は **0.50000**、履歴上の Public 表示は **0.00000**。CSV の直接提出は Notebook 限定のため HTTP 400 で拒否された。実データの検査には `train.csv`、`test.csv`、`sample_submission.csv` を使用する。公開済みのテストラベルはモデル選定に使わない。 [公式データ](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data) · [E00 Notebook](https://www.kaggle.com/code/shion5/kumite-042-e00-scoring-check)

## 実行前の仮説

| Run | E01 からの変更 | 予想 |
|---|---|---|
| E01 | 単語 unigram TF-IDF とロジスティック回帰 | 比較の基準 |
| E02 | 単語 bigram 追加 | 属性語の周辺文脈が効き、BPSN が上がる |
| E03 | 文字 3–5 gram 追加 | 綴り崩しを拾え、全体 AUC と BNSP が上がる |
| E04 | 属性言及を注釈された陰性の学習重みを2倍 | BPSN が上がり、BNSP が下がる可能性がある |
| E05 | 学習の二値ラベルを有害判定の投票比率に変更 | 注釈者間で意見が割れたコメントの情報を残すと、Subgroup AUC のべき平均が上がる |

E05 は4位 COMBAT WOMBAT の損失設計から、有害性の投票比率を二値化せずに学習する着眼点だけを取り出す。E01 の特徴量、正則化、評価時の二値化閾値を固定する。 [4位チームの公開コードと解説](https://github.com/iezepov/combat-wombat-bias-in-toxicity)

## 再現手順

リポジトリ直下で実行する。`train.py` は `RunStore` を通して OOF、fold、指標内訳、提出用 CSV を保存する。やり直しには `--run-id` で別名を使う。E02 以降は E01 の保存済み分割を検証して再利用する。基準 run を改名した場合は `--baseline` でその ID を渡す。

```powershell
dev download 42
$src = 'competitions/042-jigsaw-unintended-bias-in-toxicity-classification/src'
$env:OPENBLAS_NUM_THREADS = '4'
$env:OMP_NUM_THREADS = '4'
foreach ($trial in 'E00', 'E01', 'E02', 'E04') {
    uv run python "$src/train.py" $trial
    if ($LASTEXITCODE -ne 0) { throw "$trial が失敗しました" }
}
uv run python "$src/train.py" E03 --run-id E03-r2 --jobs 3
uv run python "$src/train.py" E05 --jobs 3
uv run python "$src/select_model.py" E01 E02 E03-r2 E04 E05
uv run python "$src/external.py" public
uv run python "$src/external.py" private
```

文字 n-gram の逐次実行は計測前に中断し、同じ分割とモデル設定で3 fold を並列にした `E03-r2` を使う。メモリを抑える場合は `--jobs 1` で実行できる。

`select_model.py` は OOF の比較と本文グループ単位のペアブートストラップで採用 run を固定する。`external.py` は選定記録を検証してから公式の公開済みテストラベルで採点する。配布版の `test.csv` は Private と ID が一致する。Public は同じ fold のモデルを再学習して本文を予測する。

提出用 Notebook は `uv run python "$src/kaggle_notebook.py" <Kaggleユーザー名> <run_id> <新規出力先>` で生成する。Windows では `$env:PYTHONUTF8 = '1'` を設定してから `kaggle kernels push -p <出力先>` で非公開 Notebook を実行する。完了したバージョンを `kaggle competitions submit jigsaw-unintended-bias-in-toxicity-classification -k <ユーザー名/Notebook名> -v <バージョン> -f submission.csv -m <説明>` で提出する。
