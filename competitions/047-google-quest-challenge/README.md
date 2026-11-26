# #047 Google QUEST Q&A Labeling

- 深さ：Standard（CPU）
- 予測対象：Stack Exchange の質問と回答に付与された30項目の評価値
- 評価指標：列ごとの Spearman 順位相関の平均（最大化）
- 公式データ：`train.csv`、`test.csv`、`sample_submission.csv`
- 提出：インターネット無効の CPU Notebook が `submission.csv` を生成する。列は `qa_id` と30項目。予測範囲は [0, 1]。

[公式の評価・提出仕様](https://www.kaggle.com/c/google-quest-challenge) · [公式データ](https://www.kaggle.com/competitions/google-quest-challenge/data)

## P0 の再現手順

```powershell
dev download 47
$d = 'competitions/047-google-quest-challenge'
uv run python "$d/src/dummy.py" <Kaggleユーザー名> "$d/outputs/p0"
kaggle kernels push -p "$d/outputs/p0"
kaggle kernels status <Kaggleユーザー名>/kumite-quest-p0
kaggle competitions submit google-quest-challenge -k <Kaggleユーザー名>/kumite-quest-p0 -v <版> -f submission.csv -m 'Kumite QUEST P0 scoring smoke test'
kaggle competitions submissions google-quest-challenge
```

2026-11-26 に公式 CSV 3本を取得した。ローカルで公開テストの476行について、ID 順、出力列、予測の値域と各列の非定数性を検証した。Kaggle では `shion5/kumite-quest-p0` の第1版の実行が完了し、2026-11-26 10:29:34 UTC に採点用として送信した。ダミーはモデル選定から除外する。採点は COMPLETE（Public 0.00586、Private 0.00372）。

## 進捗

- [x] P0. 参加・評価環境の確認
- [x] P1. 問題設定・EDA
- [x] P2. 検証分割の設計
- [x] P3. Baseline の固定
- [x] P4. 自力の仮説検証
- [x] P5. 上位解法の調査
- [x] P6. 上位解法の着眼点の再現
- [x] P7. 最終評価・エラー分析（F01 で外部評価。CV 勝者 F00 は形式エラー）
- [x] P8. 知見の保存

## EDA と比較の事前登録

学習6,079行、公開テスト476行。URL と質問文の連結で3,583グループを確定し、うち1,838グループが複数行を含んだ。5つの検証 fold は1,215–1,216行。ラベルのユニーク値数は3–17、ラベル間の絶対 Spearman の中央値は0.0565。定数列の公式処理は確認できなかったため、ローカル採点では相関0を用いる。

- E01 は出現フィールドを分けると質問系・回答系の順位予測が改善すると見込む。
- E02 は文字 n-gram で表記揺れやコード断片を捉え、単語特徴の弱い項目が改善すると見込む。
- E03 はラベル PCA で相関の強い項目が改善し、独立性の高い項目は悪化し得ると見込む。

主評価は質問グループで分けた固定5-foldの全行 OOF 平均 Spearman。学習前の候補計画は `plan-initial.json`。

## P4 の結果と P6 の事前登録

P4 の保存 OOF を再採点し、E01 を選定した。平均 Spearman は B00 0.29164、E01 0.32460、E02 0.29677、E03 0.29025。

P4 終了後、1位と7位のチーム本人による write-up を調査した。1位は Stack Exchange での事前学習と疑似ラベルを検証し、最終段で一部の列を離散化した。7位は複数 Transformer を平均し、学習ラベル分布に合わせた後処理を列ごとに選んだ。

- [1位チームの解法](https://www.kaggle.com/competitions/google-quest-challenge/writeups/bibimorph-1st-place-solution-with-code)
- [7位チームの解法](https://www.kaggle.com/competitions/google-quest-challenge/writeups/7th-place-post-processing-and-ensembling)

CPU 追試は後処理の1点。E01 を対照とし、E04 では各外側 fold の学習ラベルで値が5種類以下の列について、予測の百分位順位を学習ラベルの経験分位点に写像する。同点の予測には同じ値を割り当てる。正解の同順位を再現すると改善するという仮説を検証する。列の選定と分位点の作成に検証ラベルを使わない。候補順と許容差は `plan-review.json`。

## 実験の再現

```powershell
$d = 'competitions/047-google-quest-challenge'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
uv run python "$d/src/run.py" eda
uv run python "$d/src/run.py" train
uv run python "$d/src/run.py" select
uv run python "$d/src/run.py" train --plan review
uv run python "$d/src/run.py" select --plan review
uv run python "$d/src/run.py" export --plan review --owner <Kaggleユーザー名>
uv run python "$d/src/analyze.py" E04-review --compare E01-review
```

学習コマンドは成果物のない作業環境で実行する。公開済み run ID を使って再学習する場合は、別の作業環境で実験ログと成果物を一緒に初期化するか、新しい候補計画と run ID を使う。CV 最終選定は `selection-review.json` の E04-review。F00 の export 先は `outputs/exports/outputs/runs/F00`。Notebook 版と採点の対応は `submission-evidence.json` に記録する。

F00（提出 58942189）は非公開テストの採点で形式エラーになり、Public と Private の値が返らなかった。公開テストではローカルと Kaggle の出力が最大絶対差 1.94e-13 未満で一致し、列・ID・値域の検査も通過した。項目によっては分位点写像後の予測がほぼ定数になり、採点部分集合で順位相関が未定義になる可能性があるが、Kaggle のエラーは該当列や原因を示さない。

採点経路の切り分けとして、P4 で事前選定済みの E01 を `E01-external` として再実行する。分位点写像だけを除いた対照。別の計画・選定記録と F01 に紐づける。F00 の CV 選定と提出結果はそのまま記録し、F01 は外部採点できるモデルの評価に用いる。

採点切り分け用の F01 は `train --plan external`、`select --plan external`、`export --plan external --export-id F01` の順で生成する。アップロード後は実行完了と公開テストの出力を検査し、採点する版を明示する。

```powershell
$kaggleUser = 'Kaggleユーザー名'
kaggle kernels push -p "$d/outputs/exports/outputs/runs/F01"
kaggle kernels status "$kaggleUser/kumite-quest-final"
kaggle competitions submit google-quest-challenge -k "$kaggleUser/kumite-quest-final" -v <版> -f submission.csv -m 'Kumite QUEST F01 E01-external'
```

第2版の F01 は公式採点が成立した。OOF 0.32460、Late Public 0.30035、Late Private 0.28560。外部採点まで成立したモデルは E01-external。F00 の Late スコアは未取得。 [採点証跡](submission-evidence.json)
