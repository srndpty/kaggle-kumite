# #046 Quora Insincere Questions Classification

- URL: https://www.kaggle.com/competitions/quora-insincere-questions-classification
- カテゴリ: D / 難度: 3 / 計算負荷: M
- 今回の比較実験テーマ: 不均衡テキスト分類・閾値選択
- 深さ: Standard（CPU）

## コンペ概要

Quora の質問文から、不誠実とラベル付けされた質問を二値分類する。

## 基本情報（P0 で記入）

- 評価指標は陽性クラスの F1。
- Kaggle Notebook が整数の 0/1 を持つ `qid,prediction` の CSV を出力する。
- 公式の train・test・sample_submission を個別取得した。CLI が CSV 拡張子で保存した ZIP は展開した。

## 進捗

- [x] P0. 参加・評価環境の確認（公式 CSV 3 件を取得し、全件 0 の Notebook 版 1 は Public / Private F1 ともに 0.00000）
- [x] P1. 問題設定・EDA（予測仮説 3 件を事前登録）
- [x] P2. 検証分割の設計（リーク経路の洗い出し、fold の確定）
- [x] P3. Baseline の固定（E00 の CV と Late Submission）
- [x] P4. 自力の仮説検証（E01〜E03）
- [x] P5. 上位解法の調査（Private 1 位本人の writeup と Private 3 位本人の公開 Notebook）
- [x] P6. 上位解法の着眼点の再現（E04）
- [x] P7. 最終評価・エラー分析（E02 の公式採点と OOF の誤り分析）
- [x] P8. 知見の保存（`report.md`、K002、K016）

## 仮説

| # | 仮説 | 根拠（EDA） | 出所（自力 / 上位解法） | 結果 |
|---|---|---|---|---|
| H1 | 内側保留で閾値を選ぶと、0.5 固定より Recall と F1 が上がる | 陽性が少ないタスクで F1 を比較する | 自力 | Recall が上がり、F1 +0.09424 |
| H2 | 文字 3〜5-gram で綴りの変形を含む質問の F1 が上がる | 語内の部分一致を利用できる | 自力 | 全体 F1 +0.01973。綴り変形の効果は未分離 |
| H3 | balanced 重みの追加効果は閾値調整後には小さい | 閾値調整と重み付けはともに陽性の判定率を動かす | 自力 | F1 +0.00249 |
| H4 | 単語の大小文字を区別すると F1 が上がる | 強調や固有名を異なるトークンとして扱える | Private 1 位 | F1 +0.00310 |

## 検証設計

NFKC・casefold・空白圧縮後の文面をグループとし、陽性を含むかで層化した seed 42 の 3-fold にグループを割り付ける。外側学習行を同じ方法で再分割し、最初の fold で閾値を決め、残りで語彙・IDF・分類器を学習する。外側の判定をまとめた F1 を主指標にする。qid は照合専用とする。

学習 1,306,122 行のうち陽性は 80,810 行（6.19%）。文長の四分位点は 45、60、85 文字。正規化後の重複グループは 18 件で、うち 2 件はラベルが混在する。テスト 375,806 行のうち 10 行が学習と同一の正規化文面を持つ。

外側の検証は 435,377・435,371・435,374 行。陽性はそれぞれ 26,937・26,936・26,937 行。外側検証ラベルの変更で閾値と予測が変わらないことを合成データで確認した。

分類器は `C=1`・最大 200 反復のロジスティック回帰。TF-IDF は `min_df=3`、sublinear TF、単語と文字の各ブロック別 L2 正規化。閾値は内側保留で観測した確率を降順に調べ、F1 が同じなら最も高い値を選ぶ。

## 再現手順

Kaggle の Rules を承諾したアカウントで取得する。個別取得したファイルが ZIP の場合は、元ファイルの拡張子を `.zip` に変えて `data/` に展開する。

```powershell
$d = 'competitions/046-quora-insincere-questions-classification'
$slug = 'quora-insincere-questions-classification'
foreach ($name in 'train.csv', 'test.csv', 'sample_submission.csv') {
    kaggle competitions download -c $slug -f $name -p "$d/data"
}
```

```powershell
$owner = '<Kaggleユーザー名>'
uv run python "$d/src/train.py"                       # EDA
uv run python "$d/src/train.py" E00 --run-id E00-r2
uv run python "$d/src/train.py" E01 --run-id E01-r2 --baseline-run-id E00-r2
foreach ($trial in 'E02', 'E03', 'E04') {
    uv run python "$d/src/train.py" $trial --run-id "$trial-r2"
}
uv run python "$d/src/analyze.py" E00-r2 E01-r2
uv run python "$d/src/notebook.py" $owner "$d/outputs/final-r2" --model final --run-suffix=-r2 --selection "$d/selection-r2.json" --freeze
```

Notebook 生成時に `--model` を省略するとダミー、`--model baseline` と `--run-suffix=-r2` を指定すると E00-r2 を生成する。生成先を `kaggle kernels push -p <出力先>` で登録し、実行完了後に `kaggle competitions submit quora-insincere-questions-classification -k <Notebook ID> -v <版> -f submission.csv -m <説明>` で採点する。

選定記録は新規パスに固定し、同じ記録から再生成する場合は `--freeze` を外して新しい出力先を指定する。新しい選定では、各 run の OOF のハッシュ・二値・F1 と、CV のデータ・分割・ソースを照合する。従来の `selection.json` と E00〜E04 は当時の採点を示す記録で、CV ソースの個別ハッシュが保存されていないため、新しい Notebook の生成には個別ハッシュを持つ run が必要。

## 実験ログ

詳細は `experiments.jsonl`（`kumite.runs.RunStore` で追記。記録は書き換えず、やり直しは別の run_id）。主要な結果のみ抜粋する。

| run | 変更点（1つだけ） | OOF | Late | 判定 |
|---|---|---|---|---|
| E00 | 単語 unigram・閾値 0.5 | 0.50922 | 0.52626 | 基準 |
| E01 | 同じ確率に内側保留の閾値を適用 | 0.60346 | — | +0.09424 |
| E02 | E01 に文字 3〜5-gram を追加 | 0.62319 | 0.63519 | +0.01973、最終採用 |
| E03 | E01 に balanced 重みを追加 | 0.60595 | — | +0.00249 |
| E04 | E01 で単語の大小文字を保持 | 0.60655 | — | +0.00310 |

Late 列は Private F1。E00 の Public F1 は 0.52431。[採点 Notebook 版 2](https://www.kaggle.com/code/shion5/kumite-quora-insincere-baseline-e00)

E02 を OOF F1 で最終選定した。Public F1 は 0.62054。[選定記録](selection.json) · [最終採点 Notebook 版 1](https://www.kaggle.com/code/shion5/kumite-quora-insincere-final-e02)
