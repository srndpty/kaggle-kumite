# #049 CommonLit Readability Prize

深さは **Standard**。英語の抜粋文から読みやすさを CPU で予測する。公式指標は RMSE。Kaggle の非公開 Notebook で `id,target` の CSV を出力する。 [公式評価仕様](https://www.kaggle.com/competitions/commonlitreadabilityprize/overview/evaluation) · [公式データ仕様](https://www.kaggle.com/c/commonlitreadabilityprize/data)

## 進捗

- [x] P0 公式データ取得と定数予測の Late Submission
- [x] P1 EDA と事前仮説
- [x] P2 グループ化と固定5分割
- [x] P3 Ridge ベースラインの学習と Late Submission
- [x] P4 自力の比較実験3件
- [x] P5 上位2チームの本人資料を調査
- [x] P6 線形 SVR の単独比較
- [x] P7 最終モデルの評価と誤差分析
- [x] P8 レポートと知見の保存

## 事前仮説と結果

| Run | 比較元 | 変更と予想 | OOF RMSE |
|---|---|---|---:|
| E00 | | 学習ラベルの平均 | 1.03348 |
| E01 | E00 | 単語 unigram TF-IDF + Ridge で語彙の違いを捉え、RMSE が下がる | 0.72984 |
| E02 | E01 | 語内文字 3–5-gram を追加すると、学習語彙でカバーしにくい文で改善する | 0.70539 |
| E03 | E01 | 単語 bigram を独立の特徴群として追加すると、頻出の語の組み合わせを含む文で改善する | 0.72216 |
| E04 | E01 | 語数・文数・平均語長・平均文長を追加すると、文の構造の違いを捉えられる | **0.69913** |
| E05 | E01 | 2位解法から回帰器の比較を取り出し、線形 SVR が小さな誤差を許容することで改善するかを検証する | 0.74228 |

学習は2,834件、配布テストは7件。学習文の語数中央値は177、10・90パーセンタイルは151・198。目的値は中央値 −0.912、10・90パーセンタイルは −2.353・0.377。出典 URL とライセンスはそれぞれ学習2,004件、配布テスト4件で欠損している。

CV で選んだ E04 の Late Submission は Public 0.674、Private 0.705。

## 検証と再現

`commonlit-group-v1` は、正規化本文の一致、文字5-gram 集合の Jaccard ≥ 0.8、または非ルート出典 URL の一致を辺とする連結成分で抜粋を束ねる。2,831グループ中3グループは2件。目的値を五分位で層化し、seed 42 の5分割で全件を一度ずつ検証する。fold の件数は 567・567・567・567・566。

特徴は本文から作成し、語彙・IDF・数値特徴の標準化は学習 fold に限定する。出典 URL は分割用、ID は照合用。ライセンス・`standard_error`・公開済みの完全版 CLEAR Corpus は使用対象外。出典が判明する範囲は限られ、公式テストの現代文比率の高さは CV で再現していない。

```powershell
dev download 49
$d = 'competitions/049-commonlitreadabilityprize'
uv run python "$d/src/train.py" prepare
$trials = 'E00', 'E01', 'E02', 'E03', 'E04', 'E05'
$runs = foreach ($trial in $trials) {
    $run = "$trial-r2"
    uv run python "$d/src/train.py" cv $trial --run-id $run | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "実験 $run に失敗" }
    $run
}
uv run python "$d/src/train.py" select @runs --output "$d/outputs/selection-r2.json"
uv run python "$d/src/train.py" final "$d/outputs/selection-r2.json" "$d/outputs/final-r2.csv"
```

`selection.json` は初回の選定記録。Kaggle では同じ予測コードを組み込む。元のモデルの提出手順は次のとおり。再実行時は push が返すバージョンを使う。

```powershell
$owner = '<Kaggle username>'
$trial = (Get-Content "$d/selection.json" -Raw | ConvertFrom-Json).trial
uv run python "$d/src/notebook.py" $owner "$d/outputs/kaggle/$trial" --trial $trial
kaggle kernels push -p "$d/outputs/kaggle/$trial"
$kernel = "$owner/kumite-commonlit-$($trial.ToLowerInvariant())"
kaggle kernels status $kernel
# COMPLETE の後に、push が返したバージョンで実行
$version = Read-Host 'Notebook version'
kaggle competitions submit commonlitreadabilityprize -k $kernel -v $version -f submission.csv -m "Kumite $trial"
dev submissions 49
```

[実験ログ](experiments.jsonl) · [振り返り](report.md) · [K016](../../knowledge/K016-character-features-in-text.md)
