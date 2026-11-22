# #016 PetFinder.my Adoption Prediction

深さは **Standard**。ペットの掲載情報から譲渡速度（`AdoptionSpeed`、0〜4）を予測する。主指標は Quadratic Weighted Kappa（QWK）。提出は Kaggle Notebook で `submission.csv` を生成する。

[公式コンペ](https://www.kaggle.com/competitions/petfinder-adoption-prediction)・[データ仕様](https://www.kaggle.com/competitions/petfinder-adoption-prediction/data)

## 進捗

- [x] P0　公式データ取得・定数予測の外部採点
- [x] P1　EDA と事前仮説
- [x] P2　保護者単位の検証分割
- [x] P3　Baseline の固定と外部採点
- [x] P4　比較実験3件
- [x] P5　本人 writeup・コードの調査
- [x] P6　上位解法の着眼点のアブレーション
- [x] P7　モデル選定・最終評価・誤差分析
- [x] P8　振り返りと知見の保存

## 事前仮説

| Run | E01 からの変更 | 予想 |
|---|---|---|
| E02 | 目的関数を多クラス分類から回帰に変更 | 2段階以上の誤分類が減り、QWK が上がる |
| E03 | 内側 OOF で4つの閾値を調整 | クラス予測の偏りが減り、QWK が上がる |
| E04 | 画像の寸法・縦横比・明度の平均と有効画像数を追加 | PhotoAmt を超える情報で QWK が上がる |
| E05 | 本文の単語 TF-IDF と16次元 SVD を追加 | 掲載本文の情報で表モデルの QWK が上がる |

E01 は表特徴の CatBoost 多クラス分類。確率の期待値を固定閾値で整数化する。画像の完全一致でつながった保護者をまとめ、seed 42 の層化グループ5分割を共通で使う。入力には掲載時点の表を使い、識別子・本文は除く。閾値の学習は外側学習部分だけで行う。

公開配布データは学習 14,993 件・テスト 3,972 件。学習のクラス 0〜4 は順に 410、3,090、4,037、3,259、4,197 件。学習の保護者 5,595 組と、公開テストの保護者 1,518 組は重複ゼロだった。

学習画像 58,311 枚のうち 1,662 枚は、同じ内容の画像の2枚目以降だった。画像の完全一致で保護者を結合すると 5,558 グループになった。画像なしは341件。画像ファイル数と `PhotoAmt` の不一致および画像の読み取り失敗はともに0件だった。最大の保護者は459件を掲載していた。

## 外部採点の確認

2026-11-23 JST に、クラス2の定数予測 E00 が Public・Private ともに QWK 0.00000 で採点された。提出は非公開 Notebook のバージョン3を使用した。

表特徴 Baseline E01 は OOF QWK 0.26455。非公開 Notebook のバージョン1の外部スコアは Public 0.00000、Private 0.23544 だった。

採用モデルは **E03**。内側 OOF で調整した閾値により、OOF QWK は 0.37908（E01 比 +0.11453）。Kaggle の最終採点は Public 0.00000、Private 0.34235（E01 比 +0.10691）。[選定記録](selection.json)・[実験結果と振り返り](report.md)

## 実行手順

リポジトリのルートで実行する。

```powershell
dev download 16
$d = 'competitions/016-petfinder-adoption-prediction'
uv run python "$d/src/train.py" eda
uv run python "$d/src/train.py" dummy --run-id E00
uv run python "$d/src/train.py" baseline --run-id E01
uv run python "$d/src/train.py" regression --run-id E02
uv run python "$d/src/train.py" threshold --run-id E03
uv run python "$d/src/train.py" image --run-id E04
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
uv run python "$d/src/train.py" description --run-id E05
uv run python "$d/src/select_model.py" E02 E03 E04 E05 --output "$d/outputs/selection-recheck.json"
uv run python "$d/src/export_notebook.py" dummy "$d/outputs/kaggle-dummy/PetFinder.ipynb"
uv run python "$d/src/export_notebook.py" baseline "$d/outputs/kaggle-baseline/PetFinder.ipynb"
uv run python "$d/src/export_notebook.py" threshold "$d/outputs/kaggle-final/PetFinder.ipynb"
```

Kaggle Notebook は非公開で作成し、コンペを入力に指定して実行する。採点は完走した Notebook の版と `submission.csv` を `kaggle competitions submit petfinder-adoption-prediction -k <NotebookのID> -v <版> -f submission.csv -m <実験名>` で指定する。選定記録は提出前に固定した。実験ログの `late_score` には Private QWK を記録する。

E03 は保存済みの E01 連続予測を再利用する。分割・データ・パラメータ・成果物のハッシュを照合し、内側 OOF を外側 fold ごとに別プロセスで学習する。
