# #052 Feedback Prize – Predicting Effective Arguments

深さは Standard。英文エッセイの論証要素を Ineffective、Adequate、Effective に分類する。公式の主指標は論証行単位の multiclass LogLoss。

P0〜P8 完了。最終モデルは E03-final（同じエッセイの他の論証を追加）。OOF LogLoss は 0.751580、Late Private は 0.74798。比較の内訳は [レポート](report.md) に記録した。

[公式概要・提出仕様](https://www.kaggle.com/competitions/feedback-prize-effectiveness/overview)

## P0–P2　データと検証

均等確率を出力する非公開 CPU Notebook の Late Submission は Public、Private ともに 1.09861。

訓練は36,765論証・4,191エッセイ。上記のクラス順で件数は6,462・20,977・9,326。論証は中央値28語、四分位範囲16〜57語。エッセイ当たりの論証数の中央値は９。Ineffective の比率は Evidence が26.07%、Counterclaim が11.56%だった。

同じエッセイの論証を同一グループに置き、本文を NFKC・大小文字・空白で正規化して同一本文も束ねる。今回の正規化本文グループは4,191件。StratifiedGroupKFold ５分割、seed 42、v1 を固定した。検証件数は7,349・7,345・7,361・7,352・7,358。語彙、IDF、カテゴリは学習 fold のみで適合する。エッセイ内文脈は対象論証を除いた論証本文のみを使う。

## P3–P4　事前仮説

| Run | 構成・追加 | 予想 |
|---|---|---|
| E00 | 論証本文の単語 unigram TF-IDF（min_df=2）、多項ロジスティック回帰（C=1） | 語彙分布から有効性を推定できる |
| E01 | 論証種別 one-hot | 役割により異なるクラス分布を捉える |
| E02 | 文字 3–5 gram TF-IDF（min_df=3） | 語形・綴りで単語特徴を補う |
| E03 | 同一エッセイの他の論証を連結し単語 unigram TF-IDF（min_df=2） | 短い論証の判定を周辺文脈で補う |

優先順は E00、E01、E03、E02、P6 候補。最良 LogLoss との差 0.001 以内は実質同点とする。

P6 の E04 は冒頭と末尾の各８トークンを区別した TF-IDF を E00 に加える。出典と他のアブレーションの結果は [レポート](report.md) に記録した。

## 再現手順

リポジトリのルートで実行する。保存済み run を再実行するときは、新しい run ID と構成の対応を `RUN_TRIALS` と事前計画に登録する。

```powershell
$d = 'competitions/052-feedback-prize-effectiveness'
$plan = "$d/selection-plan-final.json"
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
uv run python "$d/src/train.py" eda
if ($LASTEXITCODE) { throw 'EDA に失敗' }
foreach ($trial in 'E00','E01','E02','E03','E04') {
    uv run python "$d/src/train.py" cv --trial $trial --run-id "$trial-final" --plan $plan
    if ($LASTEXITCODE) { throw "CV に失敗: $trial" }
}
uv run python "$d/src/compare.py" E00-final E01-final E02-final E03-final E04-final
if ($LASTEXITCODE) { throw '比較に失敗' }
$receipt = "$d/outputs/fresh-selection-final.json"
uv run python "$d/src/train.py" select --plan $plan --receipt $receipt
if ($LASTEXITCODE) { throw '選定に失敗' }
$owner = 'Kaggle のユーザー名'
uv run python "$d/src/export.py" --plan $plan --receipt $receipt --owner $owner --export-id final
if ($LASTEXITCODE) { throw 'export に失敗' }
```

ベースラインの外部評価は `selection-plan-baseline.json` で E00-external を単独選定し、export ID を baseline とする。確定済み export の Notebook を Kaggle に登録し、実行完了した版の `submission.csv` を提出する。
