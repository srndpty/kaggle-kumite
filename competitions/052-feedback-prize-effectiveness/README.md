# #052 Feedback Prize – Predicting Effective Arguments

深さは Standard。英文エッセイの論証要素を Ineffective、Adequate、Effective に分類する。公式の主指標は論証行単位の multiclass LogLoss。

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
