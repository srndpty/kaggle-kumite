# #085 Rainforest Connection Species Audio Detection 振り返り

## 採点疎通と進捗

seed 42 のダミー予測は Kaggle で採点され、lwlrap は Public 0.26175、Private 0.25054 だった。公式音声の取得は Kaggle の HTTP 429 により保留した。取得数と注釈の暫定集計は [README](README.md) に記録した。

合成 FLAC では、小型 CNN の３ fold 学習、録音別予測の保存、Quality Gate による保存 OOF の再採点を確認した。公式音声による CV、上位解法との比較、最終選定、Late Submission はデータ取得後に実施する。
