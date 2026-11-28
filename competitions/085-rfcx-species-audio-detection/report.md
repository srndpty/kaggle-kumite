# #085 Rainforest Connection Species Audio Detection 振り返り

## 採点疎通と進捗

seed 42 のダミー予測は Kaggle で採点され、lwlrap は Public 0.26175、Private 0.25054 だった。Kaggle の取得制限解除後、学習録音4,727件とテスト録音1,992件を取得した。注釈と固定３分割の集計は [README](README.md) に記録した。

合成 FLAC では、小型 CNN の３ fold 学習、録音別予測の保存、Quality Gate による保存 OOF の再採点を確認した。
