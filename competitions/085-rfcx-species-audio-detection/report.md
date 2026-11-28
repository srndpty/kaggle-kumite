# #085 Rainforest Connection Species Audio Detection 振り返り

## 採点疎通と進捗

seed 42 のダミー予測は Kaggle で採点され、lwlrap は Public 0.26175、Private 0.25054 だった。Kaggle の取得制限解除後、学習録音4,727件とテスト録音1,992件を取得した。注釈と固定３分割の集計は [README](README.md) に記録した。

合成 FLAC では、小型 CNN の３ fold 学習、録音別予測の保存、Quality Gate による保存 OOF の再採点を確認した。

## 自力の比較実験

差は E00 に対する OOF lwlrap。95%区間は同じ録音を対応づけた bootstrap 2,000回。

| Run | OOF lwlrap | 差 | 95%区間 |
|---|---:|---:|---|
| E00 | 0.347081 | — | — |
| E01 | 0.305969 | −0.041112 | −0.057007〜−0.026382 |
| E02 | 0.347568 | +0.000487 | −0.002329〜+0.003236 |
| E03 | 0.346492 | −0.000589 | −0.004875〜+0.003671 |

E01 は３ fold すべてで低下した。録音・種の最長注釈が１秒以下の陽性289件では −0.04912、１秒超の872件では −0.03846。誤検出窓への置換では、対象種に負例の重みを集中させるため、窓の内容と種別の監督分布の寄与は分離できない。E02・E03 の差は小さく、P4 の再採点では事前の許容差 0.001 に従って E00 を選んだ。

## 上位解法と再現候補

- ２位の本人解説では、マスク付き BCE から始め、録音を走査して得た疑似ラベルと注釈区間を併用して学習を反復した。 [２位の解法](https://www.kaggle.com/competitions/rfcx-species-audio-detection/writeups/selim-seferbekov-2nd-place-solution)
- 11位の本人解説では、log-mel の画像分類器に周波数軸に沿った線形勾配を加えた。 [11位の解法](https://www.kaggle.com/competitions/rfcx-species-audio-detection/writeups/cpmp-11th-place-the-0-931-magic-explained-image-cl)
- 21位の本人解説では、誤検出を使う際の co-teaching と周波数帯ごとの mixup を検証した。 [21位の解法](https://www.kaggle.com/competitions/rfcx-species-audio-detection/writeups/vasiliy-kotov-21st-place-solution-fp-co-teaching-w)

E04 は11位の周波数位置の着眼点を取り出し、E00 の log-mel に低周波から高周波へ −0.5〜+0.5 の固定オフセットを加える。学習と推論の両方に適用し、種の順位が改善すると仮定する。候補計画は E00-final、E03-final、E02-final、E01-final、E04-final の順とし、許容差 0.001 を維持する。
