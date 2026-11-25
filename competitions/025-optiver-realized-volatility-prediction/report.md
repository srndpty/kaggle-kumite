# #025 Optiver Realized Volatility Prediction 振り返り

## 固定分割での比較

| Run | E00 に追加した特徴 | OOF RMSPE | E00 との差 |
|---|---|---:|---:|
| E00-r2 | 板 RV のみ | 0.26600 | — |
| E01-r2 | 直近300秒・150秒の板 RV | **0.25466** | −0.01134 |
| E02-r2 | 約定 RV・出来高・注文数 | 0.25502 | −0.01098 |
| E03-r2 | 同時刻の他銘柄の板 RV 中央値 | 0.26352 | −0.00248 |
| E04 | 同時刻の板 RV 順位 | 0.26293 | −0.00307 |

評価は時間バケット単位の5 fold、合計428,932行。全候補の保存 OOF を再採点し、`outputs/selections/P7.json` で E01-r2 を選定した。export ID は `F00`。

## どこで効いたか

E01-r2 は baseline に対し、時間バケット単位のペア・ブートストラップ（2,000回、seed 42）で RMSPE 差 −0.01134、95% 区間 [−0.01305, −0.00984]。E02-r2 に対する差は −0.00035、95% 区間 [−0.00284, +0.00276] で、短窓の優位はこの分割では確定しない。短窓と約定特徴の組み合わせは未検証。

選定モデルの OOF RMSPE は目的値の低い四分位から順に 0.28993、0.21899、0.22988、0.27304。最小四分位では分母が小さく、相対誤差が大きかった。銘柄別では 31 が 0.82271（3,830行）で、次点の 37 は 0.33951。銘柄31の乖離の原因は切り分けていない。

## 上位解法からの再現

1位の時間バケット内順位化を板 RV に転用した E04 は、baseline 比で 0.00307 改善した。銘柄間の中央値を使った E03 と同様に追加情報があったが、この比較では短窓・約定特徴の改善幅の方が大きかった。時系列順の復元とモデルのブレンドは調査に留めた。 [1位の解法記事](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970) · [7位のコードと説明](https://github.com/michaelpoluektov/orvp)

## 外部評価

2026-11-26 に F00 v1 が Late Private RMSPE **0.25143** で採点された。Kaggle の表示上の Public 値は 0.00000。ダミーの Late Private RMSPE は 1.04387。最終 Notebook はローカルの公開テスト3行で列、ID、件数、有限性、非負性を検査し、Kaggle の公開実行と Late 採点を完了した。 [最終 Notebook](https://www.kaggle.com/code/shion5/kumite-optiver-final)

Late Private は OOF より 0.00323 低かった。CV は未観測の時間バケットを評価し、Late は後続期間の隠しデータを評価する。期間、銘柄構成、学習量の影響は切り分けていない。基準モデルの外部採点は行っていないため、外部での特徴量の改善幅は未検証。

- [実験ログ](experiments.jsonl)
- [K049 短窓の実現ボラティリティ](../../knowledge/K049-short-window-realized-volatility.md)
