# #077 Sartorius - Cell Instance Segmentation

位相差顕微鏡画像から細胞ごとのマスクを予測する。深さは Standard。接触・重複した細胞を分離し、予測マスクの重複除去が偽陽性と取りこぼしに与える影響を調べる。

- [公式データ](https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/data)
- [公式評価・提出仕様](https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/overview/evaluation)

## 進捗

- [x] P0. 公式データ取得と Notebook 採点疎通
- [x] P1. データの確認と比較仮説の記録
- [x] P2. 撮影サンプル単位の分割と採点仕様の固定
- [x] P3. Baseline の固定
- [x] P4. 自力の仮説検証
- [x] P5. 上位解法の調査
- [x] P6. 上位解法の着眼点の再現
- [x] P7. 最終評価とエラー分析
- [x] P8. レポートと知見の保存

## 評価と分割

- 学習画像 606 枚、正解マスク 73,585 件、撮影サンプル 403 件。画像内の細胞数は最小 4、中央値 46、最大 790。テスト画像の公開見本 3 枚とサンプル提出の ID を照合した。
- 2026-11-27 に非公開のダミー Notebook v1 を Late Submission に送信し、Public / Private ともに 0.000 で採点された。これは疎通用の提出で、CV 候補から除く。
- 注釈の RLE は 1 始まり・行優先。25 枚で注釈輪郭と画像勾配を照合し、全枚で行優先の境界上の平均勾配が列優先を上回った。ダミーマスクの位置は両順序で同じなので、提出側の順序の根拠には[コンペ中の訂正経緯](https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/discussion/279790)を併用した。
- IoU が各閾値を厳密に超えた対応から最大一対一マッチングを取り、画像ごとの TP / (TP + FP + FN) を IoU 0.50〜0.95 と全画像で平均する。重なった正解に一つの予測が当たる場合は、一つを TP、残りを FN とする。正解のある画像で空予測は 0。
- 同じ撮影サンプルおよび同一ファイル内容の画像をまとめ、細胞種で層化した 3-fold を seed 42 で固定した。fold 0・1・2 は各 203・202・201 枚。モデル入力は PNG のみで、分割用メタデータや外側検証ラベルを渡さない。分割ハッシュは `54166f4fe9635daf7a3795822f72680a7dac9e47c4ebbef2a28dc04d0f12acf5`。

| 細胞種 | 画像数 | 画像あたり個体数の中央値 | マスク面積の中央値 (px) | 正解マスクが重なる画像数 |
|---|---:|---:|---:|---:|
| astro | 131 | 73 | 665 | 131 |
| cort | 320 | 30 | 208 | 267 |
| shsy5y | 155 | 324 | 193 | 155 |

shsy5y は小さい細胞が密集する。astro は面積が大きく、全画像に重複がある。解像度と重複除去の影響を、全体スコアに加えて細胞種・面積・密度でも確認する。

## 比較仮説

| Run | 変更 | 予想 |
|---|---|---|
| E00 | COCO 初期化の Mask R-CNN。スコアの高いマスクから画素を割り当てる。 | 基準 |
| E01 | 短辺 640 → 800。長辺上限 1024 → 1333。 | 小さい細胞の見落としが減る。 |
| E02 | 水平反転を確率 0.5 で追加。 | 撮影方向への依存が減り、細胞種をまたいで改善する。 |
| E03 | 重複除去後に元面積の 50% 未満なら棄却。 | 断片状の偽陽性が減る。密集画像では取りこぼしが増えうる。 |

全 run でマスク閾値 0.5、検出スコア閾値 0.2、検出上限 800、batch 2、12 epoch、SGD を使用する。検出上限は EDA の最大細胞数を踏まえて決めた。E03 は E00 の後処理前 OOF を再利用する。選定時は保存した OOF を再採点し、最良から 0.001 以内なら [選定計画](selection.toml) の優先順を使う。

## 実測結果

| Run | Fold 0 | Fold 1 | Fold 2 | OOF |
|---|---:|---:|---:|---:|
| E00 | 0.21646 | 0.20562 | 0.22770 | 0.21658 |
| E01 | 0.22986 | 0.21649 | 0.24092 | 0.22907 |
| E02 | 0.21284 | 0.20183 | 0.22525 | 0.21329 |
| E03 | 0.23450 | 0.22115 | 0.24263 | 0.23275 |
| S00 | 0.23548 | 0.22237 | 0.24295 | 0.23359 |
| S01 | 0.23592 | 0.22271 | 0.24311 | 0.23390 |

P4 の全候補を保存 OOF から再採点し、E03 を選定した。P6 は E03 の構成で再学習する S00 と、検出スコアに閾値以上のマスク画素の平均確率を掛けてから重複画素を割り当てる S01 を比較する。高信頼のマスクを先に置けば、輪郭の欠けと取りこぼしを減らせると予想する。S01 は S00 と同じ推論候補を使用し、0.001 以内は S00 を優先する。[P6 の選定計画](selection-p6.toml)

P6 の全候補を再採点した。S01 は +0.00031 で事前の同点幅以内だったため、S00 を最終モデルに選定した。

F00 は S00 の構成で全量学習し、2026-11-27 に Notebook v1 を提出した。Late Public は 0.262、Late Private は 0.268。公開見本の出力は 3 画像・341 行で、ID と RLE の範囲・重複を再検査した。[振り返り](report.md)

## 実行

リポジトリルートから実行する。

以下の ID は実施時の例。レビュー後のソースで学習や export をやり直す場合は、新しい run ID を指定して比較計画を事前登録し、新しい export ID を使う。

```powershell
$d = "competitions/077-sartorius-cell-instance-segmentation"
uv run python "$d/src/train.py" smoke
uv run python "$d/src/train.py" E00
uv run python "$d/src/train.py" E01
uv run python "$d/src/train.py" E02
uv run python "$d/src/train.py" E03
uv run python "$d/src/train.py" select
```

OOF と選定記録は `outputs/`、run 記録は `experiments.jsonl` に保存する。最終 export は上位解法の比較後に確定した選定計画と記録を渡す。提出 Notebook は隠しテスト画像を実行時に読み込む。

```powershell
$plan = "$d/selection-p6.toml"
$receipt = "$d/outputs/selection-p6.json"
uv run python "$d/src/train.py" S00 --plan $plan
uv run python "$d/src/train.py" S01 --plan $plan
uv run python "$d/src/train.py" select --plan $plan --receipt $receipt
uv run python "$d/src/export.py" F00 <owner> <owner/model-dataset> --plan $plan --receipt $receipt
uv run python "$d/src/deploy.py" F00 "$d/outputs/deploy-F00"
$env:PYTHONUTF8 = "1"
Push-Location "$d/outputs/deploy-F00/dataset"
try { kaggle datasets create -p . } finally { Pop-Location }
kaggle datasets status <owner>/model-dataset
kaggle kernels push -p "$d/outputs/deploy-F00/kernel"
kaggle kernels status <owner>/kumite-077-final
kaggle kernels output <owner>/kumite-077-final -p "$d/outputs/deploy-F00/preview"
kaggle competitions submit sartorius-cell-instance-segmentation -k <owner>/kumite-077-final -v <version> -f submission.csv -m "Kumite 077 F00"
kaggle competitions submissions sartorius-cell-instance-segmentation
```

提出前に Notebook の正常終了と公開見本の `submission.csv` を確認する。エラー分析は選定 run の ID を渡して生成する。

```powershell
uv run python "$d/src/analyze.py" <selected-run> "$d/outputs/error-analysis.md"
```
