# #059 Cassava Leaf Disease Classification

深さは **Standard**。キャッサバの葉画像を 5 クラスに分類し、クラス不均衡と画像解像度の効果を調べる。指標は Accuracy。

## 進捗

- [x] P0 公式データ取得と定数提出の採点
- [x] P1 EDA と事前仮説
- [x] P2 近重複をまとめた固定 3 分割
- [x] P3 ResNet-18 ベースライン
- [x] P4 自力比較 3 件
- [x] P5 上位解法の調査
- [x] P6 推論時水平反転の単独比較
- [x] P7 最終モデルの外部採点と誤りの分析
- [x] P8 レポートと知見カード

## P0

2026-11-23、Rules 承認後に公式の学習ラベルと提出サンプルを取得した。クラス 0 の定数予測を出す非公開 CPU Notebook（インターネット無効）のバージョン 1 は採点完了。Public **0.0481**、Private **0.0507**。

[Kaggle 公式概要・提出条件](https://www.kaggle.com/competitions/cassava-leaf-disease-classification)

## 固定分割と事前仮説

`cassava-group-v1` は、64 ビット dHash のハミング距離が 4 以下の画像を連結成分でまとめ、画像 ID 順で seed 42 の層化グループ 3 分割を作る。

全 21,397 画像は 800 × 600 px で、分割作成時に全件をデコードできた。21,395 グループのうち 2 グループに画像が 2 枚ずつあり、グループ内のラベル不一致は 0 件。クラス 0〜4 の件数は順に 1,087・2,189・2,386・13,158・2,577。クラス 3 が約 61.5% を占める。撮影元をまとめるための ID は公式ラベルに含まれていないため、dHash で拾えない同一個体や撮影条件の共有は残り得る。

| 実験 | 比較元 | 変更と事前仮説 |
|---|---|---|
| E00 | | 学習 fold の最多クラスを予測 |
| E01 | E00 | ImageNet 事前学習済み ResNet-18 で葉の特徴を使うと Accuracy が上がる |
| E02 | E01 | 学習 fold の頻度の逆平方根で損失を重み付けすると少数クラスの再現率が上がる。多数クラスの誤りが増えると Accuracy は下がり得る |
| E03 | E01 | 学習時の水平反転で葉の向きへの依存が減り、Accuracy が上がる |
| E04 | E01 | 320 px で細かな病斑が残り、Accuracy が上がる |
| E05 | E01 | 予測時に元画像と水平反転画像の確率を平均すると、向きによって変わる誤りが減り、Accuracy が上がる |

E01 は 224 px、AdamW、学習率 0.0003、重み減衰 0.0001、バッチ 32、5 epoch。学習時は混合精度を使う。

OOF Accuracy は E00 **0.61495**、E01 **0.77296**、E02 **0.79679**、E03 **0.80175**、E04 **0.81881**、E05 **0.79156**（全 21,397 件）。最大の E04 を `selection.json` に固定した。実験の正確な値とハッシュは `experiments.jsonl` を参照。

## 上位解法

- 1 位は異なるアーキテクチャと事前学習条件の 4 モデルを合成した。EfficientNet B4 では重なりのあるパッチを切り出し、反転・回転・転置を用いた複数の予測を平均した。 [1 位チームの writeup](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/discussion/221957)
- 2 位はキャッサバ画像で事前学習済みの CropNet を微調整した。既存モデルの背景クラスを含む 6 出力に合わせ、画像の前処理はリサイズと値域変換を中心にした。 [2 位本人の writeup](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/writeups/devon-stanfield-2nd-place-summary-of-approach)
- E05 は 1 位の推論時拡張から、反転予測の平均だけを切り出す。変換は、水平反転も含めて推論平均を行った 14 位の資料を参考に固定した。E01 の学習済み重みをそのまま使う。 [14 位チームの writeup](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/writeups/mean-predictors-14th-place-solution-code)

## 再現

```powershell
dev download 59
$d = 'competitions/059-cassava-leaf-disease-classification'
uv run python "$d/src/train.py" prepare
foreach ($trial in 'E00', 'E01', 'E02', 'E03', 'E04') {
    uv run python "$d/src/train.py" cv $trial --run-id "$trial-r2"
    if ($LASTEXITCODE -ne 0) { throw "実験 $trial に失敗" }
}
uv run python "$d/src/train.py" cv E05 --run-id E05-r2 --baseline E01-r2
uv run python "$d/src/train.py" select E01-r2 E02-r2 E03-r2 E04-r2 E05-r2 --output "$d/outputs/selection-r2.json"
uv run python "$d/src/train.py" final "$d/outputs/selection-r2.json" "$d/outputs/submission-r2.csv"
```

ダミー Notebook は `notebook.py <Kaggle ユーザー名> <出力先>` で生成する。学習モデルの Notebook は同じコマンドに `--selection <選定 JSON>` を加える。後者は公開サンプルでローカル出力との一致を確認し、`weights/`、`kernel/`、`evidence.json` を出力する。`weights/` に移動して `kaggle datasets create -p .` で重みを非公開入力として登録する。この呼び出し方は Windows の Kaggle CLI で再開用ファイルのパスエラーを回避する。データセットの準備後、リポジトリから `kaggle kernels push -p <出力先>/kernel` で Notebook を登録する。実行完了後、返された Notebook バージョンを指定して `kaggle competitions submit cassava-leaf-disease-classification -k <Notebook ID> -v <バージョン> -f submission.csv -m <メッセージ>` を実行する。

E01 は CPU、E04 は `--gpu` を指定して提出用 Notebook を生成した。公開サンプルで、E04 の GPU 推論はローカル CPU 推論と同じ提出 CSV を生成した。

最終推論と Notebook 生成の前に、公式学習ラベルと学習 JPEG を run のデータハッシュと照合する。選定記録には公開サンプル CSV と対応する全 JPEG の SHA256 を保存し、ローカルと Notebook の両方で照合する。Kaggle が公開 ID と重ならない非公開テストへ差し替えた場合、Notebook は提出 ID と画像ファイル集合の完全一致を検査する。非公開画像の既知ハッシュは配布されていない。

`select` は既存ファイルへの書き込みを拒否する。再選定時は新しい `--output` を指定する。既存 run の `flip` は推論時水平反転を表し、E03 の学習時水平反転は実験番号と変更点に記録されている。新規 run では `train_horizontal_flip` と `inference_horizontal_flip` を個別に記録する。既存の E04 選定記録には公開入力の証跡と新しいキー名を追加した。

E04 の Late Submission は Public **0.8534**、Private **0.8528**。E01 は Public **0.8326**、Private **0.8227**。実験ログの `late_score` には Private Accuracy を記録した。 [Kaggle 提出履歴](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/submissions)

[振り返り](report.md) · [K023 左右反転](../../knowledge/K023-horizontal-flips-for-segmentation.md) · [K032 クラス重み](../../knowledge/K032-class-weighting-and-per-class-recall.md)
