# #072 Severstal: Steel Defect Detection

- URL: https://www.kaggle.com/competitions/severstal-steel-defect-detection
- カテゴリ: E / 難度: 4 / 計算負荷: L
- 今回の比較実験テーマ: 工場欠陥・分類→セグメント
- 深さは Standard。小欠陥の検出と空マスクの誤検出抑制を比較する。

## コンペ概要

鋼板の表面を撮影した JPEG から、製造時に生じた4種類の欠陥の画素マスクを推定する。1枚の画像には欠陥なし、単一クラス、複数クラスのいずれもあり得る。

## 基本情報（P0 で記入）

- 評価指標は画像×クラス単位の平均 Dice。正解と予測が両方空なら 1。
- インターネットを無効にした非公開 Notebook で `submission.csv` を生成する。RLE は1始まりの列優先。取得したサンプル CSV は各 JPEG にクラス0のプレースホルダーを1件持つため、ダミーでは実行時のテスト JPEG を列挙し、`ImageId_ClassId,EncodedPixels` 形式にクラス1〜4を出力する。
- 公式データ取得済み。学習画像12,568枚、公開環境のテスト画像5,506枚、非空注釈7,095件。学習画像5,902枚には注釈がなく、クラス別の非空注釈は1から順に897、247、5,150、801件。
- Late Submission の動作は下記の P0 記録を参照。

[公式評価](https://www.kaggle.com/competitions/severstal-steel-defect-detection/overview) · [公式データ](https://www.kaggle.com/competitions/severstal-steel-defect-detection/data) · [ダミー Notebook](https://www.kaggle.com/code/shion5/kumite-severstal-p0)

## 進捗

- [x] P0. 参加・評価環境の確認（Rules 承諾、データ取得、ダミー提出で採点されるか）
- [x] P1. 問題設定・EDA（予測仮説を3件以上。予想する結果も先に書く）
- [x] P2. 検証分割の設計（リーク経路の洗い出し、fold の確定）
- [x] P3. Baseline の固定（seed・fold・指標・前処理）
- [x] P4. 自力の仮説検証（E01〜E03）
- [x] P5. 上位解法の調査（4位、9位、10位の本人 writeup）
- [x] P6. 上位解法の着眼点の再現（E04 で独立分類器を比較）
- [x] P7. 最終評価・エラー分析（E04 を選定後に Late 採点し、CV と比較）
- [x] P8. ローカル実験の知見を `report.md` と K042 に保存

## 仮説

| # | 仮説 | 根拠（EDA） | 出所（自力 / 上位解法） | 結果 |
|---|---|---|---|---|
| H1 | 非空マスク向けの soft Dice loss を BCE に加えると、欠陥あり Dice が上がる。空マスクでの誤検出は増えうる | 欠陥なし画像が5,902枚 | 自力 | 欠陥あり Dice は +0.09924、空への誤検出率は +34.54ポイント。全体 Dice は悪化 |
| H2 | 元解像度を保つと、小欠陥の Dice が上がる | 縮小時に狭いマスクの画素が失われうる | 自力 | 小欠陥の Dice は 0.47225 → 0.48211 |
| H3 | 32画素未満の予測を空にすると、空マスクの誤検出率が下がる。小欠陥の全見逃しは増えうる | 空マスクは両方空の場合のみ Dice が1 | 自力 | 誤検出率は −0.22ポイント、小欠陥の全見逃し率は +0.84ポイント |

## 実験ログ

詳細は `experiments.jsonl`（`kumite.runs.RunStore` で追記。記録は書き換えず、やり直しは別の run_id）。主要な結果のみ抜粋する。

| run | 変更点（1つだけ） | OOF | Late | 判定 |
|---|---|---|---|---|
| E00-r2 | BCE の基準 | 0.90618 | Public 0.84655 / Private 0.85225 | 全件再学習した `late-E00` を採点 |
| E01 | BCE と非空マスク向けの soft Dice loss を等重みで併用 | 0.62354 | | 不採用。空マスクへの誤検出が増えた |
| E02 | 元解像度の入力 | 0.90695 | | 基準比 +0.00077 |
| E03 | 予測面積32画素未満のクラスを空にする | 0.90807 | | 基準比 +0.00188 |
| E04 | 独立した欠陥有無分類器でクラス予測を空にする | 0.92488 | Public 0.88413 / Private 0.88359 | 選定。基準比 OOF +0.01870、Private +0.03134 |

初回 E00 は学習中のソース更新と Windows のワーカー再起動が重なり、最初の fold の後で中断した。3 fold が揃わなかったため、基準は同じ設定の `E00-r2` として再実行する。

## P0 採点記録

2026-11-25 に全空マスクの非公開 Notebook 版3を Late Submission し、Public Dice **0.85674**、Private Dice **0.85560** を確認した（提出ID `58850794`）。版2ではサンプル CSV とテスト画像の一致を強制し、非公開再実行時に Kaggle がエラーを返した。版3は再実行時のテスト画像から直接提出キーを作る。

全空マスクのローカル Dice は `1 - 7095 / (12568 × 4) = 0.85887`。

[提出履歴](https://www.kaggle.com/competitions/severstal-steel-defect-detection/submissions)

## EDA と分割

複数クラスがアノテーションされた画像は427枚。最も多い組み合わせはクラス3・4の284枚。非空マスクの面積中央値はクラス1〜4の順に3,326、2,944、11,953.5、25,357画素。同じ鋼板の連続撮影を判定できる系列 ID は公式ファイルに含まれない。

同一バイト列の JPEG を SHA-256 でグループ化し、クラスの有無の組み合わせで層化した3 foldを seed 42 で固定した（`split_version=v1`）。完全一致する学習 JPEG は0件だった。3グループ未満の欠陥組み合わせは希少層にまとめる。分割後に縮小とマスク復号を行う。再圧縮された同一画像や同じ鋼板の別カットはハッシュで識別できない。

| fold | 画像数 | クラス1 | クラス2 | クラス3 | クラス4 |
|---|---:|---:|---:|---:|---:|
| 0 | 4,190 | 300 | 83 | 1,717 | 267 |
| 1 | 4,189 | 298 | 81 | 1,716 | 268 |
| 2 | 4,189 | 299 | 83 | 1,717 | 266 |

## 再現手順

リポジトリのルートで実行する。E00 は ImageNet 事前学習済み ResNet-18 の U-Net を使い、AdamW、学習率 `3e-4`、バッチ8、10 epoch、CUDA の混合精度で学習する。最後の epoch を採点し、元解像度に補間した独立4チャネルの確率を `0.5` で二値化する。

採点済み run と提出の証跡は当時の記録。新しい export は CV 開始時に記録した学習・推論ソースの SHA-256 を要求する。採点済み run にはこの記録がないため、現行コードで再学習・提出する場合は全候補に新しい run ID を使う。以下では `-r3` を使用する。選定は全候補の OOF ファイル、評価条件、キー、fold、再計算した平均 Dice を照合してから行う。

```powershell
$d = 'competitions/072-severstal-steel-defect-detection'
dev download 72
curl.exe --fail --location https://download.pytorch.org/models/resnet18-f37072fd.pth --output "$d/data/resnet18-f37072fd.pth"
uv run python "$d/src/train.py" --run-id E00-r3 --variant baseline
uv run python "$d/src/train.py" --run-id E01-r3 --variant dice
uv run python "$d/src/train.py" --run-id E02-r3 --variant full
uv run python "$d/src/train.py" --run-id E03-r3 --variant area --baseline E00-r3
```

各 CV run は画像×クラス別 Dice・正解面積・予測面積を `oof.csv` に保存する。欠陥あり Dice、空マスクの誤検出率、小欠陥の全見逃し率は `diagnostics.json` に出力する。小欠陥は各外側 fold の学習側で求めたクラス別の面積の下位四分位を基準にする。

ベースラインの外部評価では E00 の CV 設定で全学習画像を再学習する。重みは非公開データセットとして登録し、推論 Notebook の完了後に返された版を指定して採点する。

```powershell
$owner = 'shion5'
uv run python "$d/src/export.py" $owner E00-r3 late-E00-r3 "$d/outputs/late-E00-r3"
kaggle datasets create -p (Resolve-Path "$d/outputs/late-E00-r3/dataset").Path
kaggle kernels push -p (Resolve-Path "$d/outputs/late-E00-r3/kernel").Path
kaggle kernels status "$owner/kumite-steel-late-e00-r3"
kaggle competitions submit severstal-steel-defect-detection -k "$owner/kumite-steel-late-e00-r3" -v <版> -f submission.csv -m "Kumite Severstal E00-r3"
```

`train.py --run-id E04-r3 --variant gate --baseline E00-r3` で独立した欠陥有無分類器を試す。分類器は各 fold の学習画像だけで学習し、画像×クラスの確率が `0.5` 未満なら基準モデルのマスクを空にする。

比較が揃ったら画像単位のペア bootstrap で選ぶ。平均 OOF Dice の最高 run が E00-r2 を上回り、差の95%区間の下限が0より大きければ採用する。E00-r2 を選んだ場合は採点済みのベースラインを使う。

```powershell
uv run python "$d/src/select_model.py" E00-r3 E01-r3 E02-r3 E03-r3 E04-r3 --output "$d/outputs/selection-r3.json"
```

E03 または E04 を選定した場合は、全学習画像で学習済みの `late-E00` のセグメンテーション重みを再利用する。E04 の分類器は全学習画像で学習する。E01 または E02 を選定した場合は `--full-baseline` を外し、選定設定で全件再学習する。

```powershell
$selected = (Get-Content "$d/outputs/selection-r3.json" -Raw | ConvertFrom-Json).selected_run
uv run python "$d/src/export.py" $owner $selected final-r3 "$d/outputs/final-r3" --selection "$d/outputs/selection-r3.json" --full-baseline late-E00-r3
```

選定結果は E04。提出用成果物は `outputs/late-E00/` と `outputs/final/` に保存した。ベースラインの CPU Notebook と最終モデルの CUDA リプレイはそれぞれ公開テスト5,506枚を推論し、22,024行の提出キーと全ての非空 RLE を検証した（ベースライン3,393件、最終2,108件）。最終の CPU Notebook も8枚で32行を生成した。

両方の推論 Notebook の版1で、Kaggle 上でも22,024行の提出キーと全ての非空 RLE を検証した。Late Submission はベースライン `58855506`、最終 `58856834`。実験ログの `late_score` は全件学習した提出 run の Private Dice。

[提出の証跡](submission-evidence.json) · [提出履歴](https://www.kaggle.com/competitions/severstal-steel-defect-detection/submissions)
