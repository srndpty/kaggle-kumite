# #074 Global Wheat Detection

深さは **Standard**。圃場画像から小麦の穂を検出し、撮影元が変わったときの性能を調べる。

- 公式評価は IoU 0.50〜0.75 の6閾値で計算する画像単位の検出スコアの平均。
- 提出はインターネット無効の非公開 Kaggle Notebook。列は `image_id,PredictionString`。
- 公式配布から学習 JPEG 3,422 枚、公開テスト JPEG 10 枚、学習ラベル、サンプル提出を取得した。学習 CSV に現れる画像は 3,373 枚で、撮影元は7種類。残り49枚の撮影元は CSV から直接取得できない。

[Kaggle 公式評価仕様](https://www.kaggle.com/competitions/global-wheat-detection/overview/timeline) · [公式データ](https://www.kaggle.com/competitions/global-wheat-detection/data)

## 進捗

- [x] P0. 参加・評価環境の確認（Rules 承諾、データ取得、ダミー提出で採点されるか）
- [x] P1. 問題設定・EDA（予測仮説を3件以上。予想する結果も先に書く）
- [x] P2. 検証分割の設計（リーク経路の洗い出し、fold の確定）
- [x] P3. Baseline の固定（seed・fold・指標・前処理、初回の Late Submission）
- [x] P4. 自力の仮説検証（E01〜E04）
- [x] P5. 上位解法の調査（2位・9位の本人 writeup）
- [x] P6. 上位解法の着眼点の再現（E05）
- [x] P7. 最終評価・エラー分析（E02 の Late Submission と OOF の誤差分析）
- [x] P8. 知見の保存（レポート、K011・K023・K034）

## 仮説と比較設定

| 試行 | E00 からの変更 | 実験前の予想 |
|---|---|---|
| E00 | 基準。COCO 事前学習済み Faster R-CNN、512 px、8 エポック、バッチ2、水平反転、seed 42 | 撮影元が異なる画像への汎化を計測する |
| E01 | 画像単位ランダム 3-fold | 同じ撮影元が学習と検証に入ると OOF が上がる |
| E02 | 768 px | 小さい穂の見逃しが減る |
| E03 | 明度とコントラストの拡張 | 照明条件の違いによる性能低下が縮まる |
| E04 | 推論の NMS IoU だけを 0.50 から 0.65 へ変更 | 密集画像の見逃しが減り、重複検出が増える |

## データと検証

集計は `uv run python competitions/074-global-wheat-detection/src/audit.py` で再現できる。

- 学習 CSV は 147,793 矩形。ラベルがある画像あたりの矩形数の中央値は43、矩形面積の中央値は 5,488 px²。公開テスト JPEG とサンプル提出の画像 ID は一致した。
- `source-v1` は撮影元を分けた 3-fold。検証枚数は 1,055 / 1,151 / 1,167。撮影元を特定できない49枚は両方の分割から外した。
- 画像数は `arvalis_1` 1,055、`ethz_1` 747、`arvalis_3` 559、`rres_1` 432、`arvalis_2` 204、`usask_1` 200、`inrae_1` 176。撮影元の大きさが異なるため、全画像平均に加えて撮影元別の結果も集計する。
- 撮影元別の画像あたり矩形数の中央値は21〜69、矩形面積の中央値は 4,050〜12,160 px²。撮影元によって穂の密度とサイズが異なる。
- ファイルの SHA-256 で完全重複は0件。撮影元を特定できた画像について、64-bit の差分ハッシュでハミング距離4以下の候補は0組。
- 信頼度閾値は0.25、検出上限は画像あたり300件。各 IoU 閾値で信頼度順に一対一で照合し、正解・予測がともに空の画像は平均から外す。

## 実験ログ

詳細は `experiments.jsonl`（`kumite.runs.RunStore` で追記。記録は書き換えず、やり直しは別の run_id）。主要な結果のみ抜粋する。

| run | 変更点（1つだけ） | OOF | Late | 判定 |
|---|---|---|---|---|
| E00-r2 | Baseline | 0.445057 | Public 0.5459 / Private 0.4791 | 基準 |
| E01 | 画像単位ランダム分割 | 0.612980 | | 診断。撮影元分割より +0.167923 |
| E02 | 入力解像度 512 → 768 px | 0.486631 | Public 0.6054 / Private 0.5343 | 最終採用 |
| E03 | 明度・コントラストの拡張 | 0.462752 | | 基準より +0.017696 |
| E04 | NMS IoU 0.50 → 0.65 | 0.406032 | | 基準より −0.039024 |
| E05 | 水平反転推論と矩形融合 | 0.463676 | | 基準より +0.018619 |

初回の `E00` は Windows の一時ディレクトリ権限によって学習開始前に中断した。`E00-r2` は同じ設定での再実行。

`E00-r2` の fold スコアは 0.412320 / 0.420016 / 0.499349。OOF は全検証画像 3,373 枚の平均。

全学習画像 3,422 枚で再学習し、2026-11-24 にベースラインを提出した。実験ログの `late_score` は Private。 [ベースライン Notebook](https://www.kaggle.com/code/shion5/kumite-wheat-e00-r2) · [提出履歴](https://www.kaggle.com/competitions/global-wheat-detection/submissions)

## P0 の再現

```powershell
dev download 74
$d = 'competitions/074-global-wheat-detection'
uv run python "$d/src/dummy.py" <Kaggleユーザー名> "$d/outputs/p0"
kaggle kernels push -p "$d/outputs/p0"
```

非公開 Notebook の実行完了後、返された版と `submission.csv` を指定して Late Submission を実行する。

```powershell
kaggle competitions submit global-wheat-detection -k <ユーザー名>/kumite-wheat-p0 -v <版> -f submission.csv -m "Kumite Wheat P0"
dev submissions 74
```

2026-11-23 に非公開 Notebook `shion5/kumite-wheat-p0` の版1を採点。固定矩形の Public / Private はともに **0.0004**。

## 比較実験の実行

```powershell
uv run python "$d/src/train.py" E00 E00-r2
uv run python "$d/src/train.py" E01 E01
uv run python "$d/src/train.py" E02 E02
uv run python "$d/src/train.py" E03 E03
uv run python "$d/src/train.py" E04 E04 --baseline E00-r2
uv run python "$d/src/train.py" E05 E05 --baseline E00-r2
```

`export.py <Kaggleユーザー名> <CV run ID> <新規出力先> --run-id <再学習run ID>` は、指定した CV 設定で再学習し、公開テストの出力と、重み・提出用 Notebook を生成する。再学習には公式 CSV に矩形がない49枚も空ラベルで加える。独立した run には全学習データ・コード・CV 成果物・選定記録のハッシュと、生成物を保存する。ログの `score_source: source_cv` は引き継いだ CV スコアを示す。成果物は `outputs/runs/<再学習run ID>/` に保存し、指定した出力先にも提出用に配置する。重みを非公開データセットに登録し、`kernel/` を Kaggle で実行する。

最終比較は `analyze.py <新規選定JSON> E00-r2 E02 E03 E04 <P6のrun ID>` で記録する。ランダム分割 E01 は撮影元を分けた CV との差を見るために使う。最終提出では `export.py` に `--selection <選定JSON>` を追加する。

[選定記録](selection.json) に従い、主分割で最も高い `E02` を最終構成に固定した。 [振り返り](report.md)

最終評価は非公開 CPU Notebook の版1を使用した。当時は GPU の待ち時間を受けて CPU に切り替えた。現在は `export.py <Kaggleユーザー名> E02 <新規出力先> --run-id final-E02 --selection <選定JSON> --device cpu` で、公開テストの推論と Notebook を同じデバイス設定で生成できる。学習は CUDA を使用する。選定時と最終学習前に保存済み OOF 予測を再採点する。 [実際に採点した Notebook の証跡](submission-evidence.json)

証跡の `selection_sha256` は提出前の選定ファイルのバイト列に対応する。現在の `selection.json` を、キー順を維持した `json.dumps(..., indent=2)`、UTF-8、CRLF、末尾改行なしで再シリアライズすると同じハッシュを得られる。

提出 CSV は JPEG とサンプル提出の ID を照合し、テンプレート順で生成する。Kaggle が公開テンプレートを保持したまま JPEG 一式を非公開テストへ差し替える場合に限り、公開テンプレートのハッシュと JPEG ID の完全な非重複を確認して、非公開 JPEG の ID 順で出力する。

最終 Late Submission は **Public 0.6054 / Private 0.5343**。ベースラインとの差は Public +0.0595 / Private +0.0552。 [提出履歴](https://www.kaggle.com/competitions/global-wheat-detection/submissions)
