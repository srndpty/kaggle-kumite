# #037 Shopee - Price Match Guarantee

- URL: https://www.kaggle.com/competitions/shopee-product-matching
- カテゴリ: C / 難度: 4 / 計算負荷: L
- 今回の比較実験テーマ: 画像＋文書の埋め込み・類似検索
- 深さ: Standard

## コンペ概要

Shopee の出品タイトルと画像から、各出品と同じ商品を指す出品の集合を予測する。

## 基本情報（P0 で記入）

- 評価指標: 出品ごとの集合 F1 の平均。自己一致を含む。
- 提出形式: インターネットを無効にした Kaggle Notebook から `posting_id,matches` の `submission.csv` を生成する。
- Late Submission: 自己一致だけを返す非公開 Notebook v1 が Public 0.466、Private 0.458 で採点された。同じ v1 が2件登録され、両方で同じスコアを確認した。
- データ概要: 学習34,250行、商品グループ11,014件。公開テストは3行でサンプル提出と ID 順が一致する。参照先の画像は学習32,412ファイル、公開テスト3ファイルが揃っている。

[公式評価・提出仕様](https://www.kaggle.com/competitions/shopee-product-matching/overview/code-requirements)

## P0 の再現

```powershell
dev download 37
$d = 'competitions/037-shopee-product-matching'
uv run python "$d/src/dummy.py" <Kaggleユーザー名> "$d/outputs/p0"
kaggle kernels push -p "$d/outputs/p0"
kaggle competitions submit shopee-product-matching -k <ユーザー名>/kumite-shopee-p0 -v <版> -f submission.csv -m 'Kumite Shopee P0'
```

## 進捗

- [x] P0. 参加・評価環境の確認（Rules 承諾、データ取得、ダミー提出で採点されるか）
- [x] P1. 問題設定・EDA（予測仮説を3件以上。予想する結果も先に書く）
- [x] P2. 検証分割の設計（リーク経路の洗い出し、fold の確定）
- [x] P3. Baseline の固定（5 fold CV と初回 Late Submission の登録）
- [x] P4. 自力の仮説検証（E01〜E03）
- [x] P5. 上位解法の調査（本人の writeup 3件）
- [x] P6. 上位解法の着眼点の再現（E04 の双方向フィルタ）
- [x] P7. 最終評価・エラー分析（E04 の公式採点と OOF のエラー分析）
- [x] P8. 知見の保存（`report.md`、K038 の証跡追記）

## 仮説

| # | 仮説 | 根拠（EDA） | 出所（自力 / 上位解法） | 結果 |
|---|---|---|---|---|
| H1 | 文字に単語の類似度を加えると、同じ型番や商品名を共有する出品で候補漏れが減る | 同一タイトルを共有する出品は2,095件 | 自力 | 全体の候補 Recall は +0.00026、F1 は −0.00928 |
| H2 | 画像検索を追加すると、タイトルの表現が異なる同じ商品の候補 Recall と F1 上限が上がる | 同一画像 ID を共有する出品は3,084件 | 自力 | 全体の候補 Recall は +0.03023、F1 上限は +0.01414 |
| H3 | 候補を固定し、タイトル・画像の双方で採用を判定すると平均 F1 が上がる | 46の画像 ID が複数の商品ラベルにまたがる | 自力 | F1 は +0.07892。候補ペア3,193,074件は E02 と一致 |

## 検証分割

商品グループのサイズは中央値2、最大51。提出側は計画どおり自己一致込み最大50件とし、評価時の正解集合には学習 CSV のラベルから得た全投稿を含める。
`label_group` または同一画像 ID でつながる出品を連結し、seed 42 の GroupKFold で5分割する。行数は順に 7,087、6,658、6,781、6,759、6,965。商品ラベル・同一画像 ID が複数 fold に現れないことを確認した。タイトルが一致する別ラベルは誤一致の分析対象にする。検索対象は同じ検証 fold 内とし、語彙・閾値・分類器の学習は外側学習 fold 内に限定する。

```powershell
$d = 'competitions/037-shopee-product-matching'
uv run python "$d/src/split.py" "$d/data/train.csv" "$d/outputs/split_v1.csv"
```

## 実験ログ

詳細は `experiments.jsonl`（`kumite.runs.RunStore` で追記。記録は書き換えず、やり直しは別の run_id）。主要な結果のみ抜粋する。

| run | 変更点（1つだけ） | OOF | Late Private | 判定 |
|---|---|---|---|---|
| E00 | 文字 3–5 gram TF-IDF | 0.75837 | 0.531 | 基準 |
| E01 | E00 に単語 1–2 gram cosine を等重みで追加 | 0.74909 | | 候補 Recall は微増、F1 は低下 |
| E02 | E00 に画像 cosine の上位50候補を追加 | 0.75837 | | 候補 Recall と F1 上限が改善 |
| E03 | E02 の候補を LightGBM で再スコアリング | 0.83729 | | 候補を固定して F1 改善 |
| E04 | E03 の採用を双方向に一致するペアに限定 | 0.83944 | 0.638 | 最終モデル |

E00 は Late Public 0.540。公開テスト3件の Notebook 出力はローカルと内容が一致した。 [E00 Notebook](https://www.kaggle.com/code/shion5/kumite-shopee-e00-submission)

E04 の GPU Notebook はオフラインで公開テスト3件を処理し、ローカルと同じ出力を生成した。Late Submission #58842447 は Public 0.646、Private 0.638 で採点完了。 [E04 Notebook](https://www.kaggle.com/code/shion5/kumite-shopee-e04-submission) · [提出証跡](submission-evidence.json)

## 上位解法

E01〜E03 の計測と保存を終えてから、以下の本人による writeup を確認した。

- 1位の Upstage は ArcFace で学習したモダリティ別の埋め込みを使い、近傍の埋め込みを混ぜて検索し直す操作を繰り返した。 [1位の writeup](https://www.kaggle.com/competitions/shopee-product-matching/writeups/upstage-making-ai-beneficial-1st-place-solution-fr)
- 2位の lyakaap と tkm2261 は、検索候補を LightGBM と Graph Attention Network で判定し、両方向で採用されたペアを残した。E04 では双方向のフィルタだけを E03 に加えた。 [2位の writeup](https://www.kaggle.com/competitions/shopee-product-matching/writeups/lyaka-tkm-2nd-place-solution-matching-prediction-b)
- 7位の T Nakamura は NFNet とインドネシア語 DistilBERT を距離学習し、正規化済みの埋め込みを連結してから、類似度で重み付けした近傍でクエリを拡張した。 [7位の writeup](https://www.kaggle.com/competitions/shopee-product-matching/writeups/t-nakamura-7th-place-solution)

## 実験の再現

```powershell
$d = 'competitions/037-shopee-product-matching'
uv run python "$d/src/train.py" E00
uv run python "$d/src/train.py" E01
Invoke-WebRequest https://download.pytorch.org/models/resnet18-f37072fd.pth -OutFile "$d/outputs/resnet18-f37072fd.pth"
uv run python "$d/src/images.py" "$d/data/train.csv" "$d/data/train_images" "$d/outputs/resnet18-f37072fd.pth" "$d/outputs/train-embeddings.npz"
uv run python "$d/src/train.py" E02
uv run python "$d/src/train.py" E03
uv run python "$d/src/train.py" E04
```

各 run の候補、採用フラグ、行別 F1、fold、閾値を `outputs/runs/<run_id>/` に保存する。再実行は `--run-id E00-r2` のように別 ID を使う。

## 最終モデルの再現

外側 OOF の平均 F1 が最大の E04 を、最終提出の採点前に選んだ。 [選定記録](selection.json)

```powershell
$d = 'competitions/037-shopee-product-matching'
$out = "$d/outputs/e04-reproduction"
$owner = '<Kaggleユーザー名>'
$slug = 'kumite-shopee-e04'
uv run python "$d/src/export.py" E04 $owner $slug $out
uv run python "$d/src/inference.py" "$d/data" "$out/dataset" "$out/submission.csv"
uv run python "$d/src/analyze.py" E04 "$out/analysis.csv"
kaggle datasets create -p (Resolve-Path "$out/dataset").Path
kaggle datasets status "$owner/$slug"
kaggle kernels push -p "$out/notebook"
kaggle kernels status "$owner/$slug-submission"
kaggle competitions submit shopee-product-matching -k "$owner/$slug-submission" -v 1 -f submission.csv -m 'Kumite Shopee E04 final'
```
