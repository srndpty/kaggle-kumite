# #045 Quora Question Pairs

質問ペアが同じ意図を持つ確率を予測する。深さは **Standard**、CPU で実施する。

- [公式概要](https://www.kaggle.com/competitions/quora-question-pairs/overview)
- 指標は二値 Log Loss。提出形式は `test_id,is_duplicate`。
- 公式提出見本は 2,345,796 行。取得した test.csv は見本と同じ ID 順の行に続いて余分なデータがあるため、見本の行数を公式予測対象とする。
- E00 は定数 0.5。2026-11-19 に採点が完了し、Public / Private ともに 0.69314。

## EDA と検証

- 学習は 404,290 ペア、陽性率 36.92%。本文欠損は左1行・右2行。正規化後に複数回登場する質問は 111,853 件。
- 質問の順序を無視すると重複ペアの余分な行は 541 行、ラベルが矛盾するペアは 10 件。これらは検証でも採点対象とし、同一 fold に収める。
- 連結成分は 200,338 個で最大 17,624 ペア。seed 42 の連結成分単位の層化 5-fold を採用した。各 fold の検証は 80,857〜80,860 行、陽性率は約 36.92%。学習と検証で質問共有はゼロ。
- テストで学習と本文が完全一致する質問の数は、0件が 2,160,296 ペア、1件が 151,113 ペア、2件が 34,387 ペア。合成ペアが含まれるため、採点対象における重複率は不明。

## 事前登録

主評価は両端とも未知の質問ペア。質問 ID と正規化本文で同一質問を束ね、連結成分単位の層化5-foldを試す。検証 fold が不均衡ならノードを5群に分け、境界をまたぐ行を各 fold から除外する。

| Run | 変更点 | 期待する結果 |
|---|---|---|
| E01 | 単語 TF-IDF 類似度と長さで L2 ロジスティック回帰 | 未知質問のペアでも語彙の類似度が重複判定に役立つ |
| E02 | 文字3–5 gram の TF-IDF 類似度を追加 | 綴りや表記の差がある重複で損失が下がる |
| E03 | 単語集合の Jaccard 係数と包含率を追加 | 一方だけに条件が追加されたペアで過大な重複確率が減る |
| E04 | 数字集合と否定の不一致を追加 | 語彙が似ていても意味が異なるペアで過大な重複確率が減る |

P4 完了後に E05 を登録した。1位チームの着眼点を基に、共通する先頭・末尾の単語列の長さを、短い方の質問の単語数で割った2特徴を E01 に追加する。単語の集合が似ていても質問の構造が違うペアで損失が下がると予想する。本文だけから計算でき、同じ未知質問 CV で効果を切り出せるため採用した。[1位チームの writeup](https://www.kaggle.com/competitions/quora-question-pairs/writeups/dl-guys-1st-place-solution)

## 進捗

- [x] P0. 公式データ取得・ダミー採点
- [x] P1. EDA・事前仮説
- [x] P2. リーク点検・分割固定
- [x] P3. E01 Baseline を固定。OOF 0.57134、Public 0.49581、Private 0.49628。
- [x] P4. E02-cv・E03・E04 の OOF を測定
- [x] P5. 1位・4位の本人資料を確認
- [x] P6. E05 の先頭・末尾一致を追加。OOF は 0.56675。
- [x] P7. E03 を固定して外部評価。Public 0.48685、Private 0.48808。OOF を類似度帯・ラベル別に分析
- [x] P8. [レポート](report.md) と [K011](../../knowledge/K011-group-split-vs-random-split.md) に記録

最終モデルは **E03**。Kaggle 提出前に、主 CV の Log Loss が最小の 0.54742 であることを確認して固定した。次点との差は 0.01922 で、事前規則の 0.001 を上回ったため seed 43 の再測定は発動しなかった。

## 再現

リポジトリ直下から実行する。記録済みの run を再実行するときは別の `run_id` と `--experiment` を指定する。

```powershell
$src = 'competitions/045-quora-question-pairs/src'
uv run python "$src/eda.py"
uv run python "$src/train.py" E03-r2 --experiment E03
uv run python "$src/train.py" E02-cv-r2 --experiment E02 --cv-only
uv run python "$src/analyze.py" E03
```

分割診断は `diagnose.py` で行う。`--node-holdout` を付けるとノード単位の補助検証になる。再実行時は `--run-id` で新しい ID を指定する。

`select_model.py` は全候補の OOF を再採点し、提出 CSV と保存済み確率を照合してから、上書き禁止で `selection.json` を作る。CV の勝者に提出成果物がなければ停止する。その場合は同じ構成・seed で提出用の新しい run を作り、`--submission-run` で渡す。今回の [選定証跡](selection.json) は外部採点後の事後記録。

`diagnose.py --audit-run E01-shared-r2` は保存済みの比較分割から、各条件の学習件数と陽性数、実際に共有した質問数、学習行の重複件数と検証ペアの同一性を再構成する。
