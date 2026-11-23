# #061 Planet: Understanding the Amazon from Space

深さは **Standard**。衛星画像に天候と土地利用のタグを付け、多ラベル分類の判定閾値を比較する予定だった。

## P0 の確認結果（2026-11-23）

- 定数予測の Late Submission は採点完了。公式サンプルと同じ画像 ID 順で全件に `clear` を付けた結果は Public **0.30307**、Private **0.30000**。
- 公式 CLI のファイル一覧と一括配布を確認。一括取得で学習ラベル、サンプル提出、テスト ID 対応表、学習・テストの TIFF 用トレントを取得した。JPEG は含まれず、旧ファイル名 `train-jpg.tar.7z`、`test-jpg.tar.7z` の個別取得はともに HTTP 404。
- 公式 JPEG を使う計画は P0 で中断した。P1 以降は未着手。差し替え候補の #059 は学習ラベル取得時点で HTTP 403 のため、参加条件の確認待ち。

[Kaggle 公式データ](https://www.kaggle.com/competitions/planet-understanding-the-amazon-from-space/data) · [差し替え候補の確認結果](../059-cassava-leaf-disease-classification/README.md)

## 定数提出の再現

```powershell
dev download 61
$d = 'competitions/061-planet-understanding-the-amazon-from-space'
uv run python "$d/src/dummy.py"
dev submit 61 "$d/outputs/p0-dummy.csv" -m 'Kumite P0 constant clear'
dev submissions 61
```
