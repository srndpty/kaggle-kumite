# #015 Elo Merchant Category Recommendation

深さは **Standard**。CPU で、カード属性と取引履歴からカード単位のロイヤルティを予測する。評価指標は RMSE、提出列は `card_id,target`。 [Kaggle 公式概要](https://www.kaggle.com/competitions/elo-merchant-category-recommendation)

## EDA と検証設計

- 学習 201,917 枚、テスト 123,623 枚。学習カードの 2,207 枚（1.09%）で `target < -20`。最小値は −33.219281、99 パーセンタイルは 4.812662。
- 履歴取引は 29,112,361 行（`month_lag` は −13〜0、カードあたり中央値 55 件）。新規加盟店取引は 1,963,031 行（同 1〜2、中央値 5 件）。初回利用年が 2017 年の比率は学習・テストともに約 65%。テストで初回利用月に欠損が1件。
- カード ID は結合・照合に使い、取引はカード内で集約する。学習・テストの ID 非重複と、集約結合後の行数を検査する。分割は極端値フラグで層化した seed 42 の5分割。観測範囲は公式提供ファイルに準拠し、初回利用年別の誤差も見る。

## 比較実験の事前仮説

| Run | 比較元 | 変更・予想 |
|---|---|---|
| E01 | — | カード属性と全履歴のカード別集約を使う LightGBM 回帰を固定する。 |
| E02 | E01 | 直近1か月（`month_lag == 0`）の集約を追加。直近の行動変化が有益なら RMSE が下がる。 |
| E03 | E01 | 新規加盟店の取引集約を追加。既存履歴にない購買傾向が有益なら RMSE が下がる。 |
| E04 | E01 | 極端値の発生確率と群別平均を推定して混合。発生確率が識別できれば RMSE が下がる。 |

## 再現

```powershell
dev download 15
uv run python competitions/015-elo-merchant-category-recommendation/src/eda.py
foreach ($trial in "E00", "E01", "E02", "E03", "E04", "E05") {
    uv run python competitions/015-elo-merchant-category-recommendation/src/train.py $trial
}
dev compare 15 E01 E02 E03 E04 E05
uv run python competitions/015-elo-merchant-category-recommendation/src/select_model.py E01 E02 E03 E04 E05
```

`train.py` は E00〜E05 を受け付ける。別 seed での再試行は `--seed` と新しい `--run-id` を指定する。成果物は `outputs/runs/<run_id>/`、実験ログは `experiments.jsonl` に保存する。

## 進捗

- [x] P0. 公式データ取得・E00 の採点確認
- [x] P1. EDA と事前仮説
- [x] P2. 分割固定と実データでの検査
- [x] P3. Baseline 固定と初回提出
- [x] P4. 自力実験
- [x] P5. 上位解法調査
- [x] P6. 着眼点の再現
- [x] P7. 最終モデル固定・外部採点・エラー分析
- [x] P8. レポート・知見カード

E00 の Late Submission は Public 3.93068、Private 3.81507。最終モデル E05 は OOF RMSE 3.78302、Public 3.84414、Private 3.74788。結果と選定根拠は [レポート](report.md) を参照。
