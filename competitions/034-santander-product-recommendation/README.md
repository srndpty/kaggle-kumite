# #034 Santander Product Recommendation

深さは **Standard**。CPU で、顧客の前月保有状況から翌月に新規取得する銀行商品を最大7件推薦する。公式指標は MAP@7、提出列は `ncodpers,added_products`。 [Kaggle 公式概要](https://www.kaggle.com/c/santander-product-recommendation)

## 検証設計

- 顧客・月で直前の暦月だけを結合し、商品ごとの `0 → 1` を正解にする。欠損は未知として保持する。取得率の分母は前月が非保有で当月の状態も既知の商品・顧客に限る。候補からは保有が確認できた商品だけを外す。
- 2016年3月・4月を拡張窓 CV（`monthly-v1`）、5月を選定後のホールドアウト、6月を本番とする。各予測の学習には前月までの取得イベントを使う。CV は24商品すべての正解が確定できる顧客を対象とし、取得ゼロなら AP は0。内部評価の対象範囲と Kaggle の対象範囲は異なり得る。
- E00 の EDA では3月の採点対象は 919,070 / 925,076 行、4月は 923,414 / 928,274 行。対象者の新規取得ゼロはそれぞれ 96.98%、97.10%。両月合計の取得件数は口座引き落とし（`ind_recibo_ult1`）が 19,916 件、年金給与口座（`ind_nom_pens_ult1`）が 9,462 件、給与口座（`ind_nomina_ult1`）が 8,764 件。

## 事前仮説

| Run | 変更 | 予想 |
|---|---|---|
| E01 | 全期間の商品別取得率で並べる | 月別 MAP@7 の比較基準 |
| E02 | E01 の集計を直近3か月に限定 | 直近の需要を反映して改善 |
| E03 | E01 の集計を保有商品数 0・1・2 以上で分ける | 顧客の保有状況を反映して改善 |
| E04 | E01 の順位付けを商品別 L2 ロジスティック回帰に変更 | 商品の組み合わせを捉えて改善 |

取得率の平滑化は `(取得件数 + 0.5) / (非保有件数 + 1)`。E04 は保有不明も入力として区別し、単一クラスの商品は平滑化率を使う。商品同点は商品列名の昇順。P4 完了後に上位解法を調べる。

## 再現

リポジトリルートから実行する。各 run は初回にこの ID で実行し、再実行は別 ID で記録する。

```powershell
dev download 34
$src = "competitions/034-santander-product-recommendation/src"
uv run python "$src/train.py" --run-id E00 --variant dummy
uv run python "$src/train.py" --run-id E01 --variant baseline
uv run python "$src/train.py" --run-id E02 --variant recent
uv run python "$src/train.py" --run-id E03 --variant portfolio
uv run python "$src/train.py" --run-id E04 --variant logistic
uv run python "$src/train.py" --run-id E05 --variant seasonal
uv run python "$src/analyze.py"
$variant = (Get-Content competitions/034-santander-product-recommendation/selection.json -Raw | ConvertFrom-Json).variant
uv run python "$src/train.py" --run-id E06 --variant $variant --final
uv run python "$src/analyze.py" --summarize competitions/034-santander-product-recommendation/outputs/runs/E06/evaluation.npz
```

## 進捗

- [x] P0. E00 のダミー採点確認
- [x] P1. 問題設定・EDA・事前仮説
- [x] P2. リーク経路の確認と分割固定
- [x] P3. E01 の Baseline 固定・初回提出
- [x] P4. 自力実験
- [x] P5. 上位解法の調査
- [x] P6. 着眼点の再現
- [x] P7. 最終評価・エラー分析
- [x] P8. レポートと知見カード

2026年11月22日（日本時間）に E00 と E01 の Late Submission が採点された。E00 は全商品を同点として商品列名順に並べ、保有済み商品を除外した。E00 の Public / Private は 0.00374 / 0.00379。E01 は OOF 0.01947、Public / Private は 0.02365 / 0.02388。

## P5〜P6 ─ 上位解法と比較実験

E02〜E04 の保存後に、次の本人 writeup を確認した。

- 5位は、前年同月や直近期間などで学習した XGBoost モデルを混合し、季節性と時間変化を捉えた。 [5位本人の writeup](https://www.kaggle.com/competitions/santander-product-recommendation/writeups/jared-turkewitz-and-breakfastpirate-5-solution)
- 4位は、推薦の順位評価を意識した独自目的関数を XGBoost に実装した。 [4位本人の writeup](https://www.kaggle.com/competitions/santander-product-recommendation/writeups/yoniko-4th-place-solution)

E05 は5位の学習期間の着眼点を E01 の取得率に移す。前年同月の取得数と非保有機会だけを2倍に重み付けし、全期間のほかの観測は重み1とする。季節性があれば MAP@7 の改善を予想する。重み2はこの比較のための固定値。
