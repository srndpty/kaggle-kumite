# kaggle-kumite

Kaggle 100本ノック。終了済みの賞金付きコンペ100題で、EDA → 仮説 → 比較実験 → 評価 → 上位解法の分析を繰り返す。
ゴールは順位ではなく、100件の再現可能な ML 実験レポート集。

- コンペ一覧: [catalog/kaggle100_catalog.md](catalog/kaggle100_catalog.md)
- 進め方・規約: [AGENTS.md](AGENTS.md)

## セットアップ

必要なもの: [uv](https://docs.astral.sh/uv/)、PowerShell 7、kaggle CLI（`~/.kaggle/kaggle.json` 設定済み）。

```powershell
dev setup
```

## 使い方

```powershell
dev list          # カタログ一覧と着手状況
dev new 1         # competitions/001-.../ を生成
dev download 1    # データを data/ に取得（事前に Kaggle で Rules の承諾が必要）
dev check         # lint + 整形確認 + 型検査 + テスト
dev help          # コマンド一覧
```
