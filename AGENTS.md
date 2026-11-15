# AGENTS.md

Kaggle 100本ノック: 終了済みの賞金付きコンペ100題を解き、再現可能な ML 実験レポート集を作るリポジトリ。

## 基本方針

- 人間が読む想定の文書は日本語で書く
- DRY や責務の分離などを意識する
- ただし、個人向けプロジェクトかつ開発中のため、フォールバックや過剰な互換性残しは不要（禁物）

## 構成

- `catalog/` — 100コンペの一覧。`kaggle100_catalog.csv` を正とする
- `src/kumite/` — コンペ横断で使う共通ライブラリ（カタログ、雛形生成、実験ログなど）
- `templates/competition/` — コンペ用ディレクトリの雛形
- `competitions/NNN-<slug>/` — 1ディレクトリ1コンペ
  - `README.md`（進捗・仮説・実験抜粋）, `report.md`（振り返り）, `experiments.jsonl`（実験ログ）
  - `src/`, `notebooks/` — コンペ固有のコード
  - `data/`, `outputs/` — git 管理外。データ・モデル・提出ファイルは commit しない
- `tests/` — `src/kumite` のテスト

2つ以上のコンペで使うコードは `src/kumite` に移す。コンペ間でコピーしない。

## 開発コマンド

`dev <command>`（実体は `dev.ps1`）。一覧は `dev help`。

- `dev setup` — 依存関係の同期と pre-commit の導入
- `dev check` — lint + 整形確認 + 型検査 + テスト。CI と同一なので、commit 前に通す
- `dev format` / `dev lint` / `dev test` / `dev clean`
- `dev list` / `dev new <番号>` / `dev download <番号>`
- `dev submit <番号> <ファイル> -m <メッセージ>` / `dev submissions <番号>` / `dev compare <番号> <基準> <run_id...>`

Python は uv で管理する（`uv run ...`、依存追加は `uv add --group <group> ...`）。
ML 依存は dependency group（`tabular` など）に分ける。既定グループ（`dev` + `tabular`）は CI でも入れる。
型検査は pyright。`src/` と `tests/` は strict。`competitions/` は ML ライブラリの型情報が不完全なため
「型が不明」系のルールだけ切っている（`pyproject.toml` の `executionEnvironments`）。型エラーは無視せず、型が付く書き方に直す。

## 1コンペの進め方

各コンペの `README.md` のチェックリストに沿って進める。

- P0. 参加・評価環境の確認（ダミー提出で採点できるか。不能なら記録して差し替え）
- P1. 問題設定・EDA（予測仮説を最低3件）
- P2. 検証分割の設計
- P3. Baseline の固定
- P4. 仮説検証・Ablation
- P5. Late Submission とエラー分析
- P6. 上位解法との照合
- P7. 知見の保存

## 実験の原則

- 変更は一度に1つだけ。組み合わせ効果は片方ずつの実験と比較して判断する
- 同じ CV 分割でのペア比較を原則とする。分割を変えたら `split_version` を変える
- seed・fold・評価指標・前処理を固定し、再現できる状態で比較する
- 全ての実験を `experiments.jsonl` に記録する（`kumite.experiments.ExperimentRecord`）
- 上位解法や公開 Notebook を体系的に読むのは P6 から。それまでは自力で仮説を立てる
- 実測していないスコアや結果を文書に書かない
