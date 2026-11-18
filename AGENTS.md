# AGENTS.md

Kaggle 100本ノック: 終了済みの賞金付きコンペ100題を解き、再現可能な ML 実験レポート集を作るリポジトリ。

## 基本方針

- 人間が読む想定の文書は日本語で書く
- DRY や責務の分離などを意識する
- ただし、個人向けプロジェクトかつ開発中のため、フォールバックや過剰な互換性残しは不要（禁物）

## 構成

- `catalog/` — 100コンペの一覧。`kaggle100_catalog.csv` を正とする
- `src/kumite/` — コンペ横断で使う共通ライブラリ（カタログ、雛形生成、実験ログ、run の保存、クロスフィット、リーク検査など）
- `templates/competition/` — コンペ用ディレクトリの雛形
- `competitions/NNN-<slug>/` — 1ディレクトリ1コンペ
  - `README.md`（進捗・仮説・実験抜粋）, `report.md`（振り返り）, `experiments.jsonl`（実験ログ）
  - `src/`, `notebooks/` — コンペ固有のコード
  - `data/`, `outputs/` — git 管理外。データ・モデル・提出ファイルは commit しない
- `knowledge/` — 100コンペを横断した知見カード（何がわかったか・どの条件で有効か）
- `tests/` — `src/kumite` のテスト。コンペ固有コードのテストは、そのコンペの `src/test_*.py` に置く（`dev test` がコンペごとに別プロセスで実行する。コンペ間でモジュール名がぶつかるため）

2つ以上のコンペで使うコードは `src/kumite` に移す。コンペ間でコピーしない。

## 開発コマンド

`dev <command>`（実体は `dev.ps1`）。一覧は `dev help`。

- `dev setup` — 依存関係の同期と pre-commit の導入
- `dev check` — lint + 整形確認 + 型検査 + テスト。CI と同一なので、commit 前に通す
- `dev format` / `dev lint` / `dev test` / `dev clean`
- `dev list` / `dev new <番号>` / `dev download <番号>`
- `dev submit <番号> <ファイル> -m <メッセージ>` / `dev submissions <番号>` / `dev compare <番号> <基準> <run_id...>` / `dev orphans <番号> [--delete]`

Python は uv で管理する（`uv run ...`、依存追加は `uv add --group <group> ...`）。
ML 依存は dependency group（`tabular` など）に分ける。既定グループ（`dev` + `tabular` + `vision`）は CI でも入れる。torch は Windows では CUDA 版、Linux（CI）では CPU 版（`pyproject.toml` の `tool.uv.sources`）。
型検査は pyright。`src/` と `tests/` は strict。`competitions/` は ML ライブラリの型情報が不完全なため
「型が不明」系のルールだけ切っている（`pyproject.toml` の `executionEnvironments`）。型エラーは無視せず、型が付く書き方に直す。

## 1コンペの進め方

各コンペの `README.md` のチェックリストに沿って進める。

- P0. 参加・評価環境の確認（ダミー提出で採点できるか。不能なら記録して差し替え）
- P1. 問題設定・EDA（予測仮説を最低3件。予想する結果も先に書く）
- P2. 検証分割の設計（リーク経路を洗い出してから fold を確定する）
- P3. Baseline の固定（CV と、可能なら初回の Late Submission）
- P4. 自力の仮説検証（上位解法を読む前に 3〜6 実験。失敗した仮説も記録）
- P5. 上位解法の調査（1位だけでなく、アプローチの異なる上位解法を 2〜3 件）
- P6. 上位解法の着眼点の再現（1〜3 件を自分の Baseline でアブレーション）
- P7. 最終評価・エラー分析（事前に決めたモデルを外部評価し、CV との整合性を確認）
- P8. 知見の保存（`report.md` と `knowledge/` のカード）

### 深さ

着手時に README の「深さ」を決める。目安は Deep Dive 20件・Standard 60件・Survey 20件で、進捗を見て調整する。

| | Deep Dive | Standard | Survey |
|---|---|---|---|
| EDA | 詳細 | 主要な分布と仮説 | 最低限 |
| 比較実験 | 8〜20 件以上 | 3〜6 件 | 1〜2 件 |
| CV 設計 | 複数方式を検証 | 適切な方式を固定 | 基本方式 |
| 上位解法 | 実装して再現・差分分析 | 重要な着眼点を確認 | 概要を確認 |
| レポート | 詳細 | 1〜2 ページ | 要点のみ |

Survey でも、リーク対策・正しい検証分割・Baseline・評価指標の確認は省略しない（浅い実験と不正確な実験は別物）。
評価軸は「1コンペで何件実験したか」より「全体で何種類の独立した原理を検証できたか」。
コンペの選び方も、既に検証した原理を別条件で確かめるか、新しい原理に挑むかを意識する。

## 実験の原則

- 変更は一度に1つだけ。組み合わせ効果は片方ずつの実験と比較して判断する
- 同じ CV 分割でのペア比較を原則とする。分割を変えたら `split_version` を変える
- seed・fold・評価指標・前処理を固定し、再現できる状態で比較する
- 全ての実験を `experiments.jsonl` に記録する。保存は `kumite.runs.RunStore` を通す（`reserve` で run_id を予約してから学習し、返った予約を `publish` に渡す。run_id は英数字・`-`・`_` のみ。データ・分割・コード・成果物のハッシュが残る。強制終了の残骸は `dev orphans`）
- 記録は追記のみ。計測値は書き換えない。やり直すときは別の run_id（例: `E01-r2`）にする。後から書き足してよいのは `late_score`・`conclusion`・`next_experiment` だけ
- ラベルを使う特徴量（target encoding・近傍特徴・スタッキングなど）は、外側 fold の学習行とそのラベルだけから作る（内側 fold で OOF にする）。全行で作った OOF を後段の CV に使わない。`kumite.checks.check_label_independence` で、外側の検証ラベルを変えても入力が変わらないことを確かめる
- 予測は保存・提出の前に検査する（形状・NaN/Inf・確率の範囲と和）
- 評価の使い分け: CV は仮説検証とモデル選定の主評価。Public LB は提出パイプラインの確認と補助的な外部評価。Private LB は最終モデルの外部評価で、モデル選定には使わない。Late Submission で改善しても、CV にリークがないことの証明にはならない
- fold 間の改善の一貫性は補助的な証拠として扱う（fold 同士は学習データが重なるので独立な再現ではない）。小さい差は seed を変えた CV や行単位の損失差の bootstrap で確かめる
- 上位解法や公開 Notebook を体系的に読むのは P5 から。それまでは自力で仮説を立てる
- 上位解法の出典は、本人の writeup・公開コード > 当時の発表資料・技術ブログ > 第三者の解説 > 検索結果の抜粋 の順に優先し、どれに基づくかを記録する
- 実測していないスコアや結果を文書に書かない

## 知見の管理

- 実験ログ（`experiments.jsonl`）は「何を実行したか」の証跡、`knowledge/` は「何がわかったか」
- 知見はカード（`knowledge/K###-<slug>.md`）に、仮説・観測・適用条件・未検証の点・証跡を書く。書式は `knowledge/README.md`
- 同じ知見を別コンペで再確認したら、新しいカードを作らず既存カードに証跡を追記する。反例も同じカードに書く
