# #024 Recruit Restaurant Visitor Forecasting

- URL: https://www.kaggle.com/competitions/recruit-restaurant-visitor-forecasting
- カテゴリ: B / 難度: 3 / 計算負荷: M
- 今回の比較実験テーマ: 予約・来店・休日の系列結合
- 深さ: Standard
- この題で確かめること: K007（時系列分割とランダム分割）・K008（検証でも情報の打ち切りを再現）・K009（log 目的変数）を別条件で再確認する。新しい原理は「予約という別系列の結合と、予約日時による打ち切り」と「GW を含む予測期間に対する検証窓の位置」

## コンペ概要

**日本の飲食店の日ごとの来店客数を、過去の来店数・予約・祝日から 39 日先まで予測するコンペ。**

- 分野: 時系列・テーブルデータ（店舗 × 日の来店数の回帰。複数のデータ源の結合）
- 主催: Recruit Holdings（Air レジ・ホットペッパーグルメを運営）
- 背景（なぜこの予測が必要か）: 来店数が分かれば、飲食店は仕入れや人員配置を前もって計画できる
- 予測するもの: Air レジを使う各店舗（air）の各日の来店客数（`visitors`）
- 入力: Air レジの来店記録、Air レジ・ホットペッパーグルメ（hpg）の予約、店舗のジャンル・地域・緯度経度、祝日フラグ
- 開催期間・参加チーム数: 2017 年 11 月〜2018 年 2 月 6 日（Leaderboard の最終提出日時の範囲）。`kaggle competitions leaderboard --download` で取得した Public Leaderboard は 2,150 行（2026-11-16 取得）
- 参考スコア（Public Leaderboard）: 1 位 0.46535、上位 10% 0.47800、中央値 0.48920

出典について: 主催者・背景は Kaggle Overview の説明文を記憶に基づいて要約したもの。期間・行数・スコアは取得した Public Leaderboard から確認した。

## 基本情報（P0 で記入）

- 評価指標: RMSLE（Root Mean Squared Logarithmic Error、小さいほど良い）。Evaluation ページの記憶に基づく
- 提出形式（File / Code / 推論API）: File。`id`（`<air_store_id>_<日付>`）+ `visitors` の CSV（32,019 行）
- Late Submission 可否: 可（2026-11-16 に「全行に来店数の中央値 17」を提出し、Public 0.84082 / Private 0.83157 が返ることを確認。Public 2,010 / 2,150 位相当）
- 提出: `dev submit 24 <ファイル> -m <メッセージ>`、履歴とスコアは `dev submissions 24`
- データ概要（実測、`src/dataset.py`）:
  - `air_visit_data.csv`: 来店記録 252,108 行（2016-01-01 〜 2017-04-22、829 店舗）。来店記録のある日だけで、休業日などの行は無い
  - test（`sample_submission.csv`）: 2017-04-23 〜 05-31 の **39 日**、821 店舗（すべて来店記録がある）。GW（4/29・5/3〜5/5 の祝日）を含む
  - 予約: `air_reserve.csv` 92,378 件（314 店舗）、`hpg_reserve.csv` 約 200 万件のうち air と対応の付くもの 28,183 件（150 店舗）。**予約日時はどちらも 2017-04-22 23:00 まで**（test 期間の予約は、train 末尾までに入ったものだけ。2,209 件）
  - `date_info.csv`: 2016-01-01 〜 2017-05-31 の曜日・祝日フラグ。`air_store_info.csv`（829 店舗）・`hpg_store_info.csv`・`store_id_relation.csv`（150 組）

## 進捗

- [x] P0. 参加・評価環境の確認（Rules 承諾、データ取得、ダミー提出で採点されるか）
- [ ] P1. 問題設定・EDA（予測仮説を3件以上。予想する結果も先に書く）
- [ ] P2. 検証分割の設計（リーク経路の洗い出し、fold の確定）
- [ ] P3. Baseline の固定（seed・fold・指標・前処理。可能なら初回の Late Submission）
- [ ] P4. 自力の仮説検証（上位解法を読む前に 3〜6 実験）
- [ ] P5. 上位解法の調査（アプローチの異なる上位解法を 2〜3 件。出典の種類を記録）
- [ ] P6. 上位解法の着眼点の再現（1〜3 件をアブレーション）
- [ ] P7. 最終評価・エラー分析（事前に決めたモデルの外部評価と CV との整合性）
- [ ] P8. 知見の保存（`report.md`、`knowledge/` のカード）

## 仮説

| # | 仮説 | 根拠（EDA） | 出所（自力 / 上位解法） | 結果 |
|---|---|---|---|---|
| H1 | | | 自力 | |
| H2 | | | 自力 | |
| H3 | | | 自力 | |

## 実験ログ

詳細は `experiments.jsonl`（`kumite.runs.RunStore` で追記。記録は書き換えず、やり直しは別の run_id）。主要な結果のみ抜粋する。

| run | 変更点（1つだけ） | OOF | Late | 判定 |
|---|---|---|---|---|
| E00 | Baseline | | | 基準 |
