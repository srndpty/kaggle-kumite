# #001 Otto Group Product Classification Challenge

- URL: https://www.kaggle.com/competitions/otto-group-product-classification-challenge
- カテゴリ: A / 難度: 2 / 計算負荷: S
- 今回の比較実験テーマ: 多クラス分類：LogLoss・モデルの較正

## 基本情報（P0 で記入）

- 評価指標: multi-class logloss（小さいほど良い）
- 提出形式（File / Code / 推論API）: File。`id` + `Class_1`〜`Class_9` の確率 CSV（144,368 行）
- Late Submission 可否: 未確認（ダミー提出で確かめる）
- データ概要: 商品1件 = 1行。匿名の非負整数カウント特徴 `feat_1`〜`feat_93`、目的変数は9クラスの商品カテゴリ

## 進捗

- [ ] P0. 参加・評価環境の確認（Rules 承諾・データ取得は済。ダミー提出が未実施）
- [x] P1. 問題設定・EDA（予測仮説を3件以上）

## EDA（P1）

`uv run python competitions/001-otto-group-product-classification-challenge/src/eda.py` で再現できる。図は `outputs/eda/` に出力される。

### データの基本

- train 61,878 行、test 144,368 行（test は train の約2.3倍）。特徴量は 93 列
- 欠損なし。値はすべて 0 以上の整数で、最大値は train 352、test 398
- 特徴量が完全一致する train 行の重複はない。全特徴量が 0 の行は train に 0 行、test に 1 行

### 目的変数

| クラス | 件数 | 比率 |
|---|---:|---:|
| Class_1 | 1,929 | 3.1% |
| Class_2 | 16,122 | 26.1% |
| Class_3 | 8,004 | 12.9% |
| Class_4 | 2,691 | 4.3% |
| Class_5 | 2,739 | 4.4% |
| Class_6 | 14,135 | 22.8% |
| Class_7 | 2,839 | 4.6% |
| Class_8 | 8,464 | 13.7% |
| Class_9 | 4,955 | 8.0% |

- 不均衡で、最多クラスと最少クラスの比は 8.4 倍。Class_2 と Class_6 で約半分を占める
- 基準の logloss: 一様予測は 2.1972、クラス比をそのまま予測すると 1.9504。後のモデルはこれを大きく下回る必要がある

### 注意: id は目的変数の順に並んでいる（リーク）

- train は Class_1 → Class_9 の順に並んでおり、Spearman(id, class) は 0.982
- train と test の id は別々に 1 から振られているので、test では id に意味がない
- そのため、id は特徴量から必ず除外する。CV の前には必ずシャッフルする（シャッフルなしの KFold だと fold ごとにクラス構成が偏る）

### 特徴量の分布

- 疎なデータ: 非ゼロ率は全体で 20.7%。1行あたりの非ゼロ特徴量数は平均 19.2（最小 1、最大 68）
- 特徴量ごとの非ゼロ率は 1.9%〜64.3%。非ゼロ率 1% 未満の特徴量はない
- 右に長い裾を持つ: 非ゼロ値の中央値は 1、99% 点は 22、99.9% 点は 51。歪度は最大 36.5（feat_30）
- 相関: log1p 変換後に |r| > 0.5 となるペアが 50 組ある。最大は feat_3 と feat_46 の 0.775

### クラスごとの違い

| クラス | 行合計の平均 | 非ゼロ数の平均 |
|---|---:|---:|
| Class_1 | 57.6 | 20.5 |
| Class_2 | 41.3 | 14.8 |
| Class_3 | 45.7 | 15.6 |
| Class_4 | 40.7 | 15.5 |
| Class_5 | 26.9 | 9.4 |
| Class_6 | 68.9 | 23.7 |
| Class_7 | 72.5 | 24.0 |
| Class_8 | 88.0 | 24.3 |
| Class_9 | 70.4 | 22.1 |

- 特定クラスでだけよく出る特徴量がある。例: feat_61 は Class_6 で非ゼロ率が全クラス平均の 6.3 倍、feat_90 は Class_8 で 3.9 倍
- クラス平均プロファイル（log1p）の cosine 類似度が高いのは、Class_2 と Class_3（0.973）、Class_2 と Class_4（0.932）、Class_3 と Class_4（0.925）、Class_1 と Class_9（0.912）

### train / test の分布差

- 特徴量ごとの非ゼロ率の差は最大 0.0064（feat_70）、平均 0.0016。行合計の分布もほぼ同じ
- 分布シフトはほぼないので、train 内のランダムな分割で CV を作ればテストを代表できると考える

### P2 への申し送り

- 時刻やグループの構造はない。id を除外して StratifiedKFold（shuffle あり、5 fold、seed 固定）とする
- 1 行だけある全ゼロのテスト行は、モデルの出力にそのまま任せる（特別扱いしない）

## 検証分割（P2）

- `src/cv.py` の `make_folds` を全実験で使う。`split_version = skf5-seed42`（StratifiedKFold、shuffle あり、5 fold、seed 42）
- 各 fold は 12,375〜12,376 行。fold 間のクラス比の差は最大 0.0001
- 分割を変えるときは `split_version` を変え、それより前の実験とは直接比較しない

## Baseline（P3）

`uv run python competitions/001-otto-group-product-classification-challenge/src/train.py <run_id>` で実行する。

- 構成: 実験定義と学習ループは `src/train.py`、モデルは `src/models.py`（どのモデルも `fit_predict` という同じ関数形）
- 出力: `outputs/runs/<run_id>/` に OOF 予測、テスト予測（5 fold の平均）、`submission.csv`、fold ごとの情報。`experiments.jsonl` に1行追記される（同じ run_id の再記録はエラーになる）
- 固定したもの: seed 42、上記の分割、指標は `sklearn.metrics.log_loss`、前処理なし（生カウント、id は除外）
- E01 の LightGBM は lr 0.05、num_leaves 31、その他は既定値。最大 10,000 round で、検証 fold を使って early stopping（100 round）する。best iteration は 422〜533
  - 注意: 検証 fold で early stopping するため、OOF スコアはわずかに楽観側に偏る。全実験で同じ条件なので、比較には影響しない
- 再現性: E01 を2回実行し、OOF 予測の差が最大 6.7e-16 であることを確認した

### P4 での判断基準

- E01 の fold スコアの標準偏差は 0.0076 あり、fold によるばらつきが大きい
- そのため改善の判断は OOF の差だけでなく、5 fold のうちいくつで改善したかと、同じ fold どうしの差の平均で行う
- [x] P2. 検証分割の設計
- [x] P3. Baseline の固定（seed・fold・指標・前処理）
- [ ] P4. 仮説検証・Ablation（単一要因で3回以上）
- [ ] P5. Late Submission とエラー分析（CV と外部スコアの整合性）
- [ ] P6. 上位解法との照合
- [ ] P7. 知見の保存（`report.md`）

## 仮説

| # | 仮説 | 根拠（EDA） | 結果 |
|---|---|---|---|
| H1 | 線形モデルや NN では、log1p 変換でカウントの裾を潰すと logloss が改善する（決定木系にはほぼ影響しない） | 非ゼロ値の中央値 1 に対して 99.9% 点は 51、歪度は最大 36.5 | |
| H2 | 誤差は Class_2・3・4 の取り違えに集中する。クラスごとの logloss に分解すると、この3クラスの寄与が件数比以上に大きい | クラス平均プロファイルの cosine 類似度は 2-3 間で 0.973、2-4 間で 0.932、3-4 間で 0.925 | |
| H3 | 行単位の集約特徴（行合計・非ゼロ数）を加えると、木モデルの logloss も改善する | 非ゼロ数の平均は Class_5 の 9.4 から Class_8 の 24.3 まで、クラスで大きく異なる | |
| H4 | 木モデルの出力確率は較正がずれており、温度スケーリングなどの較正で logloss が下がる | 評価指標が logloss で、比較実験のテーマが較正のため（EDA 由来ではなく、検証用に追加した仮説） | |

## 実験ログ

詳細は `experiments.jsonl`（`kumite.experiments` で追記）。主要な結果のみ抜粋する。

| run | 変更点（1つだけ） | OOF | Late | 判定 |
|---|---|---|---|---|
| E00 | クラス比予測（統計的 baseline） | 1.95041 | | 下限の目安 |
| E01 | LightGBM、生カウント（51 秒） | 0.47393 | | 基準 |
