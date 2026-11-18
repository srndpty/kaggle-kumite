# #002 Porto Seguro’s Safe Driver Prediction

- URL: https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction
- カテゴリ: A / 難度: 2 / 計算負荷: S / 深さ: Standard
- 今回の比較実験テーマ: 不均衡二値分類、Gini と確率評価の違い、欠損符号

匿名化された契約者・車両の特徴から、翌年の保険請求の有無を予測する。主催は Porto Seguro。保険のリスク順位付けが背景にある。

## 基本情報と EDA

- 評価: Normalized Gini（大きいほど良い）。ローカルは同点を平均順位として扱う `2*AUC-1`。LogLoss・Average Precision を補助評価する。
- 開催: 2017-09-29〜2017-11-29。5,156 teams（公式概要ページ）。順位比較用の LB 分位点は未取得。
- CSV 提出: `id,target`。予測は5 foldの確率平均、sample submission の ID 順に照合する。
- 実測データ: train 595,212 行、test 892,816 行、57 特徴、陽性率 0.036447517859182946。
- 特徴完全重複 0、train/test の ID 重複 0。匿名特徴に時間・人物グループを直接表す列は確認できない。IDを特徴に使わず、層化5分割を採用する。未知の時間・グループ構造がないと証明したわけではない。
- EDA の列別分布・欠損率・train/test 平均は `outputs/eda/columns.csv`、件数・分割根拠は `summary.json`。
- Late Submission: E00 が 2026-11-18 に COMPLETE。Public 0.00001、Private -0.00706。P0 の採点確認に限って参照し、モデル選定には使わない。

## 仮説（実行前の計画で予想を固定）

| # | 仮説・予想 | 比較 |
|---|---|---|
| H1 | 数値の -1 を NaN にして大小関係から切り離すと Gini が改善する | E02 − E01 |
| H2 | 行の欠損数に契約属性の情報があり、追加すると改善する | E03 − E02 |
| H3 | 名義カテゴリを数値扱いすると不自然な順序制約で悪化する | E04 − E01 |
| H4 | 陽性重み付けは確率を歪めるが、Gini を改善する可能性がある | E05 − E01 |

いずれも自力の仮説。P4完了前に上位解法・公開Notebookを体系的に調べない。

## 検証・再現条件

- `StratifiedKFold(5, shuffle=True, random_state=42)`、`stratified5-seed42-v1`。完全重複があるデータでは group 分割を選び、別 split_version を記録する。
- 数値欠損の変換は固定規則。カテゴリ辞書は各学習foldだけで作り、-1もカテゴリに保持、未知値は欠損にする。target encoding は初期実験で使わない。
- LightGBM binary / AUC early stopping、learning_rate=0.05、num_leaves=31、最大5,000 round、patience=100、CPU 8 threads、deterministic。
- 検証foldでearly stoppingするため、CVに選択の楽観が含まれる。補助指標で停止roundを選び直さない。
- 主選定値はfold平均Gini、全体OOF Giniも記録。各foldの確率尺度が違うため両者は一致しない。
- 全 run は学習前に RunStore.reserve、保存は publish_cv_run 経由。データ・分割・コード・成果物ハッシュを記録。seed違いは別run_idとsplit_versionにする。
- 最終候補と直接の比較元を seed 43/44 でもペア比較し、3 seedの平均差が正かつ2 seed以上で改善したら採用。満たさなければ比較元を維持。Privateを最終選定に使わない。
- P5後は本人のwriteup・公開コードを優先して異なる上位解法2〜3件を調べ、重要な着眼点1〜2件を単独変更で再現する。

## 実行

```powershell
dev download 2
uv run python competitions/002-porto-seguro-safe-driver-prediction/src/eda.py
uv run python competitions/002-porto-seguro-safe-driver-prediction/src/train.py E01
uv run python competitions/002-porto-seguro-safe-driver-prediction/src/train.py E01-s43 --experiment E01 --seed 43
uv run python competitions/002-porto-seguro-safe-driver-prediction/src/analyze.py E01 E02 E03 E04 E05
dev compare 2 E01 E02 E04 E05
dev check
```

`analyze.py` は同一データ・分割の run だけ比較し、欠損有無・主要カテゴリ別指標、予測分位点、厳密単調変換 `0.1+0.8*p` の影響を保存する。順位指標には行損失平均用のbootstrapを流用しない。

## 進捗

- [x] P0. データ取得・E00ダミー提出の採点確認
- [x] P1. 基本EDA・事前仮説
- [x] P2. リーク経路の確認・分割固定
- [x] P3. Baseline固定・初回提出（採点完了）
- [x] P4. 自力実験 E02〜E05
- [x] P5. 異なる上位解法2〜3件の調査
- [x] P6. 上位解法の着眼点1〜2件の再現
- [x] P7. seed再検証・モデル固定・外部評価・エラー分析
- [x] P8. reportと知見カード

## 実験ログ

実測値は `experiments.jsonl` を正とし、主要な結果を完了時に追記する。補助指標は各runの `metrics.json`。

## P5: 上位解法の調査（P4 完了後）

2026-11-18、E02〜E05 の保存完了後に調査した。両方の全体構成の再現はせず、Standard として特徴群の除外を1件切り出す。

| 解法 | 確認した着眼点 | 出典の種類 |
|---|---|---|
| 1位 Michael Jahrer | DAE の表現を使う NN と生特徴の LightGBM を平均。calc 特徴を除外し、カテゴリは one-hot。欠損は保持 | [本人の投稿を許可を得て転載した記事](https://kaggler.com/2017/12/01/winners-solution-porto-seguro.html)。Kaggle元投稿は本文取得不能 |
| 2位 Xiaozhou Wang チーム | 他の特徴群から car/ind/reg の各特徴を予測した教師なし特徴を NN/GBDT に利用。GBDT は欠損数・カテゴリ頻度・ind 組合せ頻度も使用 | [本人の公開リポジトリ](https://github.com/xiaozhouwang/kaggle-porto-seguro)、`code/fea_eng0.py` と `code/gbm_model291.py` を確認 |

- P6 は E04 を比較元に、`ps_calc_*` の除外だけを E06 として試す。仮説はノイズ削減による改善。欠損処理・カテゴリ方式・モデル条件は変えない。
- DAE、教師なし特徴予測、NNとの平均、one-hot、頻度特徴、大量seed平均は未再現。今回の結果からその効果を主張しない。
- 外部スコアは掲載値を今回の実測として転記しない。

## 実測結果（2026-11-18）

Giniは大きいほど良い。表はseed42、外部スコアはKaggle表示精度。`late_score` はPrivateを保存し、Publicは本表とconclusionに記録する。

| run | 変更点・比較元 | fold平均Gini | 全体OOF Gini | OOF LogLoss | Public | Private |
|---|---|---:|---:|---:|---:|---:|
| E00 | 学習foldの陽性率 | 0.000000 | -0.000039 | — | 0.00001 | -0.00706 |
| E01 | LightGBM Baseline | 0.262591 | 0.262451 | 0.152456 | 0.26562 | 0.27239 |
| E02 | E01の数値 -1 をNaN化 | 0.261655 | 0.261400 | 0.152497 | — | — |
| E03 | E02に欠損数追加 | 0.261862 | 0.261682 | 0.152483 | — | — |
| E04 | E01のカテゴリを数値扱い | 0.278371 | 0.278306 | 0.152023 | — | — |
| E05 | E01の陽性重みを約26.44倍 | 0.258753 | 0.256030 | 0.588285 | — | — |
| **E06** | **E04からps_calc_*を除外** | **0.280392** | **0.280191** | **0.151923** | **0.27899** | **0.28414** |

H1・H3・H4は予想を支持せず、H2の改善は小さく未確定。とくにH3は予想と逆で、数値扱いが良かった。LightGBMのカテゴリ分割と数値分割の学習方式の違いを一括で比較した結果であり、元カテゴリに意味ある順序があると証明したものではない。

| ペア（candidate − parent） | seed42 | seed43 | seed44 | 平均 |
|---|---:|---:|---:|---:|
| E04 − E01 | +0.015781 | +0.017050 | +0.015331 | +0.016054 |
| E06 − E04 | +0.002020 | +0.002777 | +0.003166 | +0.002654 |

各値は同一seed・同一分割のfold平均Gini差。分割seedもモデルseedも変えているため、学習乱数だけの効果とは解釈しない。

最終モデルはE06のseed42・5fold確率平均。`src/select_model.py` が事前規則を実行し、提出前に `outputs/analysis/selection.json` へ採用理由と提出ファイルのSHA256を保存した。提出SHA256は `2efeb901fb02c292777f5b7013348a8ecfb6592428e2de23db7679bd5ed71a9c`。選定時ログのハッシュはLate・conclusion追記前のもの。

## エラー分析と限界

- E06の欠損なし124,931行はGini 0.255408 / LogLoss 0.180130、欠損あり470,281行は0.276451 / 0.144430。陽性率など母集団も違うので、欠損自体が精度改善の原因とは言えない。
- E06の予測を `0.1+0.8*p` に変えるとGiniは同じ0.280191だが、LogLossは0.151923→0.205791。順位が同じでも確率は悪くなりうる。これは診断計算であり、別の学習runや提出ではない。
- E00はfoldごとの学習陽性率がわずかに違うため、fold内Gini=0でも全体OOFは厳密には0にならない。ローカルは同点平均順位で、Kaggleの同点処理と完全一致する保証はない。定数予測の外部値を漏洩の兆候と解釈しない。
- E06のPublicはOOF比 -0.001201、Privateは+0.003949。方向はおおむね整合するが、fold平均と単一OOFの違いを入れ子の疑似testで切り分けてはいない。CVの非リーク性を外部スコアで証明したわけではない。
- Giniの信頼区間・独立データでの再現は未実施。3 seedは同一データの再分割で、独立した3コンペの再現ではない。
- 参考順位・LB分位点は未取得。上位解法のDAEやNN統合は未実装。今回の目的は評価軸と前処理の比較に絞った。

## 成果物と検証

- 各run: `oof.npy`, `test.npy`, `fold_info.json`, `folds.csv`, `metrics.json`, `submission.csv`。全13 runをRunStoreで保存。
- 分析: `outputs/analysis/E01-E02-E03-E04-E05.json`, `E01-E04-E06.json`。読み込みはRunStoreでハッシュ照合する。
- `dev check`: lint・整形・pyright成功、共通109件＋本コンペ7件＋Recruit15件＋State Farm32件＝163テスト成功。
- 実行環境の制約でuvの再同期を避け、既存venvに対して `UV_NO_SYNC=1` で標準の `dev check` を実行した。依存追加はない。
- 知見: [K013](../../knowledge/K013-categorical-splits-vs-numeric.md)、[K014](../../knowledge/K014-ranking-vs-probabilities.md)、[K015](../../knowledge/K015-feature-group-ablation.md)。
