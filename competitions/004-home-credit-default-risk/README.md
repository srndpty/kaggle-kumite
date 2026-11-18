# #004 Home Credit Default Risk

- URL: https://www.kaggle.com/competitions/home-credit-default-risk
- 分野: テーブルデータ・リレーショナル集約 / 主催: Home Credit Group
- 深さ: **Standard** / カタログ難度3・計算負荷M
- テーマ: 顧客の申込情報に、他社借入・過去申込・返済履歴を足すと予測が改善するか。
- 予測: 申込者の返済困難を表すTARGETの陽性確率。公式指標ROC AUC。
- 開催: 2018-05-17〜2018-08-29、7,176チーム（公式概要）。順位相当の参考スコアは未調査。

## P0・データと評価

- 2026-11-19 JST、公式データ取得成功。定数0.5のLate SubmissionはCOMPLETE、Public / Privateとも0.50000。提出はSK_ID_CURR,TARGETのCSV。
- train 307,511行、test 48,744行。陽性率0.0807288195。主表キーは一意、train/test ID共有なし。IDを除く完全重複行・train/test共通特徴行は0件。
- bureau 1,716,428行、previous_application 1,670,214行、installments_payments 13,605,401行。trainの履歴保有率は順に85.69%、94.65%、94.84%。testでは86.82%、98.06%、98.36%。
- 公式列説明HomeCredit_columns_description.csvを情報時点の根拠に使う。bureauの申込後更新17行を除外。過去申込はDAYS_DECISION<=0。返済は期日・実支払日とも<=0の観測だけを集約する。支払日欠損2,905行は除外し、未観測の支払を0としない。
- 将来の予定日を将来の実績と混同しない。bureauの終了予定日とpreviousの払出・終了日関連列は使用しない。絶対日時は復元せず、未来時点への汎化を保証しない。
- 元データとEDA成果物はdata/・outputs/eda/（git管理外）。データなしでテストできる。

## 分割・前処理・予算（本実験前に固定）

- seed42の層化3fold、確認用seed43。完全重複特徴行がある場合はグループ層化。顧客ID、fold、ハッシュ、split_versionをrunごとに保存する。
- カテゴリ辞書は学習foldだけで作る。未知カテゴリは欠損。ID・TARGETは特徴量から除外。DAYS_EMPLOYED=365243は欠損へ変換する。
- 履歴は同じ顧客内だけでラベルなしに集約して左結合。件数は履歴なしなら0、その他は欠損。結合前後でキー・行数・順序を検査する。
- 返済は顧客・契約・返済版・返済番号で複数支払をまとめ、支払合計と最終支払日から不足額・遅延日数を計算する。完済までの日数とは限らない。同一返済内の予定額・期日不一致はエラー。実データの観測済み行には不一致0件。
- LightGBM: learning_rate=.05、num_leaves=31、最大2,000round、early stopping=100、8thread、クラス重みなし、deterministic=true、force_col_wise=true。全比較共通。training_config.jsonを正とする。
- E01の1fold試走は10.70秒、ピーク1492MB、best_iteration=254。3fold×10構成に10倍の余裕を含む3212秒を目安に、2,000roundを維持する。集約費用は追加で監視。総学習時間4時間を予算とし、超過見込みなら中断して予算を相談する。
- 試走はE01-benchmarkとして保存し選定対象外。early stoppingで検証ラベルを使うため、CVが楽観側へ偏り得る。

## 自力仮説（本実験前に登録）

| run | 比較元 | 単独変更 | 予想 |
|---|---|---|---|
| E00 | — | 学習fold陽性率 | fold AUC=0.5の基準 |
| E01 | E00 | 主表LightGBM | 定数予測を上回るbaseline |
| E02 | E01 | bureau集約 | 他社借入・残高・延滞が追加情報になる |
| E03 | E01 | previous_application集約 | 過去申込の頻度・金額・承認状況が有効 |
| E04 | E01 | installments_payments集約 | 返済遅延・支払不足が有効 |
| E05 | E01 | 主表欠損数 | 欠損パターンが有効 |

- E02: 借入金額・残高・延滞金額・最大延滞額・延滞日数のmean/max/sum、借入日のmin/max、履歴件数、状態別件数。
- E03: 申込金額・信用額・ローン返済額のmean/max/sum、申込決定日のmin/max、履歴件数、承認状態別件数。
- E04: 分割返済単位の遅延日数・不足額のmean/max、件数、遅延・不足発生率。
- 欠損や延滞の原因、履歴の因果効果までは主張しない。各履歴は他の履歴と組み合わせず、E01への単独追加で比較する。

## 採否・P5以降の規則（結果を揃える前に固定）

- 主評価はfold平均AUC。OOF AUC、OOF/fold LogLoss、実行時間、ピークメモリも保存する。
- 自力E02〜E05完了後に異なる上位解法2〜3件を調査し、本人writeup・公開コードを優先して取得範囲を記録する。
- 未実装の着眼点1件を比較元への単独変更として事前登録し、E06の3foldを実行する。忠実な全解法再現とは区別する。
- seed42で最良の構成を候補とし、直接の比較元とともにseed43で確認する。両seedでfold平均AUCが改善すれば採用、そうでなければ比較元を維持。同点は短時間の構成を優先する。候補を次々に試して採用を積み重ねない。
- seed42採用runの3fold確率平均を最終提出とする。選定JSONと提出ハッシュを上書き禁止で保存してから提出する。Public/Privateは選定に使わない。
- baselineのLateはパイプライン確認。採用後は履歴有無・履歴件数・欠損量別のエラー分析、CVと外部評価の差を記録する。群間でラベル構成が異なるため、AUCだけから難易度を断定しない。

## 実行方法

```powershell
dev new 4
dev download 4
uv run python competitions/004-home-credit-default-risk/src/eda.py
uv run python competitions/004-home-credit-default-risk/src/train.py E01-benchmark --experiment E01 --benchmark
uv run python competitions/004-home-credit-default-risk/src/train.py E00 --experiment E00
uv run python competitions/004-home-credit-default-risk/src/train.py E01 --experiment E01
# E02〜E05も同じCLI。再実行は新しいrun_id。
uv run python competitions/004-home-credit-default-risk/src/analyze.py E01 E02 E03 E04 E05
# 候補と直接の比較元をseed43で実行してからselect_model.pyへ渡す。
dev check
dev orphans 4
```

全学習をRunStore.reserve内で実行し、publish_cv_runで成果物とログを保存する。実測値はexperiments.jsonlを正とし、書き換えない。

## 進捗

- [x] P0. データ取得・ダミー採点
- [x] P1. EDA・事前仮説
- [x] P2. リーク経路確認・分割固定
- [x] P3. baseline固定・初回Late
- [x] P4. 自力実験
- [x] P5. 上位解法調査
- [x] P6. 着眼点再現
- [x] P7. seed確認・最終固定・外部評価・エラー分析
- [x] P8. report・知見カード

## P5 上位解法調査（自力E02〜E05完了後）

| 解法 | 確認した着眼点 | 一次資料と取得範囲 |
|---|---|---|
| 2位 ikiri_DS / Kazuki Onodera | 全期間・1年・2年・3年の履歴集約、最初と最後の申込、主表の差・比率 | [本人公開README](https://github.com/KazukiOnodera/Home-Credit-Default-Risk)。raw README本文を直接取得。元の学習コード全体は未実行 |
| 7位チームの構成モデル / js-aguiar | LightGBM GOSS。複数テーブル集約とテーブル間比率・groupbyを統合。チーム最終解法は13モデルのアンサンブル | [本人公開コード](https://github.com/js-aguiar/home-credit-default-competition)。raw READMEとmain.pyを直接取得。GOSS・統合処理は確認したが、全チーム構成の忠実再現ではない |

1位writeupも取得を試みたが本文が空で、今回の根拠には使わない。上記2件の異なる着眼点を調査対象とする。公開コードの全体学習・検証方式は転用しない。

## P6 事前登録（E06実行前）

- E06の比較元はE02。全期間bureau集約を保持し、DAYS_CREDITが[-365,0]の履歴に同じ集約を適用した特徴群だけを追加する。DAYS_CREDIT_UPDATE<=0も維持する。
- 仮説: 直近借入状況は、全履歴の平均・最大・合計にない追加情報を持ち、fold平均AUCを改善する。
- 365日は上記2位の1年集約に基づき固定する。期間探索・他テーブル追加・モデル設定変更は行わない。元の多期間・全テーブル・最初/最後の申込は再現しない。
- seed42のE01〜E06から最良候補を1件選ぶ。E06なら直接の比較元E02、E02〜E05ならE01とseed43で確認する。選定規則は変更しない。

## 実測結果

| run | 内容 | fold平均AUC | OOF AUC | OOF LogLoss | 秒 |
|---|---|---:|---:|---:|---:|
| E00 | 学習fold陽性率 | 0.500000 | 0.499999 | 0.280546 | 3.6 |
| E01 | 主表baseline | 0.756908 | 0.756898 | 0.246684 | 33.8 |
| E02 | +他社借入集約 | 0.762628 | 0.762555 | 0.244978 | 41.8 |
| E03 | +過去申込集約 | 0.762574 | 0.762511 | 0.245057 | 37.8 |
| E04 | +返済集約 | 0.762466 | 0.762440 | 0.245079 | 62.6 |
| E05 | +主表欠損数 | 0.757045 | 0.756972 | 0.246723 | 32.4 |
| E06 | E02+直近365日集約 | 0.763416 | 0.763354 | 0.244758 | 35.2 |

- E02〜E04はE01に対して全3fold改善。E02とE03の差は約0.000054で、手法の優劣を確定しない。
- E05はAUCが+0.000137、1foldで悪化し、LogLossも微悪化。小差の頑健性は未確認で、採用しない。
- E00はfoldごとに陽性率が異なるため、全OOFでは厳密なAUC=0.5にならない。主選定値はfold平均とする。

### 別seed確認と最終固定

| seed | E02 | E06 | 差 |
|---|---:|---:|---:|
| 42 | 0.762628404 | 0.763416359 | +0.000787955 |
| 43 | 0.762301200 | 0.763863659 | +0.001562459 |

両seedで改善し、それぞれ全3foldでも改善した。foldは独立した再現ではなく、別seedも同じデータ内での確認である。

- 最終採用は**seed42のE06・3fold確率平均**。seed43は確認用のまま保持した。
- 外部評価前に`outputs/selection.json`を作成。提出sha256: `8ecf59436e0dd38aad41537bdd2e2dd2e322ce65cfee6970369e8a8f1a4962ff`。
- E01 Late: Public **0.74657** / Private **0.74435**。E06 Late: Public **0.75548** / Private **0.75557**。どちらもCOMPLETE。
- baseline比Private +0.01122。E06のfold平均CVはPrivateより0.007846高い。分布差、潜在的な顧客間依存、CV選択、OOFとfold平均の差の寄与は切り分けていない。
- E02単体は外部提出していないため、直近集約だけの外部効果は未検証。Lateの改善をリーク不在の証明には使わない。
- 試走と確認seedを含む10 runの実行時間合計320.83秒。集約・読込・予測も各runtimeに含む。最大ピークメモリ4318MBはE04。モデルファイルは保存せず、OOF・test予測と再学習に必要な設定・コード識別子を保存した。

### エラー分析

情報時点制約を適用した履歴件数で群分けした。E06とE01の同じ行群でのOOF AUCを比較する。

| 他社借入履歴件数 | 行数 | E01 | E06 |
|---|---:|---:|---:|
| 0 | 44,021 | 0.730249 | 0.730449 |
| 1〜10 | 231,213 | 0.759648 | 0.766289 |
| 11以上 | 32,277 | 0.760767 | 0.774336 |

履歴ありの群で改善が大きい。ただし群間AUCの大小から難易度や履歴の因果効果を断定しない。主表欠損20列以下では0.757363→0.763733、21列以上では0.753157→0.759837。過去申込・返済件数別の集計と高損失30行のIDは`outputs/analysis/comparison.json`に保存した。

### 成果物と再現の範囲

- 実験は10 run、OOF・test予測・分割・指標・特徴量定義・提出ファイルをハッシュ付き保存。モデルの木そのものは保存していない。再現は同じ公式データと設定からの再学習による。
- 選定時点のログを`outputs/evaluation_log_before_annotations.jsonl`に保管。その後、許可されたlate_scoreとconclusionだけを追記し、計測値と成果物ハッシュは維持した。
- bureau_balance・POS_CASH・credit_card_balance、履歴同士の組合せ、NN、target encoding、スタッキング、時系列CVは未実施。
- 振り返りは[report.md](report.md)、知見は[K017](../../knowledge/K017-customer-history-aggregation.md)・[K018](../../knowledge/K018-recent-history-aggregation.md)。
- 最終`dev check`成功。lint・整形確認・pyright（エラー0）、共通113件・#002 20件・#004 19件・#024 15件・#041 25件・#060 32件、計224テストが通過した。
- 10 run・70成果物のハッシュ照合成功。選定時点ログと提出ハッシュの一致、注記後の計測値不変、`dev orphans 4`で未確定runなしを確認した。
