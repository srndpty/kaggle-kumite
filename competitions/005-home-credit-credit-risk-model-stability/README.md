# #005 Home Credit - Credit Risk Model Stability

- 深さは Standard。申込時点の静的情報と過去申込履歴から、貸倒確率を予測する。
- 公式指標は週別 Gini の平均から、減少傾向と回帰残差のばらつきに対するペナルティを引いた値。提出は `case_id,score` を出す Kaggle Notebook。
- 定数 0.5 のダミー提出（ID 58885518）は Public・Private ともに 0.00000。対象の公式 Parquet を取得済み。 [ダミー Notebook](https://www.kaggle.com/code/shion5/kumite-home-credit-stability-p0)

## 進捗

- [x] P0. 公式データ取得とダミー採点
- [x] P1. 問題設定・EDA
- [x] P2. 時点制約と検証分割
- [x] P3. Baseline と初回モデル提出
- [x] P4. 自力実験
- [x] P5. 上位解法の調査
- [x] P6. 上位解法の追試
- [x] P7. 最終評価と誤差分析
- [x] P8. 知見の保存

## EDA

学習は 1,526,659 件、陽性率は 3.14%。週別で 825〜35,920 件、陽性率は 1.77〜5.21%。static_0 は全申込、static_cb_0 は 1,500,476 件と結合できた。作成日を制限した過去申込特徴は 79.12% の申込で利用でき、週別の割合は 70.94〜88.51% だった。static_cb_0 の一部は 97% 超が欠損しており、欠損率の変化を検証する根拠とした。

## 検証設計

週53–65、66–78、79–91の3つを検証窓とし、各窓より前の申込を学習に使う。各検証週に両クラスがあることを確認済み。主指標は窓ごとに計算した公式スコアの単純平均。39週をまとめたスコアと各ペナルティも記録する。申込週で分割し、ラベル確定遅延の影響はこの分割では評価できない。主催者は外部データも申込日時点で収集されると説明している。 [主催者の説明](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/472866)

特徴には申込 ID、週、月、生の申込日、利用時点を特定しにくい静的テーブルの日付列を含めない。過去申込は作成日が今回の判断日より前の行に限る。同日の順序は日付から判断できないため除外する。カテゴリ辞書と欠損率による列選択は各学習窓で確定する。

## 事前仮説

| run | 比較元 | 変更と期待する結果 |
|---|---|---|
| E00 | — | 静的特徴、CPU LightGBM、seed 42、固定500ラウンド |
| E01 | E00 | 過去申込の件数と経過日数を追加すると平均週別 Gini が上がる |
| E02 | E01 | 学習行を半減期26週で重み付けすると週別 Gini の低下が緩和される |
| E03 | E01 | 学習期間の前後半で欠損率差が20ポイント以上の列を除くと残差の標準偏差が下がる |

3窓平均の最大値で採用し、完全同点なら登録順が先の run を採用する。

## 上位解法の追試

1位チームは履歴の各項目を最大・最小・平均などで集約し、複数学習器を組み合わせた。13位チームは手作業と相関で特徴を選び、複数方式の予測をスタッキングした。E04 は自力実験の最良 E01 に、1位解法から着想した過去申込の融資額・カード限度額の最大値を1列だけ追加する。前回までの申込から得られる限度額の情報が、件数・経過日数だけの場合より平均週別 Gini を改善すると予想する。 [1位チームの本人解説](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/writeups/yuuniee-1st-place-solution-my-betting-strategy) · [13位チームの本人解説](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/writeups/team-dic-13th-place-solution-pmts-year-1139t-postp)

## 実測結果

| run | 3窓平均安定性スコア |
|---|---:|
| E00 | 0.50292 |
| E01 | 0.50804 |
| E02 | 0.46105 |
| E03 | 0.50346 |
| E04 | **0.51406** |

E04 を CV で選定し、最終モデル F00 の提出（ID 58886515）は Public 0.44888、Private 0.36145。初回モデル（ID 58886234）は Public 0.43314、Private 0.34425。 [最終 Notebook](https://www.kaggle.com/code/shion5/kumite-home-credit-stability-e04) · [振り返り](report.md)

## 実行

- `uv run python competitions/005-home-credit-credit-risk-model-stability/src/train.py E00` で CV。
- 同コマンドに `--owner <Kaggle ユーザー名>` を付けると、その構成を Kaggle 上で全行再学習する Notebook を `outputs/runs/F-E00/` に生成する。E00〜E04 の CV がそろったら、試行名を `best` にして最良構成の Notebook を `outputs/runs/F00/` に生成する。再試行は `--run-id` で別 ID を指定する。
- 公式データは `dev download 5` で取得できる。個別取得する場合は、Kaggle CLI の `-f` にディレクトリ区切りを `%2F` に変換したファイル名を指定し、配布時の階層に配置する。ZIP で返るファイルは展開する。
- 入力は `data/parquet_files/{train,test}/` 配下の base、static_0、static_cb_0、applprev_1。実験ログは `experiments.jsonl`。
