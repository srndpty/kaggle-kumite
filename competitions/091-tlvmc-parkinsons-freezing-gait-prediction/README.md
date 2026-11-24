# #091 Parkinson's Freezing of Gait Prediction

- URL: https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction
- カテゴリ: F / 難度: 4 / 計算負荷: M
- 今回の比較実験テーマ: 未知の被験者に対する加速度の時間窓と収録条件別の検出性能
- 深さ: Standard

## コンペ概要

装着型センサーで記録した3軸加速度から、歩行凍結の3種類を時点ごとに検出する。
主催は The Michael J. Fox Foundation。

## 基本情報（P0 で記入）

- 評価指標: 3クラスの平均 AP。DeFOG は `Valid & Task` の行が採点対象。
- 提出形式: インターネット無効の Kaggle Notebook が `Id,StartHesitation,Turn,Walking` の CSV を生成する。
- Late Submission: 公式ページに表示あり。採点の確認は P0 に記録する。
- データ概要: tDCS FOG は 128 Hz、DeFOG は 100 Hz。出典は [公式の評価・データ説明](https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/overview/evaluation)。

## 進捗

**実装・学習完了、外部評価未達。** 固定5 fold の比較実験と最終モデル E04 の選定を完了した。Kaggle の Late Submission は再提出でも形式エラーとなり、外部スコアは取得できなかった。追加の手がかりが得られるまで再提出を保留する。 [採点証跡](submission-evidence.json)

- [ ] P0. 参加・評価環境の確認（Rules 承諾、データ取得、ダミー提出で採点されるか）
- [x] P1. 問題設定・EDA（予測仮説を3件以上。予想する結果も先に書く）
- [x] P2. 検証分割の設計（リーク経路の洗い出し、fold の確定）
- [x] P3. Baseline の固定（seed・fold・指標・前処理。可能なら初回の Late Submission）
- [x] P4. 自力の仮説検証（上位解法を読む前に 3〜6 実験）
- [x] P5. 上位解法の調査（アプローチの異なる上位解法を 2〜3 件。出典の種類を記録）
- [x] P6. 上位解法の着眼点の再現（1〜3 件をアブレーション）
- [ ] P7. 最終評価・エラー分析（事前に決めたモデルの外部評価と CV との整合性）
- [x] P8. 知見の保存（`report.md`、`knowledge/` のカード）

## P0 の確認記録（2026-11-24）

公式ファイル一覧にアクセスできた。公式サンプル提出を使ったローカル実行では、286,370 行の ID が元の順序と一致し、3クラスすべてに定数 0 を出力した。
[採点確認用 Notebook](https://www.kaggle.com/code/shion5/kumite-fog-p0) は非公開・インターネット無効で実行した。サンプル提出の ID を使った v1、テスト収録から ID を生成した v2、全テスト CSV を収録 ID 順で処理し整数を出力した v3 は、いずれも採点段階で形式エラーとなった。
E01 と E04 は Kaggle 上の公開テストで実行でき、ローカルと同じ286,370行の ID と予測値を生成した。Late Submission は両方とも形式エラーとなり、スコアは返らなかった。この環境で外部採点は成立せず、P0 と P7 の外部評価は未完了。具体的な原因は確認できていない。 [採点証跡](submission-evidence.json)

2026-11-25 に、2位の Notebook と同じ ID の辞書順で E04 v2 を再提出した。公開テスト286,370行では行順だけが変わり、ID ごとの予測値は v1 と完全に一致した。再提出 #58838775 でも同じ形式エラーが返り、行順の変更では解消しなかった。個別提出 API にも該当行や列の詳細はなかった。 [2位の提出 Notebook](https://www.kaggle.com/code/takoihiraokazu/cv-ensemble-sub-0607-1) [再提出の証跡](submission-evidence.json)

公式アーカイブは約63 GB。個別ダウンロードが HTTP 429 になったため、公式アーカイブの HTTP Range 取得で不足分を回収した。本実験で使う924件の学習収録、公開テスト、2種の収録メタデータ、サンプル提出を取得した。

このディレクトリから、次のコマンドで採点確認用 Notebook を再生成できる。

```powershell
uv run python src/dummy.py <Kaggleユーザー名> outputs/p0
kaggle kernels push -p outputs/p0
```

## 仮説

| # | 仮説 | 根拠（EDA） | 出所（自力 / 上位解法） | 結果 |
|---|---|---|---|---|
| H1 | 過去1秒の加速度の平均と標準偏差を追加すると、瞬時値より全体の AP が上がる | センサーの単一時点では運動強度の変化を捉えにくい | 自力 | 全体 +0.03291 |
| H2 | 1秒窓を現在時点の前後に置くと、直前の情報だけを使う窓より StartHesitation の AP が上がる | 歩き始めの前後で信号が変わると予想する | 自力 | 対象クラス −0.00030 |
| H3 | 過去窓を1秒から3秒に延ばすと、Turn と Walking の AP が上がる | 旋回と歩行の推移をより長く観測できる | 自力 | Turn +0.03172、Walking +0.00059 |
| H4 | 収録種別ごとに別モデルを学習すると、全体 AP が上がる | サンプリング周波数・計測装置・ラベル頻度が異なる | 2位解法 | 全体 −0.00168 |

## 上位解法調査

- 2位は収録種別ごとに GRU を学習し、推論では学習時より長い系列を入力した。DeFOG ではイベント種別欠落データへの疑似ラベルも使った。 [本人の writeup](https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/writeups/waiwai-2nd-place-solution)
- 10位は両収録種別で共通の1次元 U-Net を使い、重複する長い窓と左右反転で推論を平均した。 [本人の writeup](https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/writeups/james-day-10th-place-solution-u-net-with-squeeze-e)

E05 は2位解法から収録種別ごとの学習だけを取り入れる。E04 の特徴量、学習時の間引き、fold、LightGBM の設定を固定する。

## 検証手順

| 収録種別 | 収録数 | 被験者数 | 全行 | 採点対象行 | StartHesitation 陽性 | Turn 陽性 | Walking 陽性 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeFOG | 91 | 38 | 13,525,702 | 4,090,530 | 500 | 586,829 | 98,518 |
| tDCS FOG | 833 | 62 | 7,062,672 | 7,062,672 | 304,790 | 1,678,782 | 207,838 |

両方の収録を持つ被験者 ID は0人。924件の学習収録に完全一致ファイルや `Time` の不連続はなかった。
同じ被験者の別収録が fold をまたぐと、個人の動き方や収録の癖を検証側で再利用できてしまう。被験者を単位に収録種別の構成を層化し、seed 42〜71 の候補から、各 fold に3クラスの陽性がありクラス別陽性数の偏りが最小の5 fold を固定した。
陽性の被験者間偏りは残り、最多の1人に `StartHesitation` の陽性行の77.8%が集中していた。DeFOG の1 fold には同クラスの陽性がなかった。
特徴量は収録をまたがず、マスクとラベルは特徴量から除外した。中央窓は収録全体を受け取れるバッチ推論を前提とする。
LightGBM は全試行で同じ100ラウンドとし、学習は各収録の8サンプルごと、OOF の採点は公式のマスクを適用した全行で行う。
このディレクトリで以下を実行する。

```powershell
uv run python src/prepare.py
foreach ($run in 'E00', 'E01', 'E02', 'E03', 'E04') {
    uv run python src/train.py $run
    if ($LASTEXITCODE -ne 0) { break }
}
# 上位解法の調査後に収録種別ごとの学習を検証する
uv run python src/train.py E05
uv run python src/select_model.py E01 E02 E03 E04 E05
```

提出には `src/export.py <run-id> <Kaggleユーザー名> <出力先>` を使う。生成されたモデル重みを `kaggle datasets create -p <datasetの絶対パス>` で非公開登録し、`kaggle kernels push -p <notebookの絶対パス>` で実行する。Notebook はモデル重みの SHA-256 を検証し、その実行で提供されたテスト収録から提出 ID を作る。

## 実験ログ

詳細は `experiments.jsonl`（`kumite.runs.RunStore` で追記。記録は書き換えず、やり直しは別の run_id）。主要な結果のみ抜粋する。

| run | 変更点（1つだけ） | OOF | Late | 判定 |
|---|---|---|---|---|
| E00 | 学習 fold のクラス別陽性率 | 0.07314 | | 定数基準 |
| E01 | 加速度の瞬時値と収録種別、LightGBM | 0.09644 | 形式エラー | モデル基準 |
| E02 | 過去1秒の軸別平均・標準偏差 | 0.12935 | | E01 から改善 |
| E03 | 1秒窓の中央揃え | 0.12995 | | 全体で微増、対象クラスは低下 |
| E04 | 過去窓を3秒に延長 | 0.13722 | 形式エラー | Turn・Walking が改善、最終採用 |
| E05 | E04 を収録種別ごとに学習 | 0.13554 | | E04 から低下 |
