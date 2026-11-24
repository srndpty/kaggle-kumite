# #084 Freesound Audio Tagging 2019

- 深さは **Standard**。カタログ難度4・計算負荷M。
- 可変長の環境音クリップに複数タグを付ける。主指標はクリップ内のタグ順位に基づく lwlrap。提出は fname と80タグを含む CSV を出力する非公開 Notebook。 [Kaggle 公式概要・評価・提出要件](https://www.kaggle.com/competitions/freesound-audio-tagging-2019)

## 進行状況

- [x] P0 公式データ取得と固定順位 Notebook の採点確認
- [x] P1–P2 EDA、録音の重複調査、curated 3 fold の確定
- [x] P3 定数基準と CNN Baseline の採点、Baseline の Late Submission
- [x] P4 自力仮説3件のペア比較
- [x] P5 異なる上位解法3件の調査
- [x] P6 着眼点1件の再現比較
- [x] P7 最終構成の固定・再学習・外部評価・誤差分析
- [x] P8 振り返りと知見の保存

固定タグ順位を出す [P0 Notebook](https://www.kaggle.com/code/shion5/kumite-freesound-p0) を登録した。版2の実行ログで展開済みテスト音源がマウントされないことを確認し、版3では公式 `test.zip` のメンバーを列挙する方式に切り替えた。

2026-11-24 JST、版3の Late Submission は COMPLETE。Kaggle の表示は Public **0.00000**、Private **0.06791**。提出が重複し、同じ版が2件同じスコアで採点された。P0 の実験記録は1件とした。

公式 CSV では、curated は4,970クリップ（複数タグ701件）、noisy は19,815クリップ（複数タグ3,249件）。学習 CSV の公開列は `fname` と `labels`。両集合で同じ `fname` は0件。提出テンプレートは3,361クリップ・80タグ。公式アーカイブは28,146 WAV に展開した。

## 実験前の登録

- Decoded PCM の完全一致をグループにし、curated のタグ数を概ね均衡させた seed 42 の3 foldを使う。公式データで同一録音の識別子の有無と近似重複を調査する。各検証 fold と同じグループに含まれる noisy クリップを除外する。
- 16 kHz モノラルの64-bin log-mel に小型 CNN を適用する。学習はランダムな2秒窓、推論は1秒間隔と末尾を覆う2秒窓。短尺はゼロ埋めする。AdamW、学習率0.001、weight decay 0.0001、batch 32。2秒窓の時間マスクは連続20フレームをその窓の周波数帯別平均で埋める。
- GPU 学習は試走・再学習込み4時間。RTX 4090 で200更新と1 fold の検証を含む試走は85.05秒、lwlrap は0.19429。主比較は同じ分割・seed・1,600更新で行う。試走は選定対象外。

## EDA と検証分割

- 音声長の中央値は curated 4.68秒、noisy 15秒。固定長の切り出しによる影響と、窓集約の変更を比較する。
- 完全一致で curated の4,970件は4,969グループ、noisy の19,815件は19,744グループになった。curated 検証音源に一致する noisy は3 fold とも0件。ゲイン・直流オフセット・極性を正規化した同一長の波形照合による追加一致は0組。時間ずれ・編集・録音者の共有はこの照合の対象外。
- 検証 fold の件数は1,655、1,657、1,658。全 fold に80タグがあり、最小正例数は15。再生成は固定 seed 42。詳細は `outputs/eda.json` に保存した。

| Run | 比較元 | 変更 | 予想 |
|---|---|---|---|
| E00 | — | 学習 fold のタグ頻度 | 定数予測の基準 |
| E01 | E00 | 2秒窓の log-mel CNN と平均集約 | 音響特徴でタグ順位が改善する |
| E02 | E01 | バッチの8/32を noisy に置換 | 希少タグの順位が改善する |
| E03 | E01 | 推論窓をタグ別最大値で集約 | 長尺内の短いイベントの順位が改善する |
| E04 | E01 | 学習時に時間マスク | 一部の時間帯への依存が減る |

主指標は全 curated OOF の lwlrap。最上位構成と E01 の差が0.005未満で、クリップ単位のペア bootstrap（2,000回）の95%区間がゼロを含む場合は E01 を採用する。上位解法の調査は E02〜E04 の完了後に始める。

## 実測

| Run | OOF lwlrap |
|---|---:|
| E00 | 0.06833 |
| E01 | 0.39561 |
| E02 | 0.39898 |
| E03 | 0.38802 |
| E04 | 0.36544 |
| E05 | **0.40982** |

[OOF の選定記録](selection.json) に従い E05 を最終構成に固定し、全データで再学習したモデルを外部評価に使った。 [振り返り](report.md) [知見カード](../../knowledge/K035-staged-noisy-supervision.md)

E01 の3 fold 平均による Late Submission は COMPLETE。Kaggle の表示は Public 0.00000、Private **0.37736**。 [Baseline Notebook の版2](https://www.kaggle.com/code/shion5/kumite-freesound-baseline)

`final-E05` の Late Submission も COMPLETE。表示は Public 0.00000、Private **0.26984** で、E01 比 -0.10752。CV と外部評価の比較では方向が逆転した。選定済みの構成はそのまま記録した。 [最終 Notebook の版1](https://www.kaggle.com/code/shion5/kumite-freesound-final) [採点証跡](submission-evidence.json)

## 上位解法と E05 の事前登録

- 1位の Baikulov は curated と noisy で損失を分け、低スコアの curated を手作業で再ラベル付けした。推論は attention CNN と二段目 MLP のアンサンブル。 [本人の公開コードと解説](https://github.com/lRomul/argus-freesound)
- Private 4位の Akiyama・Sato は、curated と noisy の違いをマルチタスクで扱い、予測で noisy を再ラベル付けし、異なる窓長のモデルを平均した。 [本人の論文](https://archive.nyu.edu/handle/2451/60774) [本人の公開コード](https://github.com/OsciiArt/Freesound-Audio-Tagging-2019)
- 9位チームは、軽量なモデルと64-bin log-mel を使い、noisy での事前学習から curated 学習に移った。 [本人の Kaggle writeup](https://www.kaggle.com/competitions/freesound-audio-tagging-2019/writeups/4-people-9th-place-solution-smaller-and-faster) [本人の公開コード](https://github.com/TheoViel/kaggle_freesound)

E05 では E01 の CNN と学習更新数を使い、最初の400更新を noisy のみ、続く1,200更新を curated のみにする。E02 と両データ源の延べサンプル数が同じなので、混ぜる時期の効果も比較できる。curated で仕上げることで E01 と E02 の両方を上回ると予想する。

## 実行

- 公式データを取得するには dev download 84 を実行する。
- P0 Notebook は uv run python competitions/084-freesound-audio-tagging-2019/src/dummy.py に Kaggle ユーザー名と出力先を渡して生成する。
- src/eda.py に出力 JSON を指定すると、音声長・タグ頻度・共起・完全一致・fold ごとの noisy 除外数を集計する。ゲイン、直流オフセット、極性を正規化した波形の一致も調べる。
- CV は uv run python competitions/084-freesound-audio-tagging-2019/src/train.py に run ID、--experiment、--updates、--seconds を渡す。--benchmark は最初の1 foldだけを実行する診断用。RunStore に OOF・分割・正解・重みを保存する。
- E03 には --baseline で E01 の run ID を指定する。E01 の保存済み重みを使って推論窓の集約だけを変える。
- 初回のモデル提出は src/export.py に Kaggle ユーザー名、E01 の run ID、新しい出力先、--dataset に非公開データセットのスラッグを指定して生成する。weights/ をそのデータセットとして登録し、kernel/ を実行する。
- src/analyze.py に新しい選定 JSON、E01 の run ID、比較 run ID を指定して、OOF の再採点と長さ・タグ数別の差を記録する。最終提出用の再学習には src/export.py に選定 run ID と --selection、--refit-run、--seconds を指定する。

## レビュー対応

- 再学習前に E01〜E05 の成果物と採点値を照合し、候補一覧、差の区間、スライス、採否を再計算する。レポートに使用した既存の `selection.json` は、この手順で再構成できることを確認した。
- 新しいモデル提出 Notebook は配布テンプレートの3,361件の ID と順序を固定する。音源 ZIP は、配布分と完全一致する構成、配布分と互いに素な追加評価分1,120件、両者を合わせた4,481件のいずれかを許容する。差異の件数はログに出力する。1,120件は主催者が公表した総テスト4,481件から、実際に配布された3,361件を引いた数。 [主催者のデータセット論文](https://dcase.community/documents/workshop2019/proceedings/DCASE2019Workshop_Fonseca_33.pdf)
- `src/diagnose.py` に過去の E01、final-E05 の公開出力 CSV と新しい出力先を順に渡すと、保存済み E05 の3 fold 平均を公開音源で推論し、タグ順位・予測分布を比較する。E05 を `src/export.py` に指定すると、同じ3 fold 平均の診断 Notebook を生成できる。
- 事後診断 `E05-fold-review` の Private は **0.38638**。同じ3 fold 平均の E01 比 **+0.00902**。当初の選定と final-E05 の Private 0.26984 は歴史的な結果として保持した。 [診断結果](fold-diagnostic.json)
