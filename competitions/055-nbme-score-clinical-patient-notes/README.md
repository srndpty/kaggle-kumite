# #055 NBME - Score Clinical Patient Notes

- URL: https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes
- カテゴリ: D / 難度: 4 / 計算負荷: M
- 今回の比較実験テーマ: 臨床ノートから根拠span抽出
- 深さ: Standard

## コンペ概要

臨床ノートと採点項目を入力し、項目の根拠に当たるノート上の文字区間を抽出する。1ペアに対し、区間はゼロ件以上。

- 分野: テキスト・LLM
- 主催: National Board of Medical Examiners
- 開催期間: 2022-02-01〜2022-05-03

## 基本情報（P0）

- 評価指標: 文字単位の micro-F1（最大化）。区間は元の本文の文字インデックスによる半開区間。
- 提出形式: インターネット無効の Kaggle Notebook が `submission.csv` を生成する。列は `id,location`。
- データ: 公式 CSV 5点の取得を確認。公式仕様では10ケース、注釈付き1,000ノート。
- Late Submission: 非公開 Notebook `shion5/kumite-nbme-dummy` v1 を提出し、2026-11-26 に Public・Private とも 0.00000 で採点された。全件空欄の採点疎通用。

## 進捗

CV run の公開には共通 Quality Gate が必須。評価単位・分割・採点関数を決めたら、
[Quality Gate 仕様](../../docs/agent/experiment-quality-gates.md) に沿って学習・選定・最終 export の入口を実装する。

- [x] P0. 参加・評価環境の確認（Rules 承諾、データ取得、ダミー提出で採点されるか）
- [x] P1. 問題設定・EDA（予測仮説を3件以上。予想する結果も先に書く）
- [x] P2. 検証分割の設計（リーク経路の洗い出し、fold の確定）
- [x] P3. Baseline の固定（seed・fold・指標・前処理。可能なら初回の Late Submission）
- [x] P4. 自力の仮説検証（上位解法を読む前に 3〜6 実験）
- [x] P5. 上位解法の調査（アプローチの異なる上位解法を 2〜3 件。出典の種類を記録）
- [x] P6. 上位解法の着眼点の再現（1〜3 件をアブレーション）
- [x] P7. 最終評価・エラー分析（事前に決めたモデルの外部評価と CV との整合性）
- [x] P8. 知見の保存（`report.md`、`knowledge/` のカード）

## 仮説

| # | 仮説 | 根拠（EDA） | 出所（自力 / 上位解法） | 結果 |
|---|---|---|---|---|
| H1 | 項目説明で抽出精度が上がる | ノートごとに複数項目があり、項目 ID だけでは語義を伝えられない | 自力 | E01 の OOF 0.70386。B00 が上回った |
| H2 | 半分以上が正解と重なるトークンだけを陽性にすると境界の過検出が減る | 文字境界とトークン境界が異なる | 自力 | E02 の OOF 0.75607。B00 を下回った |
| H3 | 両側が陽性の空白を連結すると区間内の取りこぼしが減る | トークン間の空白も文字単位の評価対象 | 自力 | E03 が全 OOF で +0.03960 |

## 実験ログ

詳細は `experiments.jsonl`（`kumite.runs.RunStore` で追記。記録は書き換えず、やり直しは別の run_id）。主要な結果のみ抜粋する。

| run | 変更点（1つだけ） | OOF | Late | 判定 |
|---|---|---|---|---|
| B00 | 項目説明付き DistilBERT | 0.76145 | | 基準 |
| E01 | 項目説明を項目 ID に置換 | 0.70386 | | B00 を下回った |
| E02 | トークンの半分以上が正解と重なる場合だけ陽性 | 0.75607 | | B00 を下回った |
| E03 | 両側が陽性の空白を連結 | 0.80106 | | 全5 fold で改善 |

## EDA と検証分割

- 1,000 ノート、14,300 ペア。空ラベルは 4,399 ペア、複数区間は 2,664 ペア。
- ノート文字数は最小 200、中央値 856、95 パーセンタイルと最大は 950。同一本文の別ノート ID は 0 件。
- seed 42 でノートを単位にケース層化し、同一本文は同じグループに束ねる。各 fold は各ケース 20 ノート。分割 ID は `note-group-v1-seed42`、SHA-256 は `d1f55da27eb1a82dbf5d449ef34d3460d41806092ab93408cc677496234d22d5`。
- 同じノートに対する別項目の注釈が学習側へ漏れる経路を、ノート単位の分割で遮断する。検証側の正解は採点にだけ使い、項目説明と本文からトークン予測を作る。
- 比較前に `selection-plan.json` に候補順と同点許容差 0.001 を固定する。

## 再現手順

公式データの取得は `dev download 55`。集計と分割確認は `src/data.py`、保存 OOF のエラー分析は `src/analyze.py E03-r2-pkg`。学習する場合は、別 run ID と同点許容差を選定計画に保存・コミットしてから `src/train.py <B00|E01|E02|E03|E04> --run-id <ID>` を実行する。E03 と E04 には `--baseline <B00のID>` を渡す。各スクリプトはリポジトリのルートから `uv run python competitions/055-nbme-score-clinical-patient-notes/<スクリプト>` で実行する。

## 上位解法と P6

P4 の3件を公開した後、2026-11-27 に本人の writeup を確認した。

- 1位はラベルなしノートで MLM、擬似ラベル、AWP を組み合わせ、複数の DeBERTa をブレンドした。文字区間の端の空白を除去した。 [1位チームの writeup](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/writeups/ryuichi-currypurin-1st-solution)
- 3位は開始・内部・終了の複数ターゲット、Meta Pseudo Labels と蒸留を使った。期間を問う項目では明示的に単位が合わない予測も除去した。 [3位チームの writeup](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/writeups/lakecity-3rd-place-solution-meta-pseudo-labels-kno)
- 18位は改行・略語をトークンとして残し、モデルの文字単位予測を加重平均した。 [18位の writeup](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/writeups/michaln-18th-place-with-only-kaggle-and-free-colab)

E04 は B00 の出力に対し、2か月を問う項目309の予測区間が「2週」または「年」を表す英語表現に重なる場合だけ除外する。明示的な単位違いによる過検出が減ると予想する。全候補を新しい選定計画に揃えるため、B00・E01・E02・E03 は別 ID で再実行する。E03-r2 と E04 は B00-r2 の保存確率を使う。


E04 と B00-r2 の全 OOF はともに 0.76102。保存区間も全件一致し、この規則で消える予測はなかった。E03-r2 は 0.80082。

## 最終選定

保存 OOF から再採点した全候補を比較し、同点許容差 0.001 で E03-r2 を選定した。E02 は再実行で B00 を上回った。差の向きが揃わなかったため、陽性ラベルを絞る効果は未確定とした。 [選定記録](selection.json)

| 候補 | 再採点 OOF micro-F1 |
|---|---:|
| B00-r2 | 0.76102 |
| E02-r2 | 0.76269 |
| E03-r2 | **0.80082** |
| E01-r2 | 0.70842 |
| E04 | 0.76102 |

F01 の履歴には、Kaggle Dataset の ZIP 自動展開への対応時に、保存 OOF を再登録した `-pkg` run が含まれる。選定記録とスコアは当時の証跡として保存する。 [初回選定](selection-f00.json) · [F01 の選定](selection.json)

今後は、検証済み export の Notebook にファイル同一性検査だけを追加する場合、`src/deployment.py <export ID> <改訂 ID>` を使う。元の Notebook の重み参照行に SHA-256 検査を挿入し、元の CV・export 証跡、Notebook とモデルのハッシュを `outputs/deployments/<改訂 ID>/deployment-evidence.json` に記録する。既存 CV の再登録が不要な提出用改訂の手順である。推論や学習の処理を変更する場合は、更新後のソースで CV を実行し、新たな選定記録から `src/export.py` を実行する。

新規 export と提出用改訂の Notebook は、Kaggle が展開したモデルの全ファイルを、ZIP から採取した SHA-256 と読み込み前に照合する。F01 v1 の採点は、ファイル単位の実行時照合を導入する前に行った。

## 外部評価

2026-11-27 に F01 の Kaggle Late Submission が完了した。Public micro-F1 は **0.80488**、Private は **0.80957**。Notebook v1 のサンプル5行の列・ID 順・区間をローカルで検査した後に提出した。 [採点証跡](submission-audit.json) · [振り返り](report.md)

2026-11-27 に dropout を有効にして再学習した F02 は、OOF **0.79770**、Late Public **0.81418**、Late Private **0.81945**。 [再計測の選定](selection-dropout.json) · [採点証跡](submission-audit-dropout.json)
