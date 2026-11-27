# #040 OTTO – Multi-Objective Recommender System

- URL: https://www.kaggle.com/competitions/otto-recommender-system
- カテゴリ: C / 難度: 5 / 計算負荷: XL
- 今回の比較実験テーマ: 候補幅と再ランキングの教師集合が行動別 Recall@20 に与える影響
- 深さ: Standard

## コンペ概要

匿名の購買セッションの先頭部分から、次のクリックと残りのカート追加・注文の商品を予測する。

候補幅を変える比較では、教師に入るセッションを固定した対照を置く。
[#033 H&M の観測](../../knowledge/K050-candidate-count-vs-ranker.md) をセッション推薦で確かめる。

## 基本情報（P0 で記入）

- 評価指標は行動別 Recall@20 の加重平均。クリック 0.10、カート追加 0.30、注文 0.60。
- ファイル提出。列は `session_type,labels` で、行動ごとに予測商品 ID を空白区切りで最大20件並べる。
- Kaggle 公式ページに Late Submission の表示がある。
- 公式配布は完全な学習セッションの `train.jsonl`、途中で切断されたセッションの `test.jsonl`、`sample_submission.csv`。

[公式評価仕様](https://www.kaggle.com/competitions/otto-recommender-system) · [公式データ](https://www.kaggle.com/competitions/otto-recommender-system/data)

### 採点疎通

```powershell
dev download 40
$d = 'competitions/040-otto-recommender-system'
uv run python "$d/src/dummy.py"
dev submit 40 "$d/outputs/p0-dummy.csv" -m 'Kumite P0 action popularity single aid'
dev submissions 40
```

2026-11-27 に公式3ファイルを取得し、ダミー提出の採点完了を確認した。
学習ファイルの先頭10万セッションで行動別の最頻商品を1件ずつ求め、公式テンプレートの全行に割り当てた。
Late Public は 0.00017、Late Private は 0.00018。ダミー提出はモデル選定の対象外。

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
| H1 | 40候補の再ランキングで、特に注文 Recall@20 が上がる | 注文の正解の約60%が入力内に再出現 | 自力 | 注文 Recall@20 +0.01526 |
| H2 | 教師を固定して候補を40件から100件に増やすと、候補から到達可能な Recall@20 が上がる。実際の指標の伸びは上限より小さい | 人気候補だけの正解カバーは小さい | 自力 | 加重スコアの上限 +0.0071、実測 +0.00122 |
| H3 | 41〜100位に初めて正解が入る教師を加えると、教師の構成と正例率が変わり、注文 Recall@20 が下がる | #033 で教師を揃えた対照との差を観測 | 自力 | 注文 Recall@20 −0.00004、加重 OOF −0.00020 |

## 実験ログ

詳細は `experiments.jsonl`（`kumite.runs.RunStore` で追記。記録は書き換えず、やり直しは別の run_id）。主要な結果のみ抜粋する。

| run | 変更点（1つだけ） | OOF | Late | 判定 |
|---|---|---|---|---|
| B00 | 直近商品と行動別人気順の上位20件 | 0.47611 | 未提出 | 基準 |
| E01 | 40候補を行動別 LambdaRank で再ランキング | 0.48521 | — | B00 比 +0.00911 |
| E02 | 教師を固定して100候補に拡張 | 0.48643 | — | E01 比 +0.00122 |
| E03 | 41〜100位で初めて正解が入る教師を追加 | 0.48623 | — | E02 比 −0.00020 |
| E04-final | E02 と同じ候補・教師に経過秒を1列追加 | 0.48647 | — | E02-final 比 +0.00004 |
| F00 | E02-final の設定で再学習・提出 | 0.48643（選定時） | Public 0.48852 / Private 0.49002 | 最終評価 |

## 検証設計

`weekly-prefix-v1` は公式学習データの最終イベントを終点とする連続2週を使う。
各週に完全に収まる2イベント以上のセッションが対象。seed 42 とセッション ID で疑似切断し、
先頭部分を入力、最初の後続クリックと後続のカート追加・注文の商品集合を正解にする。
窓の開始時点を境に、完結済みセッションだけを人気集計と教師生成に使う。
教師となる疑似切断セッションの人気特徴量も、教師窓の開始時点で固定する。

| run | 変更点 | 教師グループ |
|---|---|---|
| B00 | 直近順と行動別人気順で上位20件を返す | 学習なし |
| E01 | 40候補を行動別 LambdaRank で並べ替える | 40候補内に正解があるグループ |
| E02 | 候補を100件に増やす | E01 と同じグループ |
| E03 | 100候補で教師グループを広げる | 100候補内に正解があるグループ |

候補の先頭40件を固定し、候補幅の差を E01 と E02、教師集合の差を E02 と E03 で測る。
P4 の選定順と同点許容差は [選定計画](selection-p4.toml) に記録する。
P5 で上位解法を調査し、P6 で着眼点を1件追加する。最終計画で既存候補を比較する際は新しい run ID で実行する。

公式データの分割 Parquet 変換は `uv run python "$d/src/prepare.py"` で実行する。
再ランキングの教師は各教師窓でセッション ID が100の倍数のものに固定し、候補幅を広げても学習メモリを抑える。
検証と提出推論は各対象窓の全件を使う。教師の候補順位と行動別人気は教師窓の直前7日間から作る。
セッション内に出ない人気候補は、同じ特徴量を持つ行のスコアを再利用する。
注文・100候補の固定2万セッションでは、スコアを毎回計算したときと上位20件が全件一致し、推論時間は 6.93 秒から 2.52 秒になった。

```powershell
$d = 'competitions/040-otto-recommender-system'
uv run python "$d/src/workflow.py" cv B00 B00
uv run python "$d/src/workflow.py" cv E01 E01
uv run python "$d/src/workflow.py" cv E02 E02
uv run python "$d/src/workflow.py" cv E03 E03
uv run python "$d/src/workflow.py" select "$d/outputs/selection-p4.json"
uv run python "$d/src/analyze.py" E02 E03
```

P6 では E02 に候補商品の最終イベントから予測時点までの経過秒を1列追加する。
入力に出現しない候補の商品は −1。候補幅、教師グループ、モデル設定は E02 と揃える。
事前に確定した [最終選定計画](selection-final.toml) で、P4 の4候補も新しい ID で再実行する。

```powershell
foreach ($method in 'B00', 'E01', 'E02', 'E03', 'E04') {
    uv run python "$d/src/workflow.py" --plan "$d/selection-final.toml" cv "$method-final" $method
    if ($LASTEXITCODE -ne 0) { throw "CV failed for $method" }
}
uv run python "$d/src/workflow.py" --plan "$d/selection-final.toml" select "$d/outputs/selection-final.json"
uv run python "$d/src/workflow.py" --plan "$d/selection-final.toml" export F00 "$d/outputs/selection-final.json"
dev submit 40 "$d/outputs/exports/outputs/runs/F00/submission.csv" -m 'Kumite F00 E02-final'
dev submissions 40
```

## EDA

公式学習データを 12,899,779 セッション・216,716,096 イベント、公式テストの観測部分を
1,671,803 セッション・6,928,123 イベントとして変換した。以下は `uv run python "$d/src/eda.py"`
で各検証窓のセッション ID が100の倍数の標本を集計した結果。

| 検証窓 | 標本数 | イベント数の中央値 / 90%点 | 正解が入力セッションにもある割合（クリック / カート / 注文） |
|---|---:|---:|---:|
| 1週目 | 16,828 | 4 / 16 | 32.4% / 32.6% / 60.0% |
| 2週目 | 18,012 | 4 / 18 | 32.6% / 31.3% / 60.5% |

直前週の完結セッションに登場しない商品は正解の 6.4–8.7% だった。
直前週の行動別人気上位20件に入る正解は各窓・各行動で 1.1% 未満だった。
注文は入力セッション内に正解が多いので、直近順を基準とし、行動別の再ランキングで上積みを検証する。
