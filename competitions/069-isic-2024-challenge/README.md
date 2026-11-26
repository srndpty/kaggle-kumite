# #069 ISIC 2024 - Skin Cancer Detection with 3D-TBP

- 深さは Standard。今回は診察時メタデータを入力に、病変ごとの悪性確率を推定する。
- 主評価は TPR 0.80 以上の非標準化 pAUC（最大 0.20）。提出はオフライン Kaggle Notebook が生成する `isic_id,target` の CSV。
- 公式の train/test メタデータと提出サンプルを取得済み。定数 0.5 のダミー Notebook v2 を 2026-11-27 に提出し、Late Public・Private とも 0.02000 で採点された。v1 は出力待ちが続いたため、入力ディレクトリを直接参照する v2 で再実行した。 [公式評価](https://www.kaggle.com/competitions/isic-2024-challenge/overview/conference) · [P0 Notebook](https://www.kaggle.com/code/shion5/kumite-isic-2024-dummy)

## 進捗

- [x] P0. 参加・評価環境の確認
- [x] P1. 問題設定・EDA
- [x] P2. 検証分割の設計
- [x] P3. Baseline の固定
- [x] P4. 自力の仮説検証
- [x] P5. 上位解法の調査
- [x] P6. 上位解法の着眼点の再現
- [x] P7. 最終評価・エラー分析
- [x] P8. 知見の保存

## EDA と検証分割

学習データは 401,059 病変、陽性 393 件（0.098%）、1,042 患者のうち陽性患者は 259 人。患者当たり病変数の中央値は 241.5、範囲は 1〜9,184。提供元7群の陽性率は 0.020〜0.156%、年齢欠損率は 0〜2.33%。公開サンプルの3患者には1病変ずつある。

未知患者への適用を想定し、患者で分ける `StratifiedGroupKFold` 5分割、seed 42 を `patient-v1` とした。陽性数は 78 / 78 / 83 / 77 / 77。患者の交差ゼロ、全病変の検証1回、各 fold の両クラスを検査する。診断・生検由来の列、病変・患者 ID、施設やライセンスをモデル入力から除外する。患者 ID は分割とラベルを使わない患者内集約に使う。カテゴリの語彙は fold 学習側のみで確定する。

## 事前仮説と選定

E02 の患者内相対特徴は、推論対象のバッチに含まれる同一患者の病変だけから、正解ラベルを使わず計算する。CV でも患者を分割間で共有しない。未観測の病変は集約に含まれず、1病変ずつ推論する場合は病変数が1になり、観測値の中央値からの差は0になる。

| Run | 単独の変更 | 期待する結果 |
|---|---|---|
| B00 | 病変の診察時メタデータを使う CPU LightGBM、500ラウンド | 比較の基準 |
| E01 | 短径/長径比、色差の合成値を追加 | 単独測定値では捉えにくい形状と色を表せるため pAUC が上がる |
| E02 | 患者内病変数と、サイズ・色差・境界の中央値からの差を追加 | 同一患者内の目立つ病変を検出しやすく、病変が多い患者で改善する |
| E03 | 学習時の陽性重みを5倍にする | 高感度域の pAUC が上がりうる。log loss は悪化しうる |

全 OOF の pAUC で比較し、最良値との差が 0.0005 以内なら表の上から優先する。Kaggle スコアは外部評価に使う。P6 候補を加える場合は新しい計画を確定し、既存構成も別 ID で再計測する。 [選定計画](selection-plan.json) · [Quality Gate 仕様](../../docs/agent/experiment-quality-gates.md)

## 実行方法

リポジトリルートから実行する。CSV 3件をコンペの `data/` に配置する。Kaggle CLI が学習 CSV を ZIP コンテナで返した場合は展開してから配置する。

```powershell
$entry = "competitions/069-isic-2024-challenge/src/workflow.py"
uv run python $entry train B00
uv run python $entry train E01
uv run python $entry train E02
uv run python $entry train E03
```

## P4 の実測と P5 の調査

| Run | OOF pAUC | log loss | 予測 0 の件数 |
|---|---:|---:|---:|
| B00 | 0.020184 | 0.062310 | 400,631 |
| E01 | 0.019982 | 0.059973 | 400,693 |
| E02 | 0.020239 | 0.059283 | 400,675 |
| E03 | 0.022024 | 0.180435 | 398,908 |

P4 を終えてから以下を調べた。

- 1位チームの公開コードは、外部データと合成画像を活用した EVA・EdgeNeXt の予測を、上段の GBDT に入れる。 [本人の公開コード](https://github.com/ilyanovo/isic-2024)
- 5位チームは、GBDT 側の陽性重みを調整し、不確定な診断にはソフトラベルの扱いを導入した。 [本人の writeup](https://www.kaggle.com/competitions/isic-2024-challenge/writeups/kanna-hashimoto-friends-2-5th-place-solution)
- 21位は、画像分類器の出力を GBDT に入力し、各 fold の学習時に陽性対陰性を 1:5 に揃えた。 [本人の writeup](https://www.kaggle.com/competitions/isic-2024-challenge/writeups/rush98-21st-place-solution)

P6 の `E04` は、21位の陰性抽出だけを `B00` に適用する。学習 fold ごとに全陽性を残し、陰性を陽性の5倍まで seed 42 で非復元抽出する。外側検証の分布は保持する。陽性が希少な状況で予測の飽和が減り、pAUC が上がると予想する。 [P6 選定計画](selection-plan-p6.json)

今後の E04 系 run は、学習に採用した陽性数、抽出陰性数、陽性率、seed と、採用した全 `isic_id` をソートした JSON 配列の SHA-256 を fold 別に `fold_info.json` に保存する。保存済み run の分析は元の CV 証跡と正本データ・分割からの再採点で検証する。過去の選定から export を作り直す場合は、Quality Gate が学習時と同じソースを要求する。

P6 の再計測・最終選定・export は次の順で行う。

```powershell
$root = "competitions/069-isic-2024-challenge"
$entry = "$root/src/workflow.py"
$plan = "$root/selection-plan-p6.json"
$receipt = "$root/selection-p6.json"
foreach ($run in "B00-p6", "E01-p6", "E02-p6", "E03-p6", "E04") {
    uv run python $entry train $run --plan $plan
}
uv run python $entry select --plan $plan --receipt $receipt
uv run python $entry export F00 --plan $plan --receipt $receipt
```

最終選定は E04。F00 では検証済み構成を全学習データで学習し直し、公開サンプル3件で提出 Notebook の出力と ID 順を照合した。 [最終選定記録](selection-p6.json) · [最終 Notebook](https://www.kaggle.com/code/shion5/kumite-isic-2024-final) · [振り返り](report.md)

F00 に対応する Kaggle Notebook v1 はプレビュー実行がキャンセルされた。入力ファイルの探索を直接参照に変更した F01 を `uv run python competitions/069-isic-2024-challenge/src/relocate.py` で公開した。F00 とモデルの SHA-256 は同じ。F01 の Notebook は2種類の Kaggle 入力ディレクトリ配置を再現し、公開サンプル3件の出力が F00 と一致することを確認した。Kaggle Notebook v2 が F01 に対応する。

2026-11-27 に F01 の Late Submission が完了した。OOF pAUC は **0.15543**、Late Public は **0.15980**、Late Private は **0.14107**。Kaggle 上の公開サンプル出力を照合してから提出した。 [採点証跡](submission-audit.json)
