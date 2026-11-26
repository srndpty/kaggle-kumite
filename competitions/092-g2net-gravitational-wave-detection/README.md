# #092 G2Net Gravitational Wave Detection

- 深さは Standard。カテゴリ F、難度 5、計算負荷 L。
- 3 検出器の各 2 秒間の波形から、ブラックホール連星の合体に伴う重力波信号の有無を予測する。公式指標は ROC AUC。提出はテスト ID ごとの確率を入れた `id,target` の CSV。 [Kaggle 公式概要・評価仕様](https://www.kaggle.com/c/23249)
- 公式サンプルの全 226,000 行に 0.5 を入れた疎通提出は 2026-11-26 に採点完了し、Late Public・Private AUC ともに 0.50000 だった。この値は定数予測の疎通結果。選定候補には含めない。
- Kaggle の一括アーカイブは 74,086,993,965 バイト。目録には学習波形 560,000 件、テスト波形 226,000 件と、ラベル・提出テンプレートの CSV 各 1 件がある。Windows の旧 Kaggle CLI が応答をメモリに蓄積したため、同じ公式 API からディスクへストリーミングした。公式 ZIP の波形は CRC・ID・形状・有限値を照合し、CSV の順序で float64 の NumPy 配列へまとめる。

## 進捗

- [ ] P0. 公式波形一式を取得・照合する。Late 採点の疎通は完了。
- [ ] P1. 学習とテストの波形・周波数・重複を確認する。
- [ ] P2. 正本データでグループ分割 `v1` とハッシュを確定する。
- [ ] P3. B00 の CV を確定し、初回 Late 提出を行う。
- [ ] P4. E01〜E03 の比較を実施する。
- [ ] P5. アプローチの異なる上位解法を 2〜3 件調査する。
- [ ] P6. 1 件の工夫をアブレーションする。
- [ ] P7. 最終 export の外部評価と OOF のエラー分析を行う。
- [ ] P8. `report.md` と知見カードに実測結果を保存する。

## 実波形の小規模疎通

学習 CSV の先頭 1,280 行とテスト CSV の先頭 256 行を公式アーカイブから取り出した。この範囲では定数チャネルと完全重複は 0 件だった。25〜500 Hz に含まれるパワーの割合の中央値は、学習の Hanford・Livingston・Virgo の順に 0.093%、0.123%、0.845% だった（波形ハッシュに基づく標本）。これは全件 EDA の前に得た観測。

RTX 4090 で学習 1,024 行・検証 256 行・テスト 256 行を使い、B00 の CUDA 疎通を 9.5 秒、ピーク割り当て 144 MiB で確認した。E03 も 3 エポックの学習と有限な確率の出力を確認した。本番では外側 fold の学習行を全件使い、バッチサイズ 256、3 エポックで比較する。

## 事前仮説

| run | 比較対象 | 変更 | 期待結果 |
|---|---|---|---|
| B00 | — | Hanford の波形に固定倍率を適用し、小型 1D CNN で学習 | 基準 |
| E01 | B00 | 25〜500 Hz の帯域制限 | 帯域外ノイズを抑え、AUC が向上する |
| E02 | E01 | 検出器別の白色化を追加 | 周波数ごとのノイズ強度差を補正し、AUC が向上する |
| E03 | E02 | Livingston と Virgo を追加 | 複数検出器の信号を使うことで、AUC が向上する |

白色化のパワースペクトルは各外側 fold の学習波形だけで推定する。公式アーカイブの目録と CSV の列を確認した。元ノイズ区間や注入元を示すキーは配布ファイルに含まれない。完全重複でグループ化し、seed 42 の層化グループ 3-fold を使う。別ファイルに共有されるノイズ区間はこの分割で検出できない。

## 実行手順

リポジトリ直下から実行する。以下では `$g2net` を `competitions/092-g2net-gravitational-wave-detection/src/run.py` に設定する。公式一括 ZIP をコンペ直下の `data/official.zip` に保存し、`uv run python competitions/092-g2net-gravitational-wave-detection/src/prepare.py` で `data/packed/` を生成する。ZIP と配列を合わせて約 152 GB の空き容量が必要。

1. `uv run python $g2net inspect` で ID・波形・分割を照合し、`outputs/data-manifest.npz` を生成する。
2. `uv run python $g2net smoke` で実波形 1,024 行の学習と検証・テストの推論を疎通してから、バッチサイズとエポック数を固定する。`selection-plans.json` をコミットしてから、`uv run python $g2net cv baseline B00-initial`、`select baseline`、`export baseline F00-baseline` を順に実行し、確定した export 内の `submission.csv` を `dev submit 92` で提出する。B00 は比較用計画でも同一構成で個別に実行する。
3. `cv p4 B00`、`cv p4 E01`、`cv p4 E02`、`cv p4 E03`、`select p4` の順に進める。再採点した OOF AUC の最高値から 0.001 以内の候補を、B00、E01、E02、E03 の順で優先する。
4. P5 後に P6 の対照と変更後を別の run ID にする。`selection-p6.json` に `p6` キーを置き、`candidates`、`tolerance`、run ID ごとの `configs` を固定する。P6 の実験前に計画と必要なモデル変更をコミットし、`cv p6 <run_id>` で両方を実行する。
5. `select p6`、`export p6 F00`、`analyze p6` で選定・提出生成・検出器別 RMS 区間の AUC 集計を行う。確定した提出 CSV の Late Private スコアを `dev submissions 92` で確認して報告する。

CV の公開、保存済み OOF の独立再採点、選定、export は [共通 Quality Gate](../../docs/agent/experiment-quality-gates.md) を通す。提出予測は選定済み run の 3-fold 平均確率を使用する。
