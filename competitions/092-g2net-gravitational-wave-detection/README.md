# #092 G2Net Gravitational Wave Detection

- 深さは Standard。カテゴリ F、難度 5、計算負荷 L。
- 3 検出器の各 2 秒間の波形から、ブラックホール連星の合体に伴う重力波信号の有無を予測する。公式指標は ROC AUC。提出はテスト ID ごとの確率を入れた `id,target` の CSV。 [Kaggle 公式概要・評価仕様](https://www.kaggle.com/c/23249)
- 公式サンプルの全 226,000 行に 0.5 を入れた疎通提出は 2026-11-26 に採点完了し、Late Public・Private AUC ともに 0.50000 だった。この値は定数予測の疎通結果。選定候補には含めない。
- Kaggle の一括アーカイブは 74,086,993,965 バイト。目録には学習波形 560,000 件、テスト波形 226,000 件と、ラベル・提出テンプレートの CSV 各 1 件がある。Windows の旧 Kaggle CLI が応答をメモリに蓄積したため、同じ公式 API からディスクへストリーミングした。公式 ZIP の波形は CRC・ID・形状・有限値を照合し、CSV の順序で float64 の NumPy 配列へまとめる。

## 進捗

- [x] P0. 公式波形一式の取得・照合と Late 採点の疎通。
- [x] P1. 学習とテストの波形・周波数・重複を確認し、事前仮説を記録。
- [x] P2. 正本データでグループ分割 `v1` とハッシュを確定。
- [x] P3. B00 と同じ構成の B00-initial を CV で確定し、初回 Late 提出を採点。
- [x] P4. E01〜E03 を比較し、保存 OOF を再採点して E03 を選定。
- [x] P5. 1 位と 2 位の本人の解法・公開コードを調査。
- [x] P6. Hanford と Livingston の入れ替えを E04・E05 で比較。
- [x] P7. E04 を選定し、F00 の Late 採点と OOF のエラー分析を完了。
- [x] P8. `report.md` と知見カード K051・K052 に実測結果を保存。

## 実波形の小規模疎通

学習 CSV の先頭 1,280 行とテスト CSV の先頭 256 行を公式アーカイブから取り出した。この範囲では定数チャネルと完全重複は 0 件だった。25〜500 Hz に含まれるパワーの割合の中央値は、学習の Hanford・Livingston・Virgo の順に 0.093%、0.123%、0.845% だった（波形ハッシュに基づく標本）。これは全件 EDA の前に得た観測。

RTX 4090 で学習 1,024 行・検証 256 行・テスト 256 行を使い、B00 の CUDA 疎通を 9.5 秒、ピーク割り当て 144 MiB で確認した。E03 も 3 エポックの学習と有限な確率の出力を確認した。本番では外側 fold の学習行を全件使い、バッチサイズ 256、3 エポックで比較する。

## 全件 EDA と検証分割

- 学習 560,000 行とテスト 226,000 行を照合した。検出器ごとの定数チャネル、各データ内の完全重複、学習とテスト間の完全重複はいずれも 0 件。
- 波形を 10²⁰ 倍した RMS の中央値は、学習の Hanford・Livingston・Virgo の順に 0.70849、0.70834、0.17829。テストは 0.70754、0.70783、0.17829。25〜500 Hz の帯域内パワー比の中央値は、波形ハッシュによる標本で学習 0.141%、0.143%、0.850%、テスト 0.140%、0.146%、0.853% だった。
- 検証行数は fold 順に 186,667、186,667、186,666、陽性率は各 fold 約 49.99%。全学習 ID に検証 fold が 1 つずつ対応し、同じ波形指紋のグループは学習側と検証側をまたがない。
- データハッシュ `f6a151a7eaac8778a606689e7ea6601801bc86ac393eb41b3635f113b8b7435e`、分割ハッシュ `49f00d2caf47ed8d92be192a9f7eaacce459a783da8ff3773b80505ce5e6c77c`。ID と分割の全対応は `outputs/data-manifest.npz` で再生成できる。

## 事前仮説

| run | 比較対象 | 変更 | 期待結果 |
|---|---|---|---|
| B00 | — | Hanford の波形に固定倍率を適用し、小型 1D CNN で学習 | 基準 |
| E01 | B00 | 25〜500 Hz の帯域制限 | 帯域外ノイズを抑え、AUC が向上する |
| E02 | E01 | 検出器別の白色化を追加 | 周波数ごとのノイズ強度差を補正し、AUC が向上する |
| E03 | E02 | Livingston と Virgo を追加 | 複数検出器の信号を使うことで、AUC が向上する |

白色化のパワースペクトルは各外側 fold の学習波形だけで推定する。公式アーカイブの目録と CSV の列を確認した。元ノイズ区間や注入元を示すキーは配布ファイルに含まれない。完全重複でグループ化し、seed 42 の層化グループ 3-fold を使う。別ファイルに共有されるノイズ区間はこの分割で検出できない。

## 実測結果

| run | OOF AUC | Late Public AUC | 備考 |
|---|---:|---:|---|
| B00-initial | 0.50067 | 0.50281 | fold 別 0.49887 / 0.50254 / 0.50157。export は F00-baseline。 |
| B00 | 0.50067 | — | B00-initial と同じ OOF AUC。 |
| E01 | 0.72396 | — | fold 別 0.73170 / 0.73428 / 0.73820。 |
| E02 | 0.73009 | — | fold 別 0.72835 / 0.72956 / 0.73244。 |
| E03 | 0.78815 | — | fold 別 0.78765 / 0.78794 / 0.78972。P4 の選定構成。 |
| E04 | 0.78815 | 0.80164 | P6 対照。E03 と保存 OOF・テスト予測のハッシュが一致。F00 の Late Private AUC は 0.79573。 |
| E05 | 0.78642 | — | Hanford と Livingston を学習時に入れ替え。fold 別 0.78622 / 0.78562 / 0.78934。 |

B00-initial は未処理の Hanford 波形を固定倍率だけで入力した。CV では定数予測に近く、Late Public は提出経路の確認にのみ使用した。E01 は B00 から全体 AUC が 0.22328 改善した。E02 は E01 から全体 AUC が 0.00614 上がり、fold 内の AUC は全 fold で下がった。fold 別の予測中央値は E01 で 0.287〜0.456、E02 で 0.409〜0.410。全体指標は fold をまたぐ予測順位も含むため、改善と解釈できる範囲を区別する。

## 上位解法と再現実験

- 1 位 KDL は、帯域を絞った 3 チャネル波形を入力する 1D CNN で Hanford と Livingston の入れ替えなどを試した。最終構成では人工波形で事前学習し、検出器別のエンコーダーで得た特徴を ResNet34 で統合した。 [本人の解法](https://www.kaggle.com/competitions/g2net-gravitational-wave-detection/writeups/kdl-top-1-solution-deep-learning-part) · [本人の発表資料](https://indico.ego-gw.it/event/464/contributions/4135/attachments/2277/4019/G2Net%20Kaggle%20Competition%20Winner%20Talk.pdf)
- 2 位 EventHorizon は、固定の時間周波数変換と学習可能な前段を比較し、前段と 2D CNN を組み合わせた。1D CNN も検証した。 [本人の解法](https://www.kaggle.com/competitions/g2net-gravitational-wave-detection/writeups/aillis-jp-eventhorizon-2nd-place-solution-trainabl) · [本人の公開コード](https://github.com/analokmaus/kaggle-g2net-public)
- P6 は E03 の再実行 E04 を対照とする。E05 は白色化後の学習波形で Hanford と Livingston を行ごとに確率 0.5 で入れ替える。Virgo と検証・テスト時のチャネル順は固定する。両 LIGO 検出器で共通する信号の学習が促され、AUC が上がると予想する。P6 の事前計画は E04、E05 の順で、許容差は 0.001。

## 実行手順

リポジトリ直下から実行する。以下では `$g2net` を `competitions/092-g2net-gravitational-wave-detection/src/run.py` に設定する。公式一括 ZIP をコンペ直下の `data/official.zip` に保存し、`uv run python competitions/092-g2net-gravitational-wave-detection/src/prepare.py` で `data/packed/` を生成する。ZIP と配列を合わせて約 152 GB の空き容量が必要。

1. `uv run python $g2net inspect` で ID・波形・分割を照合し、`outputs/data-manifest.npz` を生成する。
2. `uv run python $g2net smoke` で実波形 1,024 行の学習と検証・テストの推論を疎通してから、バッチサイズとエポック数を固定する。`selection-plans.json` をコミットしてから、`uv run python $g2net cv baseline B00-initial`、`select baseline`、`export baseline F00-baseline` を順に実行し、確定した export 内の `submission.csv` を `dev submit 92` で提出する。B00 は比較用計画でも同一構成で個別に実行する。
3. `cv p4 B00`、`cv p4 E01`、`cv p4 E02`、`cv p4 E03`、`select p4` の順に進める。再採点した OOF AUC の最高値から 0.001 以内の候補を、B00、E01、E02、E03 の順で優先する。
4. P5 後に P6 の対照と変更後を別の run ID にする。`selection-p6.json` に `p6` キーを置き、`candidates`、`tolerance`、run ID ごとの `configs` を固定する。P6 の実験前に計画と必要なモデル変更をコミットし、`cv p6 <run_id>` で両方を実行する。
5. `select p6`、`export p6 F00`、`analyze p6` で選定・提出生成・検出器別 RMS 区間の AUC 集計を行う。確定した提出 CSV の Late Private スコアを `dev submissions 92` で確認して報告する。

最終モデルと外部評価の比較は [振り返り](report.md) に記録した。CV の公開、保存済み OOF の独立再採点、選定、export は [共通 Quality Gate](../../docs/agent/experiment-quality-gates.md) を通す。提出予測は選定済み run の 3-fold 平均確率を使用する。

## 過去 run の再現範囲

CV 証跡で照合できるのは、保存済み OOF・テスト予測、分割、ソース・設定と再採点結果。各 fold の白色化係数と学習済み重みは保存していない。F00 は照合済みの fold 平均予測を CSV にした提出であり、保存成果物からモデルを復元して再推論した結果は含まない。過去の export を再生成する場合は、その CV 証跡に対応するコミット（F00 は `0e9d98e`、F00-baseline は `33a4eeb`）のソースを使う。今後、保存成果物からの再推論が必要な場合は、新しい run ID で fold 別の白色化係数と重みを保存し、前処理設定と併せて検証する。

## 公式データの取得証跡

2026-11-26（JST）に Kaggle 公式 API の `/api/v1/competitions/data/download-all/g2net-gravitational-wave-detection` から取得した。HTTP Range で取得した部分を結合して `data/official.zip` に保存し、`prepare.py` で各エントリの CRC、CSV と波形 ID の対応、形状、有限性を照合して配列に変換した。ZIP 自体の SHA-256 はレビュー時に保存済みファイルから計測した。[Kaggle 公式データ](https://www.kaggle.com/c/g2net-gravitational-wave-detection/data)

| 対象 | 値 |
|---|---|
| ZIP のバイト数 | 74,086,993,965 |
| ZIP の SHA-256 | `d6827df165ffa4458cd43c9bca7361de14e6431286d66034abdf7a165a3b92f7` |
| 変換後の CV データハッシュ | `f6a151a7eaac8778a606689e7ea6601801bc86ac393eb41b3635f113b8b7435e` |

CV データハッシュは ID 順の波形指紋と学習ラベルに基づく。再取得時は `Get-FileHash -Algorithm SHA256` で ZIP を照合し、`run.py inspect` で変換後のデータハッシュを確認できる。
