# #090 Child Mind Institute – Detect Sleep States

腕時計の加速度系列から入眠・起床の時点を検出する。深さは **Standard**。比較の主題は、イベントの時刻精度と重複抑制が Event Detection AP に与える影響。 [Kaggle 公式仕様](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/overview)

## 進捗

- [x] P0. データ取得とダミーの Late Submission 採点
- [x] P1. EDA と仮説
- [x] P2. 固定3-fold
- [x] P3. ベースラインと初回モデル提出
- [x] P4. 自力の比較実験
- [x] P5. 上位解法の調査
- [x] P6. 着眼点の再現
- [x] P7. 最終評価・エラー分析
- [x] P8. レポートと知見

## P0–P2

公式 Parquet と注釈 CSV を取得した。固定予測 Notebook v1 の Late Submission は Public 0.000 / Private 0.000 で完了した。 [採点証跡](submission-evidence.json)

学習センサーは277系列、127,946,340行。anglez と enmo の null は0件。注釈は入眠と起床それぞれ7,254行で、step の欠損は入眠2,463行、起床2,460行。観測済みイベントのみ採点し、候補生成は系列全体で行う。評価は2種×10許容幅の Event Detection AP。系列長は37,080〜1,433,880ステップ（中央値412,920）、系列内の step の欠番は0。観測された9,585イベントは全件、同じ step のセンサー時刻と一致した。7,254夜のうち両イベントありは4,790夜、片方のみは5夜、両方欠損は2,459夜。正解イベントのない8系列は、公式採点と同様に AP の集計対象外となる。

配布列に被験者 ID がないため、系列 ID 単位で seed 42 の3-foldを使う。系列をまたいだ特徴量計算や同一系列の fold 間共有を避ける。欠番による区間境界を移動窓と局所ピークがまたがないようにする。学習の負例は両イベントが観測された夜の入眠から起床までと、その前後3時間から抽出する。

## 事前仮説

| Run | 変更 | 予想 |
|---|---|---|
| E01 | 前後30分の統計量を追加 | 活動状態の持続を捉え、特に入眠の AP が改善する |
| E02 | 正例を前後3分に拡張 | 広い許容幅で再現率が上がり、1分幅では時刻精度が下がり得る |
| E03 | 重複抑制を15分から60分に拡張 | 重複による誤検出は減るが、短い間隔の真のイベントを落とし得る |

## 実行

コンペのディレクトリで `uv run python src/train.py E00` を実行する。E01、E02、E03、E04 も同様に実行する。E03 は E00 の保存済みビンスコアを使う。実験ログは `experiments.jsonl`、分割・ビンスコア・予測・重みは `outputs/runs/` に保存する。

## 上位解法と再現仮説

- 1位の本人発表では、注釈時刻の分単位の偏りと許容幅を踏まえて、イベント候補の出し方を設計した。 [1位の発表資料](https://speakerdeck.com/unonao/shui-mian-konpe-1st-place-solution)
- 16位の本人 writeup は、移動統計・スペクトログラム・周期時刻を系列モデルに入力し、睡眠状態の推定結果でイベント候補を再採点した。 [16位の解法](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/writeups/nikita-babich-16th-place-solution)

E04 では16位の着眼点から、現地時刻を1日周期で符号化した特徴量だけを E00 に加える。入眠・起床が1日の特定の時間帯に集中するなら、センサーに似た変化が生じる他の時間帯の誤検出が減り、両イベントの AP が改善すると予想する。

## 選定と提出の再現

`uv run python src/select_model.py E00 E01 E02 E03 E04` で、主分割の保存 OOF から `selection.json` を新規生成する。保管済み選定記録はそのまま参照する。

`uv run python src/export.py E04 <Kaggleユーザー名> outputs/final` で、候補数を記録日数あたり各イベント8件に制限したローカル予測、非公開データセット用の重み、CPU Notebook を生成する。Kaggle CLI の `datasets create -p dataset` と `kernels push -p notebook` は `outputs/final` を作業ディレクトリにして実行する。E00 も同じ操作で再現できる。 [選定記録](selection.json) [採点証跡](submission-evidence.json)

公式メトリックとの照合を再実行する場合は、公式 Notebook の出力 `metric.py` を `outputs/reference/metric.txt` に置いてから `dev test` を実行する。 [公式メトリック Notebook](https://www.kaggle.com/code/metric/event-detection-ap)

主分割上位2件の再検証では `uv run python src/train.py E01 --seed 2026 --run-id E01-seed2026` と `uv run python src/train.py E04 --seed 2026 --run-id E04-seed2026` を実行する。この分割では E01 が上回った。 [比較結果](report.md)

提出時の候補制限を OOF で確認するには、保存済みの主分割 run を使い、`uv run python src/train.py E00 --capped --run-id E00-capped` と `uv run python src/train.py E04 --capped --run-id E04-capped` を実行する。
