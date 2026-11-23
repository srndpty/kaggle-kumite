# #010 BNP Paribas Cardif Claims Management

匿名化された請求属性から、迅速に処理できる請求かどうかを予測する。
深さは Standard、CPU で実施。公式指標は Log Loss、提出列は
`ID,PredictedProb`。 [Kaggle 公式概要](https://www.kaggle.com/competitions/bnp-paribas-cardif-claims-management/overview)

## 進捗

- [x] P0. 公式データ取得・ダミー採点
- [x] P1. 問題設定・EDA
- [x] P2. 検証分割の設計
- [x] P3. Baseline の固定と提出
- [x] P4. 自力の仮説検証
- [x] P5. 上位解法の調査
- [x] P6. 上位解法の着眼点の再現
- [x] P7. 最終評価・エラー分析
- [x] P8. 知見の保存

## P0 とデータ

#018 の公式データを取得できなかったため、本コンペに差し替えた。
初回の公式 CLI は HTTP 403 だった。Rules 承諾後に取得でき、定数 0.5 の
ダミー提出は Public / Private ともに 0.69314 で採点された。

学習 114,321 行、テスト 114,393 行。特徴量は数値 112 列とカテゴリ 19 列。
学習行のうち 96,565 行に欠損があり、陽性率は 76.12%。特徴行の完全重複は 0 件。

## 検証と事前仮説

ID を特徴から除外。会員・契約の公式キーや時点は提供されていないため、
特徴の完全重複を検査したうえで seed 42 の層化 5 fold を固定した。
中央値、カテゴリ辞書、欠損指示を追加する列の集合は各 fold の学習行から決める。

| run | E00 からの変更 | 事前の予想 |
| --- | --- | --- |
| E00 | CPU LightGBM。数値を中央値補完し、カテゴリを学習 fold の辞書で符号化 | 比較基準 |
| E01 | 数値欠損を LightGBM の欠損分岐に渡す | 欠損行を別に分岐でき、Log Loss が下がる |
| E02 | 数値列の欠損指示を追加する | 補完値と真の中央値を区別でき、Log Loss が下がる |
| E03 | カテゴリ欠損を独立カテゴリにする | 未知カテゴリと区別でき、カテゴリ欠損のある行で改善する |
| E04 | 数値列 `v40` と `v50` の刻み幅を概算し、単位を合わせた差を追加する | 時間差に予測情報があれば Log Loss が下がる |

E04 の着想は 1 位解法。差の特徴だけを検証し、学習・検証・テストの行を横断する
ラベル特徴は使用しない。 [Dexter's Lab 本人解説](https://www.kaggle.com/competitions/bnp-paribas-cardif-claims-management/writeups/dexter-s-lab-1-dexter-s-lab-winning-solution)

## 再現手順

```powershell
dev download 10
$script = "competitions/010-bnp-paribas-cardif-claims-management/src/train.py"
uv run python $script --eda
foreach ($variant in 'E00','E01','E02','E03','E04') {
    uv run python $script --variant $variant
}
uv run python competitions/010-bnp-paribas-cardif-claims-management/src/select_model.py
```

既存 run を再実行する場合は別の `--run-id` を指定する。実験ログは
`experiments.jsonl`、分割・OOF・提出ファイルは `outputs/runs/` に保存する。

## 結果

| run | OOF Log Loss | Public | Private |
| --- | ---: | ---: | ---: |
| E00（採用） | 0.473624 | 0.47044 | 0.46881 |
| E01 | 0.473412 | — | — |
| E02 | 0.473885 | — | — |
| E03 | 0.473723 | — | — |
| E04 | 0.474063 | — | — |

E00 を基準とする行単位のペア bootstrap で、Log Loss 差の 95% 区間が
全体に負の候補だけを採用する。条件を満たす候補がなく E00 を採用した。
最終評価には同一 E00 ファイルの採点済みスコアを使用した。
