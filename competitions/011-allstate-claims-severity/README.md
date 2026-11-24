# #011 Allstate Claims Severity

- 深さは Standard。カタログ掲載済みの採用コンペを例外として扱う。
- 匿名化されたカテゴリ 116 列と連続値 14 列から、保険金支払額を予測する。
- 公式評価は元尺度の MAE。提出は `id,loss` の CSV。[Kaggle 公式概要](https://www.kaggle.com/competitions/allstate-claims-severity/overview/description)

## 進捗

- [x] P0. 公式データ取得とダミー提出の採点
- [x] P1. 問題設定・EDA と予測仮説
- [x] P2. リーク経路の確認と分割の固定
- [x] P3. Baseline の固定と Late Submission
- [x] P4. 自力の仮説検証
- [x] P5. 上位解法の調査
- [x] P6. 上位解法の着眼点の再現
- [x] P7. 最終評価とエラー分析
- [x] P8. 知見の保存

## 公式データと P0

公式 ZIP の直下にある 3 CSV を使用する。同梱の入れ子 ZIP にも同名のファイルがあり、そちらの train/test には末尾に不正な行がある。通常の `dev download 011` は内側の ZIP を続けて展開するため、以下の手順で直下の 3 ファイルを取り出す。

```powershell
$dest = 'competitions/011-allstate-claims-severity/data'
kaggle competitions download -c allstate-claims-severity -p $dest
uv run python -c "from zipfile import ZipFile; from pathlib import Path; p = Path('$dest'); ZipFile(p / 'allstate-claims-severity.zip').extractall(p, ['train.csv', 'test.csv', 'sample_submission.csv'])"
uv run python competitions/011-allstate-claims-severity/src/dummy.py
```

2026-11-25 JST に、中央値 2115.57 のダミー提出が採点された。Late Public MAE は 1783.61708、Late Private MAE は 1801.33448。

## EDA と検証設計

- 学習 188,318 行、テスト 125,546 行。欠損は両方とも 0。テストにしかないカテゴリ水準は列ごとの合計で 37 件。
- 損失額の 25% 点は 1204.46、中央値は 2115.57、75% 点は 3864.05、99% 点は 13981.20。最大は 121012.25 で、上位 1% の行が全損失額の 6.07% を占める。
- `id` 昇順の学習行を 5 区間に分けると、損失中央値は 2110.43〜2124.95。各区間にテスト行が 24,862〜25,455 行あり、順序ドリフトの兆候は小さい。
- 学習内で特徴量の完全一致が 1 行、学習とテスト間で 4 行あった。学習内の重複を fold の両側に置く経路を防ぐため、ID とラベルを除く全特徴の一致をグループとし、seed 42 のシャッフル付き 5-fold GroupKFold を固定した。カテゴリ語彙は fold の学習側だけから作り、テスト重複行に学習ラベルを転記しない。

## 実行前の仮説

| Run | 変更 | 予想 |
|---|---|---|
| E00 | 元尺度 L2 | fold 内の学習中央値より OOF MAE が下がる |
| E01 | 元尺度 L1 | 大口損失の影響が弱まり、全体 MAE が下がる |
| E02 | log1p 変換と L2 | 中央帯の過大予測が減り、上位 1% の過小予測が増える |
| E03 | 平方根変換と L2 | E02 より上位 1% の過小予測が小さく、E00 より中央帯の過大予測が少ない |

```powershell
uv run python competitions/011-allstate-claims-severity/src/train.py --eda
uv run python competitions/011-allstate-claims-severity/src/train.py --variant E00
```

他の run も `--variant` に run 名を指定する。再実行には `--run-id` で別の ID を指定する。計測値は `experiments.jsonl` に追記する。

## 実験結果

E00 の OOF MAE は 1188.20699。全 5 fold で、その fold の学習中央値を使う定数予測を上回った。初回モデル提出の Late Public MAE は 1167.01466、Late Private MAE は 1175.88415。

| Run | OOF MAE | E00 との差 |
|---|---:|---:|
| E00 | 1188.20699 | 0.00000 |
| E01 | 1159.18261 | −29.02438 |
| E02 | 1152.19527 | −36.01172 |
| E03 | 1153.57530 | −34.63169 |
| E04 | 1189.21125 | +1.00426 |

## 上位解法と再現実験

- 2 位の Alexey Noskov は、クラスタ中心までの距離を使う非線形特徴、カテゴリの組み合わせ、複数モデルのスタッキングを組み合わせた。距離特徴の効果は線形回帰でも個別に報告している。 [本人の解法記事](https://www.kaggle.com/competitions/allstate-claims-severity/writeups/alexey-noskov-2nd-place-solution)・[本人へのインタビュー](https://medium.com/kaggle-blog/allstate-claims-severity-competition-2nd-place-winners-interview-alexey-noskov-f4e4ce18fcfc)
- 3 位の Faron は、入力表現と目的変数変換を変えた XGBoost・Keras を大量に学習し、2 段目で Keras をバギングした。単体のチューニングよりモデル間の多様性を重視している。 [本人の解法記事](https://www.kaggle.com/competitions/allstate-claims-severity/writeups/faron-faron-s-3rd-place-solution)
- 1 位の本人記事は取得できなかったため、手法を確認できた上位記事のうち、最上位の 2 位から着眼点を選んだ。 [1 位解法記事](https://www.kaggle.com/competitions/allstate-claims-severity/writeups/bishwarup-b-1st-place-solution)

E04 は E00 にクラスタ中心までの距離だけを加える。各 fold の学習側で連続値 14 列を標準化し、MiniBatchKMeans の 16 中心を学習する。検証・テスト行には同じ変換を使う。既存の木に連続値の同時分布を表現する特徴を加えることで、MAE が下がると予想した。

## 最終評価

OOF MAE と事前に決めたブートストラップ基準で E02 を採用した。Late Public MAE は 1130.46572、Late Private MAE は 1144.12303。再現用の選定コマンドは以下のとおり。帯別の誤差と仮説の判定は振り返りに記録した。 [振り返り](report.md)

```powershell
uv run python competitions/011-allstate-claims-severity/src/select_model.py E00 E01 E02 E03 E04
```
