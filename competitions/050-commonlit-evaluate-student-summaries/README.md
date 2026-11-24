# #050 CommonLit — Evaluate Student Summaries

深さは **Standard**。生徒の要約から内容 `content` と表現 `wording` の評価を予測する。原文との対応を使うと未知の課題で改善するかを検証する。評価は2列の RMSE の平均（MCRMSE、低い方が良い）。提出はインターネット無効の非公開 CPU Notebook。 [評価仕様](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/overview/evaluation) · [公式データ](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/data)

## 進捗

- [x] P0. 公式データ取得と定数予測の Late Submission
- [x] P1. EDA と事前仮説
- [x] P2. 課題単位の検証分割を固定
- [x] P3. ベースラインと Late Submission
- [x] P4. 自力の比較実験4件
- [x] P5. 上位2チームの本人資料を確認
- [x] P6. 原文に対する要約の長さ比の比較
- [x] P7. 最終評価・誤差分析
- [x] P8. レポートと知見の保存

## データと検証

学習7,165件は4課題（39c16e が2,057件、3b9047 が2,009件、814d6b が1,103件、ebad26 が1,996件）。配布テストは学習と異なる2課題の4件。課題別の要約語数中央値は50、65、59、63語。表現スコアの中央値は約 −0.28〜0.50 と課題で異なる。正規化した要約の完全一致グループに78行が含まれ、課題をまたぐ一致と原文全体をそのまま写した要約は各0件だった。

`prompt-loo-v1` は毎回1課題を保留する4分割。検証要約と正規化後に完全一致する学習行を除外する。変換器と回帰器は各 fold の学習行で適合する。原文と設問は推論対象の行に付属するものを使う。主指標は全 OOF 行の MCRMSE。課題別と目的列別の値も見る。

P0 では学習ラベルの列平均を提出し、Public **0.85073**、Private **0.98867** を確認した。 [ダミー Notebook](https://www.kaggle.com/code/shion5/kumite-summaries-p0) · [提出履歴](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/submissions)

## 実験

| Run | 変更と事前の予想 | OOF MCRMSE |
|---|---|---:|
| E00 | 語数・文数・平均語長・平均文長を標準化して Ridge に入力し、列平均より改善する | 0.85492 |
| E01 | E00 に要約の単語 unigram TF-IDF を追加し、表現の RMSE が下がる | 0.89089 |
| E02 | E00 に原文との unigram・bigram 一致率を追加し、内容の RMSE が下がる | 0.75594 |
| E03 | E00 に設問との単語一致率を追加し、内容の RMSE が下がる | 0.87529 |
| E04 | E00 を seed 42 の行単位4分割で評価し、課題の共有で MCRMSE が低く出る | 0.72724 |
| E05 | E00 に要約語数 ÷ 原文語数を追加し、原文に対する情報量の違いを捉えて内容の RMSE が下がる | 1.59356 |

E00 の学習ラベル列平均による参照 OOF は 1.04993。E04 は分割の診断に使う。

2026-11-24 に各 Notebook の版1を採点した。E00 は Public 0.61688 / Private 0.65148。課題単位 CV で選定した E02 は Public **0.56152** / Private **0.63399**。実験ログの `late_score` は Private を記録する。 [E00 Notebook](https://www.kaggle.com/code/shion5/kumite-summaries-e00) · [E02 Notebook](https://www.kaggle.com/code/shion5/kumite-summaries-e02) · [振り返り](report.md)

Notebook の書き出し時は、配布 CSV 5件、保存済み推論ソース、生成 Notebook のハッシュを `evidence.json` に記録する。Kaggle 実行時は学習 CSV 2件のハッシュを照合し、テスト側は列構成・課題の対応・サンプル提出と要約の ID 集合を検査する。上の採点値は提出済みの版1の実測値。

## 上位解法と再現対象

- 1位 Lucky shake は学習課題を増やすために原文・課題・要約を生成し、擬似ラベルを使った。公開推論コードの LightGBM 特徴には要約と原文の語数比がある。E05 ではこの比だけを加える。語数の定義は E00 のトークン化に揃える。 [本人 writeup](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/writeups/lucky-shake-1st-a-brief-review-of-the-competition-) · [本人の推論コード](https://www.kaggle.com/code/itsmeanszao/debertav3-lgbm-no-autocorrect)
- 2位 Ivan Aerlic は、設問や原文をエンコーダーの文脈に使い、回帰ヘッドの平均プーリングは要約部分に限定した。課題の多様性を増やす1位と比べ、予測に使う表現を設計する着眼点が異なる。 [本人の解法解説](https://www.youtube.com/watch?v=BHi155SztPg) · [主催者らの論文](https://aclanthology.org/2024.bea-1.9/)

## 再現手順

```powershell
dev download 50
$d = 'competitions/050-commonlit-evaluate-student-summaries'
uv run python "$d/src/train.py" audit
foreach ($trial in 'E00','E01','E02','E03','E04','E05') {
    uv run python "$d/src/train.py" cv $trial
}
uv run python "$d/src/train.py" select E00 E01 E02 E03 E05
uv run python "$d/src/dummy.py" <Kaggleユーザー名> "$d/outputs/p0"
uv run python "$d/src/notebook.py" <Kaggleユーザー名> "$d/outputs/baseline" --baseline
uv run python "$d/src/notebook.py" <Kaggleユーザー名> "$d/outputs/final" --selection "$d/selection.json"
```

再実行時は `cv <試行名> --run-id <試行名>-r2` を使い、選定には再実行した run ID と `--output <新規パス>` を渡す。再実行したベースラインの書き出しには `--baseline E00-r2` を指定する。各出力先の Notebook を `kaggle kernels push -p <出力先>` で登録する。実行完了後に `kaggle competitions submit commonlit-evaluate-student-summaries -k <ユーザー名>/<Notebook名> -v <版> -f submission.csv -m <メッセージ>` で採点する。
