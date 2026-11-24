# #054 U.S. Patent Phrase to Phrase Matching

- 深さ: Standard
- 評価: Pearson 相関
- 入力: 特許の基準語句、比較語句、CPC コード。出力は類似度。
- 提出: インターネット無効の非公開 Kaggle Notebook で `id,score` の `submission.csv` を生成する。

[公式概要・提出仕様](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/overview)

## 進捗

- [x] P0. 公式データ取得とダミー採点
- [x] P1. EDA と仮説の事前登録
- [x] P2. 未知 anchor を評価する固定分割
- [x] P3. Baseline と初回 Late Submission
- [x] P4. 自力の比較実験
- [x] P5. 上位解法の調査
- [x] P6. 着眼点を一つ再現
- [x] P7. 最終評価・エラー分析
- [x] P8. レポートと知見の保存

## データと採点経路

P0 の公式採点は Public 0.0085、Private −0.0151。公式学習データは 36,473 行、正規化済み anchor は 733 種、CPC は 106 種。スコア 0、0.25、0.5、0.75、1 の件数は順に 7,471、11,519、12,300、4,029、1,154。公開テスト 36 行はすべて学習データに含まれるため、公開 CSV は提出形式の確認に使う。

## 検証と仮説

NFKC、小文字化、空白の正規化を行い、共通 anchor と同一・逆順ペアでつながる行を連結した。732 成分を GroupKFold で分け、検証行数は 7,295 / 7,295 / 7,295 / 7,294 / 7,294。各分割の語彙とカテゴリは学習側で適合する。モデル選択には全行 OOF Pearson を使い、同点なら先行 run を採用する。

| 比較 | 事前仮説 |
|---|---|
| E02 ← E01 | 文字 3〜5 gram の TF-IDF コサイン類似度を加えると、表記揺れのあるペアの誤差が下がる。 |
| E03 ← E02 | CPC のセクションと完全コードを加えると、分野別のスコア分布差を捉えられる。正規化した無順序ペアのうち 2,744 組が複数 CPC に現れ、うち 675 組には異なるスコアがある。 |
| E04 ← E03 | CPC セクションと類似度の交互作用を加えると、同程度の語句一致に対する分野別のスコア差を捉えられる。 |

## EDA の補足

anchor と target の文字数中央値はともに 15。四分位範囲は anchor が 12–19、target が 11–20。正規化後の完全一致は 279 行。CPC セクション別の平均スコアは C の 0.3359 から B の 0.3786 まで分布した。検証側 target が学習側にも現れる割合は fold により 15.99–18.23% だった。

## 実験結果

| run | 比較元 | 特徴量の追加 | OOF Pearson | 差の95%区間 |
|---|---|---|---:|---:|
| E01 | — | 長さ・完全一致・単語 Jaccard の Ridge | 0.42722 | — |
| E02 | E01 | 文字 TF-IDF コサイン | 0.48577 | [+0.04974, +0.06794] |
| E03 | E02 | CPC セクション・完全コードの one-hot | 0.48614 | [−0.00149, +0.00234] |
| E04 | E03 | CPC セクションと類似度の交互作用 | 0.48722 | [−0.00043, +0.00261] |
| E05 | E04 | 同一 anchor・CPC の他候補に対する平均コサイン | **0.48840** | [+0.00032, +0.00201] |

区間は連結成分を単位に同じ成分を引き直すペア・ブートストラップ 2,000 回から求めた。E01 の Late Public は 0.4206、Private は 0.4204。OOF で選んだ E05 の Late Public は 0.4645、Private は 0.4864。

## 再現

```powershell
$d = 'competitions/054-us-patent-phrase-to-phrase-matching'
dev download 54
foreach ($run in 'E01','E02','E03','E04','E05') {
    uv run python "$d/src/train.py" $run --run-id "${run}-r2"
    if ($LASTEXITCODE -ne 0) { break }
}
uv run python "$d/src/analyze.py" --run-suffix=-r2
uv run python "$d/src/export.py" <Kaggleユーザー名> "$d/outputs/final" --run-suffix=-r2
kaggle kernels push -p "$d/outputs/final"
kaggle competitions submit us-patent-phrase-to-phrase-matching -k <ユーザー名>/kumite-patent-final -v <版> -f submission.csv -m 'Kumite Patent E05 final'
```

公式採点に用いた run は記録済みの E01〜E05。上の例は再実行のため `-r2` を付ける。再実行済みなら、未使用の共通接尾辞に置き換える。初回採点の Notebook は `src/dummy.py` で生成し、Baseline は `src/export.py --baseline` で生成する。
