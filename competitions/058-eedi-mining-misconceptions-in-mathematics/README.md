# #058 Eedi - Mining Misconceptions in Mathematics

深さは **Standard**。数学の四択問題で、誤答肢の背景にある誤概念を順位付きで予測する。評価は MAP@25。候補を拾う効果と、その候補の順位を改善する効果を同じ問題分割で比較する。 [公式概要](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/overview/abstract)

## 進捗

- [x] P0. 公式データ取得とダミー採点
- [x] P1. EDA と仮説の登録
- [x] P2. 問題グループ単位の分割
- [x] P3. ベースラインと Late Submission
- [x] P4. 自力の仮説検証
- [x] P5. 上位解法の調査
- [x] P6. 上位解法の着眼点の再現
- [x] P7. 最終評価・エラー分析
- [x] P8. 知見の保存

## データと検証分割

公式学習データは 1,869 問。誤答肢のうち誤概念ラベル付きは 4,370 行、ラベルなしは 1,237 行。学習データに現れる 1,604 種類のラベルのうち 747 種類は1回だけ出現した。ラベル付き行の多い分野は Linear Equations（112行）、Area of Simple Shapes（95行）、BIDMAS と Linear Sequences (nth term)（各88行）。

誤答肢を展開してから同じ問題の行を束ね、正規化した問題文と選択肢の並べ替えが一致する問題も同一グループにした。計 1,868 グループを seed 42 の GroupKFold 5分割（v1）に固定した。分割前から全候補の名前が公式に配布されるため、全 fold でその一覧を検索する。正解ラベルを外側学習 fold で観測できたのは 3,206 行、観測できなかったのは 1,164 行。

## 事前仮説

| run | 変更 | 予想 |
|---|---|---|
| E00 | 誤概念名で fit した文字 3–5 gram TF-IDF とコサイン類似度 | ベースライン |
| E01 | E00 の検索器を固定 E5 に差し替える | 語彙が一致しない誤答で Recall@100 が上がる |
| E02 | E01 と E00 の上位100件を等重みで RRF 融合する | 手掛かりが補完し、E01 より Recall@100 が上がる |
| E03 | E02 の上位100候補を LambdaRank で並べ替える | Recall@100 を保ち、MAP@25 が上がる |

## Kaggle 採点

2026-11-24 にインターネット無効の非公開 CPU Notebook を採点した。P0 は固定の誤概念 ID 25件で Public 0.00183 / Private 0.00077。E00 は OOF 0.12625、Public 0.11230 / Private 0.13592。 [P0 Notebook](https://www.kaggle.com/code/shion5/kumite-eedi-p0) · [E00 Notebook](https://www.kaggle.com/code/shion5/kumite-eedi-e00) · [提出履歴](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/submissions)

最終 E04 は Public **0.13175** / Private **0.16252**。Kaggle Notebook の実行結果は配布テスト9行でローカル CSV と一致した。実験ログの `late_score` は Private 値。 [E04 Notebook](https://www.kaggle.com/code/shion5/kumite-eedi-e04) · [提出証跡](submission-evidence.json)

## 再実行

リポジトリのルートで実行する。

```powershell
dev download 58
$d = 'competitions/058-eedi-mining-misconceptions-in-mathematics'
uv run hf download intfloat/e5-small-v2 config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json vocab.txt --revision ffb93f3bd4047442299a41ebb6fa998a38507c52 --local-dir "$d/outputs/e5"
uv run python "$d/src/train.py" p0 "$d/outputs/p0-new" --owner <Kaggleユーザー名>
uv run python "$d/src/train.py" cv E00 --run-id E00-r2
```

各 CV run の OOF、候補一覧、fold 割当と指標は `outputs/runs/<run-id>/` に保存される。実験記録は `experiments.jsonl` に追記する。Kaggle では `kaggle kernels push -p <Notebook出力先>` で登録し、完了した版を `kaggle competitions submit eedi-mining-misconceptions-in-mathematics -k <ユーザー名>/<Notebook名> -v <版> -f submission.csv -m <メッセージ>` で採点する。


CV をすべて再計算して、別名の選定ファイルを生成する。

```powershell
foreach ($trial in @('E01', 'E02', 'E03', 'E04')) {
    uv run python "$d/src/train.py" cv $trial --run-id "$trial-r2"
}
uv run python "$d/src/train.py" select E00-r2 E01-r2 E02-r2 E03-r2 E04-r2 --baseline E00-r2 --output "$d/outputs/selection-r2.json"
```

最終版の再現には、ローカルの E5 の6ファイルを非公開 Kaggle Dataset に登録して添付する。保存済みの選定記録と OOF を使った実施時のコマンドは次のとおり。

```powershell
uv run python "$d/src/train.py" final E04 final-E04 --owner shion5 --dataset shion5/kumite-eedi-e5-small-v2 --selection "$d/selection.json"
kaggle kernels push -p "$d/outputs/runs/final-E04"
```

`final` は公開テスト用の提出 CSV と Notebook を同じ実行で生成し、CV run と選定記録のハッシュを保存する。E04 の学習済み資材は固定 E5 で、クエリと誤概念名から順位を計算する。報告書に全実験と誤差分析をまとめた。 [実験レポート](report.md)

提出を生成する前に、現在の推論ソースを採用 CV run のハッシュと照合する。選定記録を指定する場合は E00〜E04 各1件の保存済み OOF を再採点して記録全体を照合する。E00 以外の候補は固定 E5 リビジョンと同一のモデル資材を使う。過去の E00 を現在の推論ソースで再提出する場合は、先に E00 を別 run ID で再計算する。
