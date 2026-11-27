# #085 Rainforest Connection Species Audio Detection

深さは Standard。熱帯雨林の録音を入力とし、鳥・カエル24種の出現を録音単位で予測する。公式の主指標は lwlrap。

[公式概要](https://www.kaggle.com/competitions/rfcx-species-audio-detection/overview) · [公式データ](https://www.kaggle.com/competitions/rfcx-species-audio-detection/data)

## 採点と検証

P0 の seed 42 ダミー提出は Public 0.26175、Private 0.25054。公式ファイル一覧で FLAC と注釈・提出サンプルの合計は 22,843,987,549 バイト。取得前の空きは約 288 GB。Kaggle の個別ファイル API ではディレクトリの区切りをパーセントエンコードして取得する。

**公式音声の取得を保留中。** 2026-11-27、個別ファイルの取得中に Kaggle が HTTP 429 を返した。公式アーカイブから必要な範囲だけ取得する要求も 429 となり、応答は `Retry-After: 53850`（約15時間）を指定した。照合済みの取得分は学習 1 / 4,727 件、テスト 112 / 1,992 件。注釈２ファイルと提出サンプルは取得済み。取得制限の解除後、残りの音声を取得してから公式データで分割と学習を実行する。次点は #052。

注釈 CSV の集計では真陽性1,216区間、誤検出7,781区間。真陽性がある録音は1,132件、そのうち複数種が出現する録音は27件。種別の陽性録音数は30〜90件。真陽性区間長の中央値は1.856秒、四分位範囲は1.093〜3.344秒。同一録音・同一種で真陽性区間と誤検出区間が直接重なる注釈は０件。５秒窓での重なりと録音の重複は、全 FLAC 取得後に確認する。

録音の PCM MD5 と音声形式で同一音声をまとめ、種別の真陽性数を均衡させる固定３分割（seed 42、v1）を使用する。検証・推論は全録音を走査し、注釈がある区間を入力として渡さない。公式に録音場所を特定できる情報を確認できない間は、未知の録音場所への汎化は未検証として扱う。

## 比較実験の事前登録

| Run | 変更 | 事前仮説 |
|---|---|---|
| E00 | 32 kHz モノラル、64-bin log-mel、５秒窓、SmallCNN。５秒間隔で走査し、最大値で集約 | 真陽性の鳴き声から録音中の種を推定できる |
| E01 | 背景窓の半数を、同じ種の真陽性と重ならない誤検出窓に置換 | 紛らわしい音による順位の誤りが減る |
| E02 | 推論間隔を2.5秒に変更 | 境界をまたぐ短いイベントで改善する |
| E03 | 窓上位３件の平均で集約 | 単発誤検出を抑え、短いイベントでは悪化し得る |

学習は各 fold で800更新、バッチ16、AdamW、seed 42。真陽性窓では注釈された種のみに損失を課す。背景・誤検出の１窓あたりの総負例重みは0.25とし、誤検出窓では注釈対象種だけに割り当てる。E01 でも正例抽出の乱数列と正例・負例のバッチ件数は同じ。

最良 OOF lwlrap から0.001以内なら E00、E03、E02、E01、P6 候補の順で採用する。P5 の資料調査は自力実験の記録後に行う。

## 実行

ルートから次のように実行する。音声はコンペの data/train と data/test、CSV は data 直下に置く。

```powershell
$d = 'competitions/085-rfcx-species-audio-detection'
uv run python "$d/src/train.py" eda
foreach ($trial in 'E00','E01','E02','E03') {
    uv run python "$d/src/train.py" cv --trial $trial --run-id $trial
    if ($LASTEXITCODE -ne 0) { throw "CV に失敗: $trial" }
}
uv run python "$d/src/compare.py" E00 E01 E02 E03
uv run python "$d/src/train.py" select --receipt "$d/outputs/p4-selection.json"
```

外部評価用のベースラインは `selection-plan-baseline.json` で E00 を E00-external として実行し、単独で選定する。最終候補は P5 の後に別の事前計画へ登録し、全候補に新しい run ID を使う。選定後に `export --plan <事前計画> --receipt <選定記録> --run-id <export ID>` を使い、Quality Gate が照合した fold 平均のテスト予測から提出 CSV を生成する。
