# #083 レビュー対応

既存12記録・120成果物、分割、候補固定・最終選定、提出ファイルを維持した。追加学習・Late Submissionは行っていない。以下は保存済みデータ・OOFによる事後診断で、モデル選定には使用しない。

## P2-2：音声cacheの真正性

`load_prepared(deep=True)`でtrain/testの両npyについてSHA256を照合してから配列を使用する。学習・予測再生成・音声条件の分析・最終選定の再検証・監査から共通処理を呼ぶ。従来のtrain.py内の重複した照合は削除した。音声を使わないmetadataの読み込みは従来どおり軽量に行える。

同じ形状・dtypeのままcacheの1値を変更するネガティブテストを追加。train/testそれぞれの改変について、予測再生成・監査が推論やfold処理の前に拒否することを確認した。実データでもdeep照合を通した後、最終checkpointからの再生成確率の最大差0・提出ラベル一致を再確認した。

## P2-3：runと実験定義の整合性

選定で許可するrun IDと実験ID・seedの対応を明示し、SmallCNN・Accuracy・学習クラス・12出力クラスの順序・benchmark/random_splitフラグを検査する。学習設定の保存ファイルと記録、checkpointの出力数・線形周波数フラグも照合する。候補5件と確認seedの両runを再検証した。

初期E01〜E04にはtraining_classesの記録がない。この4件だけは、その事実を維持したまま、12クラスの宣言とcheckpointの12出力を確認する。E05や確認seedでは学習クラスの記録を必須にした。既存runへのmetadataの後付けはしていない。

ID・モデル・実験ID・seed・クラス順序・各フラグの不一致、および正しい宣言と異なるcheckpointを拒否するテストを追加した。`--verify-existing`による再検証結果は、既存selection.jsonと一致した。

## P2-1：既存foldの構成

seed42・43の全foldで、学習側／検証側それぞれの話者数、話者グループ数、12クラス件数、silence窓数、背景ファイル、発話長、RMSの分布を集計した。RMSは正規化後の各1秒窓から計算し、長い背景録音のRMSを流用しない。分布の件数・平均・標準偏差・min・四分位点・p95・maxはoutputs/review/diagnostics.jsonに保存した。

| seed / fold | 検証話者数 | 検証silence窓 | 検証背景録音（窓数） | silence RMS中央値：学習→検証 |
|---|---:|---:|---|---:|
| 42 / 0 | 636 | 121 | pink_noise (60), running_tap (61) | 0.02561 → 0.07382 |
| 42 / 1 | 614 | 122 | dude_miaowing (61), exercise_bike (61) | 0.07114 → 0.01755 |
| 42 / 2 | 631 | 155 | doing_the_dishes (95), white_noise (60) | 0.03219 → 0.04276 |
| 43 / 0 | 626 | 121 | exercise_bike (61), white_noise (60) | 0.04276 → 0.03219 |
| 43 / 1 | 625 | 156 | doing_the_dishes (95), dude_miaowing (61) | 0.07381 → 0.01987 |
| 43 / 2 | 630 | 121 | pink_noise (60), running_tap (61) | 0.02561 → 0.07382 |

録音数は各foldで2本でも、窓数・音源の種類・RMSは揃っていない。検証発話のRMS中央値は0.04360〜0.04773、発話長中央値はいずれも1秒だった。これだけでseed変動の原因を特定しない。

E01のseed42→43のOOF Accuracy差は-0.071177。そのうちsilenceの直接寄与差は-0.000584にとどまり、silenceの正答率変動だけでは全体の約7.1ポイントの変動を説明できない。背景音が学習や他クラスの予測へ与える間接的な影響は未分離である。

## 12クラス別のOOF Accuracyへの寄与

以下はseed42。寄与は「そのクラスの正解行数／全65,119行」で、合計がOOF Accuracyになる。fold平均とは重み付けが僅かに異なる。seed43も同じ形式でJSONへ保存した。

| クラス | 件数 | E01寄与 | E05-r2寄与 | 差 |
|---|---:|---:|---:|---:|
| yes | 2,377 | 0.032003 | 0.034706 | +0.002703 |
| no | 2,375 | 0.029423 | 0.032279 | +0.002856 |
| up | 2,375 | 0.032510 | 0.030329 | -0.002181 |
| down | 2,359 | 0.030037 | 0.030176 | +0.000138 |
| left | 2,353 | 0.034245 | 0.034368 | +0.000123 |
| right | 2,367 | 0.033431 | 0.033139 | -0.000292 |
| on | 2,367 | 0.033155 | 0.033001 | -0.000154 |
| off | 2,357 | 0.029223 | 0.033723 | +0.004499 |
| stop | 2,380 | 0.034153 | 0.033708 | -0.000445 |
| go | 2,372 | 0.030452 | 0.031619 | +0.001167 |
| silence | 398 | 0.005252 | 0.005436 | +0.000184 |
| unknown | 41,039 | 0.532379 | 0.564090 | +0.031711 |

seed42の改善+0.040311の大部分はunknownの寄与+0.031711。seed43もunknownの寄与+0.042092が全体改善+0.039865を上回り、他クラスの悪化を補っていた。

## silence／非silenceと構成比の感度分析

| seed / run | silence Recall（=クラス内Accuracy） | 非silenceの12クラスAccuracy | 非silenceと判定するRecall |
|---|---:|---:|---:|
| 42 / E01 | 0.859296 | 0.856244 | 0.999737 |
| 42 / E05-r2 | 0.889447 | 0.896618 | 0.999521 |
| 43 / E01-s43 | 0.763819 | 0.785217 | 0.999552 |
| 43 / E05-s43 | 0.801508 | 0.825095 | 0.999212 |

最後の列は二値のsilence検出としてのRecallであり、非silence内の語を取り違えた場合も正解と数える。12クラスAccuracyとは区別する。

各クラスの条件付き正答率と非silence内の相対クラス比率はCVのまま固定し、silence比率だけをP0から分かる約9.949%へ変更した。

`仮定下のAccuracy = p_silence × silence Recall + (1 − p_silence) × 非silence内Accuracy`

| seed | 元のOOF差 | 比率変更後のE01 | 比率変更後のE05 | 比率変更後の差 |
|---|---:|---:|---:|---:|
| 42 | +0.040311 | 0.856548 | 0.895904 | +0.039356 |
| 43 | +0.039865 | 0.783088 | 0.822749 | +0.039661 |

**この仮定では、silence比率だけを変えても差の符号は反転しない。** 構成比の違いは実測できたが、それだけでPrivateの-0.00529を説明できたわけではない。これはPrivate Accuracyの推定ではなく、条件付き正答率を固定した事後感度分析である。未収録語・音響条件・他のクラス構成比・fold平均の効果は引き続き未分離とする。

## 検証と再実行

レビュー対応後の`dev check`成功。lint・整形・型検査エラー0、#083は48件、全体284テスト。既存証跡128ファイル（ログ、120成果物、固定JSONなど）の変更前後のSHA256一致を確認した。元の評価・選定・学習結果を書き換えず、レビュー出力はoutputs/review/へ分けて保存した。

```powershell
uv run python competitions/083-tensorflow-speech-recognition-challenge/src/review_diagnostics.py
uv run python competitions/083-tensorflow-speech-recognition-challenge/src/select_model.py --verify-existing
uv run python competitions/083-tensorflow-speech-recognition-challenge/src/predict.py E05-r2 --output competitions/083-tensorflow-speech-recognition-challenge/outputs/review/reproduction.json
uv run python competitions/083-tensorflow-speech-recognition-challenge/src/verify.py --output competitions/083-tensorflow-speech-recognition-challenge/outputs/review/audit.json
```

出力先は上書き禁止。同じ診断を再実行する場合は`--output`で未使用のパスを指定する。
