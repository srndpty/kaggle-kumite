# #083 TensorFlow Speech Recognition Challenge

- URL: https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge
- 分野: 音声 / 主催: Google Brain / 深さ: **Standard** / カタログ難度2・計算負荷S
- テーマ: 音声表現・時間移動・背景雑音の効果と、未知話者への汎化。
- 予測: 10コマンドとunknown・silenceの12クラス。公式指標Accuracy、提出はfname,labelのCSV。
- 開催: 2017-11-15〜2018-01-16。公式概要の表示は1,313チーム（[公式概要](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/overview)の索引本文で確認）。順位相当スコアは未調査。

## P0

- 2026-11-19 JST、公式データ取得・定数silenceのLate Submission成功。P0はCOMPLETE、Public **0.09950** / Private **0.09949**。
- P0はRunStoreで予約・保存済み。oof_score=0はスキーマ上の未評価値で、CV実測値ではない。選定対象外。
- train.7z 1,121,103,842 bytes、test.7z 2,640,679,130 bytes。sample_submissionも7z形式。
- データ・モデル・成果物はdata/・outputs/（git管理外）。終了後公開のtestラベルは使わない。

## 分割・前処理（本実験前に登録）

- IDは発話ファイルの相対パス。10コマンド以外はunknown。背景録音の重複しない1秒窓をsilenceとし、端数を除外する。
- 話者IDは_nohash_前。完全重複WAVで話者が異なる場合はグループを統合。seed42の層化グループ3fold、確認seed43。背景録音は元ファイル単位で各foldへ均等配分する。
- 各foldの学習・検証に12クラスがあること、話者・背景録音・重複グループの非重複を検査する。ID・分割・音声ハッシュを保存する。
- 近似重複・録音環境依存は残り得る。test話者との対応は復元しない。
- 16kHz・モノラル・1秒。短い音声は末尾ゼロ埋め、長い音声は中央切り出し。標準化は学習foldだけの周波数別平均・標準偏差。
- log-melは40bin・窓400・hop160・FFT512、対数下限1e-6。線形周波数も40binの三角フィルタで入力形状を揃える。
- CNNはConv-BN-ReLU-Poolを32/64/128ch、global average pooling、12クラス出力。AdamW・lr=.001・weight_decay=.0001・batch128、クラス均等の復元抽出。検証の構成比は変更しない。
- 20epoch固定で検証による打ち切り・checkpoint選択なし。training_config.jsonを正とする。試走で予算超過の見込みなら本実験前に全構成共通のepoch数を下げる。
- GPU学習合計4時間。試走・seed確認・ランダム分割診断を含む。超過見込みなら停止し残作業を明記する。

## 自力仮説（本実験前に登録）

| run | 比較元 | 単独変更 | 予想 |
|---|---|---|---|
| E00 | — | 学習foldクラス頻度 | 定数予測の基準 |
| E01 | E00 | log-mel＋小型CNN | 定数予測より改善 |
| E02 | E01 | log線形周波数40bin | melの方が音声の識別に有利と予想 |
| E03 | E01 | ±100msのゼロ埋め時間移動 | 発話位置への頑健性が改善 |
| E04 | E01 | 学習側背景音を確率0.5、SNR5〜20dBで混合 | 雑音への頑健性が改善 |

- E03・E04は学習時のみ適用し、組み合わせない。雑音源は学習foldのsilence窓だけ。
- E01-randomは行単位層化3foldの診断。別split_versionとし、話者重複を記録する。主選定には使わない。
- E01-benchmarkは主分割の1fold試走。選定対象外。

## P5以降・選定規則（本実験前に固定）

- E02〜E04完了後に、異なるアプローチの上位解法を2〜3件調査。本人writeup・コードを優先し取得範囲を記録する。
- 未検証の着眼点1件をE05として比較元・変更・予想を実行前に登録する。全解法の再現とは区別する。
- 主評価はfold平均Accuracy。OOF Accuracy・macro F1・クラス別Recall・LogLoss・混同行列・時間・メモリも保存する。
- seed42のE01〜E05から最良候補1件を選ぶ。同点は実行時間の短い構成。候補と直接の比較元をseed43で確認し、両seedで改善した場合だけ採用。同点・不一致なら比較元を維持する。
- 最終は採用構成のseed42・3fold確率平均のargmax。選定JSON・提出ハッシュを固定してから外部評価し、Public/Privateで選び直さない。
- OOFは1モデル、提出は3モデル平均。silenceの割合・種類が公式testと一致する保証はない。外部改善をリーク不在の証明に使わない。

## 実行方法

```powershell
dev new 83
dev download 83
7z x competitions/083-tensorflow-speech-recognition-challenge/data/sample_submission.7z -ocompetitions/083-tensorflow-speech-recognition-challenge/data -y
7z x competitions/083-tensorflow-speech-recognition-challenge/data/train.7z -ocompetitions/083-tensorflow-speech-recognition-challenge/data -y
7z x competitions/083-tensorflow-speech-recognition-challenge/data/test.7z -ocompetitions/083-tensorflow-speech-recognition-challenge/data -y
uv run python competitions/083-tensorflow-speech-recognition-challenge/src/dataset.py
uv run python competitions/083-tensorflow-speech-recognition-challenge/src/train.py E01-benchmark --experiment E01 --benchmark
uv run python competitions/083-tensorflow-speech-recognition-challenge/src/train.py E00 --experiment E00
uv run python competitions/083-tensorflow-speech-recognition-challenge/src/train.py E01 --experiment E01
# E02〜E04も同じCLI。再実行は新しいrun_id。確認用は --seed 43。
uv run python competitions/083-tensorflow-speech-recognition-challenge/src/train.py E01-random --experiment E01 --random-split
dev check
dev orphans 83
```

## 進捗

- [x] P0. データ取得・ダミー採点
- [x] P1. EDA・事前仮説
- [x] P2. リーク経路確認・分割固定
- [x] P3. baseline固定・初回Late
- [x] P4. 自力実験
- [x] P5. 上位解法調査
- [x] P6. 着眼点再現
- [x] P7. seed確認・最終固定・外部評価・エラー分析
- [x] P8. report・知見カード

## P1・P2 実データの確認

- 発話64,721行と背景録音6本から作ったsilence398窓、計65,119行。test158,538行。入力は全て16kHz。
- unknown41,039行、各コマンド2,353〜2,380行、silence398行。検証はこの構成比を保持する。
- 完全重複発話421行。重複ハッシュのラベル不一致は0件。同一WAVは話者とともに同じグループに置いた。最大グループ205行。
- train/testで共有するWAVハッシュ57種類（train57行・test3,765行）。56種類は非ゼロ音声。対応するtrainラベルをtestへ転記したり、予測を置換したりしていない。Late評価には完全に独立でない行が含まれる。
- 全ゼロの発話1件はbird（unknown）。元ラベルを維持した。音声全体のゼロ率はラベルの正しさを保証しない。
- seed42の各foldに全12クラスがあり、学習側と検証側のグループ共有は0。分割ハッシュは923df92f0ecaf862a47c13740077f28a863070b52342a130528a188050b44120。
- prepared/summary.jsonの背景音声の長さ・RMSは元録音の値。実際の学習入力は1秒窓。発話の長さなどの詳細はoutputs/eda_review.jsonに保存する。
- 配布train/README.mdと_background_noise_/README.mdを、話者IDと背景音声の利用法の一次資料として確認した。
- 発話者は1,881グループ。発話長は0.3715625〜1.0秒、平均0.984582秒。
- E01-benchmark（1fold・20epoch）は61.219秒。前処理・test推論込みの値を28foldへそのまま掛けても1,714秒程度で、4時間予算内と判断した。20epochを維持する。試走のAccuracyは選定に使わない。
- E01の初回LateはCOMPLETE。Public **0.79221** / Private **0.80277**。fold平均CV **0.856265294**。外部評価値は以後の構成選定に使わない。

## P5 上位解法調査（E02〜E04確定後）

| 解法 | 着眼点 | 一次資料と取得範囲 |
|---|---|---|
| 1位チームのSteffen Eberbach構成 | raw波形のdepthwise 1D CNN、蒸留、疑似ラベル、TTA。拡張でCVが改善してもLBは改善しない場合があった | [本人公開README](https://github.com/see--/speech_recognition)。README本文を取得。チーム全構成・学習コード全体は未実行 |
| 4位チームのdimart構成 | 31クラスで学習。追加の二語合成クラスと疑似ラベルで未知語に対応。話者と背景録音を分けて検証 | [本人公開README](https://github.com/dimart/TensorFlow-Speech-Recognition-Challenge-Solution)。README本文を取得。今回の直接の着眼点。学習コードの取得は失敗し、コード全体は未確認 |
| 10位 xuyuan・tugstugi | 画像用CNNを32×32 mel画像へ適用、STFT上で拡張。終了後のMixup実験は当時の解法と区別 | [本人公開README](https://github.com/tugstugi/pytorch-speech-commands)。README本文を取得。全モデルの再学習は未実施 |

最初に検索で見つかった4位解法のforkは出典にせず、fork元のdimartリポジトリへ遡った。転載・検索抜粋だけからの実装とは区別する。

## P6 事前登録（E05実行前）

- 比較元はE01。unknownの20語を細分化し、10コマンド＋silence＋20語の31クラスで学習する。推論では20語の確率を合算してunknownへ戻す。
- 仮説: 多様な語を単一unknownクラスへ押し込むより、語を識別させる補助教師情報が有効で、12クラスAccuracyが改善する。
- 変更は教師ラベルの細分化とそれに必要な出力層のみ。音声表現・標準化・20epoch・optimizer・12クラス単位の均等抽出は維持する。31クラス均等抽出には変えない。
- 固定した語彙は公式配布READMEの30語とsilence。外側の検証ラベルから語彙を作らない。testラベルや疑似ラベルは使わない。
- 元解法の二語合成・追加未知語クラス・TTA・大規模モデル・多段スタッキングは再現しない。未知の新語への対応そのものを検証したとは主張しない。
- 機械可読な登録はp5_registration.json。E01〜E05の最良候補1件だけを、直接の比較元とseed43で確認する。
- 最初のE05は2fold目の採点時、float32のunknown確率合算が1.0000001192092896となり停止した。失敗時のコード・ログ・事前登録をRunStoreへ保存し、metric=execution_success、値0として記録した。Accuracyは未確定で選定対象外。
- E05-r2はfloat64での合算・再正規化だけを修正した再実行。元の学習仮説・モデル・分割・設定は変えない。失敗の実測学習時間は取得できなかったためruntime_secは未記録とし、ログのファイル時刻差＋60秒の181秒を保守的な予算消費として別に計上する。

## seed42の実測結果

| run | 内容 | fold平均Accuracy | OOF Accuracy | 秒 |
|---|---|---:|---:|---:|
| E00 | 学習foldクラス頻度 | 0.630216 | 0.630215 | 11.7 |
| E01 | log-mel＋小型CNN | 0.856265 | 0.856263 | 138.3 |
| E02 | log線形周波数 | 0.851366 | 0.851349 | 156.3 |
| E03 | 時間移動 | 0.833897 | 0.833904 | 178.8 |
| E04 | 雑音混合 | 0.813623 | 0.813603 | 172.5 |
| E05-r2 | 31クラス学習→確率合算 | 0.896580 | 0.896574 | 159.8 |
| E01-random | 行単位分割の診断 | 0.715365 | 0.715367 | 161.2 |

- E05-r2はE01より全3foldで改善。差は+0.054731・+0.017088・+0.049125。foldは独立した再現ではない。
- OOF macro F1は0.809160→0.854574、LogLossは0.439210→0.324159。unknown Recallは0.844757→0.895075。ただしupは0.891368→0.831579へ悪化し、全クラスの改善ではない。
- 自力の3変更は全てfold平均で悪化。時間移動・雑音混合は固定20epochでの比較であり、学習時間・拡張強度を変えた場合は未検証。
- ランダム分割は話者を知る発話が各fold約99.90〜99.93%、背景録音共有は100%。主分割の共有は0%。今回はランダム分割のAccuracyが低く、差の原因を話者だけに帰属しない。
- seed43確認前にoutputs/candidate.jsonへE05-r2と比較元E01を固定した。

## 別seed確認・最終固定

| seed | E01 | 31クラス学習 | fold平均の差 |
|---|---:|---:|---:|
| 42 | 0.856265294 | 0.896579984（E05-r2） | +0.040314690 |
| 43 | 0.785056992 | 0.824944929（E05-s43） | +0.039887937 |

- 両seedで改善し、事前規則で**seed42のE05-r2・3fold確率平均**を採用。seed43は確認用で、最終予測へ混ぜない。
- seed43では1foldが-0.003129と悪化した。全6fold中5foldで改善。同じデータ内の確認であり、foldやseedを独立した外部再現とは扱わない。
- CVの絶対値はseed間で大きく変わる。モデル選定・精度推定の安定性に限界がある。
- 外部評価前にoutputs/selection.jsonを作成。提出sha256: 8f1852c75b519de8ee0a4f204dbe224881841c2a3abe4d1e37e56aed9570953f。
- 保存した全3foldの重みからtest予測を再生成し、確率の最大差0・提出ラベル完全一致を確認した。環境を変えた再学習のビット一致は保証しない。

再現・選定・分析コマンド:

```powershell
# P5の事前登録・実装後
uv run python competitions/083-tensorflow-speech-recognition-challenge/src/train.py E05-r2 --experiment E05
uv run python competitions/083-tensorflow-speech-recognition-challenge/src/select_model.py --freeze
uv run python competitions/083-tensorflow-speech-recognition-challenge/src/train.py E01-s43 --experiment E01 --seed 43
uv run python competitions/083-tensorflow-speech-recognition-challenge/src/train.py E05-s43 --experiment E05 --seed 43
uv run python competitions/083-tensorflow-speech-recognition-challenge/src/select_model.py
uv run python competitions/083-tensorflow-speech-recognition-challenge/src/predict.py E05-r2
uv run python competitions/083-tensorflow-speech-recognition-challenge/src/analyze.py E01 E05-r2
uv run python competitions/083-tensorflow-speech-recognition-challenge/src/split_diagnostic.py
uv run python competitions/083-tensorflow-speech-recognition-challenge/src/verify.py
```

同じrun IDや固定JSONを上書きしない。既存成果物からの予測再生成はpredict.py、新規の学習は別run IDを使う。7zの展開には7-Zipが必要。

## 最終外部評価・保存

- E05-r2のLateはCOMPLETE。Public **0.78371** / Private **0.79748**。E01比でPublic -0.00850、Private -0.00529。CVの改善は外部評価へ再現しなかった。最終固定後にモデルを選び直していない。
- 最終fold平均CVとPrivateの差は約0.09910。両seed・全foldで学習／検証の元ラベルは同じ31種で、未収録語への汎化は直接測っていない。
- CVのsilence比率は約0.611%。P0の定数silence Accuracyから分かるPrivate採点対象のsilence比率は約9.949%（表示の丸めあり）。少なくともクラス構成比が異なる。未収録語・音響条件・予測平均の寄与は未分離。
- seed42の音声長3区分・正規化RMS4分位では全区分でOOF改善したが、全語・全クラスで改善したわけではない。語・話者別の指標、混同行列、高損失30行はoutputs/analysis/に保存した。
- 12記録・120成果物のハッシュ、分割の再生成、保存予測からのCV再計算、選定済み提出ハッシュの照合に成功。未確定runなし。
- Late注記前ログはoutputs/evaluation_log_before_annotations.jsonl。更新したのはlate_score・conclusionだけで、元の計測値・成果物ハッシュは不変。
- 記録されたruntime合計1,377.47秒。失敗の保守的予算計上181秒を加えて約26.0分。GPU学習4時間の予算内。
- 最終dev check成功。lint・整形・pyrightエラー0、共通113件・#002 20件・#004 31件・#024 15件・#041 25件・#060 32件・#083 18件、計254テスト。
- 振り返りは[report.md](report.md)、知見は[K011](../../knowledge/K011-group-split-vs-random-split.md)・[K019](../../knowledge/K019-fine-grained-supervision-cv-vs-external.md)。
