# #041 Jigsaw Toxic Comment Classification Challenge

- URL: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge
- カテゴリ: D / 難度: 2 / 計算負荷: S / 深さ: Standard
- テーマ: 単語・文字の疎な表現と軽量Transformer、多ラベル不均衡分類。
- 選定理由: 既存5件（表形式・時系列・画像）に対して、未経験のNLPを追加する。順位指標と確率評価（K014）も別条件で検証する。

Wikipediaのコメントから6種類の毒性を予測する。ラベルは排他的ではなく、各列を独立した確率として扱う。主評価は6列のROC AUCの平均（大きいほど良い）。補助指標はラベル別AUC・AP・LogLoss。閾値の最適化は行わない。

## P0・EDA・検証設計

- 公式データ取得済み。train 159,571行、test 153,164行。
- P0の定数提出は、RunStoreで予約・ハッシュ保存したP0-r2で採点を再確認済み（Public/Privateとも0.50000）。モデル選定には使用しない。最初のoutputs/p0の確認ファイルはRunStoreを通していなかったため、P0-r2で作り直した。旧ファイルは初回確認の証跡として残す。
- 陽性率: toxic 9.584%、severe_toxic 1.000%、obscene 5.295%、threat 0.300%、insult 4.936%、identity_hate 0.880%。
- NFKC・casefold・空白正規化後の重複258行。正規化した文面をグループとして、グループごとのラベルORを多ラベル層化3分割し、元の全行へ戻す。同一文面はfoldを跨がない。文字列正規化は重複検査用で、TF-IDF入力の追加クリーニングとは分ける。
- train/testで共通の正規化文面70種類。行は削除せず、testラベルを参照せず通常の推論をする。
- 文字長の中央値はtrain 205、test 180。trainの90%点889、99%点3444、最大5000。Transformerの256token切り捨ては長文での制約となる。
- 空白だけの文はtrain 0件、test 1件。test内の正規化文面重複177行。TF-IDFとTransformerのどちらも空文を通常の入力として予測し、有限値・形状・確率範囲を検査した。
- 単語2文字以上・小文字化した語彙はtrain 189,775種類、test 246,855種類、共通82,663種類。trainにない単語はtestの単語出現数の3.610%（EDA上の診断だけで、学習語彙の作成には使わない）。ラベル共起はoutputs/eda/label_cooccurrence.csv。
- 語彙・IDFは学習foldだけから作る。testを語彙作成に含めない。公開test_labels.csvは入力ファイル一覧に含めず、最終評価にも使わない。
- 投稿者・記事・時刻の列はない。同一文面以外の潜在的なグループ依存は除去できたと主張しない。
- 分割はseed 42、確認は43。分割を変える場合はsplit_versionも変更。foldはIDとともに成果物に保存する。

## 実行前に固定した仮説

最初の実験実行前にsrc/pipeline.pyへ登録した仮説を以下に示す。

| run | 比較元 | 単独変更 | 予想 |
|---|---|---|---|
| E00 | — | 学習foldのラベル別陽性率 | fold AUC 0.5の基準 |
| E01 | E00 | 単語unigram TF-IDF + LR | 単語だけで有効なbaselineになる |
| E02 | E01 | 単語bigram追加 | 短い文脈を拾い改善する |
| E03 | E01 | 文字char_wb 3〜5gram追加 | 綴りの変形に強くなる |
| E04 | E01 | LRのclass_weightをbalancedへ | 希少ラベルの順位改善の可能性と確率悪化を比較する |
| E05 | E01 | DistilBERTへモデル方式を変更 | 文脈を拾うが切り捨てと計算費用が増える |

TF-IDF: min_df=3、sublinear_tf、L2、float32。単語最大10万・文字最大15万。LR: C=1、liblinear、max_iter=1000。未収束はエラーで終了する。E03は2種類のL2正規化行列をそのまま結合し、結合後の再正規化はしない。

DistilBERT: distilbert/distilbert-base-uncased、最大256 token、6出力、BCEWithLogitsLoss。AdamW、lr=2e-5、weight_decay=0.01、batch=32、bf16、warmup=10%、2epoch固定。検証ラベルでepochを選ばない。RTX 4090上の40step測定では0.05275秒/step、余裕を含む3fold見積もり1473秒だったため2epochを採用した。計算時間の上限は4時間。事前学習重み・tokenizerのrevisionとファイルハッシュを記録する。

## 採否と最終評価の規則（結果を揃える前に固定）

- 主選定値はfold平均macro AUC。全体OOFも併記するが選定には使わない。
- P4後に異なる上位解法2〜3件を調査し、出典で確認できた着眼点1件を比較元への単独変更として事前登録する。
- 最良候補と直接の比較元をseed43で再実行。seed42・43の両方で改善したら採用し、そうでなければ比較元を維持。同点は短い実行時間を優先。
- seed42の採用モデル・3fold確率平均を最終提出として固定する。Privateは選択に使わない。AUCに行損失平均用bootstrapを流用しない。
- 選定結果はoutputs/selection.jsonへ提出ファイルのハッシュとともに保存し、上書きしない。

## 実行方法

```powershell
dev download 41
uv run python competitions/041-jigsaw-toxic-comment-classification-challenge/src/eda.py
uv run python competitions/041-jigsaw-toxic-comment-classification-challenge/src/train.py E01
uv run python competitions/041-jigsaw-toxic-comment-classification-challenge/src/transformer.py --benchmark
uv run python competitions/041-jigsaw-toxic-comment-classification-challenge/src/train.py E05
uv run python competitions/041-jigsaw-toxic-comment-classification-challenge/src/train.py E01-s43 --experiment E01 --seed 43
uv run python competitions/041-jigsaw-toxic-comment-classification-challenge/src/analyze.py E01 E02 E03 E04 E05
uv run python competitions/041-jigsaw-toxic-comment-classification-challenge/src/blend.py E06
uv run python competitions/041-jigsaw-toxic-comment-classification-challenge/src/blend.py E06-s43 --seed 43
uv run python competitions/041-jigsaw-toxic-comment-classification-challenge/src/select_model.py E06 E05
uv run python competitions/041-jigsaw-toxic-comment-classification-challenge/src/analyze.py E06 --errors
dev check
```

再実行は新しいrun_idを使う。全学習はRunStore.reserveの中で実行し、publish_cv_runでOOF・test・分割・指標・提出ファイルを一体保存する。データ・分割・コード・成果物をハッシュで識別する。モデルとデータはgit管理外。

## 進捗

- [x] P0. 取得・ダミー提出の採点確認
- [x] P1. EDA・事前仮説
- [x] P2. リーク経路確認・分割固定
- [x] P3. baseline固定
- [x] P4. 自力実験
- [x] P5. 異なる上位解法2〜3件の調査
- [x] P6. 着眼点の再現
- [x] P7. seed確認・最終モデル固定・外部評価・エラー分析
- [x] P8. reportと知見カード

実測値はexperiments.jsonlを正とする。進行中の結果は学習が完了してから追記する。


## 上位解法の調査（P5）

自力のE02〜E04完了後に調査した。E05は調査前に設定を固定して開始済みで、調査後も設定を変えていない。

| 順位 | 着眼点 | 出典と取得範囲 |
|---|---|---|
| 2位 neongen / Computer says no | RNN・DPCNN・GBMの多様性。複数の事前学習埋め込み、往復翻訳によるtrain/test augmentation、約30モデルの平均 | [本人writeup](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/writeups/neongen-computer-says-no-2nd-place-solution-overvi)。通常のページ取得は本文が空だったため検索サービスが返した索引本文による確認。公開コードは未確認 |
| 27位 Sotoxic | 単語と文字のモデルを平均。綴りの変形に対する別の見方を足す。別手段として品詞埋め込み・bad_comment補助ラベル | [本人公開コードのReadme](https://github.com/zake7749/DeepToxic/blob/master/Readme.md)を直接取得。単語・文字の平均は本人が明記している |
| 34位 Peter Hurford | 文字・単語TF-IDFのLR、文字数・大文字比率・句読点・品詞などの特徴量、複数のNNとLightGBMによる統合 | [本人公開コード](https://github.com/peterhurford/kaggle-toxic_comment)。README、feature_engineering.py、lr.pyを直接確認。元の全体解法は再実行していない |

当時のモデル・データ加工・CVの結果をそのまま転用しない。特に公開コードの全データ語彙作成や後段のOOF統合は、今回のfold内学習・ラベル独立性の規約に合わせる必要がある。

## 上位解法の着眼点の再現（P6、E06実行前の登録）

- E06: E05（DistilBERT）のOOF・test確率と、E03（単語＋文字TF-IDF）の確率を1:1で平均する。比較元はE05、追加するのは固定平均の後処理だけ。
- 仮説: 単語の文脈を読むモデルと文字の表現を使うモデルで誤り方が異なり、平均するとE05のfold平均macro AUCを改善する。
- 27位の文字/単語モデルの平均という原理の小規模な再現であり、元のRNN構造・埋め込み・スコアの忠実再現ではない。E03には単語特徴も含む。
- 係数0.5は実行前に固定。ラベルを使って係数・ラベル別の重みを学習しないため、追加の内側CVは要らない。元runは同じデータ・分割・seed・ラベル順であることを検査し、予測と分割の成果物ハッシュを照合する。
- E03/E05/E06のseed42で最も良い構成を候補とし、その直接の比較元とのseed43ペア比較で採否を決める。E06の比較元はE05、E03/E05の比較元はE01。候補が採用規則を満たさなければ直接の比較元を維持する。複数候補を順に試して採用を積み重ねない。


## 実測結果（seed42）

| run | 内容 | fold平均macro AUC | OOF macro AUC | ラベル平均OOF LogLoss | 実行秒 |
|---|---|---:|---:|---:|---:|
| E00 | ラベル別陽性率 | 0.500000 | 0.499539 | 0.141069 | 6 |
| E01 | 単語unigram + LR | 0.979929 | 0.979914 | 0.054970 | 80 |
| E02 | +単語bigram | 0.978583 | 0.978568 | 0.058175 | 180 |
| E03 | +文字3〜5gram | 0.985127 | 0.985120 | 0.048473 | 714 |
| E04 | balanced重み | 0.981131 | 0.981112 | 0.106705 | 81 |
| E05 | DistilBERT | 0.990429 | 0.990363 | 0.037216 | 1073 |
| E06 | E05とE03を1:1平均 | 0.990703 | 0.990689 | 0.038606 | 3（元モデルの学習時間は別） |

E00の各foldのAUCは0.5だが、foldによって学習側の陽性率が違うため、OOF全体の順位は完全な同点にはならない。採否は事前規則どおりfold平均で判断する。

- bigram追加は全3foldで悪化。文字追加は全3foldで改善。文字の効果を綴りの変形への頑健性だけで説明できたわけではない。
- balanced重みはAUCを小幅に改善する一方、6ラベルすべてでLogLossが悪化した。確率が陽性側へ寄る影響と順位の変化を分けて見る（K014）。
- DistilBERTはモデル・事前学習・表現・共有分類層をまとめて変えた比較であり、改善を文脈利用だけの因果効果と解釈しない。2018年当時の解法の忠実再現ではなく、現在利用できる事前学習モデルとの比較。
- E06は2/3foldでE05を改善、1foldで悪化。threatのOOF AUC +0.002641・AP +0.039581の寄与が大きく、3ラベルでAUCは悪化、全ラベルでLogLossは悪化した。OOFのLogLossはE05より悪い。seed43でもfold平均AUCが改善し、事前規則で採用した。
- E01のLate: Public 0.97489 / Private 0.97462。CVより低い。外部評価は選定に使わない。


## エラー分析（外部ラベル不使用）

OOF予測を短文（100文字以下）・中程度・長文（1,000文字超）・非ASCII文字を含む文に分けて比較した。区分間でラベル構成が異なるため、区分間のAUCだけから難易度を断定しない。

| 区分 | 行数 | E01 | E05 | E06 |
|---|---:|---:|---:|---:|
| 100文字以下 | 42,052 | 0.975041 | 0.987919 | 0.988551 |
| 101〜1,000文字 | 104,159 | 0.982152 | 0.991242 | 0.991481 |
| 1,000文字超 | 13,360 | 0.982672 | 0.985282 | 0.989256 |
| 非ASCIIを含む | 17,215 | 0.977635 | 0.992161 | 0.993553 |

E06の改善は長文で大きい。先頭256tokenだけを見るE05に対して、全文章を使う文字TF-IDFが補う可能性がある。ただし切り捨ての有無だけを変えた比較ではなく、原因は分離できていない。train/testで共通の文面を持つtrain行76件は、一部ラベルが片クラスなのでmacro AUCを計算しない。最大損失の30件のIDとラベル別指標はoutputs/analysisに保存した。


## 最終選定（P7）

候補はseed42の主評価が最良だったE06。直接の比較元E05との差をseed43でも確認した。

| seed | E05 | E06 | 差 |
|---|---:|---:|---:|
| 42 | 0.990429 | 0.990703 | +0.000274 |
| 43 | 0.990776 | 0.990918 | +0.000142 |

両seedで改善したので、事前規則どおり**seed42のE06**（DistilBERTの3fold平均と文字・単語TF-IDF+LRの3fold平均を1:1で平均）を採用した。seed43の方がスコアは高いが、確認用なので提出には使わない。

提出前にoutputs/selection.jsonへ選定条件と提出sha256（c52fb64b7d70f1230523ab30f7046ae2a1c234e213c891c7e741d86ed6bb7bfe）を保存した。モデルと係数を固定してから外部評価へ提出した。

文字特徴の単独効果もseed43で確認済み（E01-s43 0.979964→E03-s43 0.985175、+0.005211）。同じコンペの別分割での支持であって、独立したデータでの再現ではない。


### 最終外部評価・検証

- E06 Late Submission: **Public 0.98572 / Private 0.98621**（COMPLETE）。baseline E01のPrivate 0.97462に対して+0.01159。
- CV（fold平均0.990703）よりPrivateは0.004493低い。baselineでも同じ方向の差があった（約0.00531）。分布差、未知のグループ依存、モデル選択、OOFとfold平均の違いの寄与は分離していない。Lateで改善したことはCVにリークがない証明ではない。
- E05単体は外部提出していないので、E06の平均処理だけの外部効果は未検証。上位解法の忠実な再現スコアや順位相当は主張しない。
- 選定時点のログsha256はselection.jsonに保存した。その後に許可されたlate_score・conclusionだけを追記し、計測値と成果物は変更していない。Public/Privateの採点結果はoutputs/submissions.txtにも保存した。
- dev check成功: lint・整形・型検査、共通113件・#002 20件・#024 15件・#041 20件・#060 32件、合計200テスト。外部モデル取得やGPUを必要としないCPUテストで検証した。
- dev orphans 41で未確定run・一時ディレクトリなし。データ・予測・提出ファイルはgit管理外。
