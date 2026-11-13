# Kaggle 100本ノック — 賞金付き本番コンペ・学習カタログ v1.0

基準日: 2026-11-14。Getting Started／Playground／Community／Recruitmentを除外し、過去に賞金が設定されたFeatured／Researchの終了済みコンペから選定。各コンペのURLはKaggleのOverviewに対応する。すべてのページでログイン後のLate Submission実スコアリングを個別に完走確認したものではない（コンペごとにスモークテストを必須とする）。Prizeは**過去の開催時の賞金**を意味し、Late Submissionで受け取れるものではない。

**凡例**: 難易度1–5（主観的な学習上の難しさ）、計算負荷 S=軽め M=中 L=大 XL=相当に大きい。これらは目安で、データ量・モデル選択で大幅に変わる。

**学習完了条件**: EDA（欠損・分布・目的変数・リーク） / 妥当なCV / baseline / 単一要因ablation 3回以上 / エラー分析 / 可能な場合はLate Submission 1回以上 / 終了後の解法との比較 / A4 1ページ程度の振り返り。

**Late Submission事前確認**: ①公式OverviewにLate Submission表示 ②Rules承諾後にDataが読める ③正しい提出方法（File/Code/推論API）を確認 ④最小baselineでスコアが返る ⑤ダメなら評価用独立holdoutを固定して代替コンペへ回す。特に2020年頃以前のKernels-onlyや現在の推論APIでは挙動差の可能性がある。


## A. 表形式・ビジネス予測（18）

| # | コンペ | 今回の比較実験 | 難度 | 負荷 |
|---:|---|---|:---:|:---:|
| 001 | [Otto Group Product Classification Challenge](https://www.kaggle.com/competitions/otto-group-product-classification-challenge) | 多クラス分類：LogLoss・モデルの較正 | 2 | S |
| 002 | [Porto Seguro’s Safe Driver Prediction](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction) | 不均衡二値分類：GiniとAUC・欠損符号 | 2 | S |
| 003 | [Santander Customer Transaction Prediction](https://www.kaggle.com/competitions/santander-customer-transaction-prediction) | 匿名特徴：分布の異常・変数ごとの効果 | 3 | M |
| 004 | [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) | リレーショナル集約・顧客単位CV | 3 | M |
| 005 | [Home Credit - Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability) | 時点を跨ぐ頑健性・安定性指標 | 4 | L |
| 006 | [American Express - Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction) | 履歴集約と系列モデルの比較 | 4 | L |
| 007 | [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) | 不正検出：ID・時間ドリフト・リーク | 4 | L |
| 008 | [Microsoft Malware Prediction](https://www.kaggle.com/competitions/microsoft-malware-prediction) | 巨大カテゴリ列とメモリ節約 | 3 | L |
| 009 | [TalkingData AdTracking Fraud Detection Challenge](https://www.kaggle.com/competitions/talkingdata-adtracking-fraud-detection) | クリックログの頻度・時刻集約 | 4 | L |
| 010 | [BNP Paribas Cardif Claims Management](https://www.kaggle.com/competitions/bnp-paribas-cardif-claims-management) | 匿名数値・カテゴリ欠損処理 | 2 | M |
| 011 | [Allstate Claims Severity](https://www.kaggle.com/competitions/allstate-claims-severity) | 右裾の長い目的変数と損失関数 | 2 | M |
| 012 | [Mercedes-Benz Greener Manufacturing](https://www.kaggle.com/competitions/mercedes-benz-greener-manufacturing) | 小標本回帰・高カーディナリティ | 2 | S |
| 013 | [Bosch Production Line Performance](https://www.kaggle.com/competitions/bosch-production-line-performance) | 工場ログ・希少イベント・工程系列 | 4 | L |
| 014 | [Zillow Prize: Zillow’s Home Value Prediction](https://www.kaggle.com/competitions/zillow-prize-1) | 不動産回帰・位置特徴・ドメイン分割 | 3 | M |
| 015 | [Elo Merchant Category Recommendation](https://www.kaggle.com/competitions/elo-merchant-category-recommendation) | 履歴特徴集約・外れ値への耐性 | 3 | M |
| 016 | [PetFinder.my Adoption Prediction](https://www.kaggle.com/competitions/petfinder-adoption-prediction) | 順序分類・画像と表の融合 | 3 | M |
| 017 | [Riiid Answer Correctness Prediction](https://www.kaggle.com/competitions/riiid-test-answer-prediction) | 学習履歴の時系列特徴・逐次推論 | 5 | L |
| 018 | [2019 Data Science Bowl](https://www.kaggle.com/competitions/data-science-bowl-2019) | ゲームイベント集約・QWKと順序分類 | 4 | M |

## B. 時系列・金融・予測（14）

| # | コンペ | 今回の比較実験 | 難度 | 負荷 |
|---:|---|---|:---:|:---:|
| 019 | [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales) | 店別売上・祝日・時系列CV | 2 | S |
| 020 | [Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting) | 大規模小売・商品階層・販促効果 | 4 | L |
| 021 | [M5 Forecasting - Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) | 階層時系列・ラグ・WRMSSE | 4 | L |
| 022 | [M5 Forecasting - Uncertainty](https://www.kaggle.com/competitions/m5-forecasting-uncertainty) | 分位点予測・予測区間と較正 | 5 | L |
| 023 | [ASHRAE - Great Energy Predictor III](https://www.kaggle.com/competitions/ashrae-energy-prediction) | 建物×気象・欠損・時差とリーク | 3 | L |
| 024 | [Recruit Restaurant Visitor Forecasting](https://www.kaggle.com/competitions/recruit-restaurant-visitor-forecasting) | 予約・来店・休日の系列結合 | 3 | M |
| 025 | [Optiver Realized Volatility Prediction](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction) | 板情報・秒単位の特徴量・RV | 4 | L |
| 026 | [Optiver - Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close) | オンライン推論API・時系列検証 | 5 | L |
| 027 | [Jane Street Market Prediction](https://www.kaggle.com/competitions/jane-street-market-prediction) | 期待効用・重み付き目的関数 | 5 | L |
| 028 | [Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting) | 非定常分布・遅延ラベル・オンラインAPI | 5 | XL |
| 029 | [Ubiquant Market Prediction](https://www.kaggle.com/competitions/ubiquant-market-prediction) | 金融の分布変化・相関係数最適化 | 5 | L |
| 030 | [G-Research Crypto Forecasting](https://www.kaggle.com/competitions/g-research-crypto-forecasting) | 暗号資産・時系列評価API・ドリフト | 5 | L |
| 031 | [LANL Earthquake Prediction](https://www.kaggle.com/competitions/LANL-Earthquake-Prediction) | 信号窓の統計量・時系列リーク防止 | 4 | L |
| 032 | [MITSUI&CO. Commodity Prediction Challenge](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge) | コモディティ・時系列安定性・ラグ制約 | 5 | XL |

## C. 推薦・検索・ランキング（8）

| # | コンペ | 今回の比較実験 | 難度 | 負荷 |
|---:|---|---|:---:|:---:|
| 033 | [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) | 二段階推薦・候補生成とMAP@12 | 4 | XL |
| 034 | [Santander Product Recommendation](https://www.kaggle.com/competitions/santander-product-recommendation) | 顧客別多ラベル・既存保有商品の除外 | 3 | M |
| 035 | [WSDM - KKBox's Music Recommendation Challenge](https://www.kaggle.com/competitions/kkbox-music-recommendation-challenge) | ユーザー×楽曲CTRと負例生成 | 3 | M |
| 036 | [WSDM - KKBox's Churn Prediction Challenge](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge) | 解約予測・契約更新系列 | 3 | M |
| 037 | [Shopee - Price Match Guarantee](https://www.kaggle.com/competitions/shopee-product-matching) | 画像＋文書の埋め込み・類似検索 | 4 | L |
| 038 | [Google Landmark Retrieval 2020](https://www.kaggle.com/competitions/landmark-retrieval-2020) | Metric Learning・近傍検索・mAP | 5 | XL |
| 039 | [Google Landmark Recognition 2020](https://www.kaggle.com/competitions/landmark-recognition-2020) | 大規模分類・未知クラス拒否 | 5 | XL |
| 040 | [OTTO – Multi-Objective Recommender System](https://www.kaggle.com/competitions/otto-recommender-system) | セッション推薦・Recall@Kと複数目的 | 5 | XL |

## D. NLP・LLM（18）

| # | コンペ | 今回の比較実験 | 難度 | 負荷 |
|---:|---|---|:---:|:---:|
| 041 | [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) | TF-IDF vs Transformer・多ラベル毒性分類 | 2 | S |
| 042 | [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification) | 部分集団での公平性・バイアス評価 | 4 | M |
| 043 | [Jigsaw Multilingual Toxic Comment Classification](https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification) | 多言語・翻訳・ドメイン適応 | 4 | L |
| 044 | [Jigsaw Rate Severity of Toxic Comments](https://www.kaggle.com/competitions/jigsaw-toxic-severity-rating) | ペアワイズ順位学習 | 4 | M |
| 045 | [Quora Question Pairs](https://www.kaggle.com/competitions/quora-question-pairs) | 重複質問・文字列類似・二塔モデル | 2 | M |
| 046 | [Quora Insincere Questions Classification](https://www.kaggle.com/competitions/quora-insincere-questions-classification) | 不均衡テキスト分類・閾値選択 | 3 | M |
| 047 | [Google QUEST Q&A Labeling](https://www.kaggle.com/competitions/google-quest-challenge) | 多目的回帰・順位相関 | 4 | M |
| 048 | [Tweet Sentiment Extraction](https://www.kaggle.com/competitions/tweet-sentiment-extraction) | span抽出・token位置復元 | 3 | M |
| 049 | [CommonLit Readability Prize](https://www.kaggle.com/competitions/commonlitreadabilityprize) | 文章の難易度回帰・小標本CV | 3 | M |
| 050 | [CommonLit - Evaluate Student Summaries](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries) | 文章評価回帰・文章長の影響 | 3 | M |
| 051 | [Feedback Prize - Evaluating Student Writing](https://www.kaggle.com/competitions/feedback-prize-2021) | 談話単位系列タグ付け・F1 | 4 | L |
| 052 | [Feedback Prize - Predicting Effective Arguments](https://www.kaggle.com/competitions/feedback-prize-effectiveness) | 論証分類・事前抽出結果との接続 | 3 | M |
| 053 | [Feedback Prize - English Language Learning](https://www.kaggle.com/competitions/feedback-prize-english-language-learning) | 複数連続値・MCRMSE・CPU効率 | 3 | M |
| 054 | [U.S. Patent Phrase to Phrase Matching](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching) | 意味類似・文脈依存・相関評価 | 3 | M |
| 055 | [NBME - Score Clinical Patient Notes](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes) | 臨床ノートから根拠span抽出 | 4 | M |
| 056 | [LLM - Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text) | 生成文検出・ドメイン汎化 | 4 | L |
| 057 | [Kaggle - LLM Science Exam](https://www.kaggle.com/competitions/kaggle-llm-science-exam) | RAG・検索文書と選択肢スコア | 4 | L |
| 058 | [Eedi - Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics) | 検索ランキング・誤概念の候補生成 | 5 | L |

## E. Computer Vision（24）

| # | コンペ | 今回の比較実験 | 難度 | 負荷 |
|---:|---|---|:---:|:---:|
| 059 | [Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification) | 画像分類・転移学習・TTA | 2 | M |
| 060 | [State Farm Distracted Driver Detection](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection) | 運転手単位GroupCVと見かけの高精度 | 3 | M |
| 061 | [Planet: Understanding the Amazon from Space](https://www.kaggle.com/competitions/planet-understanding-the-amazon-from-space) | 衛星画像の多ラベル分類・F2閾値 | 3 | M |
| 062 | [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge) | 医療画像物体検出・bboxとmAP | 4 | L |
| 063 | [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection) | CT多ラベル・患者単位CV | 4 | L |
| 064 | [RSNA Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection) | 高解像度画像・強い不均衡 | 5 | XL |
| 065 | [RSNA 2022 Cervical Spine Fracture Detection](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection) | 3D CT・スライス集約 | 5 | XL |
| 066 | [RSNA 2023 Abdominal Trauma Detection](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection) | CT系列・患者レベル評価 | 5 | XL |
| 067 | [RSNA 2024 Lumbar Spine Degenerative Classification](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification) | MRI・部位別多クラス・重み付きLogLoss | 5 | XL |
| 068 | [SIIM-ISIC Melanoma Classification](https://www.kaggle.com/competitions/siim-isic-melanoma-classification) | DICOM・画像＋メタデータ融合 | 3 | M |
| 069 | [ISIC 2024 - Skin Cancer Detection with 3D-TBP](https://www.kaggle.com/competitions/isic-2024-challenge) | 極端な不均衡・pAUC・Tabular融合 | 4 | L |
| 070 | [TGS Salt Identification Challenge](https://www.kaggle.com/competitions/tgs-salt-identification-challenge) | 二値セグメンテーション・IoU閾値 | 3 | M |
| 071 | [Carvana Image Masking Challenge](https://www.kaggle.com/competitions/carvana-image-masking-challenge) | 精緻なマスク・U-Net・TTA | 3 | L |
| 072 | [Severstal: Steel Defect Detection](https://www.kaggle.com/competitions/severstal-steel-defect-detection) | 工場欠陥・分類→セグメント | 4 | L |
| 073 | [Airbus Ship Detection Challenge](https://www.kaggle.com/competitions/airbus-ship-detection) | 航空写真・小物体・RLE | 4 | L |
| 074 | [Global Wheat Detection](https://www.kaggle.com/competitions/global-wheat-detection) | 物体検出・ラベル品質とNMS | 3 | M |
| 075 | [VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection) | 多疾患検出・複数アノテータ | 5 | XL |
| 076 | [TensorFlow - Help Protect the Great Barrier Reef](https://www.kaggle.com/competitions/tensorflow-great-barrier-reef) | 動画連続フレームと魚群検出 | 4 | L |
| 077 | [Sartorius - Cell Instance Segmentation](https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation) | 細胞インスタンス分割・重なり | 4 | L |
| 078 | [HuBMAP + HPA - Hacking the Human Body](https://www.kaggle.com/competitions/hubmap-organ-segmentation) | 臓器組織切片・高解像度分割 | 4 | L |
| 079 | [HuBMAP - Hacking the Human Vasculature](https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature) | 微小血管・インスタンス分割 | 5 | XL |
| 080 | [UW-Madison GI Tract Image Segmentation](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation) | MRIスライス間文脈・3クラス分割 | 4 | M |
| 081 | [Happywhale - Whale and Dolphin Identification](https://www.kaggle.com/competitions/happy-whale-and-dolphin) | 個体識別・ArcFace・未知個体 | 5 | L |
| 082 | [Google Research - Identify Contrails to Reduce Global Warming](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming) | 衛星時系列画像・細線領域分割 | 4 | L |

## F. 音声・波形・センサ（10）

| # | コンペ | 今回の比較実験 | 難度 | 負荷 |
|---:|---|---|:---:|:---:|
| 083 | [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge) | 音声コマンド・log-mel・CNN | 2 | S |
| 084 | [Freesound Audio Tagging 2019](https://www.kaggle.com/competitions/freesound-audio-tagging-2019) | 環境音・弱いラベル・複数タグ | 4 | M |
| 085 | [Rainforest Connection Species Audio Detection](https://www.kaggle.com/competitions/rfcx-species-audio-detection) | 野生生物・ノイズ・低頻度イベント | 4 | M |
| 086 | [BirdCLEF 2021 - Birdcall Identification](https://www.kaggle.com/competitions/birdclef-2021) | 鳥声・音源混在・窓分割 | 3 | M |
| 087 | [BirdCLEF 2023](https://www.kaggle.com/competitions/birdclef-2023) | 録音場所・ドメインシフト・閾値 | 4 | L |
| 088 | [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) | 動物横断の多ラベル音響認識・推論制約 | 5 | L |
| 089 | [HMS - Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification) | EEG・スペクトログラム・ソフトラベル | 4 | L |
| 090 | [Child Mind Institute - Detect Sleep States](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states) | 加速度系列・イベント検出AP | 4 | M |
| 091 | [Parkinson's Freezing of Gait Prediction](https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction) | ウェアラブル系列・イベント開始／終了 | 4 | M |
| 092 | [G2Net Gravitational Wave Detection](https://www.kaggle.com/competitions/g2net-gravitational-wave-detection) | 物理信号・雑音・スペクトル変換 | 5 | L |

## G. 科学・分子・最適化（8）

| # | コンペ | 今回の比較実験 | 難度 | 負荷 |
|---:|---|---|:---:|:---:|
| 093 | [Mechanisms of Action (MoA) Prediction](https://www.kaggle.com/competitions/lish-moa) | 化合物応答・多ラベルNN・LogLoss | 3 | M |
| 094 | [OpenVaccine: COVID-19 mRNA Vaccine Degradation Prediction](https://www.kaggle.com/competitions/stanford-covid-vaccine) | RNA配列・二次構造・GNN | 4 | M |
| 095 | [Open Problems - Multimodal Single-Cell Integration](https://www.kaggle.com/competitions/open-problems-multimodal) | 遺伝子・タンパク質・次元削減 | 5 | XL |
| 096 | [Open Problems – Single-Cell Perturbations](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations) | 摂動の外挿・細胞型一般化 | 5 | XL |
| 097 | [Stanford Ribonanza RNA Folding](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding) | RNA配列の位置別予測・構造特徴 | 5 | XL |
| 098 | [NeurIPS 2024 - Predict New Medicines with BELKA](https://www.kaggle.com/competitions/leash-BELKA) | 分子表現・ECFP vs GNN | 4 | L |
| 099 | [Vesuvius Challenge - Ink Detection](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection) | 3D X線・体積パッチ・微弱信号分割 | 5 | XL |
| 100 | [Santa 2023 - The Polytope Permutation Puzzle](https://www.kaggle.com/competitions/santa-2023) | 群・探索・最適化アルゴリズム | 5 | M |

## おすすめの最初の10本（カタログ番号）

1. #001 Otto Group — 多クラス、LogLoss、基本的CV
2. #012 Mercedes-Benz — 小標本・過学習と回帰
3. #002 Porto Seguro — 不均衡・AUCと特徴量エンジニアリング
4. #019 Rossmann — 時系列CV、ラグ・休日
5. #041 Jigsaw Toxic — TF-IDF と Transformerの比較
6. #059 Cassava — 転移学習・TTA
7. #083 TensorFlow Speech — 音声特徴・スペクトログラム
8. #033 H&M — 候補生成とランキング（縮小版）
9. #070 TGS Salt — U-Netとセグメンテーション指標
10. #093 MoA — 多ラベル・確率較正

## 1題あたりの進行テンプレート

- **P0 参加可否**: ルールとデータの確認、最小ダミー提出、提出IDとスコアの記録。
- **P1 理解**: 目的変数、予測対象、評価指標を図示。train/test/validationの関係を整理。
- **P2 EDA**: ターゲット分布、特徴量の欠損・分布・相関、時刻・ID・群、外れ値、train/testシフト。
- **P3 CV**: Random/KFold/Stratified/Group/TimeSeries等からリークを避けた方法を選ぶ。ホールドアウトは最後まで封印。
- **P4 Baseline**: 統計的ベースラインと、代表モデル1つを固定seedで実行。
- **P5 Ablations**: 単一変更を最低3つ、各検証foldの結果・時間・メモリを記録。
- **P6 Kaggle submit**: 頻度を抑え、CV-LBギャップの診断。公開スコアをチューニング目的にはしない。
- **P7 事後調査**: 終了後の上位解法を見て差分を1つずつ再現し、真の改善かアンサンブルによる改善かを区別。
- **P8 振り返り**: 再利用できる知見、失敗した仮説、再現可能なコードを残す。

## 実験管理項目

`competition_id, run_id, timestamp, data_hash, split_version, seed, model, parameters, changed_one_thing, fold_scores, mean, std, oof_score, late_score, time_sec, peak_vram_gb, conclusions, leakage_risks, code_commit`

**推奨**: パイプラインと特徴量定義、split定義、OOF予測とテスト予測をハッシュ付きで保存。train/testまたぎのfit、ターゲット符号化のfold外利用、ユーザー/患者重複などはnegative testで拒否。スコアは現行のコード・データバージョンの結果だけを扱う。
