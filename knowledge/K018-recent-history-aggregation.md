# K018: 全履歴に直近期間の集約を加える

- 状態: 仮説
- 分類: 特徴量 / 時間窓

## 仮説

全期間の履歴統計に、同じ統計量を直近期間へ限定した特徴を追加すると、最近の状態を補える。

## 観測

Home Creditで全期間bureau集約のE02へ直近365日集約だけを加えたE06は、fold平均AUCがseed42で+0.000787955、seed43で+0.001562459。それぞれ全3foldで改善した。別seedは同じデータ内の確認であり、別コンペでの再現ではない。

## 適用条件

顧客内の他社借入履歴、DAYS_CREDITが[-365,0]、更新日が申込時点以前。全期間集約を維持し、同じ集約群を追加した。LightGBM・層化3fold・学習設定は同じ。365日は2位本人公開READMEの1年集約に基づき事前固定し、期間探索していない。

## 未検証

他期間、別テーブル、全期間集約の置換、異なるモデル・データでの効果。改善を最近の行の情報だけに帰属できるかは未確定。E02の外部提出がないため、直近集約単独の外部効果は未検証。最終E06のPrivate 0.75557は全体構成の評価である。

## 次の検証機会

Elo・American Expressなどで、全履歴を保持したまま期間集約の追加を比較する。期間を変える場合は事前登録して比較回数を制限する。

## 証跡

| コンペ | run | 結果 |
|---|---|---|
| #004 Home Credit | E02→E06 | seed42 AUC 0.762628404→0.763416359 |
| #004 Home Credit | E02-s43→E06-s43 | seed43 AUC 0.762301200→0.763863659 |

| [#015 Elo](../competitions/015-elo-merchant-category-recommendation/report.md) | E01 → E02 / E05 | 直近1か月集約で OOF RMSE −0.00722、全14月の件数で −0.05761 |

[2位本人公開README](https://github.com/KazukiOnodera/Home-Credit-Default-Risk) / [実験ログ](../competitions/004-home-credit-default-risk/experiments.jsonl)
