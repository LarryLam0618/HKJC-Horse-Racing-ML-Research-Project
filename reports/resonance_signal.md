# 即場賠率共振訊號 — 規格(等數據儲夠先做)

**背景.** 成場市場效率調查(`reports/` 其餘檔案)嘅結論:靜態收市 SP 打唔穿。
剩返唯一有真實上升空間嘅槓桿係**即場資金流** —— 開閘前最後幾分鐘,大戶倒入去嘅
「啡燈資金」包含咗收市盤未 price 嘅資訊(臨場場地偏差、馬匹狀態、內部消息)。
`live_odds_snapshots` 由而家開始向前儲(`scripts/log_odds_day.sh` + launchd),
儲夠 **1–2 季**(≈ 150–300 個賽馬日)先夠樣本做以下嘅回測。

---

## 一、由 `live_odds_snapshots` 派生嘅特徵

view 每行 = 一場一個泳池一次刷新(dedup on `last_update_time`)。
每匹馬每場,由佢嘅時間序列計:

| 特徵 | 定義 | 直覺 |
|---|---|---|
| `odds_open` | 最早一個 snapshot 嘅 WIN 賠率 | 晨早盤 |
| `odds_close` | 最後一個(sell 關閉前)WIN 賠率 | = SP,同 `results.win_odds` 對齊 |
| `drift_full` | `log(odds_open / odds_close)` | 全程有冇被追捧(+ = 縮飛 = 被買) |
| `drift_last5` | `log(odds_5min_before_close / odds_close)` | **最後 5 分鐘**縮幾多 — 智慧資金窗口 |
| `drift_last1` | 最後 1 分鐘 | 最尖 |
| `steam` | `drift_last5 > k` 且成交/`odds_drop` 配合 | 「被隊」flag |
| `drift_vs_field` | 該馬 `drift_last5` 減全場中位 | 相對縮飛(全場一齊縮 = 冇資訊) |
| `late_reversal` | 晨早縮、臨場反彈(drift_full < 0 但 drift_last5 > 0) | 資金撤走 → 負訊號 |
| `pool_share_delta` | 該馬佔 WIN 池比例嘅尾段變化 | 直接睇資金流向 |
| `place_win_gap` | PLA 隱含機率 ÷ WIN 隱含機率 尾段變化 | 位置盤同獨贏盤背馳 = 內幕買位 |

> **注意 leakage.** 決策時間定喺收市前 T 分鐘(例如 T = 3)。所有特徵只可以用
> `snapshot_ts <= off_time − T` 嘅數據。`drift_last1` 呢啲要 T < 1 先用得 —— 實際上
> 落注 cutoff 要現實(閘前 2–3 分鐘),`drift_last3` 係穩陣上限。

---

## 二、共振規則(核心假設)

`late_rel3`(分段末段速度殘差,t≈3.2,見 `weapon23_integration`)係**基本面**side;
`drift_last3_vs_field`(相對尾段縮飛)係**資金流**side。兩個獨立。

**共振 = 兩邊同時指向同一隻馬。**

- 只喺「`late_rel3` 排場內前 3 **同時** `drift_last3_vs_field` 排場內前 3」嘅馬落注。
- 假設:呢個交集嘅命中率會**遠高**過任何一邊單獨 —— 因為兩個獨立訊號 confirm。
- 副作用:落注數自然收窄到一年幾十注,**但每注質素高** → 解到之前「樣本稀疏 +
  全部係 30-1 冷門」嘅死症。

備選 confirm 組合:`late_rel3` × `steam`、`hidden3` × `pool_share_delta↑`、
`bw_up_fresh`(體重回歸,t≈2.8,正交)× `drift_last3`。

---

## 三、回測方案(儲夠數據之後)

1. 用 `reports/w456.py` 個一體化條件 logit 出基本面 side 嘅場內機率 `p_fund_resid`
   (市場做 baseline control)。
2. 由 `live_odds_snapshots` 出上面啲 drift 特徵。
3. **決策 cutoff = 收市前 3 分鐘**。用嗰刻嘅賠率做 `q`,唔用 SP。
4. 分三組回測:
   - 只用基本面殘差(baseline,預期 ≈ 之前結果 −10%~打和)
   - 只用資金流 drift
   - **共振交集**(主假設)
5. 指標:ROI + 逐盤 bootstrap CI + 逐季 + 命中率 + 平均賠率 + 注數/年。
6. 通過門檻:共振組 ROI 顯著 > 0(bootstrap 5% 分位 > 0)**而且**逐季一致
   (唔可以好似 `final_three_steps` 咁一半年份 −80%)。

---

## 四、實際操作備忘

- **cutoff 現實啲**:HKJC 閘前約 2–3 分鐘停售。`drift_last1` 通常攞唔到,唔好靠。
- **polling 密度**:`scripts/log_odds_day.sh` 預設 90 秒一轉。想尾段密啲,
  收市前 15 分鐘應該手動或者改 wrapper 去 15–20 秒一轉(`HKJC_ODDS_INTERVAL`)。
  而家夠 `drift_last3`;要 `drift_last1` 就要收市前提到 10 秒。
- **`odds_drop` 欄**:gateway 自己俾嘅「跌幅」值 —— 可能已經係一個現成 steam 代理,
  儲落嚟一齊評估。
- **成交量**:whitelisted query **冇** per-pool investment(B3 未 capture)。
  只有 meeting-level `totalInvestment`。想要資金流量而唔只係賠率,要 recon B3
  (參考 CLAUDE.md trackwork 個 browser-recon 做法)。
- **儲幾耐先夠**:每個賽馬日 ~10 場。一季 ~88 日 ≈ 880 場。共振交集可能得每場
  ~0.3 注 → 一季 ~260 注。要 2 季 ~500 注先講得到 bootstrap CI。
