"""武器2+3 跟進:整合模型 EV 篩選策略嘅細節 + 穩健性檢查."""
from __future__ import annotations
import numpy as np, pandas as pd, json
from scipy.optimize import minimize
SD = "/private/tmp/claude-501/-Users-suetyinglam-Downloads-HKJC-Horse-Racing-ML-Research-Project-main/81331658-de3f-4006-8783-feab579385e5/scratchpad"

# rebuild feature frame (same as w23.py) ---------------------------------------
res = pd.read_csv("results.csv", low_memory=False)
races = pd.read_csv("races.csv", low_memory=False)
res = res[res.race_date >= "2008-04-01"].copy()
res["dt"] = pd.to_datetime(res.race_date)
res["rid"] = res.race_date + "_" + res.venue + "_" + res.race_no.astype(str)
res["saddle"] = res.saddle.astype(str)
res["fin"] = pd.to_numeric(res.finish_pos, errors="coerce")
res = res.dropna(subset=["fin"]); res["fin"] = res.fin.astype(int)
races["rid"] = races.race_date + "_" + races.venue + "_" + races.race_no.astype(str)
res = res.merge(races[["rid", "distance_m"]], on="rid", how="left")
fs = res.groupby("rid").saddle.transform("size")
res["field_size"] = fs
res["win"] = (res.fin == 1).astype(int)
res = res.sort_values(["dt", "rid", "saddle"]).reset_index(drop=True)

sec = pd.read_csv("sectionals_clean.csv", low_memory=False)
sec["rid"] = sec.race_date + "_" + sec.venue + "_" + sec.race_no.astype(str)
sec["saddle"] = sec.saddle.astype(str)
sec = sec.dropna(subset=["section_index", "section_time_s"])
sec["section_index"] = sec.section_index.astype(int)
last = sec.loc[sec.groupby(["rid", "saddle"]).section_index.idxmax()][
    ["rid", "saddle", "section_time_s", "finishing_order"]].rename(columns={"section_time_s": "last400"})
early = sec[sec.section_index == 1][["rid", "saddle", "running_position"]].rename(
    columns={"running_position": "early_pos"})
run = last.merge(early, on=["rid", "saddle"], how="left")
run["late_rel"] = run.groupby("rid").last400.transform("median") - run.last400
run["hidden"] = ((run.finishing_order >= 6) & (run.late_rel >= 0.4)).astype(int)
run = run.merge(res[["rid", "saddle", "dt", "horse_id"]], on=["rid", "saddle"], how="left")
run = run.dropna(subset=["dt"]).sort_values("dt")
run["late_rel_mean3"] = (run.groupby("horse_id").late_rel.shift(1)
                         .groupby(run.horse_id).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True))
run["hidden_cnt3"] = (run.groupby("horse_id").hidden.shift(1)
                      .groupby(run.horse_id).rolling(3, min_periods=1).sum().reset_index(level=0, drop=True))
res = res.merge(run[["rid", "saddle", "late_rel_mean3", "hidden_cnt3"]], on=["rid", "saddle"], how="left")

res["draw"] = pd.to_numeric(res.draw, errors="coerce")
res["draw_pct"] = res.draw / res.field_size
res["mkt_raw"] = 1 / res.win_odds
res["mkt_p"] = res.mkt_raw / res.groupby("rid").mkt_raw.transform("sum")
res["season_y"] = res.race_date.map(lambda s: int(s[:4]) if int(s[5:7]) >= 9 else int(s[:4]) - 1)
res["ret1"] = np.where(res.win == 1, res.win_odds, 0.0)

d = res.dropna(subset=["mkt_p", "win_odds"]).copy()
d = d[d.race_date >= "2011-09-01"]
FACT = ["late_rel_mean3", "hidden_cnt3", "draw_pct"]     # only the clean positive-residual signals
for c in FACT:
    x = pd.to_numeric(d[c], errors="coerce"); d[c + "_z"] = ((x - x.mean()) / x.std()).fillna(0.0)
Z = [c + "_z" for c in FACT]
d["lmkt"] = np.log(d.mkt_p.clip(1e-6))

def struct(sub):
    o = np.argsort(sub.rid.values, kind="stable"); r = sub.rid.values[o]
    st = np.r_[0, np.where(r[1:] != r[:-1])[0] + 1]
    ct = np.diff(np.r_[st, len(r)]); wn = sub.win.values[o]
    wr = np.full(len(st), -1)
    for i, (s, c) in enumerate(zip(st, ct)):
        seg = wn[s:s + c]
        if seg.any(): wr[i] = s + seg.argmax()
    return o, st, ct, wr
def fit(sub, zc):
    o, st, ct, wr = struct(sub); Zm = sub[zc].values[o]; lm = sub.lmkt.values[o]
    v = wr >= 0; wv = wr[v]
    def nll(t):
        s = t[0] * lm + Zm @ t[1:]
        sm = np.maximum.reduceat(s, st)
        e = np.exp(s - np.repeat(sm, ct))
        return float(np.sum((np.log(np.add.reduceat(e, st)) + sm)[v] - s[wv]))
    return minimize(nll, np.r_[1.0, np.zeros(len(zc))], method="L-BFGS-B").x

d = d.sort_values("season_y")
d["p_int"] = np.nan
for sy in range(2014, 2027):
    tr = d[d.season_y < sy]; te = d[d.season_y == sy]
    if len(te) == 0: continue
    th = fit(tr, Z)
    sc = th[0] * te.lmkt.values + te[Z].values @ th[1:]
    ex = np.exp(sc - pd.Series(sc, index=te.rid.values).groupby(level=0).transform("max").values)
    d.loc[te.index, "p_int"] = ex / pd.Series(ex, index=te.rid.values).groupby(level=0).transform("sum").values

bt = d.dropna(subset=["p_int"]).copy()
bt["ev"] = bt.p_int * bt.win_odds - 1
print(f"OOS races {bt.rid.nunique():,}  runners {len(bt):,}  {bt.race_date.min()}..{bt.race_date.max()}")
print("整合模型只用 3 個殘差訊號:late_rel_mean3 + hidden_cnt3 + draw_pct  (市場做 baseline)\n")

def detail(sub, lbl):
    if not len(sub):
        print(f"{lbl}: 冇注"); return
    n = len(sub); hit = sub.win.mean(); avg_odds = sub.win_odds.mean()
    stake = n; ret = sub.ret1.sum(); roi = (ret - stake) / stake
    # bootstrap over races
    g = sub.groupby("rid").agg(c=("win", "size"), r=("ret1", "sum"))
    m = len(g); cc, rv = g.c.values.astype(float), g.r.values
    rs = np.random.RandomState(0); o = np.empty(3000)
    for i in range(3000):
        ix = rs.randint(0, m, m); tc = cc[ix].sum(); o[i] = (rv[ix].sum() - tc) / tc if tc else 0
    lo, hi = np.percentile(o, [2.5, 97.5]); pneg = (o < 0).mean()
    print(f"{lbl}")
    print(f"  注={n}  場={m}  命中率={hit:.1%}  平均賠率={avg_odds:.1f}  "
          f"ROI={roi:+.1%}  [95% CI {lo:+.1%},{hi:+.1%}]  P(ROI<0)={pneg:.0%}")
    yr = sub.assign(pl=sub.ret1 - 1).groupby("season_y").pl.sum()
    print("  逐季 P/L(每注 1 蚊):", {int(k): round(v, 1) for k, v in yr.items()})
    top = sub.nlargest(5, "ret1")[["race_date", "win_odds", "win", "ret1"]]
    print("  最大 5 注回報:", [f"{r.race_date}@{r.win_odds:.0f}({'中' if r.win else '輸'})" for _, r in top.iterrows()])

for T in (0.0, 0.05, 0.10):
    detail(bt[bt.ev >= T], f"\n=== EV >= {T:.0%} ===")

# leave-one-year-out robustness of EV>=5%
print("\n=== EV>=5% 去除單一年份後嘅 ROI(睇邊年撐住個結果)===")
s5 = bt[bt.ev >= 0.05]
base = (s5.ret1.sum() - len(s5)) / len(s5)
print(f"  全部: {base:+.1%}")
for sy in sorted(s5.season_y.unique()):
    sub = s5[s5.season_y != sy]
    print(f"  去除 {sy}-{sy+1}: {(sub.ret1.sum()-len(sub))/len(sub):+.1%}  (該年 {len(s5[s5.season_y==sy])} 注)")

# fractional kelly on EV>=5%
print("\n=== EV>=5% 分數 Kelly(bank 1000, 1/4 Kelly, 10% cap)===")
s5 = s5.copy()
b = s5.win_odds.values - 1
f = np.clip((s5.p_int.values * s5.win_odds.values - 1) / b, 0, None)
stake = np.minimum(0.25 * f, 0.10) * 1000
ok = stake >= 1
ss = s5[ok]; stk = stake[ok]
rr = np.where(ss.win.values == 1, stk * ss.win_odds.values, 0.0)
g = pd.DataFrame({"rid": ss.rid.values, "c": stk, "r": rr}).groupby("rid").sum()
roi = (g.r.sum() - g.c.sum()) / g.c.sum()
print(f"  注={len(ss)}  turnover={g.c.sum():,.0f}  P/L={g.r.sum()-g.c.sum():+,.0f}  ROI={roi:+.1%}")
