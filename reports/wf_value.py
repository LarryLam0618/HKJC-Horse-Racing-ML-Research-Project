"""Value-bet / overlay test on WALK-FORWARD OOS preds, full history 2010-2026."""
from __future__ import annotations
import numpy as np, pandas as pd, json
from scipy.optimize import minimize
SD = "/private/tmp/claude-501/-Users-suetyinglam-Downloads-HKJC-Horse-Racing-ML-Research-Project-main/81331658-de3f-4006-8783-feab579385e5/scratchpad"

d = pd.read_parquet(f"{SD}/preds_wf.parquet")
d = d.dropna(subset=["p_win", "mkt_p", "win_odds"]).copy()
d["wd1"] = d.win_odds                       # WIN return per HK$1 (dividend/10 == win_odds)
d["ret1"] = np.where(d.win_lbl == 1, d.wd1, 0.0)
d["mkt_p"] = d.mkt_p / d.groupby("rid").mkt_p.transform("sum")
d["p_win"] = d.p_win / d.groupby("rid").p_win.transform("sum")
d["edge"] = d.p_win / d.mkt_p
d["season"] = d.season_y.map(lambda y: f"{y}-{str(y+1)[2:]}")
print(f"races {d.rid.nunique():,}  runners {len(d):,}  {d.race_date.min()} .. {d.race_date.max()}")

REPORT = {}

# 1. calibration + accuracy, overall and by era
print("\n" + "=" * 96 + "\n1. 準繩度:  模型 vs 市場  (log-loss / Brier, 越低越準)\n" + "=" * 96)
def acc(sub):
    out = {}
    for nm, col in [("model", "p_win"), ("market", "mkt_p")]:
        p = sub[col].clip(1e-6, 1 - 1e-6)
        out[nm] = dict(logloss=round(-(sub.win_lbl*np.log(p)+(1-sub.win_lbl)*np.log(1-p)).mean(), 4),
                       brier=round(((p-sub.win_lbl)**2).mean(), 4))
    return out
print("ALL      ", acc(d))
for era, lo, hi in [("2010-2015", 2010, 2016), ("2016-2020", 2016, 2021), ("2021-2026", 2021, 2027)]:
    print(f"{era}", acc(d[(d.season_y >= lo) & (d.season_y < hi)]))
REPORT["accuracy"] = {"all": acc(d),
    "eras": {e: acc(d[(d.season_y >= lo) & (d.season_y < hi)]) for e, lo, hi in
             [("2010-2015", 2010, 2016), ("2016-2020", 2016, 2021), ("2021-2026", 2021, 2027)]}}

# 2. conditional logit market vs model, overall + per era
print("\n" + "=" * 96 + "\n2. 條件 logit:  win ~ a*logit(market) + b*logit(model)   b>0 顯著 => 模型有增量\n" + "=" * 96)
def cond_logit(sub):
    lm = np.log(sub.mkt_p.clip(1e-6)).values
    lg = np.log(sub.p_win.clip(1e-6)).values
    y = sub.win_lbl.values
    codes, _ = pd.factorize(sub.rid.values)
    idx = {}
    for i, c in enumerate(codes):
        idx.setdefault(c, []).append(i)
    ii = [np.array(v) for v in idx.values()]
    yw = [(y[a].argmax() if y[a].any() else -1) for a in ii]
    def nll(th):
        a, b = th; sc = a*lm + b*lg; tot = 0.0
        for a_i, w in zip(ii, yw):
            if w < 0: continue
            s = sc[a_i]; s -= s.max()
            tot -= s[w] - np.log(np.exp(s).sum())
        return tot
    r = minimize(nll, [1.0, 0.0], method="Nelder-Mead")
    a, b = r.x
    h = 1e-3; H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            f = []
            for si, sj in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                tp = r.x.copy(); tp[i] += si*h; tp[j] += sj*h; f.append(nll(tp))
            H[i, j] = (f[0]-f[1]-f[2]+f[3])/(4*h*h)
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return dict(a=round(a, 3), a_t=round(a/se[0], 1), b=round(b, 3), b_t=round(b/se[1], 1))
o = cond_logit(d); print("ALL      ", o); REPORT["cond_logit_all"] = o
REPORT["cond_logit_era"] = {}
for era, lo, hi in [("2010-2015", 2010, 2016), ("2016-2020", 2016, 2021), ("2021-2026", 2021, 2027)]:
    r = cond_logit(d[(d.season_y >= lo) & (d.season_y < hi)])
    print(f"{era}", r); REPORT["cond_logit_era"][era] = r

# 3. overlay buckets
print("\n" + "=" * 96 + "\n3. 按 edge 分組:  實際勝率 vs 模型 vs 市場 + flat ROI\n" + "=" * 96)
d["ebin"] = pd.cut(d.edge, [0, .6, .8, .95, 1.05, 1.25, 1.6, 2.5, 99],
                   labels=["<0.6", "0.6-0.8", "0.8-0.95", "≈市場", "1.05-1.25", "1.25-1.6", "1.6-2.5", "2.5+"])
def bootroi(g, n=800):
    rr = g.groupby("rid").agg(c=("win_lbl", "size"), r=("ret1", "sum"))
    m = len(rr); rs = np.random.RandomState(0); cc, rv = rr.c.values.astype(float), rr.r.values
    o = np.empty(n)
    for i in range(n):
        ix = rs.randint(0, m, m); tc = cc[ix].sum(); o[i] = (rv[ix].sum()-tc)/tc if tc else 0
    return np.percentile(o, [5, 95])
rows = []
for k, g in d.groupby("ebin", observed=True):
    roi = g.ret1.sum()/len(g) - 1
    lo, hi = bootroi(g)
    rows.append(dict(edge=str(k), n=len(g), actual=round(g.win_lbl.mean()*100, 1),
                     model=round(g.p_win.mean()*100, 1), market=round(g.mkt_p.mean()*100, 1),
                     roi=round(roi*100, 1), lo=round(lo*100, 1), hi=round(hi*100, 1)))
bt = pd.DataFrame(rows); print(bt.to_string(index=False)); REPORT["overlay_buckets"] = rows

# 4. strategies overall + per season
print("\n" + "=" * 96 + "\n4. Overlay 策略  (flat, edge>=門檻),  逐季\n" + "=" * 96)
def strat_roi(sub):
    if not len(sub): return None
    rr = sub.groupby("rid").agg(c=("win_lbl", "size"), r=("ret1", "sum"))
    m = len(rr); cc, rv = rr.c.values.astype(float), rr.r.values
    roi = (rv.sum()-cc.sum())/cc.sum()
    rs = np.random.RandomState(0); o = np.empty(500)
    for i in range(500):
        ix = rs.randint(0, m, m); tc = cc[ix].sum(); o[i] = (rv[ix].sum()-tc)/tc if tc else 0
    lo, hi = np.percentile(o, [5, 95])
    return dict(n=len(sub), races=m, roi=round(roi*100, 1), lo=round(lo*100, 1), hi=round(hi*100, 1))
for T in (1.0, 1.15, 1.3, 1.6):
    print(f"\n-- edge >= {T}  flat --")
    print("ALL            ", strat_roi(d[d.edge >= T]))
    REPORT.setdefault("strat", {})[f"edge>={T}"] = {"ALL": strat_roi(d[d.edge >= T])}
    for sy in sorted(d.season_y.unique()):
        r = strat_roi(d[(d.edge >= T) & (d.season_y == sy)])
        if r: print(f"  {sy}-{sy+1}: ", r)
        REPORT["strat"][f"edge>={T}"][f"{sy}-{sy+1}"] = r

# 5. model-top-pick WIN by season (does form pick ever beat market pick?)
print("\n" + "=" * 96 + "\n5. 每季:  買模型頭馬 vs 買市場大熱  (flat WIN, 全部場)\n" + "=" * 96)
def toppick(sub, by, asc):
    picks = sub.sort_values(by, ascending=asc).groupby("rid").head(1)
    return strat_roi(picks)
srows = []
for sy in sorted(d.season_y.unique()):
    s = d[d.season_y == sy]
    mdl = toppick(s, "p_win", False); mkt = toppick(s, "win_odds", True)
    srows.append(dict(season=f"{sy}-{sy+1}", races=mdl["races"],
                      model_roi=mdl["roi"], model_hit=round((s.sort_values("p_win", ascending=False)
                          .groupby("rid").head(1).win_lbl.mean())*100, 1),
                      market_roi=mkt["roi"], market_hit=round((s.sort_values("win_odds")
                          .groupby("rid").head(1).win_lbl.mean())*100, 1)))
sd = pd.DataFrame(srows); print(sd.to_string(index=False)); REPORT["season_toppick"] = srows

json.dump(REPORT, open(f"{SD}/wf_value_report.json", "w"), ensure_ascii=False, default=str)
print(f"\nsaved {SD}/wf_value_report.json")
