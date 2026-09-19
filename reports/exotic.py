"""#2 彩池測試:用整合模型嘅『排名』打盤連贏Q / 三重彩Trio / 第一四連環.  真派彩 2024-09..2026-07."""
from __future__ import annotations
import numpy as np, pandas as pd, json, itertools
from scipy.optimize import minimize
SD = "/private/tmp/claude-501/-Users-suetyinglam-Downloads-HKJC-Horse-Racing-ML-Research-Project-main/81331658-de3f-4006-8783-feab579385e5/scratchpad"

res = pd.read_csv("results.csv", low_memory=False)
races = pd.read_csv("races.csv", low_memory=False)
res = res[res.race_date >= "2008-04-01"].copy()
res["dt"] = pd.to_datetime(res.race_date)
res["rid"] = res.race_date + "_" + res.venue + "_" + res.race_no.astype(str)
res["saddle"] = res.saddle.astype(str)
res["fin"] = pd.to_numeric(res.finish_pos, errors="coerce")
res = res.dropna(subset=["fin"]); res["fin"] = res.fin.astype(int)
races["rid"] = races.race_date + "_" + races.venue + "_" + races.race_no.astype(str)
races["distance_m"] = pd.to_numeric(races.distance_m, errors="coerce")
def cord(s):
    s = str(s)
    if s.startswith("Class "):
        try: return float(s.split()[1])
        except Exception: return np.nan
    if "Group" in s: return 0.5
    if "Griffin" in s: return 4.5
    return 3.0
races["class_ord"] = races.race_class.map(cord)
res = res.merge(races[["rid", "distance_m", "class_ord"]], on="rid", how="left")
fs = res.groupby("rid").saddle.transform("size")
res["field_size"] = fs
res["win"] = (res.fin == 1).astype(int)
res = res.sort_values(["dt", "rid", "saddle"]).reset_index(drop=True)

# --- sectionals -> late_rel3
sec = pd.read_csv("sectionals_clean.csv", low_memory=False)
sec["rid"] = sec.race_date + "_" + sec.venue + "_" + sec.race_no.astype(str)
sec["saddle"] = sec.saddle.astype(str)
sec = sec.dropna(subset=["section_index", "section_time_s"])
sec["section_index"] = sec.section_index.astype(int)
lastt = sec.loc[sec.groupby(["rid", "saddle"]).section_index.idxmax()][
    ["rid", "saddle", "section_time_s", "finishing_order"]].rename(columns={"section_time_s": "last400"})
run = lastt.merge(res[["rid", "saddle", "dt", "horse_id"]], on=["rid", "saddle"], how="left").dropna(subset=["dt"]).sort_values("dt")
run["late_rel"] = run.groupby("rid").last400.transform("median") - run.last400
run["hidden"] = ((run.finishing_order >= 6) & (run.late_rel >= 0.4)).astype(int)
run["late_rel3"] = (run.groupby("horse_id").late_rel.shift(1)
                    .groupby(run.horse_id).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True))
run["hidden3"] = (run.groupby("horse_id").hidden.shift(1)
                  .groupby(run.horse_id).rolling(3, min_periods=1).sum().reset_index(level=0, drop=True))
res = res.merge(run[["rid", "saddle", "late_rel3", "hidden3"]], on=["rid", "saddle"], how="left")

# --- weight dynamics
res["bw"] = pd.to_numeric(res.declared_weight, errors="coerce")
res["bw_chg"] = res.bw - res.groupby("horse_id").bw.shift(1)
res["days_off"] = (res.dt - res.groupby("horse_id").dt.shift(1)).dt.days
res["bw_up_fresh"] = ((res.bw_chg > 8) & (res.days_off > 45)).astype(float)

def prior(df, key, lbl):
    g = df.groupby(key); c = g.cumcount(); s = g[lbl].cumsum() - df[lbl]
    return c, s / c.where(c > 0)
_, res["j_winr"] = prior(res, "jockey_name", "win")
res["h_runs"], res["h_winr"] = prior(res, "horse_id", "win")
res["avg_fin_l3"] = (res.groupby("horse_id").fin.shift(1)
                     .groupby(res.horse_id).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True))
res["draw"] = pd.to_numeric(res.draw, errors="coerce")
res["draw_pct"] = res.draw / res.field_size
res["mkt_raw"] = 1 / res.win_odds
res["mkt_p"] = res.mkt_raw / res.groupby("rid").mkt_raw.transform("sum")
res["season_y"] = res.race_date.map(lambda s: int(s[:4]) if int(s[5:7]) >= 9 else int(s[:4]) - 1)

FACT = ["late_rel3", "hidden3", "bw_up_fresh", "draw_pct", "avg_fin_l3", "j_winr", "h_winr"]
d = res.dropna(subset=["mkt_p", "win_odds"]).copy()
d = d[d.race_date >= "2011-09-01"]
for c in FACT:
    x = pd.to_numeric(d[c], errors="coerce"); d[c + "_z"] = ((x - x.mean()) / x.std()).fillna(0.0)
Z = [c + "_z" for c in FACT]
d["lmkt"] = np.log(d.mkt_p.clip(1e-6))

def struct(sub):
    o = np.argsort(sub.rid.values, kind="stable"); r = sub.rid.values[o]
    st = np.r_[0, np.where(r[1:] != r[:-1])[0] + 1]; ct = np.diff(np.r_[st, len(r)])
    wn = sub.win.values[o]; wr = np.full(len(st), -1)
    for i, (s, c) in enumerate(zip(st, ct)):
        seg = wn[s:s + c]
        if seg.any(): wr[i] = s + seg.argmax()
    return o, st, ct, wr
def fit(sub, zc, use_mkt=True):
    o, st, ct, wr = struct(sub); Zm = sub[zc].values[o]
    lm = sub.lmkt.values[o] if use_mkt else np.zeros(len(sub))
    v = wr >= 0; wv = wr[v]; k = len(zc)
    def nll(t):
        s = (t[0] * lm if use_mkt else 0) + Zm @ t[1:]
        sm = np.maximum.reduceat(s, st); e = np.exp(s - np.repeat(sm, ct))
        return float(np.sum((np.log(np.add.reduceat(e, st)) + sm)[v] - s[wv]))
    return minimize(nll, np.r_[1.0, np.zeros(k)], method="L-BFGS-B").x

d = d.sort_values("season_y")
for col, um in [("p_int", True), ("p_fund", False)]:
    d[col] = np.nan
    for sy in (2024, 2025):
        tr = d[d.season_y < sy]; te = d[d.season_y == sy]
        if len(te) == 0: continue
        th = fit(tr, Z, use_mkt=um)
        sc = (th[0] * te.lmkt.values if um else 0) + te[Z].values @ th[1:]
        ex = np.exp(sc - pd.Series(sc, index=te.rid.values).groupby(level=0).transform("max").values)
        d.loc[te.index, col] = ex / pd.Series(ex, index=te.rid.values).groupby(level=0).transform("sum").values

bt = d[d.race_date >= "2024-09-01"].dropna(subset=["p_int"]).copy()
print(f"OOS races {bt.rid.nunique():,}  ({bt.race_date.min()}..{bt.race_date.max()})")

# --- exotic dividends
div = pd.read_csv("dividends.csv", low_memory=False)
div["rid"] = div.race_date + "_" + div.venue + "_" + div.race_no.astype(str)
div["dividend"] = pd.to_numeric(div.dividend, errors="coerce")
def divmap(pool):
    s = div[div.pool == pool]
    return {r: (frozenset(c.split(",")), v) for r, c, v in zip(s.rid, s.combination, s.dividend)}
DQ, DT, D4 = divmap("QUINELLA"), divmap("TRIO"), divmap("FIRST 4")

fin_map = {rid: g.set_index("saddle").fin.to_dict() for rid, g in bt.groupby("rid")}

def sim(ranker, pool, box_n, need_k, dm):
    rows = []
    for rid, g in bt.groupby("rid"):
        g = g.dropna(subset=[ranker])
        if len(g) <= box_n:
            continue
        picks = list(g.nlargest(box_n, ranker).saddle)
        from math import comb
        ncomb = comb(box_n, need_k)
        cost = 10.0 * ncomb
        if rid not in dm:
            continue
        wincombo, dv = dm[rid]
        hit = wincombo.issubset(set(picks)) and len(wincombo) == need_k
        ret = dv if hit else 0.0
        rows.append((rid, bt[bt.rid == rid].race_date.iloc[0], cost, ret, int(hit)))
    df = pd.DataFrame(rows, columns=["rid", "date", "cost", "ret", "hit"])
    if not len(df):
        return None
    roi = (df.ret.sum() - df.cost.sum()) / df.cost.sum()
    m = len(df); rs = np.random.RandomState(0); cc, rv = df.cost.values, df.ret.values
    o = np.empty(3000)
    for i in range(3000):
        ix = rs.randint(0, m, m); tc = cc[ix].sum(); o[i] = (rv[ix].sum() - tc) / tc if tc else 0
    lo, hi = np.percentile(o, [2.5, 97.5])
    return dict(races=m, hit=df.hit.mean(), roi=roi, lo=lo, hi=hi, stake=df.cost.sum(), pnl=df.ret.sum() - df.cost.sum())

CONF = [
    ("QUINELLA 連贏", "q", DQ, 2, [(3, "box3"), (4, "box4"), (5, "box5")]),
    ("TRIO 三重彩", "t", DT, 3, [(4, "box4"), (5, "box5"), (6, "box6")]),
    ("FIRST4 第一四連環", "f", D4, 4, [(5, "box5"), (6, "box6"), (7, "box7")]),
]
RANKERS = [("市場 win_odds", "mkt_p"), ("整合模型 p_int", "p_int"), ("純基本面 p_fund", "p_fund")]
REP = {}
for pname, pk, dm, need_k, boxes in CONF:
    print("\n" + "=" * 92)
    print(f"{pname}  (box top-N,  中 = 頭 {need_k} 名嘅組合喺你 N 揀之內)")
    print("=" * 92)
    print(f"{'ranker':22s} {'box':6s} {'注/場':>7s} {'場數':>6s} {'命中率':>8s} {'ROI':>8s} {'95% CI':>18s}  {'盈虧':>9s}")
    for rname, rcol in RANKERS:
        for bn, blab in boxes:
            r = sim(rcol, pk, bn, need_k, dm)
            if r is None:
                continue
            from math import comb
            print(f"{rname:22s} {blab:6s} {comb(bn,need_k):7d} {r['races']:6d} {r['hit']:7.1%} "
                  f"{r['roi']:+7.1%}  [{r['lo']:+.0%},{r['hi']:+.0%}]  {r['pnl']:+9,.0f}")
            REP[f"{pk}_{rcol}_{blab}"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}

json.dump(REP, open(f"{SD}/exotic_report.json", "w"), ensure_ascii=False)
print("\nsaved exotic_report.json")
