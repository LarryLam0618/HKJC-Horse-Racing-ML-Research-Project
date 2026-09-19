"""武器二 + 武器三:整合分段速度 + 一體化條件 logit(市場做 baseline control)."""
from __future__ import annotations
import numpy as np, pandas as pd, json
from scipy.optimize import minimize
SD = "/private/tmp/claude-501/-Users-suetyinglam-Downloads-HKJC-Horse-Racing-ML-Research-Project-main/81331658-de3f-4006-8783-feab579385e5/scratchpad"

# ---------------------------------------------------------------- base data
res = pd.read_csv("results.csv", low_memory=False)
races = pd.read_csv("races.csv", low_memory=False)
res = res[res.race_date >= "2008-04-01"].copy()
res["dt"] = pd.to_datetime(res.race_date)
res["rid"] = res.race_date + "_" + res.venue + "_" + res.race_no.astype(str)
res["saddle"] = res.saddle.astype(str)
res["fin"] = pd.to_numeric(res.finish_pos, errors="coerce")
res = res.dropna(subset=["fin"]); res["fin"] = res.fin.astype(int)
races["rid"] = races.race_date + "_" + races.venue + "_" + races.race_no.astype(str)
def cord(s):
    s = str(s)
    if s.startswith("Class "):
        try: return float(s.split()[1])
        except Exception: return np.nan
    if "Group" in s: return 0.5
    if "Griffin" in s: return 4.5
    return 3.0
races["class_ord"] = races.race_class.map(cord)
res = res.merge(races[["rid", "distance_m", "class_ord", "going", "surface"]], on="rid", how="left")
fs = res.groupby("rid").saddle.transform("size")
res["field_size"] = fs
res["win"] = (res.fin == 1).astype(int)
res["n_place"] = np.where(fs >= 7, 3, np.where(fs >= 5, 2, 0))
res["plc"] = ((res.fin <= res.n_place) & (res.n_place > 0)).astype(int)
res = res.sort_values(["dt", "rid", "saddle"]).reset_index(drop=True)

# ---------------------------------------------------------------- 武器二: sectionals -> late-speed residual
sec = pd.read_csv("sectionals_clean.csv", low_memory=False)
sec["rid"] = sec.race_date + "_" + sec.venue + "_" + sec.race_no.astype(str)
sec["saddle"] = sec.saddle.astype(str)
sec = sec.dropna(subset=["section_index", "section_time_s"])
sec["section_index"] = sec.section_index.astype(int)
# last section (final 400m) time per runner
last = sec.loc[sec.groupby(["rid", "saddle"]).section_index.idxmax()][
    ["rid", "saddle", "horse_id", "section_time_s", "finishing_order"]].rename(
    columns={"section_time_s": "last400"})
# early position = running_position at section_index == 1
early = sec[sec.section_index == 1][["rid", "saddle", "running_position"]].rename(
    columns={"running_position": "early_pos"})
run = last.merge(early, on=["rid", "saddle"], how="left")
# within-race normalisation -> removes track/going/distance/day
run["late_rel"] = run.groupby("rid").last400.transform("median") - run.last400   # +ve = faster finish than field
run["late_z"] = run.late_rel / run.groupby("rid").last400.transform("std").replace(0, np.nan)
run["pos_gain"] = run.early_pos - run.finishing_order                             # +ve = came from behind
run["fin_run"] = run.finishing_order
# "hidden run": finished 6th+ but ran a top-tier closing sectional
run["hidden"] = ((run.fin_run >= 6) & (run.late_rel >= 0.4)).astype(int)
run = run.merge(res[["rid", "saddle", "dt", "horse_id"]].rename(columns={"horse_id": "hid2"}),
                on=["rid", "saddle"], how="left")
run = run.dropna(subset=["dt"]).sort_values("dt")
# lag per horse: aggregate over PRIOR runs
def roll_prior(col, fn, w=3):
    return (run.groupby("horse_id")[col].shift(1)
            .groupby(run.horse_id).rolling(w, min_periods=1).agg(fn).reset_index(level=0, drop=True))
run["late_rel_mean3"] = roll_prior("late_rel", "mean")
run["late_rel_best3"] = roll_prior("late_rel", "max")
run["late_z_mean3"] = roll_prior("late_z", "mean")
run["pos_gain_mean3"] = roll_prior("pos_gain", "mean")
run["hidden_cnt3"] = roll_prior("hidden", "sum")
run["sec_hist"] = run.groupby("horse_id").cumcount()
secfeat = run[["rid", "saddle", "late_rel_mean3", "late_rel_best3", "late_z_mean3",
               "pos_gain_mean3", "hidden_cnt3", "sec_hist"]]
res = res.merge(secfeat, on=["rid", "saddle"], how="left")
print(f"sectional feature coverage: {res.late_rel_best3.notna().mean():.1%}")

# ---------------------------------------------------------------- fundamental as-of features
def prior(df, key, lbl):
    g = df.groupby(key); c = g.cumcount(); s = g[lbl].cumsum() - df[lbl]
    return c, s / c.where(c > 0)
res["h_runs"], res["h_winr"] = prior(res, "horse_id", "win")
_, res["h_plcr"] = prior(res, "horse_id", "plc")
res["j_runs"], res["j_winr"] = prior(res, "jockey_name", "win")
_, res["j_plcr"] = prior(res, "jockey_name", "plc")
res["t_runs"], res["t_winr"] = prior(res, "trainer_name", "win")
res["jt"] = res.jockey_name.astype(str) + "|" + res.trainer_name.astype(str)
_, res["jt_plcr"] = prior(res, "jt", "plc")
res["prev_dt"] = res.groupby("horse_id").dt.shift(1)
res["days_since"] = (res.dt - res.prev_dt).dt.days
res["career_n"] = res.h_runs + 1
res["avg_fin_l3"] = (res.groupby("horse_id").fin.shift(1)
                     .groupby(res.horse_id).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True))
res["last_fin"] = res.groupby("horse_id").fin.shift(1)
res["prev_dist"] = res.groupby("horse_id").distance_m.shift(1)
res["dist_chg"] = res.distance_m - res.prev_dist
res["draw"] = pd.to_numeric(res.draw, errors="coerce")
res["draw_pct"] = res.draw / res.field_size
res["wt"] = pd.to_numeric(res.actual_weight, errors="coerce")
res["wt_rel"] = res.wt - res.groupby("rid").wt.transform("mean")

# market
res["mkt_raw"] = 1 / res.win_odds
res["mkt_p"] = res.mkt_raw / res.groupby("rid").mkt_raw.transform("sum")
res["season_y"] = res.race_date.map(lambda s: int(s[:4]) if int(s[5:7]) >= 9 else int(s[:4]) - 1)
res["ret1"] = np.where(res.win == 1, res.win_odds, 0.0)

FACTORS = ["late_rel_best3", "late_rel_mean3", "pos_gain_mean3", "hidden_cnt3",
           "avg_fin_l3", "last_fin", "h_plcr", "h_winr", "j_winr", "jt_plcr",
           "draw_pct", "wt_rel", "days_since", "dist_chg", "career_n"]
d = res.dropna(subset=["mkt_p", "win_odds"]).copy()
d = d[d.race_date >= "2011-09-01"]                       # need horse history + sectionals
# standardise factors globally (fill na with 0 = neutral)
for c in FACTORS:
    x = pd.to_numeric(d[c], errors="coerce")
    mu, sd = x.mean(), x.std()
    d[c + "_z"] = ((x - mu) / sd).fillna(0.0)
ZCOLS = [c + "_z" for c in FACTORS]
d["lmkt"] = np.log(d.mkt_p.clip(1e-6))
print(f"\nrows {len(d):,}  races {d.rid.nunique():,}  {d.race_date.min()}..{d.race_date.max()}")

# ---------------------------------------------------------------- 武器三: simultaneous conditional logit
def _race_struct(sub):
    o = np.argsort(sub.rid.values, kind="stable")
    rid_s = sub.rid.values[o]
    starts = np.r_[0, np.where(rid_s[1:] != rid_s[:-1])[0] + 1]
    counts = np.diff(np.r_[starts, len(rid_s)])
    win = sub.win.values[o]
    # winner row (global sorted index) per race, -1 if none
    wr = np.full(len(starts), -1)
    for gi, (st, ct) in enumerate(zip(starts, counts)):
        seg = win[st:st + ct]
        if seg.any():
            wr[gi] = st + seg.argmax()
    return o, starts, counts, wr

def fit_clogit(sub, zcols, want_se=True):
    """score_i = a*lmkt_i + sum b_k z_ki ; within-race softmax MLE. Vectorised."""
    o, starts, counts, wr = _race_struct(sub)
    Z = sub[zcols].values[o]
    lm = sub.lmkt.values[o]
    valid = wr >= 0
    wr_v = wr[valid]
    k = len(zcols)
    rep = counts  # for repeating per-race values back to rows

    def nll(th):
        s = th[0] * lm + Z @ th[1:]
        smax = np.maximum.reduceat(s, starts)
        e = np.exp(s - np.repeat(smax, rep))
        se_ = np.add.reduceat(e, starts)
        lse = np.log(se_) + smax
        return float(np.sum(lse[valid] - s[wr_v]))

    r = minimize(nll, np.r_[1.0, np.zeros(k)], method="L-BFGS-B",
                 options={"maxiter": 300})
    th = r.x
    if not want_se:
        return th, None
    eps = 1e-4
    def grad(t):
        g = np.zeros(k + 1)
        for i in range(k + 1):
            tp = t.copy(); tp[i] += eps; tm = t.copy(); tm[i] -= eps
            g[i] = (nll(tp) - nll(tm)) / (2 * eps)
        return g
    H = np.zeros((k + 1, k + 1))
    for i in range(k + 1):
        tp = th.copy(); tp[i] += eps; tm = th.copy(); tm[i] -= eps
        H[:, i] = (grad(tp) - grad(tm)) / (2 * eps)
    H = (H + H.T) / 2
    try:
        se = np.sqrt(np.diag(np.linalg.inv(H)))
    except Exception:
        se = np.full(k + 1, np.nan)
    return th, se

print("\n" + "=" * 100)
print("武器三:一體化條件 logit   score = a*log(市場機率) + Σ b*z(因子)   [市場做 control]")
print("=" * 100)
def report_fit(sub, name):
    th, se = fit_clogit(sub, ZCOLS)
    print(f"\n--- {name}  (races={sub.rid.nunique():,}) ---")
    print(f"  {'market a':22s} {th[0]:+.3f}  (t={th[0]/se[0]:+.1f})")
    rows = []
    for i, c in enumerate(FACTORS):
        b, t = th[i + 1], th[i + 1] / se[i + 1]
        flag = "  <== 正殘差訊號" if (b > 0 and t > 2) else ("  (負,剔除)" if (b < 0 and t < -2) else "")
        print(f"  {c:22s} {b:+.3f}  (t={t:+.1f}){flag}")
        rows.append(dict(factor=c, beta=round(b, 3), t=round(t, 1)))
    return dict(market_a=round(th[0], 3), market_t=round(th[0] / se[0], 1), factors=rows)

REP = {}
REP["all"] = report_fit(d, "全期 2011-2026")
REP["eras"] = {}
for e, lo, hi in [("2011-2016", 2011, 2016), ("2016-2021", 2016, 2021), ("2021-2026", 2021, 2027)]:
    REP["eras"][e] = report_fit(d[(d.season_y >= lo) & (d.season_y < hi)], e)

# ---------------------------------------------------------------- walk-forward betting with the integrated model
print("\n" + "=" * 100)
print("整合模型走前測試:每季用之前訓練,揀 overlay 落注,對比市場")
print("=" * 100)
d = d.sort_values("season_y")
d["p_int"] = np.nan
for sy in range(2014, 2027):
    tr = d[d.season_y < sy]; te = d[d.season_y == sy]
    if len(te) == 0 or tr.rid.nunique() < 2000:
        continue
    th, _ = fit_clogit(tr, ZCOLS, want_se=False)
    sc = th[0] * te.lmkt.values + te[ZCOLS].values @ th[1:]
    ex = np.exp(sc - pd.Series(sc, index=te.rid.values).groupby(level=0).transform("max").values)
    den = pd.Series(ex, index=te.rid.values).groupby(level=0).transform("sum").values
    d.loc[te.index, "p_int"] = ex / den

bt = d.dropna(subset=["p_int"]).copy()
bt["edge_int"] = bt.p_int / bt.mkt_p
bt["ev_int"] = bt.p_int * bt.win_odds - 1
def roi_block(sub, lbl):
    if not len(sub):
        print(f"  {lbl:40s} 冇注"); return
    rr = sub.groupby("rid").agg(c=("win", "size"), r=("ret1", "sum"))
    m = len(rr); cc, rv = rr.c.values.astype(float), rr.r.values
    roi = (rv.sum() - cc.sum()) / cc.sum()
    rs = np.random.RandomState(0); o = np.empty(500)
    for i in range(500):
        ix = rs.randint(0, m, m); tc = cc[ix].sum(); o[i] = (rv[ix].sum() - tc) / tc if tc else 0
    lo, hi = np.percentile(o, [5, 95])
    print(f"  {lbl:40s} n={len(sub):6d} races={m:5d}  ROI={roi:+6.1%}  [CI {lo:+.1%},{hi:+.1%}]")
print(f"\nOOS races {bt.rid.nunique():,}  ({bt.race_date.min()}..{bt.race_date.max()})")
for T in (0.0, 0.05, 0.10, 0.20):
    roi_block(bt[bt.ev_int >= T], f"整合模型 EV>={T:.0%}")
print("  -- 對照 --")
roi_block(bt[bt.p_int / bt.mkt_p >= 1.15], "整合模型 edge>=1.15")
# per-season for EV>=5%
print("\n  逐季 (整合模型 EV>=5%):")
for sy in sorted(bt.season_y.unique()):
    roi_block(bt[(bt.ev_int >= 0.05) & (bt.season_y == sy)], f"  {sy}-{sy+1}")

json.dump(REP, open(f"{SD}/w23_report.json", "w"), ensure_ascii=False, default=str)
print(f"\nsaved {SD}/w23_report.json")
