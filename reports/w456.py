"""最後三步:1 步速情景模型  3 正交殘差訊號(體重動態 + 練馬師部署).  (步 2 = M7 logger 已存在,只能向前儲)."""
from __future__ import annotations
import numpy as np, pandas as pd, json
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
res = res.merge(races[["rid", "distance_m", "class_ord", "venue"]].rename(columns={"venue": "vv"}),
                on="rid", how="left")
fs = res.groupby("rid").saddle.transform("size")
res["field_size"] = fs
res["win"] = (res.fin == 1).astype(int)
res = res.sort_values(["dt", "rid", "saddle"]).reset_index(drop=True)

# ================================================================= 步 1: PACE MODEL
sec = pd.read_csv("sectionals_clean.csv", low_memory=False)
sec["rid"] = sec.race_date + "_" + sec.venue + "_" + sec.race_no.astype(str)
sec["saddle"] = sec.saddle.astype(str)
sec = sec.dropna(subset=["section_index", "section_time_s"])
sec["section_index"] = sec.section_index.astype(int)
sec["nsec"] = sec.groupby(["rid", "saddle"]).section_index.transform("max")
sec = sec.merge(races[["rid", "distance_m"]], on="rid", how="left")

lastt = sec.loc[sec.groupby(["rid", "saddle"]).section_index.idxmax()][
    ["rid", "saddle", "section_time_s", "finishing_order"]].rename(columns={"section_time_s": "last400"})
first = sec[sec.section_index == 1][["rid", "saddle", "section_time_s", "running_position"]].rename(
    columns={"section_time_s": "sec1", "running_position": "early_pos"})
run = lastt.merge(first, on=["rid", "saddle"], how="left").merge(
    sec[["rid", "distance_m", "nsec"]].drop_duplicates("rid"), on="rid", how="left")

# race-level EARLY PACE PRESSURE = how fast the leader went early (min sec1 time), z within (distance,nsec)
lead_early = run.groupby("rid").sec1.min().rename("lead_sec1")
run = run.merge(lead_early, on="rid")
grp = run.groupby(["distance_m", "nsec"]).lead_sec1
run["pace_press"] = -((run.lead_sec1 - grp.transform("mean")) / grp.transform("std"))  # +ve = fast early = pressure

# within-race closing residual (last 400m vs field median)
run["late_rel"] = run.groupby("rid").last400.transform("median") - run.last400
# PACE-ADJUSTED closing figure: closing hard when pace was hot is worth much more
run["pace_close"] = run.late_rel * (1.0 + np.clip(run.pace_press, 0, 2.5))
# front-runner who withstood a fast pace and still placed
run["led_held_hp"] = ((run.early_pos <= 3) & (run.finishing_order <= 3) & (run.pace_press > 0.4)).astype(int)
# plain hidden run (fast finish despite bad placing) but ONLY counts in a truly-run race
run["hidden_hp"] = ((run.finishing_order >= 6) & (run.late_rel >= 0.35) & (run.pace_press > 0.0)).astype(int)

run = run.merge(res[["rid", "saddle", "dt", "horse_id"]], on=["rid", "saddle"], how="left")
run = run.dropna(subset=["dt"]).sort_values("dt")
def lag(col, fn, w=4):
    return (run.groupby("horse_id")[col].shift(1)
            .groupby(run.horse_id).rolling(w, min_periods=1).agg(fn).reset_index(level=0, drop=True))
run["pace_close3"] = lag("pace_close", "mean")
run["late_rel3"] = lag("late_rel", "mean")
run["led_held3"] = lag("led_held_hp", "sum")
run["hidden_hp3"] = lag("hidden_hp", "sum")
secf = run[["rid", "saddle", "pace_close3", "late_rel3", "led_held3", "hidden_hp3"]]
res = res.merge(secf, on=["rid", "saddle"], how="left")

# ================================================================= 步 3a: 體重動態 (declared_weight = 馬匹體重 lbs)
res["bw"] = pd.to_numeric(res.declared_weight, errors="coerce")
res["bw_prev"] = res.groupby("horse_id").bw.shift(1)
res["bw_chg"] = res.bw - res.bw_prev
res["bw_career_avg"] = (res.groupby("horse_id").bw.shift(1)
                        .groupby(res.horse_id).expanding().mean().reset_index(level=0, drop=True))
res["bw_vs_avg"] = res.bw - res.bw_career_avg
res["days_off"] = (res.dt - res.groupby("horse_id").dt.shift(1)).dt.days
# 加返體重 + 抖夠(returning bigger & fresh) vs 暴跌
res["bw_up_fresh"] = ((res.bw_chg > 8) & (res.days_off > 45)).astype(float)
res["bw_drop"] = np.clip(-res.bw_chg, 0, None)   # 暴跌幾多磅

# ================================================================= 步 3b: 練馬師部署殘差 (轉降班時機)
def prior_rate(df, key, lbl):
    g = df.groupby(key); c = g.cumcount(); s = g[lbl].cumsum() - df[lbl]
    return c, s / c.where(c > 0)
res["prev_class"] = res.groupby("horse_id").class_ord.shift(1)
res["class_drop"] = (res.class_ord - res.prev_class).clip(lower=0)   # +ve = dropping in grade (easier)
res["is_drop"] = (res.class_drop > 0).astype(int)
# trainer strike rate specifically ON class-drop runners, as-of
drp = res[res.is_drop == 1].copy()
_, drp["tr_drop_sr"] = prior_rate(drp, "trainer_name", "win")
res = res.merge(drp[["rid", "saddle", "tr_drop_sr"]], on=["rid", "saddle"], how="left")
res["tr_drop_sr"] = res.tr_drop_sr.fillna(0.0)
res["drop_x_trsr"] = res.is_drop * res.tr_drop_sr

# baseline form controls (so residual is真殘差)
res["j_runs"], res["j_winr"] = prior_rate(res, "jockey_name", "win")
res["h_runs"], res["h_winr"] = prior_rate(res, "horse_id", "win")
res["avg_fin_l3"] = (res.groupby("horse_id").fin.shift(1)
                     .groupby(res.horse_id).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True))
res["draw"] = pd.to_numeric(res.draw, errors="coerce")
res["draw_pct"] = res.draw / res.field_size
res["mkt_raw"] = 1 / res.win_odds
res["mkt_p"] = res.mkt_raw / res.groupby("rid").mkt_raw.transform("sum")
res["season_y"] = res.race_date.map(lambda s: int(s[:4]) if int(s[5:7]) >= 9 else int(s[:4]) - 1)
res["ret1"] = np.where(res.win == 1, res.win_odds, 0.0)

FACT = ["pace_close3", "late_rel3", "led_held3", "hidden_hp3",       # 步1 pace
        "bw_chg", "bw_vs_avg", "bw_up_fresh", "bw_drop",             # 步3a weight
        "is_drop", "drop_x_trsr",                                    # 步3b trainer deploy
        "draw_pct", "avg_fin_l3", "j_winr"]                          # controls / known
d = res.dropna(subset=["mkt_p", "win_odds"]).copy()
d = d[d.race_date >= "2011-09-01"]
for c in FACT:
    x = pd.to_numeric(d[c], errors="coerce"); d[c + "_z"] = ((x - x.mean()) / x.std()).fillna(0.0)
Z = [c + "_z" for c in FACT]
d["lmkt"] = np.log(d.mkt_p.clip(1e-6))
print(f"rows {len(d):,}  races {d.rid.nunique():,}  {d.race_date.min()}..{d.race_date.max()}")

# ---- orthogonality check
print("\n=== 訊號之間嘅相關(要接近 0 先算正交)===")
key = ["pace_close3_z", "bw_chg_z", "bw_vs_avg_z", "is_drop_z", "drop_x_trsr_z"]
print(d[key].corr().round(2).to_string())

# ---- simultaneous conditional logit
def struct(sub):
    o = np.argsort(sub.rid.values, kind="stable"); r = sub.rid.values[o]
    st = np.r_[0, np.where(r[1:] != r[:-1])[0] + 1]; ct = np.diff(np.r_[st, len(r)])
    wn = sub.win.values[o]; wr = np.full(len(st), -1)
    for i, (s, c) in enumerate(zip(st, ct)):
        seg = wn[s:s + c]
        if seg.any(): wr[i] = s + seg.argmax()
    return o, st, ct, wr
def fit(sub, zc, se=True):
    o, st, ct, wr = struct(sub); Zm = sub[zc].values[o]; lm = sub.lmkt.values[o]
    v = wr >= 0; wv = wr[v]; k = len(zc)
    def nll(t):
        s = t[0] * lm + Zm @ t[1:]; sm = np.maximum.reduceat(s, st)
        e = np.exp(s - np.repeat(sm, ct))
        return float(np.sum((np.log(np.add.reduceat(e, st)) + sm)[v] - s[wv]))
    r = minimize(nll, np.r_[1.0, np.zeros(k)], method="L-BFGS-B")
    th = r.x
    if not se: return th, None
    eps = 1e-4
    def g(t):
        gg = np.zeros(k + 1)
        for i in range(k + 1):
            a = t.copy(); a[i] += eps; b = t.copy(); b[i] -= eps
            gg[i] = (nll(a) - nll(b)) / (2 * eps)
        return gg
    H = np.zeros((k + 1, k + 1))
    for i in range(k + 1):
        a = th.copy(); a[i] += eps; b = th.copy(); b[i] -= eps
        H[:, i] = (g(a) - g(b)) / (2 * eps)
    H = (H + H.T) / 2
    try: sd = np.sqrt(np.diag(np.linalg.inv(H)))
    except Exception: sd = np.full(k + 1, np.nan)
    return th, sd

th, sd = fit(d, Z)
print("\n" + "=" * 88)
print("一體化條件 logit   score = a*log(市場) + Σ b*z(因子)   [全期 2011-2026]")
print("=" * 88)
print(f"  {'market a':20s} {th[0]:+.3f}  (t={th[0]/sd[0]:+.1f})")
REP = {"market_a": round(th[0], 3), "market_t": round(th[0] / sd[0], 1), "factors": []}
for i, c in enumerate(FACT):
    b, t = th[i + 1], th[i + 1] / sd[i + 1]
    fl = "  <== 正殘差訊號" if (b > 0 and t > 2) else ("  (顯著負)" if (b < 0 and t < -2) else "")
    print(f"  {c:20s} {b:+.3f}  (t={t:+.1f}){fl}")
    REP["factors"].append(dict(factor=c, beta=round(b, 3), t=round(t, 1)))

# ---- walk-forward backtest with augmented model
d = d.sort_values("season_y"); d["p_int"] = np.nan
for sy in range(2014, 2027):
    tr = d[d.season_y < sy]; te = d[d.season_y == sy]
    if len(te) == 0: continue
    th_, _ = fit(tr, Z, se=False)
    sc = th_[0] * te.lmkt.values + te[Z].values @ th_[1:]
    ex = np.exp(sc - pd.Series(sc, index=te.rid.values).groupby(level=0).transform("max").values)
    d.loc[te.index, "p_int"] = ex / pd.Series(ex, index=te.rid.values).groupby(level=0).transform("sum").values
bt = d.dropna(subset=["p_int"]).copy()
bt["ev"] = bt.p_int * bt.win_odds - 1
print("\n" + "=" * 88 + "\n走前落注測試(每季重新 fit)\n" + "=" * 88)
def blk(sub, lbl):
    if not len(sub): print(f"  {lbl:30s} 冇注"); return
    g = sub.groupby("rid").agg(c=("win", "size"), r=("ret1", "sum"))
    m = len(g); cc, rv = g.c.values.astype(float), g.r.values
    roi = (rv.sum() - cc.sum()) / cc.sum()
    rs = np.random.RandomState(0); o = np.empty(3000)
    for i in range(3000):
        ix = rs.randint(0, m, m); tc = cc[ix].sum(); o[i] = (rv[ix].sum() - tc) / tc if tc else 0
    lo, hi = np.percentile(o, [2.5, 97.5])
    print(f"  {lbl:30s} 注={len(sub):5d} 場={m:5d} 命中={sub.win.mean():.1%} "
          f"平均賠率={sub.win_odds.mean():.1f}  ROI={roi:+.1%}  [95%CI {lo:+.1%},{hi:+.1%}]  P(<0)={(o<0).mean():.0%}")
print(f"OOS races {bt.rid.nunique():,}  {bt.race_date.min()}..{bt.race_date.max()}")
for T in (0.0, 0.05, 0.10, 0.20):
    blk(bt[bt.ev >= T], f"整合(13因子) EV>={T:.0%}")
blk(bt[bt.p_int / bt.mkt_p >= 1.15], "整合 edge>=1.15")
print("\n  逐季 EV>=5%:")
for sy in sorted(bt.season_y.unique()):
    blk(bt[(bt.ev >= 0.05) & (bt.season_y == sy)], f"  {sy}-{sy+1}")
json.dump(REP, open(f"{SD}/w456_report.json", "w"), ensure_ascii=False)
print("\nsaved w456_report.json")
