"""draw_bias -- does the market under-price track-specific barrier bias?

Standalone study in the style of ``w456.py``: build strictly **as-of** draw-bias signals from the
data lake, add the closing-line market probability as a control, fit one within-race conditional
logit and read the t-stats, then walk-forward (refit per season) and report EV-gated ROI.

Signals (all keyed by venue x surface x distance, "the track configuration"):
  draw_win_bias    prior win share of this draw vs the share its field sizes imply
                   (obs_wins + k) / (exp_wins + k) - 1, shrunk with k expected wins
  draw_place_bias  same for top-3
  draw_bias_x_big  draw_win_bias x (field_size - 12) -- outside draws bite harder in big fields

**As-of discipline (the pace_press lesson):** every prior count is `cum_sum - current` on a
frame sorted by (race_date, race_no), so a race only sees races run before it. No whole-history
normalisation anywhere -- the z-scores fed to the logit are expanding too.

Run from the repo root with the venv:  .venv/bin/python reports/draw_bias.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize

from hkjc.common.config import get_config
from hkjc.features.build import _read_raw

OUT = Path(__file__).with_name("draw_bias_report.json")
K_SHRINK = 10.0  # expected wins of prior weight (~10 races' worth for a 12-runner draw)
CONTROL_BASE = ["draw_rel"]  # already in BASELINE_FEATURES -> must show incremental value
FACT = ["draw_win_bias", "draw_place_bias", "draw_bias_x_big"]

# ------------------------------------------------------------------ load
cfg = get_config()
res = _read_raw(
    cfg,
    "results",
    columns=["race_date", "venue", "race_no", "saddle", "draw", "finish_pos", "win_odds"],
)
races = _read_raw(cfg, "races", columns=["race_date", "venue", "race_no", "distance_m", "surface"])
df = (
    res.join(races.unique(["race_date", "venue", "race_no"]), on=["race_date", "venue", "race_no"])
    .drop_nulls(["draw", "finish_pos", "distance_m"])
    .filter(pl.col("race_date") >= pl.date(2008, 4, 1))
    .with_columns(
        field_size=pl.len().over(["race_date", "venue", "race_no"]),
        win=(pl.col("finish_pos") == 1).cast(pl.Float64),
        plc=(pl.col("finish_pos") <= 3).cast(pl.Float64),
        cfg_key=pl.concat_str(
            [pl.col("venue"), pl.col("surface").fill_null("TURF"), pl.col("distance_m").cast(pl.Utf8)],
            separator="|",
        ),
    )
    .filter(pl.col("field_size") >= 6)
    .sort(["race_date", "race_no", "saddle"])
)

# ------------------------------------------------------------------ as-of draw bias
g = ["cfg_key", "draw"]
exp_w = 1.0 / pl.col("field_size")
exp_p = 3.0 / pl.col("field_size")
df = df.with_columns(
    _ow=pl.col("win").cum_sum().over(g) - pl.col("win"),
    _ew=exp_w.cum_sum().over(g) - exp_w,
    _op=pl.col("plc").cum_sum().over(g) - pl.col("plc"),
    _ep=exp_p.cum_sum().over(g) - exp_p,
    _n=pl.col("win").cum_count().over(g) - 1,
).with_columns(
    draw_win_bias=(pl.col("_ow") + K_SHRINK) / (pl.col("_ew") + K_SHRINK) - 1.0,
    draw_place_bias=(pl.col("_op") + 3 * K_SHRINK) / (pl.col("_ep") + 3 * K_SHRINK) - 1.0,
    draw_rel=(pl.col("draw") - 1) / (pl.col("field_size") - 1),
)
df = df.with_columns(
    draw_bias_x_big=pl.col("draw_win_bias") * (pl.col("field_size") - 12).cast(pl.Float64),
    prior_n=pl.col("_n"),
)

# market control: overround-adjusted implied win prob from the closing line
df = df.with_columns(inv=1.0 / pl.col("win_odds")).with_columns(
    mkt_p=pl.col("inv") / pl.col("inv").sum().over(["race_date", "venue", "race_no"])
)
d = df.drop_nulls(["mkt_p"]).to_pandas()
d["rid"] = d.race_date.astype(str) + "_" + d.venue + "_" + d.race_no.astype(str)
d["lmkt"] = np.log(d.mkt_p.clip(1e-4, 1))
d["season_y"] = d.race_date.map(lambda x: x.year + (1 if x.month >= 9 else 0))
d = d.sort_values(["race_date", "race_no", "saddle"]).reset_index(drop=True)

# expanding z-scores (as-of): standardise each factor by its own history up to that date
for c in FACT + CONTROL_BASE:
    s = d[c].astype(float)
    m = s.expanding(min_periods=500).mean().shift(1)
    sd = s.expanding(min_periods=500).std().shift(1)
    d[c + "_z"] = ((s - m) / sd).fillna(0.0).clip(-5, 5)
Z = [c + "_z" for c in CONTROL_BASE + FACT]

print(f"rows {len(d):,}  races {d.rid.nunique():,}  {d.race_date.min()}..{d.race_date.max()}")
print(f"configurations (venue|surface|distance): {d.cfg_key.nunique()}")
print("\n=== raw bias sanity: ST|TURF|1000 mean draw_win_bias by draw (latest season) ===")
last = d[(d.season_y == d.season_y.max()) & (d.cfg_key == "ST|TURF|1000")]
print(last.groupby("draw").draw_win_bias.mean().round(3).to_string())

print("\n=== factor correlations (want ~0 with draw_rel to be incremental) ===")
print(d[Z].corr().round(2).to_string())


# ------------------------------------------------------------------ conditional logit
def struct(sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    o = np.argsort(sub.rid.values, kind="stable")
    r = sub.rid.values[o]
    st = np.r_[0, np.where(r[1:] != r[:-1])[0] + 1]
    ct = np.diff(np.r_[st, len(r)])
    wn = sub.win.values[o]
    wr = np.full(len(st), -1)
    for i, (s, c) in enumerate(zip(st, ct, strict=True)):
        seg = wn[s : s + c]
        if seg.any():
            wr[i] = s + seg.argmax()
    return o, st, ct, wr


def fit(sub: pd.DataFrame, zc: list[str], se: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    o, st, ct, wr = struct(sub)
    zm = sub[zc].values[o]
    lm = sub.lmkt.values[o]
    v = wr >= 0
    wv = wr[v]
    k = len(zc)

    def nll(t: np.ndarray) -> float:
        s = t[0] * lm + zm @ t[1:]
        sm = np.maximum.reduceat(s, st)
        e = np.exp(s - np.repeat(sm, ct))
        return float(np.sum((np.log(np.add.reduceat(e, st)) + sm)[v] - s[wv]))

    r = minimize(nll, np.r_[1.0, np.zeros(k)], method="L-BFGS-B")
    th = r.x
    if not se:
        return th, None
    eps = 1e-4

    def grad(t: np.ndarray) -> np.ndarray:
        gg = np.zeros(k + 1)
        for i in range(k + 1):
            a = t.copy()
            a[i] += eps
            b = t.copy()
            b[i] -= eps
            gg[i] = (nll(a) - nll(b)) / (2 * eps)
        return gg

    h = np.zeros((k + 1, k + 1))
    for i in range(k + 1):
        a = th.copy()
        a[i] += eps
        b = th.copy()
        b[i] -= eps
        h[:, i] = (grad(a) - grad(b)) / (2 * eps)
    h = (h + h.T) / 2
    try:
        sd = np.sqrt(np.diag(np.linalg.inv(h)))
    except np.linalg.LinAlgError:
        sd = np.full(k + 1, np.nan)
    return th, sd


fit_from = d.season_y.min() + 3  # give the expanding counts ~3 seasons to warm up
dd = d[d.season_y >= fit_from]
th, sd = fit(dd, Z)
assert sd is not None
print("\n" + "=" * 88)
print(f"conditional logit  score = a*log(market) + sum b*z(factor)   [seasons {fit_from}-{d.season_y.max()}]")
print("=" * 88)
print(f"  {'market a':20s} {th[0]:+.3f}  (t={th[0] / sd[0]:+.1f})")
rep: dict[str, object] = {
    "market_a": round(float(th[0]), 3),
    "market_t": round(float(th[0] / sd[0]), 1),
    "factors": [],
}
for i, c in enumerate(CONTROL_BASE + FACT):
    b, t = th[i + 1], th[i + 1] / sd[i + 1]
    flag = "  <== residual signal" if (b > 0 and t > 2) else ("  (significantly negative)" if t < -2 else "")
    tag = "(control)" if c in CONTROL_BASE else ""
    print(f"  {c:20s} {b:+.3f}  (t={t:+.1f}){flag} {tag}")
    rep["factors"].append({"factor": c, "beta": round(float(b), 3), "t": round(float(t), 1)})  # type: ignore[attr-defined]

# ------------------------------------------------------------------ walk-forward
d["p_int"] = np.nan
for sy in range(fit_from + 2, int(d.season_y.max()) + 1):
    tr = d[(d.season_y >= fit_from) & (d.season_y < sy)]
    te = d[d.season_y == sy]
    if te.empty:
        continue
    th_, _ = fit(tr, Z, se=False)
    sc = th_[0] * te.lmkt.values + te[Z].values @ th_[1:]
    mx = pd.Series(sc, index=te.rid.values).groupby(level=0).transform("max").values
    ex = np.exp(sc - mx)
    d.loc[te.index, "p_int"] = ex / pd.Series(ex, index=te.rid.values).groupby(level=0).transform("sum").values
bt = d.dropna(subset=["p_int"]).copy()
bt["ev"] = bt.p_int * bt.win_odds - 1
bt["ret1"] = bt.win * bt.win_odds


def blk(sub: pd.DataFrame, lbl: str) -> dict[str, float] | None:
    if sub.empty:
        print(f"  {lbl:30s} no bets")
        return None
    gr = sub.groupby("rid").agg(c=("win", "size"), r=("ret1", "sum"))
    m = len(gr)
    cc, rv = gr.c.values.astype(float), gr.r.values
    roi = (rv.sum() - cc.sum()) / cc.sum()
    rs = np.random.RandomState(0)
    o = np.empty(3000)
    for i in range(3000):
        ix = rs.randint(0, m, m)
        tc = cc[ix].sum()
        o[i] = (rv[ix].sum() - tc) / tc if tc else 0
    lo, hi = np.percentile(o, [2.5, 97.5])
    print(
        f"  {lbl:30s} bets={len(sub):5d} races={m:5d} hit={sub.win.mean():.1%} "
        f"avg odds={sub.win_odds.mean():.1f}  ROI={roi:+.1%}  [95%CI {lo:+.1%},{hi:+.1%}]  P(<0)={(o < 0).mean():.0%}"
    )
    return {"bets": int(len(sub)), "roi": round(float(roi), 4), "ci": [round(float(lo), 4), round(float(hi), 4)]}


print("\n" + "=" * 88 + "\nwalk-forward (refit each season), market + draw-bias model\n" + "=" * 88)
print(f"OOS races {bt.rid.nunique():,}  {bt.race_date.min()}..{bt.race_date.max()}")
rep["walk_forward"] = {}
for thr in (0.0, 0.05, 0.10, 0.20):
    r_ = blk(bt[bt.ev >= thr], f"market+draw  EV>={thr:.0%}")
    if r_:
        rep["walk_forward"][f"ev>={thr:.2f}"] = r_  # type: ignore[index]
print("\n  by season, EV>=5%:")
for sy in sorted(bt.season_y.unique()):
    blk(bt[(bt.ev >= 0.05) & (bt.season_y == sy)], f"  {sy - 1}-{sy}")

OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
print(f"\nsaved {OUT.name}")
