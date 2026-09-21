"""trial_signal -- does the market under-price barrier-trial form?

Standalone study in the style of ``w456.py`` / ``draw_bias.py``: strictly **as-of** signals from
the barrier-trial archive (#4, backfilled to 2010-11 via ``hkjc scrape-trials --since``), the
closing-line market and the layoff as controls, one within-race conditional logit for t-stats,
then a per-season walk-forward with EV-gated betting.

Signals (from trials strictly before race_date; "since last race" = trial_date > prev race):
  bt_days_since     days from the horse's latest trial to the race (capped at 120; 120 if none)
  bt_margin_last    latest trial: seconds behind the batch winner (0 = won the trial)
  bt_rank_last      latest trial: (rank-1)/(n-1) within its batch (0 = fastest)
  bt_n_between      number of trials between the previous race and this one
  bt_failed_between 1 if any trial since the previous race was Failed / Required-to-...
  bt_easy_win       1 if the latest trial was won and the comment says so ("easily"/"impressive")
  bt_first_up_trial 1 if returning from a >=60-day break *with* a trial in the gap

Controls: log market probability and log(1 + days since last run) -- trials cluster around
spells, so the layoff must be partialled out before crediting a trial.

**As-of discipline:** trial rows are joined per horse and filtered on trial_date < race_date;
"last trial" is an asof join backward; the z-scores fed to the logit are expanding.

Run from the repo root with the venv:  .venv/bin/python reports/trial_signal.py
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

OUT = Path(__file__).with_name("trial_signal_report.json")
CAP_DAYS = 120
CONTROLS = ["layoff"]
FACT = [
    "bt_days_since",
    "bt_margin_last",
    "bt_rank_last",
    "bt_n_between",
    "bt_failed_between",
    "bt_easy_win",
    "bt_first_up_trial",
]
EASY = r"(?i)\b(easily|impressive|comfortabl|strongly|well in hand)"

# ------------------------------------------------------------------ load
cfg = get_config()
res = _read_raw(
    cfg,
    "results",
    columns=["race_date", "venue", "race_no", "saddle", "horse_id", "finish_pos", "win_odds"],
)
runs = (
    res.drop_nulls(["finish_pos", "horse_id"])
    .filter(pl.col("race_date") >= pl.date(2008, 4, 1))
    .sort(["horse_id", "race_date", "race_no"])
    .with_columns(
        prev_race=pl.col("race_date").shift(1).over("horse_id"),
        field_size=pl.len().over(["race_date", "venue", "race_no"]),
        win=(pl.col("finish_pos") == 1).cast(pl.Float64),
    )
    .with_columns(
        days_since_last_run=(pl.col("race_date") - pl.col("prev_race")).dt.total_days(),
        _rid=pl.concat_str(
            [pl.col("race_date").cast(pl.Utf8), pl.col("venue"), pl.col("race_no").cast(pl.Utf8)],
            separator="_",
        ),
        _row=pl.int_range(pl.len()),
    )
)

trials = pl.read_parquet(f"{cfg.paths.raw_dir}/barrier_trials/*.parquet", missing_columns="insert")
tr = (
    trials.drop_nulls(["horse_id", "trial_date"])
    .with_columns(
        bkey=pl.concat_str(
            [pl.col("trial_date").cast(pl.Utf8), pl.col("location"), pl.col("batch").cast(pl.Utf8)],
            separator="|",
        )
    )
    .with_columns(
        _n=pl.col("time_s").is_not_null().sum().over("bkey"),  # timed runners only
        _rank=pl.col("time_s").rank("min").over("bkey"),
        _best=pl.col("time_s").min().over("bkey"),
    )
    .with_columns(
        margin=(pl.col("time_s") - pl.col("_best")).clip(lower_bound=0.0),
        rank_rel=pl.when(pl.col("_rank").is_null())
        .then(None)
        .when(pl.col("_n") > 1)
        .then((pl.col("_rank") - 1) / (pl.col("_n") - 1))
        .otherwise(0.0),
        failed=pl.col("result").fill_null("").str.contains("(?i)fail|required").cast(pl.Float64),
        easy=(
            (pl.col("_rank") == 1).fill_null(value=False)
            & pl.col("comment").fill_null("").str.contains(EASY)
        ).cast(pl.Float64),
    )
    .select("horse_id", "trial_date", "margin", "rank_rel", "failed", "easy")
    .sort(["horse_id", "trial_date"])
)
print(f"trials: {tr.height:,} rows  {tr['trial_date'].min()} -> {tr['trial_date'].max()}")

# ------------------------------------------------------------------ as-of trial features
# (a) latest trial strictly before the race: asof backward, then drop same-day matches.
runs_sorted = runs.sort("race_date")
last = runs_sorted.join_asof(
    tr.rename({"trial_date": "last_trial"}).sort("last_trial"),
    left_on="race_date",
    right_on="last_trial",
    by="horse_id",
    strategy="backward",
    allow_exact_matches=False,  # strictly prior
).with_columns(
    _gap=(pl.col("race_date") - pl.col("last_trial")).dt.total_days(),
)
last = last.with_columns(
    _valid=(pl.col("_gap") > 0) & (pl.col("_gap") <= CAP_DAYS),
).with_columns(
    bt_days_since=pl.when(pl.col("_valid")).then(pl.col("_gap")).otherwise(CAP_DAYS).cast(pl.Float64),
    bt_margin_last=pl.when(pl.col("_valid")).then(pl.col("margin")).otherwise(None),
    bt_rank_last=pl.when(pl.col("_valid")).then(pl.col("rank_rel")).otherwise(None),
    bt_easy_win=pl.when(pl.col("_valid")).then(pl.col("easy")).otherwise(0.0),
)

# (b) trials between the previous race and this one: horse join + window filter + aggregate.
between = (
    runs.select("_row", "horse_id", "race_date", "prev_race")
    .join(tr.select("horse_id", "trial_date", "failed"), on="horse_id", how="inner")
    .filter(
        (pl.col("trial_date") < pl.col("race_date"))
        & (
            pl.col("prev_race").is_null()
            | (pl.col("trial_date") > pl.col("prev_race"))
        )
        & ((pl.col("race_date") - pl.col("trial_date")).dt.total_days() <= 365)
    )
    .group_by("_row")
    .agg(bt_n_between=pl.len().cast(pl.Float64), bt_failed_between=pl.col("failed").max())
)
feat = (
    last.join(between, on="_row", how="left")
    .with_columns(
        bt_n_between=pl.col("bt_n_between").fill_null(0.0),
        bt_failed_between=pl.col("bt_failed_between").fill_null(0.0),
    )
    .with_columns(
        bt_first_up_trial=(
            (pl.col("days_since_last_run").fill_null(999) >= 60) & (pl.col("bt_n_between") > 0)
        ).cast(pl.Float64),
        layoff=(1.0 + pl.col("days_since_last_run").fill_null(365).cast(pl.Float64)).log(),
        inv=1.0 / pl.col("win_odds"),
    )
    .with_columns(mkt_p=pl.col("inv") / pl.col("inv").sum().over(["race_date", "venue", "race_no"]))
)

d = feat.drop_nulls(["mkt_p"]).to_pandas()
d["rid"] = d["_rid"]
d["lmkt"] = np.log(d.mkt_p.clip(1e-4, 1))
d["season_y"] = d.race_date.map(lambda x: x.year + (1 if x.month >= 9 else 0))
d = d.sort_values(["race_date", "race_no", "saddle"]).reset_index(drop=True)
# runners with no recent trial: margin / rank unknown -> fill with the as-of running median
for c in ("bt_margin_last", "bt_rank_last"):
    med = d[c].expanding(min_periods=200).median().shift(1)
    d[c] = d[c].fillna(med).fillna(0.0)

trial_seasons = d[d.bt_n_between > 0].season_y
print(f"runner-rows {len(d):,}  races {d.rid.nunique():,}  {d.race_date.min().date()}..{d.race_date.max().date()}")
print(f"rows with a trial since last race: {(d.bt_n_between > 0).mean():.1%}  (seasons {trial_seasons.min()}-{trial_seasons.max()})")
print("\n=== coverage by season: share of runners with >=1 trial since last race ===")
print((d.groupby("season_y").bt_n_between.apply(lambda s: (s > 0).mean())).round(3).to_string())

# expanding z-scores (as-of)
for c in FACT + CONTROLS:
    s = d[c].astype(float)
    m = s.expanding(min_periods=500).mean().shift(1)
    sd = s.expanding(min_periods=500).std().shift(1)
    d[c + "_z"] = ((s - m) / sd.replace(0, np.nan)).fillna(0.0).clip(-5, 5)
Z = [c + "_z" for c in CONTROLS + FACT]

print("\n=== raw sanity: win rate by bt_failed_between / bt_easy_win (seasons with trials) ===")
cov = d[d.season_y >= trial_seasons.min()]
print(cov.groupby("bt_failed_between").win.agg(["mean", "size"]).round(3).to_string())
print(cov.groupby("bt_easy_win").win.agg(["mean", "size"]).round(3).to_string())
print("\n=== factor correlations ===")
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


# fit only on seasons where the trial archive exists (+1 season warm-up for the windows)
fit_from = int(trial_seasons.min()) + 1
dd = d[d.season_y >= fit_from]
th, sd = fit(dd, Z)
assert sd is not None
print("\n" + "=" * 88)
print(f"conditional logit  score = a*log(market) + sum b*z(factor)   [seasons {fit_from}-{d.season_y.max()}]")
print("=" * 88)
print(f"  {'market a':20s} {th[0]:+.3f}  (t={th[0] / sd[0]:+.1f})")
rep: dict[str, object] = {
    "seasons": [fit_from, int(d.season_y.max())],
    "market_a": round(float(th[0]), 3),
    "market_t": round(float(th[0] / sd[0]), 1),
    "factors": [],
}
for i, c in enumerate(CONTROLS + FACT):
    b, t = th[i + 1], th[i + 1] / sd[i + 1]
    flag = "  <== residual signal" if (abs(t) > 2 and c not in CONTROLS) else ""
    tag = "(control)" if c in CONTROLS else ""
    print(f"  {c:20s} {b:+.3f}  (t={t:+.1f}){flag} {tag}")
    rep["factors"].append({"factor": c, "beta": round(float(b), 3), "t": round(float(t), 1)})  # type: ignore[attr-defined]

# ------------------------------------------------------------------ walk-forward
d["p_int"] = np.nan
for sy in range(fit_from + 2, int(d.season_y.max()) + 1):
    trn = d[(d.season_y >= fit_from) & (d.season_y < sy)]
    te = d[d.season_y == sy]
    if te.empty:
        continue
    th_, _ = fit(trn, Z, se=False)
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


print("\n" + "=" * 88 + "\nwalk-forward (refit each season), market + layoff + trial model\n" + "=" * 88)
print(f"OOS races {bt.rid.nunique():,}  {bt.race_date.min().date()}..{bt.race_date.max().date()}")
rep["walk_forward"] = {}
for thr in (0.0, 0.05, 0.10, 0.20):
    r_ = blk(bt[bt.ev >= thr], f"market+trial  EV>={thr:.0%}")
    if r_:
        rep["walk_forward"][f"ev>={thr:.2f}"] = r_  # type: ignore[index]
print("\n  by season, EV>=5%:")
for sy in sorted(bt.season_y.unique()):
    blk(bt[(bt.ev >= 0.05) & (bt.season_y == sy)], f"  {sy - 1}-{sy}")

OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
print(f"\nsaved {OUT.name}")
