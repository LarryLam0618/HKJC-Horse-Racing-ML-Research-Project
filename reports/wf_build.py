"""Walk-forward OOS predictions 2010-2026 (train on prior seasons, predict next)."""
from __future__ import annotations
import numpy as np, pandas as pd
import lightgbm as lgb
RNG = 42
OUT = "/private/tmp/claude-501/-Users-suetyinglam-Downloads-HKJC-Horse-Racing-ML-Research-Project-main/81331658-de3f-4006-8783-feab579385e5/scratchpad/preds_wf.parquet"

res = pd.read_csv("results.csv", low_memory=False)
races = pd.read_csv("races.csv", low_memory=False)
res = res[res.race_date >= "2008-04-01"].copy()
res["dt"] = pd.to_datetime(res.race_date)
res["rid"] = res.race_date + "_" + res.venue + "_" + res.race_no.astype(str)
res["saddle"] = res.saddle.astype(str)
res["fin"] = pd.to_numeric(res.finish_pos, errors="coerce")
res = res.dropna(subset=["fin"]); res["fin"] = res.fin.astype(int)
races["rid"] = races.race_date + "_" + races.venue + "_" + races.race_no.astype(str)
def class_ord(s):
    s = str(s)
    if s.startswith("Class "):
        try: return float(s.split()[1])
        except Exception: return np.nan
    if "Group" in s: return 0.0
    if "Griffin" in s: return 4.5
    return 3.0
races["class_ord"] = races.race_class.map(class_ord)
races["is_turf"] = (races.surface == "TURF").astype(int)
GOING = {"GOOD": 0, "GOOD TO FIRM": 1, "FIRM": 2, "GOOD TO YIELDING": -1, "YIELDING": -2,
         "YIELDING TO SOFT": -3, "SOFT": -4, "HEAVY": -5, "WET SLOW": -3, "WET FAST": 0}
races["going_ord"] = races.going.map(GOING).fillna(0)
res = res.merge(races[["rid", "distance_m", "class_ord", "is_turf", "going_ord"]], on="rid", how="left")

fs = res.groupby("rid").saddle.transform("size")
res["field_size"] = fs
res["n_place"] = np.where(fs >= 7, 3, np.where(fs >= 5, 2, 0))
res["win_lbl"] = (res.fin == 1).astype(int)
res["plc_lbl"] = ((res.fin <= res.n_place) & (res.n_place > 0)).astype(int)
res = res.sort_values(["dt", "rid", "saddle"]).reset_index(drop=True)

def prior_rate(df, key, lbl):
    g = df.groupby(key); cnt = g.cumcount(); won = g[lbl].cumsum() - df[lbl]
    return cnt, won / cnt.where(cnt > 0)
res["h_runs"], res["h_winrate"] = prior_rate(res, "horse_id", "win_lbl")
_, res["h_plcrate"] = prior_rate(res, "horse_id", "plc_lbl")
res["j_runs"], res["j_winrate"] = prior_rate(res, "jockey_name", "win_lbl")
_, res["j_plcrate"] = prior_rate(res, "jockey_name", "plc_lbl")
res["t_runs"], res["t_winrate"] = prior_rate(res, "trainer_name", "win_lbl")
_, res["t_plcrate"] = prior_rate(res, "trainer_name", "plc_lbl")
res["jt"] = res.jockey_name.astype(str) + "|" + res.trainer_name.astype(str)
_, res["jt_plcrate"] = prior_rate(res, "jt", "plc_lbl")
res["jh"] = res.jockey_name.astype(str) + "|" + res.horse_id.astype(str)
res["jh_runs"], res["jh_plcrate"] = prior_rate(res, "jh", "plc_lbl")
res["j_form"] = (res.groupby("jockey_name").win_lbl.shift(1)
                 .groupby(res.jockey_name).rolling(200, min_periods=20).mean().reset_index(level=0, drop=True))
res["t_form"] = (res.groupby("trainer_name").win_lbl.shift(1)
                 .groupby(res.trainer_name).rolling(200, min_periods=20).mean().reset_index(level=0, drop=True))
res["prev_dt"] = res.groupby("horse_id").dt.shift(1)
res["days_since"] = (res.dt - res.prev_dt).dt.days
res["days_career"] = (res.dt - res.groupby("horse_id").dt.transform("min")).dt.days
res["fin_l1"] = res.groupby("horse_id").fin.shift(1)
res["avg_fin_l3"] = (res.groupby("horse_id").fin.shift(1)
                     .groupby(res.horse_id).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True))
res["win_l3"] = (res.groupby("horse_id").win_lbl.shift(1)
                 .groupby(res.horse_id).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True))
res["prev_dist"] = res.groupby("horse_id").distance_m.shift(1)
res["dist_diff"] = res.distance_m - res.prev_dist
res["hv"] = res.horse_id + "|" + res.venue
_, res["hv_plcrate"] = prior_rate(res, "hv", "plc_lbl")
res["hd"] = res.horse_id + "|" + res.distance_m.astype(str)
_, res["hd_plcrate"] = prior_rate(res, "hd", "plc_lbl")
res["draw"] = pd.to_numeric(res.draw, errors="coerce")
res["draw_pct"] = res.draw / res.field_size
res["wt"] = pd.to_numeric(res.actual_weight, errors="coerce")
res["wt_rel"] = res.wt - res.groupby("rid").wt.transform("mean")
res["mon"] = res.dt.dt.month

FEATS = ["h_runs", "h_winrate", "h_plcrate", "j_runs", "j_winrate", "j_plcrate", "j_form",
         "t_runs", "t_winrate", "t_plcrate", "t_form", "jt_plcrate", "jh_runs", "jh_plcrate",
         "days_since", "days_career", "fin_l1", "avg_fin_l3", "win_l3", "dist_diff",
         "hv_plcrate", "hd_plcrate", "draw", "draw_pct", "wt", "wt_rel", "distance_m",
         "class_ord", "is_turf", "going_ord", "field_size", "mon"]

def season(dstr):
    y, m = int(dstr[:4]), int(dstr[5:7])
    return y if m >= 9 else y - 1     # season start year

res["season_y"] = res.race_date.map(season)
res["p_win"] = np.nan
res["p_plc"] = np.nan

for sy in range(2010, 2027):
    tr_mask = res.season_y < sy
    te_mask = res.season_y == sy
    if te_mask.sum() == 0 or tr_mask.sum() < 5000:
        continue
    tr = res[tr_mask]
    for lbl, col in [("win_lbl", "p_win"), ("plc_lbl", "p_plc")]:
        m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=63,
                               subsample=0.8, colsample_bytree=0.8, min_child_samples=80,
                               random_state=RNG, n_jobs=-1, verbose=-1)
        m.fit(tr[FEATS], tr[lbl])
        res.loc[te_mask, col] = m.predict_proba(res.loc[te_mask, FEATS])[:, 1]
    print(f"season {sy}-{sy+1}: train {tr_mask.sum():,}  test {te_mask.sum():,}")

oos = res.dropna(subset=["p_win"]).copy()
oos["p_win"] = oos.p_win / oos.groupby("rid").p_win.transform("sum")
oos["p_plc"] = (oos.p_plc / oos.groupby("rid").p_plc.transform("sum")
                * oos.groupby("rid").n_place.transform("first"))
oos["mkt_p_raw"] = 1 / oos.win_odds
oos["mkt_p"] = oos.mkt_p_raw / oos.groupby("rid").mkt_p_raw.transform("sum")
keep = ["rid", "race_date", "venue", "saddle", "horse_id", "jockey_name", "fin", "field_size",
        "n_place", "win_lbl", "plc_lbl", "win_odds", "distance_m", "is_turf", "draw",
        "season_y", "p_win", "p_plc", "mkt_p"]
oos[keep].to_parquet(OUT)
print(f"\nOOS rows {len(oos):,}  races {oos.rid.nunique():,}  "
      f"{oos.race_date.min()} .. {oos.race_date.max()}  -> {OUT}")
