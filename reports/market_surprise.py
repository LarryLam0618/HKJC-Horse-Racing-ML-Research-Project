"""market_surprise.py -- 薄 join：w456.py 嘅 13-factor model vs 即時市場（一次性、觀察用途）。

刻意保持「薄」：唔碰 w456.py 本身，唔起 feature store / model zoo，淨係：
  1. 原封不動抄 w456.py 嘅 feature engineering 邏輯，喺全歷史數據 fit 一次 model。
  2. 用返一條 plain GraphQL POST（同 hkjc-api-master 個 oddsTracker.js 今日成程用緊
     嗰種方式一樣，唔係 M7 個 byte-exact whitelist client）攞返即日賽事卡：runners、
     draw、體重(currentWeight)、練馬師/騎師、即時 WIN 賠率 -- 一次過攞晒，唔使再另外
     叫 M7 個 client。
  3. 逐隻馬將距離最近一次落場計返嘅因子，對齊做「as-of 今日」。
  4. score = a*log(market_p) + factors·betas，每場 softmax 攞 model_p。
  5. Market Surprise = model_p - market_p，輸出 CSV + 印排行。

注意（誠實講清楚啲近似位）：
  - j_winr / drop_x_trsr 用嘅係「呢個騎師/練馬師去到最新一場為止嘅平均勝率」，
    唔係訓練時嗰個逐場 leave-one-out 累積值，兩者好接近但唔係 byte-exact。
  - 體重(currentWeight)喺賽日好早（截飛前好耐）可能仲未公佈，呢種情況 bw_chg 等
    因子會係 NaN -> z-score 用 0（中性），唔會令個 model 報錯，但都要意識到早於
    公佈體重之前，「體重回歸」呢個訊號其實用唔到。
  - 淨係用 WIN 賠率做 market_p（同 w456.py 訓練時一致），QIN 唔關呢個 model 事。

用法：
    python market_surprise.py --date 2026-09-06 --venue ST
"""

from __future__ import annotations

import argparse
import os
import sys

import httpx
import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPHQL_URL = "https://info.cld.hkjc.com/graphql/base/"

# 呢條 query 一字不改抄自 hkjc-api-master/src/query/horseRacingQuery.ts 嘅 horseQuery ---
# HKJC 個 gateway 會 whitelist 已知嘅 query 字串，自己砌一條字眼唔同嘅 query 會俾佢
# WHITELIST_ERROR 拒絕；呢條已經喺今日全程用嚟 poll ST 場次，證實冇問題。
HORSE_QUERY = """
fragment raceFragment on Race {
  id
  no
  status
  raceName_en
  raceName_ch
  postTime
  country_en
  country_ch
  distance
  wageringFieldSize
  go_en
  go_ch
  ratingType
  raceTrack {
    description_en
    description_ch
  }
  raceCourse {
    description_en
    description_ch
    displayCode
  }
  claCode
  raceClass_en
  raceClass_ch
  judgeSigns {
    value_en
  }
}

fragment racingBlockFragment on RaceMeeting {
  jpEsts: pmPools(
    oddsTypes: [WIN, PLA, TCE, TRI, FF, QTT, DT, TT, SixUP]
    filters: ["jackpot", "estimatedDividend"]
  ) {
    leg {
      number
      races
    }
    oddsType
    jackpot
    estimatedDividend
    mergedPoolId
  }
  poolInvs: pmPools(
    oddsTypes: [WIN, PLA, QIN, QPL, CWA, CWB, CWC, IWN, FCT, TCE, TRI, FF, QTT, DBL, TBL, DT, TT, SixUP]
  ) {
    id
    leg {
      races
    }
  }
  penetrometerReadings(filters: ["first"]) {
    reading
    readingTime
  }
  hammerReadings(filters: ["first"]) {
    reading
    readingTime
  }
  changeHistories(filters: ["top3"]) {
    type
    time
    raceNo
    runnerNo
    horseName_ch
    horseName_en
    jockeyName_ch
    jockeyName_en
    scratchHorseName_ch
    scratchHorseName_en
    handicapWeight
    scrResvIndicator
  }
}

query raceMeetings($date: String, $venueCode: String) {
  timeOffset {
    rc
  }
  activeMeetings: raceMeetings {
    id
    venueCode
    date
    status
    races {
      no
      postTime
      status
      wageringFieldSize
    }
  }
  raceMeetings(date: $date, venueCode: $venueCode) {
    id
    status
    venueCode
    date
    totalNumberOfRace
    currentNumberOfRace
    dateOfWeek
    meetingType
    totalInvestment
    country {
      code
      namech
      nameen
      seq
    }
    races {
      ...raceFragment
      runners {
        id
        no
        standbyNo
        status
        name_ch
        name_en
        horse {
          id
          code
        }
        color
        barrierDrawNumber
        handicapWeight
        currentWeight
        currentRating
        internationalRating
        gearInfo
        racingColorFileName
        allowance
        trainerPreference
        last6run
        saddleClothNo
        trumpCard
        priority
        finalPosition
        deadHeat
        winOdds
        jockey {
          code
          name_en
          name_ch
        }
        trainer {
          code
          name_en
          name_ch
        }
      }
    }
    obSt: pmPools(oddsTypes: [WIN, PLA]) {
      leg {
        races
      }
      oddsType
      comingleStatus
    }
    poolInvs: pmPools(
      oddsTypes: [WIN, PLA, QIN, QPL, CWA, CWB, CWC, IWN, FCT, TCE, TRI, FF, QTT, DBL, TBL, DT, TT, SixUP]
    ) {
      id
      leg {
        number
        races
      }
      status
      sellStatus
      oddsType
      investment
      mergedPoolId
      lastUpdateTime
    }
    ...racingBlockFragment
    pmPools(oddsTypes: []) {
      id
    }
    jkcInstNo: foPools(oddsTypes: [JKC], filters: ["top"]) {
      instNo
    }
    tncInstNo: foPools(oddsTypes: [TNC], filters: ["top"]) {
      instNo
    }
  }
}
"""

FACT = [
    "pace_close3", "late_rel3", "led_held3", "hidden_hp3",
    "bw_chg", "bw_vs_avg", "bw_up_fresh", "bw_drop",
    "is_drop", "drop_x_trsr",
    "draw_pct", "avg_fin_l3", "j_winr",
]


# ───────────────────────── 攞即日賽事卡 ─────────────────────────

def load_live_win_odds(odds_json_path: str) -> dict[str, dict[str, float]]:
    """讀 oddsTracker.js（hkjc-api-master）已經儲低嘅 win_qin_history.json，攞返
    最後一個 snapshot 嘅 WIN 賠率。main horseQuery 個 runners.winOdds 呢個時候會係
    空字串 -- 真正嘅即時賠率喺獨立嗰條 horseOddsQuery/pmPools，即係呢個 tracker
    已經成程幫你 poll 緊嗰啲。回傳 {raceNo(str): {saddle(str, 例如"01"): oddsValue}}。
    """
    import json
    with open(odds_json_path, encoding="utf-8") as f:
        raw = json.load(f)
    snapshots = raw if isinstance(raw, list) else raw.get("snapshots", [])
    if not snapshots:
        raise RuntimeError(f"{odds_json_path} 冇任何 snapshot")
    last = snapshots[-1]
    print(f"[live odds] 用返 {last['datetime']} 嗰個 snapshot（共 {len(snapshots)} 個時間點）")
    out = {}
    for race in last["races"]:
        out[str(race["raceNo"])] = {k: v for k, v in (race.get("win") or {}).items()}
    return out


def fetch_card(date_str: str, venue: str) -> dict:
    resp = httpx.post(
        GRAPHQL_URL,
        json={"query": HORSE_QUERY, "variables": {"date": date_str, "venueCode": venue}},
        headers={"Content-Type": "application/json"},
        timeout=20.0,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("errors"):
        raise RuntimeError(str(payload["errors"]))
    meetings = (payload.get("data") or {}).get("raceMeetings") or []
    if not meetings:
        raise RuntimeError(f"HKJC 冇 {date_str} {venue} 呢個場次")
    return meetings[0]


# ───────────────────────── 重用 w456.py 嘅 feature engineering（原封不動）─────────────────────────

def cord(s):
    s = str(s)
    if s.startswith("Class "):
        try:
            return float(s.split()[1])
        except Exception:
            return np.nan
    if "Group" in s:
        return 0.5
    if "Griffin" in s:
        return 4.5
    return 3.0


def prior_rate(df, key, lbl):
    g = df.groupby(key)
    c = g.cumcount()
    s = g[lbl].cumsum() - df[lbl]
    return c, s / c.where(c > 0)


def lag(run, col, fn, w=4):
    return (run.groupby("horse_id")[col].shift(1)
            .groupby(run.horse_id).rolling(w, min_periods=1).agg(fn).reset_index(level=0, drop=True))


def struct(sub):
    o = np.argsort(sub.rid.values, kind="stable")
    r = sub.rid.values[o]
    st = np.r_[0, np.where(r[1:] != r[:-1])[0] + 1]
    ct = np.diff(np.r_[st, len(r)])
    wn = sub.win.values[o]
    wr = np.full(len(st), -1)
    for i, (s, c) in enumerate(zip(st, ct)):
        seg = wn[s:s + c]
        if seg.any():
            wr[i] = s + seg.argmax()
    return o, st, ct, wr


def fit_logit(sub, zc):
    o, st, ct, wr = struct(sub)
    Zm = sub[zc].values[o]
    lm = sub.lmkt.values[o]
    v = wr >= 0
    wv = wr[v]
    k = len(zc)

    def nll(t):
        s = t[0] * lm + Zm @ t[1:]
        sm = np.maximum.reduceat(s, st)
        e = np.exp(s - np.repeat(sm, ct))
        return float(np.sum((np.log(np.add.reduceat(e, st)) + sm)[v] - s[wv]))

    r = minimize(nll, np.r_[1.0, np.zeros(k)], method="L-BFGS-B")
    return r.x


def build_history(proj_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """跟返 w456.py 個邏輯，計返成個歷史 res（逐次落場）+ run（sectionals，未 lag 之前）。"""
    res = pd.read_csv(f"{proj_dir}/results.csv", low_memory=False)
    races = pd.read_csv(f"{proj_dir}/races.csv", low_memory=False)
    res = res[res.race_date >= "2008-04-01"].copy()
    res["dt"] = pd.to_datetime(res.race_date)
    res["rid"] = res.race_date + "_" + res.venue + "_" + res.race_no.astype(str)
    res["saddle"] = res.saddle.astype(str)
    res["fin"] = pd.to_numeric(res.finish_pos, errors="coerce")
    res = res.dropna(subset=["fin"])
    res["fin"] = res.fin.astype(int)

    races = races.copy()
    races["rid"] = races.race_date + "_" + races.venue + "_" + races.race_no.astype(str)
    races["distance_m"] = pd.to_numeric(races.distance_m, errors="coerce")
    races["class_ord"] = races.race_class.map(cord)
    res = res.merge(
        races[["rid", "distance_m", "class_ord", "venue"]].rename(columns={"venue": "vv"}),
        on="rid", how="left",
    )
    res["field_size"] = res.groupby("rid").saddle.transform("size")
    res["win"] = (res.fin == 1).astype(int)
    res = res.sort_values(["dt", "rid", "saddle"]).reset_index(drop=True)

    sec = pd.read_csv(f"{proj_dir}/sectionals_clean.csv", low_memory=False)
    sec["rid"] = sec.race_date + "_" + sec.venue + "_" + sec.race_no.astype(str)
    sec["saddle"] = sec.saddle.astype(str)
    sec = sec.dropna(subset=["section_index", "section_time_s"])
    sec["section_index"] = sec.section_index.astype(int)
    sec["nsec"] = sec.groupby(["rid", "saddle"]).section_index.transform("max")
    sec = sec.merge(races[["rid", "distance_m"]], on="rid", how="left")

    lastt = sec.loc[sec.groupby(["rid", "saddle"]).section_index.idxmax()][
        ["rid", "saddle", "section_time_s", "finishing_order"]
    ].rename(columns={"section_time_s": "last400"})
    first = sec[sec.section_index == 1][["rid", "saddle", "section_time_s", "running_position"]].rename(
        columns={"section_time_s": "sec1", "running_position": "early_pos"}
    )
    run = lastt.merge(first, on=["rid", "saddle"], how="left").merge(
        sec[["rid", "distance_m", "nsec"]].drop_duplicates("rid"), on="rid", how="left"
    )

    lead_early = run.groupby("rid").sec1.min().rename("lead_sec1")
    run = run.merge(lead_early, on="rid")
    grp = run.groupby(["distance_m", "nsec"]).lead_sec1
    run["pace_press"] = -((run.lead_sec1 - grp.transform("mean")) / grp.transform("std"))

    run["late_rel"] = run.groupby("rid").last400.transform("median") - run.last400
    run["pace_close"] = run.late_rel * (1.0 + np.clip(run.pace_press, 0, 2.5))
    run["led_held_hp"] = ((run.early_pos <= 3) & (run.finishing_order <= 3) & (run.pace_press > 0.4)).astype(int)
    run["hidden_hp"] = ((run.finishing_order >= 6) & (run.late_rel >= 0.35) & (run.pace_press > 0.0)).astype(int)

    run = run.merge(res[["rid", "saddle", "dt", "horse_id"]], on=["rid", "saddle"], how="left")
    run = run.dropna(subset=["dt"]).sort_values("dt")
    run["pace_close3"] = lag(run, "pace_close", "mean")
    run["late_rel3"] = lag(run, "late_rel", "mean")
    run["led_held3"] = lag(run, "led_held_hp", "sum")
    run["hidden_hp3"] = lag(run, "hidden_hp", "sum")
    secf = run[["rid", "saddle", "pace_close3", "late_rel3", "led_held3", "hidden_hp3"]]
    res = res.merge(secf, on=["rid", "saddle"], how="left")

    res["bw"] = pd.to_numeric(res.declared_weight, errors="coerce")
    res["bw_prev"] = res.groupby("horse_id").bw.shift(1)
    res["bw_chg"] = res.bw - res.bw_prev
    res["bw_career_avg"] = (res.groupby("horse_id").bw.shift(1)
                            .groupby(res.horse_id).expanding().mean().reset_index(level=0, drop=True))
    res["bw_vs_avg"] = res.bw - res.bw_career_avg
    res["days_off"] = (res.dt - res.groupby("horse_id").dt.shift(1)).dt.days
    res["bw_up_fresh"] = ((res.bw_chg > 8) & (res.days_off > 45)).astype(float)
    res["bw_drop"] = np.clip(-res.bw_chg, 0, None)

    res["prev_class"] = res.groupby("horse_id").class_ord.shift(1)
    res["class_drop"] = (res.class_ord - res.prev_class).clip(lower=0)
    res["is_drop"] = (res.class_drop > 0).astype(int)
    drp = res[res.is_drop == 1].copy()
    _, drp["tr_drop_sr"] = prior_rate(drp, "trainer_name", "win")
    res = res.merge(drp[["rid", "saddle", "tr_drop_sr"]], on=["rid", "saddle"], how="left")
    res["tr_drop_sr"] = res.tr_drop_sr.fillna(0.0)
    res["drop_x_trsr"] = res.is_drop * res.tr_drop_sr

    res["j_runs"], res["j_winr"] = prior_rate(res, "jockey_name", "win")
    res["avg_fin_l3"] = (res.groupby("horse_id").fin.shift(1)
                        .groupby(res.horse_id).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True))
    res["draw"] = pd.to_numeric(res.draw, errors="coerce")
    res["draw_pct"] = res.draw / res.field_size
    res["mkt_raw"] = 1 / res.win_odds
    res["mkt_p"] = res.mkt_raw / res.groupby("rid").mkt_raw.transform("sum")
    res["ret1"] = np.where(res.win == 1, res.win_odds, 0.0)
    return res, run


def fit_model(res: pd.DataFrame):
    d = res.dropna(subset=["mkt_p", "win_odds"]).copy()
    d = d[d.race_date >= "2011-09-01"]
    stats = {}
    for c in FACT:
        x = pd.to_numeric(d[c], errors="coerce")
        mu, sd = x.mean(), x.std()
        stats[c] = (mu, sd if sd and not np.isnan(sd) else 1.0)
        d[c + "_z"] = ((x - mu) / stats[c][1]).fillna(0.0)
    Z = [c + "_z" for c in FACT]
    d["lmkt"] = np.log(d.mkt_p.clip(1e-6))
    th = fit_logit(d, Z)
    print(f"[model] fit 咗 {len(d):,} 行 / {d.rid.nunique():,} 場 "
          f"({d.race_date.min()}..{d.race_date.max()})，market_a={th[0]:+.3f}")
    return th, stats


# ───────────────────────── as-of-今日 嘅因子 ─────────────────────────

def horse_asof_factors(res, run, j_winr_map, tr_drop_sr_map, *, horse_id, today,
                        today_weight, today_class_ord, today_draw, today_field_size,
                        jockey_name, trainer_name):
    hist = res[res.horse_id == horse_id].sort_values("dt")
    prev = hist.iloc[-1] if len(hist) else None

    bw_prev = prev.bw if prev is not None else np.nan
    bw_chg = (today_weight - bw_prev) if (today_weight is not None and pd.notna(bw_prev)) else np.nan
    bw_career_avg = hist.bw.mean() if len(hist) else np.nan
    bw_vs_avg = (today_weight - bw_career_avg) if (today_weight is not None and pd.notna(bw_career_avg)) else np.nan
    # 用 prev["dt"] 唔用 prev.dt -- 呢個 row 係 Series，attribute 形式嘅 .dt 會撞正
    # pandas 嘅 datetime accessor 命名空間，唔係取返 "dt" 呢個 column 嘅值。
    days_off = (today - prev["dt"]).days if prev is not None else np.nan
    bw_up_fresh = float(pd.notna(bw_chg) and pd.notna(days_off) and bw_chg > 8 and days_off > 45)
    bw_drop = max(-bw_chg, 0) if pd.notna(bw_chg) else 0.0

    prev_class = prev.class_ord if prev is not None else np.nan
    class_drop = max(today_class_ord - prev_class, 0) if pd.notna(prev_class) else 0.0
    is_drop = int(class_drop > 0)
    trsr = tr_drop_sr_map.get(trainer_name, 0.0)
    drop_x_trsr = is_drop * trsr

    j_winr = j_winr_map.get(jockey_name, 0.0)
    avg_fin_l3 = hist.fin.tail(3).mean() if len(hist) else np.nan
    draw_pct = (today_draw / today_field_size) if today_field_size else np.nan

    rr = run[run.horse_id == horse_id].sort_values("dt")
    pace_close3 = rr.pace_close.tail(4).mean() if len(rr) else np.nan
    late_rel3 = rr.late_rel.tail(4).mean() if len(rr) else np.nan
    led_held3 = rr.led_held_hp.tail(4).sum() if len(rr) else 0.0
    hidden_hp3 = rr.hidden_hp.tail(4).sum() if len(rr) else 0.0

    return dict(
        pace_close3=pace_close3, late_rel3=late_rel3, led_held3=led_held3, hidden_hp3=hidden_hp3,
        bw_chg=bw_chg, bw_vs_avg=bw_vs_avg, bw_up_fresh=bw_up_fresh, bw_drop=bw_drop,
        is_drop=is_drop, drop_x_trsr=drop_x_trsr, draw_pct=draw_pct, avg_fin_l3=avg_fin_l3, j_winr=j_winr,
        n_prior_runs=len(hist),
    )


# ───────────────────────── 主流程 ─────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--venue", required=True)
    ap.add_argument(
        "--odds-json",
        default=None,
        help="oddsTracker.js 產生嘅 <venue>_win_qin_history.json 路徑（default 猜 "
             "~/Downloads/hkjc-api-master/data/<date>/<venue>_win_qin_history.json）",
    )
    args = ap.parse_args()
    odds_json = args.odds_json or os.path.expanduser(
        f"~/Downloads/hkjc-api-master/data/{args.date}/{args.venue}_win_qin_history.json"
    )

    print(f"[1/4] 讀歷史數據 + 重建 features（{PROJ_DIR}）...")
    res, run = build_history(PROJ_DIR)

    print("[2/4] Fit model（全歷史，同 w456.py 一致嘅邏輯）...")
    th, stats = fit_model(res)
    j_winr_map = res.groupby("jockey_name").win.mean().to_dict()
    tr_drop_sr_map = res[res.is_drop == 1].groupby("trainer_name").win.mean().to_dict()

    print(f"[3/4] 攞 {args.date} {args.venue} 即日賽事卡（體重/檔位/人腳）+ 即時 WIN 賠率...")
    meeting = fetch_card(args.date, args.venue)
    live_odds = load_live_win_odds(odds_json)
    today = pd.Timestamp(args.date)

    rows = []
    for race in meeting["races"]:
        if race.get("status") and "ABAND" in str(race["status"]).upper():
            continue
        race_odds = live_odds.get(str(race["no"]))
        if not race_odds:
            continue
        runners = [r for r in (race.get("runners") or []) if race_odds.get(str(r["no"]).zfill(2))]
        if not runners:
            continue
        field_size = len(runners)
        class_ord_today = cord(race.get("raceClass_en") or "")

        market_raw = {r["no"]: 1.0 / float(race_odds[str(r["no"]).zfill(2)]) for r in runners}
        mkt_sum = sum(market_raw.values())

        race_rows = []
        for r in runners:
            horse = r.get("horse") or {}
            horse_id = horse.get("id")
            if not horse_id:
                continue
            weight = r.get("currentWeight")
            weight = float(weight) if weight not in (None, "", 0) else None
            draw = r.get("barrierDrawNumber")
            draw = float(draw) if draw not in (None, "") else None
            jockey_name = (r.get("jockey") or {}).get("name_en")
            trainer_name = (r.get("trainer") or {}).get("name_en")

            feats = horse_asof_factors(
                res, run, j_winr_map, tr_drop_sr_map,
                horse_id=horse_id, today=today,
                today_weight=weight, today_class_ord=class_ord_today,
                today_draw=draw, today_field_size=field_size,
                jockey_name=jockey_name, trainer_name=trainer_name,
            )
            market_p = market_raw[r["no"]] / mkt_sum
            z = {c: (0.0 if pd.isna(feats[c]) else (feats[c] - stats[c][0]) / stats[c][1]) for c in FACT}
            score = th[0] * np.log(max(market_p, 1e-6)) + sum(th[i + 1] * z[c] for i, c in enumerate(FACT))

            race_rows.append(dict(
                raceNo=race["no"], saddle=r["no"], horse=r.get("name_en"), horse_id=horse_id,
                jockey=jockey_name, trainer=trainer_name,
                winOdds=race_odds[str(r["no"]).zfill(2)], marketP=market_p, score=score,
                nPriorRuns=feats["n_prior_runs"], weightAvailable=weight is not None,
            ))

        if not race_rows:
            continue
        scores = np.array([x["score"] for x in race_rows])
        model_p = np.exp(scores - scores.max())
        model_p = model_p / model_p.sum()
        for x, mp in zip(race_rows, model_p):
            x["modelP"] = round(float(mp), 4)
            x["marketP"] = round(float(x["marketP"]), 4)
            x["marketSurprise"] = round(float(mp - x["marketP"]), 4)
            rows.append(x)

    if not rows:
        print("冇任何場次計到（可能未截飛/冇賠率）。")
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values(["raceNo", "marketSurprise"], ascending=[True, False])
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"market_surprise_{args.date}_{args.venue}.csv")
    df.to_csv(out_path, index=False)
    print(f"[4/4] 已產生 {len(df)} 行 → {out_path}")

    n_no_weight = (~df.weightAvailable).sum()
    if n_no_weight:
        print(f"⚠️  {n_no_weight} 隻馬暫時攞唔到體重(currentWeight)，bw_* 因子當中性(0)處理，"
              f"呢啲馬嘅 Market Surprise 準確度會打折扣。")

    print("\n📊 Market Surprise 排行（絕對值最大 top 10）：")
    top = df.reindex(df.marketSurprise.abs().sort_values(ascending=False).index).head(10)
    for _, r in top.iterrows():
        print(f"  R{r.raceNo} {r.saddle}號 {r.horse:<20s} model={r.modelP:.1%}  "
              f"market={r.marketP:.1%}  surprise={r.marketSurprise:+.1%}  "
              f"(過往{int(r.nPriorRuns)}戰)")


if __name__ == "__main__":
    main()
