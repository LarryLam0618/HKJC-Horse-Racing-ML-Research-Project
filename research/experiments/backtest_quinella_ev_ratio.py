"""Quinella backtest comparing two edge-filtering methods:

  Method A (existing): edge_diff = model_prob - market_prob >= threshold
  Method B (new):       ev_ratio = model_prob * dividend      >= min_ev_ratio

Both methods already use the odds (dividend) -- market_prob is derived from
dividend via market_prob = 1/dividend. The difference is HOW the edge is
measured:

  - edge_diff favours high-probability / low-odds combinations (a fixed
    probability-point margin matters less in relative terms when odds are
    low).
  - ev_ratio favours combinations where the raw expected-value multiple is
    highest, regardless of whether that comes from a big probability edge
    on a short-priced pair or a small edge on a long-priced pair.

This script re-uses the expanding-window Platt calibration from
backtest_quinella_expanding_calibration.py (folds 5-16, zero leakage) and
reports both filtering methods side by side so you can see whether EV-ratio
filtering selects a materially different (and better/worse) set of bets.

Run (after backtest_quinella_expanding_calibration.py has produced
     quinella_bet_level_results_expanding.csv... actually this script
     recomputes from scratch since it needs per-pair dividend, not just
     the winning summary):
  python3 features/backtest_quinella_ev_ratio.py
"""

from __future__ import annotations

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

try:
    import duckdb
except ImportError:
    raise ImportError("請先安裝 duckdb: pip install duckdb")

OUTPUT_DIR = Path("output/quinella")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = Path("data/processed/hkjc.duckdb")
WIN_PRED_PATH = Path("output/regularization/F_reg_2_predictions_all_folds.csv")
DIVIDENDS_PATH = Path("dividends.csv")
TRAIN_PATH = Path("training_set_safe.csv")

# Method A: absolute probability-difference thresholds (existing approach)
EDGE_DIFF_THRESHOLDS = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15]

# Method B: EV-ratio thresholds. ev_ratio = model_prob * dividend.
# 1.0 = break-even in expectation; >1.0 = positive expected value.
EV_RATIO_THRESHOLDS = [1.00, 1.05, 1.10, 1.15, 1.20, 1.30]

CALIBRATION_BLOCKS = [
    (list(range(1, 5)), list(range(5, 9))),
    (list(range(1, 9)), list(range(9, 13))),
    (list(range(1, 13)), list(range(13, 17))),
]


def load_results_from_duckdb(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"搵唔到 {db_path}")
    conn = duckdb.connect(str(db_path))
    query = """
    SELECT race_date, venue, race_no, horse_id, finish_pos, draw
    FROM results
    WHERE finish_pos IS NOT NULL
    ORDER BY race_date, venue, race_no, finish_pos
    """
    df = conn.execute(query).fetchdf()
    conn.close()
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df = df.dropna(subset=["race_date", "finish_pos"])
    df["finish_pos"] = df["finish_pos"].astype(int)
    return df


def load_win_predictions_with_draw(pred_path: Path, train_path: Path) -> pd.DataFrame:
    if not pred_path.exists():
        raise FileNotFoundError(f"搵唔到 {pred_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"搵唔到 {train_path}")

    pred_df = pd.read_csv(pred_path)
    train_df = pd.read_csv(train_path)

    required_pred = ["race_date", "venue", "race_no", "horse_id", "win_label",
                      "pred_win_prob_raw", "fold"]
    required_train = ["race_date", "venue", "race_no", "horse_id", "draw"]
    missing_pred = [c for c in required_pred if c not in pred_df.columns]
    missing_train = [c for c in required_train if c not in train_df.columns]
    if missing_pred:
        raise ValueError(f"Prediction file 缺少欄位：{missing_pred}")
    if missing_train:
        raise ValueError(f"Training file 缺少欄位：{missing_train}")

    pred_df = pred_df.copy()
    train_df = train_df.copy()
    pred_df["race_date"] = pd.to_datetime(pred_df["race_date"], errors="coerce")
    train_df["race_date"] = pd.to_datetime(train_df["race_date"], errors="coerce")

    merge_keys = ["race_date", "venue", "race_no", "horse_id"]
    df = pred_df.merge(train_df[merge_keys + ["draw"]], on=merge_keys, how="left")

    if df["draw"].isna().any():
        na_count = int(df["draw"].isna().sum())
        print(f"⚠️ 有 {na_count} 行 merge 後仍然缺少 draw，將剔除呢啲行。")
        df = df.dropna(subset=["draw"]).copy()

    df["draw"] = df["draw"].astype(int)
    df = df.sort_values(["race_date", "venue", "race_no", "horse_id"]).reset_index(drop=True)
    return df


def load_dividends(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"搵唔到 {path}")
    df = pd.read_csv(path)
    required = ["race_date", "venue", "race_no", "pool", "combination", "dividend"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺少欄位：{missing}")
    df = df.copy()
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df = df.dropna(subset=["race_date", "dividend"])
    df["dividend"] = pd.to_numeric(df["dividend"], errors="coerce")
    return df


def fit_apply_platt(train_y, train_prob, test_prob):
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(train_prob.reshape(-1, 1), train_y)
    return lr.predict_proba(test_prob.reshape(-1, 1))[:, 1]


def build_expanding_calibrated_predictions(win_df: pd.DataFrame) -> pd.DataFrame:
    all_blocks = []
    for block_num, (calib_folds, eval_folds) in enumerate(CALIBRATION_BLOCKS, start=1):
        calib_df = win_df[win_df["fold"].isin(calib_folds)].copy()
        eval_df = win_df[win_df["fold"].isin(eval_folds)].copy()
        if calib_df.empty or eval_df.empty:
            continue

        train_y = calib_df["win_label"].astype(int).to_numpy()
        train_prob = calib_df["pred_win_prob_raw"].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()
        test_prob = eval_df["pred_win_prob_raw"].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()

        eval_df = eval_df.copy()
        eval_df["pred_win_prob_platt_oof"] = fit_apply_platt(train_y, train_prob, test_prob)
        eval_df["calibration_block"] = block_num
        all_blocks.append(eval_df)

    return pd.concat(all_blocks, ignore_index=True)


def get_winning_quinella(results_race: pd.DataFrame):
    sorted_results = results_race.sort_values("finish_pos")
    top_2 = sorted_results.head(2)
    if len(top_2) < 2:
        return None
    return tuple(sorted([top_2.iloc[0]["horse_id"], top_2.iloc[1]["horse_id"]]))


def compute_quinella_prob_from_win_probs(win_probs: pd.Series, horse_ids: list) -> dict:
    probs = win_probs.values
    quinella_dict = {}
    for idx_i, idx_j in combinations(range(len(horse_ids)), 2):
        p_i, p_j = probs[idx_i], probs[idx_j]
        if p_i < 1.0 and p_j < 1.0:
            p_quinella = p_i * p_j * (1.0 / (1.0 - p_i) + 1.0 / (1.0 - p_j))
        else:
            p_quinella = 0.0
        pair = tuple(sorted([horse_ids[idx_i], horse_ids[idx_j]]))
        quinella_dict[pair] = p_quinella
    return quinella_dict


def parse_combination(comb_str: str):
    parts = str(comb_str).split(",")
    if len(parts) == 2:
        return tuple(sorted([int(parts[0].strip()), int(parts[1].strip())]))
    return None


def compute_market_quinella_dict(dividends_race: pd.DataFrame, draw_to_horse: dict) -> dict:
    quinella_df = dividends_race[dividends_race["pool"] == "QUINELLA"].copy()
    if quinella_df.empty:
        return {}
    quinella_df["combination_parsed"] = quinella_df["combination"].apply(parse_combination)
    quinella_df = quinella_df.dropna(subset=["combination_parsed"])
    if len(quinella_df) == 0:
        return {}
    quinella_df["horse_i"] = quinella_df["combination_parsed"].apply(
        lambda x: draw_to_horse.get(x[0], None)
    )
    quinella_df["horse_j"] = quinella_df["combination_parsed"].apply(
        lambda x: draw_to_horse.get(x[1], None)
    )
    quinella_df = quinella_df.dropna(subset=["horse_i", "horse_j"])
    if len(quinella_df) == 0:
        return {}
    market_dict = {}
    for _, row in quinella_df.iterrows():
        pair = tuple(sorted([row["horse_i"], row["horse_j"]]))
        market_dict[pair] = (1.0 / row["dividend"], row["dividend"])
    return market_dict


def collect_all_pair_bets(race_df, results_race, dividends_race, draw_to_horse):
    """Return every model-vs-market pair for this race, with both edge metrics
    pre-computed, regardless of threshold. Filtering happens later so we can
    compare Method A and Method B on the exact same candidate pool."""
    winning_combo = get_winning_quinella(results_race)
    if winning_combo is None:
        return []

    win_probs = race_df.set_index("horse_id")["pred_win_prob_platt_oof"]
    horse_ids = race_df["horse_id"].tolist()
    model_quinella_dict = compute_quinella_prob_from_win_probs(win_probs, horse_ids)
    market_quinella_dict = compute_market_quinella_dict(dividends_race, draw_to_horse)
    if not market_quinella_dict:
        return []

    rows = []
    for pair, model_prob in model_quinella_dict.items():
        if pair not in market_quinella_dict:
            continue
        market_prob, dividend = market_quinella_dict[pair]
        rows.append({
            "horse_i": pair[0], "horse_j": pair[1],
            "model_prob": model_prob, "market_prob": market_prob,
            "dividend": dividend,
            "edge_diff": model_prob - market_prob,
            "ev_ratio": model_prob * dividend,
            "is_winner": pair == winning_combo,
        })
    return rows


def summarize(bets_df: pd.DataFrame, filter_col: str, thresholds: list, stake: float = 10.0):
    out = []
    for t in thresholds:
        sub = bets_df[bets_df[filter_col] >= t]
        bet_count = len(sub)
        win_count = int(sub["is_winner"].sum())
        total_stake = bet_count * stake
        total_return = float(sub.loc[sub["is_winner"], "dividend"].sum()) if bet_count else 0.0
        net_profit = total_return - total_stake
        roi = net_profit / total_stake if total_stake > 0 else np.nan
        hit_rate = win_count / bet_count if bet_count > 0 else np.nan
        out.append({
            "threshold": t, "bet_count": bet_count, "win_count": win_count,
            "hit_rate": hit_rate, "total_stake": total_stake,
            "total_return": total_return, "net_profit": net_profit, "roi": roi,
        })
    return pd.DataFrame(out)


def main() -> None:
    print("📂 Loading data...")
    results_df = load_results_from_duckdb(DB_PATH)
    win_df = load_win_predictions_with_draw(WIN_PRED_PATH, TRAIN_PATH)
    dividends_df = load_dividends(DIVIDENDS_PATH)

    print("🔁 Building expanding-window Platt calibration (folds 5-16, zero leakage)...")
    calibrated_df = build_expanding_calibrated_predictions(win_df)

    results_by_race = results_df.groupby(["race_date", "venue", "race_no"])
    calib_by_race = calibrated_df.groupby(["race_date", "venue", "race_no"])

    print("📊 Collecting all candidate pairs per race (both edge metrics)...")
    all_rows = []
    for (race_date, venue, race_no), results_race in results_by_race:
        if (race_date, venue, race_no) not in calib_by_race.groups:
            continue
        race_df = calib_by_race.get_group((race_date, venue, race_no))
        dividends_race = dividends_df[
            (dividends_df["race_date"] == race_date) &
            (dividends_df["venue"] == venue) &
            (dividends_df["race_no"] == race_no)
        ]
        if dividends_race.empty:
            continue
        draw_to_horse = dict(zip(race_df["draw"], race_df["horse_id"]))
        rows = collect_all_pair_bets(race_df, results_race, dividends_race, draw_to_horse)
        for r in rows:
            r["race_date"] = race_date
            r["venue"] = venue
            r["race_no"] = race_no
        all_rows.extend(rows)

    pairs_df = pd.DataFrame(all_rows)
    print(f"📊 Total candidate pairs (all races, all combos, pre-filter): {len(pairs_df)}")

    summary_a = summarize(pairs_df, "edge_diff", EDGE_DIFF_THRESHOLDS)
    summary_b = summarize(pairs_df, "ev_ratio", EV_RATIO_THRESHOLDS)

    print("\n" + "=" * 80)
    print("方法 A：edge_diff = model_prob - market_prob（你原本嘅方法）")
    print("=" * 80)
    print(summary_a.to_string(index=False))

    print("\n" + "=" * 80)
    print("方法 B：ev_ratio = model_prob * dividend（你今次提出嘅方法）")
    print("=" * 80)
    print(summary_b.to_string(index=False))

    # Overlap check at roughly comparable thresholds
    print("\n" + "=" * 80)
    print("重疊度檢查：edge_diff>=0.05 vs ev_ratio>=1.10 揀嘅係唔係同一批注？")
    print("=" * 80)
    set_a = set(zip(
        pairs_df[pairs_df["edge_diff"] >= 0.05]["race_date"],
        pairs_df[pairs_df["edge_diff"] >= 0.05]["race_no"],
        pairs_df[pairs_df["edge_diff"] >= 0.05]["horse_i"],
        pairs_df[pairs_df["edge_diff"] >= 0.05]["horse_j"],
    ))
    set_b = set(zip(
        pairs_df[pairs_df["ev_ratio"] >= 1.10]["race_date"],
        pairs_df[pairs_df["ev_ratio"] >= 1.10]["race_no"],
        pairs_df[pairs_df["ev_ratio"] >= 1.10]["horse_i"],
        pairs_df[pairs_df["ev_ratio"] >= 1.10]["horse_j"],
    ))
    overlap = set_a & set_b
    print(f"方法 A 揀中: {len(set_a)} 注")
    print(f"方法 B 揀中: {len(set_b)} 注")
    print(f"兩者重疊: {len(overlap)} 注")
    if set_a or set_b:
        jaccard = len(overlap) / len(set_a | set_b) if (set_a | set_b) else 0
        print(f"Jaccard 相似度: {jaccard:.2%}")

    pairs_df.to_csv(OUTPUT_DIR / "quinella_all_candidate_pairs_ev_analysis.csv", index=False)
    summary_a.to_csv(OUTPUT_DIR / "quinella_summary_method_a_edge_diff.csv", index=False)
    summary_b.to_csv(OUTPUT_DIR / "quinella_summary_method_b_ev_ratio.csv", index=False)

    print("\n✅ Outputs saved to output/quinella/:")
    print(" - quinella_all_candidate_pairs_ev_analysis.csv (raw pairs, both metrics)")
    print(" - quinella_summary_method_a_edge_diff.csv")
    print(" - quinella_summary_method_b_ev_ratio.csv")


if __name__ == "__main__":
    main()
