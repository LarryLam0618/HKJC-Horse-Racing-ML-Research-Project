"""Quinella backtest using expanding-window Platt calibration blocks.

*** CRITICAL FIX (2026-08-19) ***
Prior versions of this script mapped dividends.csv QUINELLA combination
numbers to horses using `draw` (starting gate position). This has been
PROVEN WRONG by verify_combination_mapping_key.py:

    Key      Match rate against official dividend records
    draw     1.64%   (28 / 1711 races)
    saddle   99.88%  (1709 / 1711 races)

  Cross-tabulation on the 1683 races where draw and saddle disagree:
    draw_pair correct:   0 races   (0.0%)
    saddle_pair correct: 1681 races (99.9%)

This means EVERY prior quinella backtest result (the 21-bet run, the
37-bet expanding-calibration run, the per-block breakdown, and the
EV-ratio vs edge-diff comparison) matched model probabilities to the
WRONG horse's market price in the vast majority of races. Those results
are void and must not be cited going forward -- this script fixes the
mapping key from `draw` to `saddle` and must be re-run to get valid numbers.

Why this exists (original rationale, still valid):
  Your existing pipeline only has Platt-calibrated probabilities for folds 9-16,
  because Platt scaling was fit on folds 1-8 and applied to folds 9-16 ONCE.
  That leaves folds 1-8 unusable for backtesting (their raw probabilities are
  uncalibrated). This script fixes that WITHOUT introducing leakage, by
  re-running Platt calibration multiple times in an expanding-window
  walk-forward fashion:

      Block 1: calibrate on folds  1-4  ->  evaluate on folds  5-8
      Block 2: calibrate on folds  1-8  ->  evaluate on folds  9-12
      Block 3: calibrate on folds 1-12  ->  evaluate on folds 13-16

  Every evaluation fold is scored using a Platt model fit ONLY on strictly
  earlier folds. No evaluation fold ever contributes to its own calibration.

Run (after train_lightgbm_native.py / run_lightgbm_regularization_tests.py
     have produced F_reg_2_predictions_all_folds.csv):
  python3 features/backtest_quinella_expanding_calibration.py
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

THRESHOLDS = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15]

CALIBRATION_BLOCKS = [
    (list(range(1, 5)), list(range(5, 9))),
    (list(range(1, 9)), list(range(9, 13))),
    (list(range(1, 13)), list(range(13, 17))),
]


def load_results_from_duckdb(db_path: Path) -> pd.DataFrame:
    """Load race results with finish_pos and saddle from DuckDB (read-only)."""
    if not db_path.exists():
        raise FileNotFoundError(f"搵唔到 {db_path}")

    try:
        conn = duckdb.connect(str(db_path), read_only=True)
    except duckdb.IOException as e:
        raise RuntimeError(
            f"無法以 read-only 模式打開 {db_path}。若另一個 process 持有 exclusive 鎖，"
            f"請先執行 `lsof {db_path}` 搵到 PID 並 kill 佈。原始錯誤: {e}"
        ) from e

    query = """
    SELECT race_date, venue, race_no, horse_id, finish_pos, saddle
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


def load_win_predictions_with_saddle(pred_path: Path, train_path: Path) -> pd.DataFrame:
    """Load RAW win model predictions (all 16 folds) and merge with saddle.

    NOTE: merges on `saddle`, not `draw` -- see module docstring for why.
    """
    if not pred_path.exists():
        raise FileNotFoundError(f"搵唔到 {pred_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"搵唔到 {train_path}")

    pred_df = pd.read_csv(pred_path)
    train_df = pd.read_csv(train_path)

    required_pred = ["race_date", "venue", "race_no", "horse_id", "win_label",
                      "pred_win_prob_raw", "fold"]
    required_train = ["race_date", "venue", "race_no", "horse_id", "saddle"]

    missing_pred = [c for c in required_pred if c not in pred_df.columns]
    missing_train = [c for c in required_train if c not in train_df.columns]
    if missing_pred:
        raise ValueError(f"Prediction file 缺少欄位：{missing_pred}")
    if missing_train:
        raise ValueError(
            f"Training file 缺少欄位：{missing_train}。"
            f"若 training_set_safe.csv 冇 'saddle' 欄位，需要由 results 表 join 番過去。"
        )

    pred_df = pred_df.copy()
    train_df = train_df.copy()
    pred_df["race_date"] = pd.to_datetime(pred_df["race_date"], errors="coerce")
    train_df["race_date"] = pd.to_datetime(train_df["race_date"], errors="coerce")

    merge_keys = ["race_date", "venue", "race_no", "horse_id"]
    df = pred_df.merge(train_df[merge_keys + ["saddle"]], on=merge_keys, how="left")

    if df["saddle"].isna().any():
        na_count = int(df["saddle"].isna().sum())
        print(f"⚠️ 有 {na_count} 行 merge 後仍然缺少 saddle，將剔除呢啲行。")
        df = df.dropna(subset=["saddle"]).copy()

    df["saddle"] = df["saddle"].astype(int)
    df = df.sort_values(["race_date", "venue", "race_no", "horse_id"]).reset_index(drop=True)
    return df


def load_dividends(path: Path) -> pd.DataFrame:
    """Load dividends data."""
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
    """Fit Platt scaling on calibration fold(s), apply to evaluation fold(s)."""
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(train_prob.reshape(-1, 1), train_y)
    return lr.predict_proba(test_prob.reshape(-1, 1))[:, 1]


def build_expanding_calibrated_predictions(win_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each calibration block, fit Platt on calibration folds and apply to
    evaluation folds ONLY. Concatenate all evaluation-fold results.
    """
    all_blocks = []

    for block_num, (calib_folds, eval_folds) in enumerate(CALIBRATION_BLOCKS, start=1):
        calib_df = win_df[win_df["fold"].isin(calib_folds)].copy()
        eval_df = win_df[win_df["fold"].isin(eval_folds)].copy()

        if calib_df.empty or eval_df.empty:
            print(f"⚠️ Block {block_num}: calibration 或 evaluation fold 無資料，跳過")
            continue

        train_y = calib_df["win_label"].astype(int).to_numpy()
        train_prob = calib_df["pred_win_prob_raw"].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()
        test_prob = eval_df["pred_win_prob_raw"].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()

        eval_df = eval_df.copy()
        eval_df["pred_win_prob_platt_oof"] = fit_apply_platt(train_y, train_prob, test_prob)
        eval_df["calibration_block"] = block_num
        eval_df["calibration_folds_used"] = ",".join(map(str, calib_folds))

        print(
            f"📦 Block {block_num}: calibrate on folds {calib_folds} "
            f"-> evaluate on folds {eval_folds} "
            f"({len(calib_df)} calib rows, {len(eval_df)} eval rows)"
        )

        all_blocks.append(eval_df)

    if not all_blocks:
        raise ValueError("所有 calibration blocks 都無資料，無法繼續")

    combined = pd.concat(all_blocks, ignore_index=True)
    return combined


def get_winning_quinella(results_race: pd.DataFrame) -> tuple:
    """Get winning quinella combination (1st and 2nd place) from results."""
    sorted_results = results_race.sort_values("finish_pos")
    top_2 = sorted_results.head(2)
    if len(top_2) < 2:
        return None
    horse_1st = top_2.iloc[0]["horse_id"]
    horse_2nd = top_2.iloc[1]["horse_id"]
    return tuple(sorted([horse_1st, horse_2nd]))


def compute_quinella_prob_from_win_probs(win_probs: pd.Series, horse_ids: list) -> dict:
    """p_quinella(i,j) = p_i * p_j * (1/(1-p_i) + 1/(1-p_j))"""
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


def parse_combination(comb_str: str) -> tuple:
    """Parse combination string like '2,8' into tuple (2, 8)."""
    parts = str(comb_str).split(",")
    if len(parts) == 2:
        return tuple(sorted([int(parts[0].strip()), int(parts[1].strip())]))
    return None


def compute_market_quinella_dict(dividends_race: pd.DataFrame, saddle_to_horse: dict) -> dict:
    """Returns {(horse_i, horse_j): (market_prob, dividend)}.

    Maps combination numbers to horses via `saddle` (saddle cloth / racing
    number) -- NOT `draw`. See module docstring: draw match rate was 1.64%
    vs saddle's 99.88% against official dividend records.
    """
    quinella_df = dividends_race[dividends_race["pool"] == "QUINELLA"].copy()
    if quinella_df.empty:
        return {}

    quinella_df["combination_parsed"] = quinella_df["combination"].apply(parse_combination)
    quinella_df = quinella_df.dropna(subset=["combination_parsed"])
    if len(quinella_df) == 0:
        return {}

    quinella_df["horse_i"] = quinella_df["combination_parsed"].apply(
        lambda x: saddle_to_horse.get(x[0], None)
    )
    quinella_df["horse_j"] = quinella_df["combination_parsed"].apply(
        lambda x: saddle_to_horse.get(x[1], None)
    )
    quinella_df = quinella_df.dropna(subset=["horse_i", "horse_j"])
    if len(quinella_df) == 0:
        return {}

    market_dict = {}
    for _, row in quinella_df.iterrows():
        pair = tuple(sorted([row["horse_i"], row["horse_j"]]))
        market_dict[pair] = (1.0 / row["dividend"], row["dividend"])
    return market_dict


def backtest_quinella_race(
    race_df: pd.DataFrame,
    results_race: pd.DataFrame,
    dividends_race: pd.DataFrame,
    saddle_to_horse: dict,
    threshold: float,
) -> dict:
    """Backtest quinella betting for a single race using Platt-calibrated probs."""
    winning_combo = get_winning_quinella(results_race)
    if winning_combo is None:
        return None

    win_probs = race_df.set_index("horse_id")["pred_win_prob_platt_oof"]
    horse_ids = race_df["horse_id"].tolist()
    model_quinella_dict = compute_quinella_prob_from_win_probs(win_probs, horse_ids)

    market_quinella_dict = compute_market_quinella_dict(dividends_race, saddle_to_horse)
    if not market_quinella_dict:
        return None

    bets = []
    for pair, model_prob in model_quinella_dict.items():
        if pair not in market_quinella_dict:
            continue
        market_prob, dividend = market_quinella_dict[pair]
        edge = model_prob - market_prob
        if edge >= threshold:
            bets.append({
                "horse_i": pair[0],
                "horse_j": pair[1],
                "model_prob": model_prob,
                "market_prob": market_prob,
                "edge": edge,
                "dividend": dividend,
                "is_winner": pair == winning_combo,
            })

    if not bets:
        return {
            "bet_count": 0, "win_count": 0, "hit_rate": np.nan,
            "total_stake": 0.0, "total_return": 0.0, "net_profit": 0.0, "roi": np.nan,
        }

    bets_df = pd.DataFrame(bets)
    bet_count = len(bets_df)
    win_count = int(bets_df["is_winner"].sum())
    hit_rate = win_count / bet_count if bet_count > 0 else np.nan

    total_stake = float(bet_count) * 10.0  # HKJC minimum stake $10
    winning_bets = bets_df[bets_df["is_winner"]]
    total_return = float(winning_bets["dividend"].sum()) if len(winning_bets) > 0 else 0.0
    net_profit = total_return - total_stake
    roi = net_profit / total_stake if total_stake > 0 else np.nan

    return {
        "bet_count": bet_count, "win_count": win_count, "hit_rate": hit_rate,
        "total_stake": total_stake, "total_return": total_return,
        "net_profit": net_profit, "roi": roi,
    }


def main() -> None:
    print("📂 Loading results from DuckDB...")
    results_df = load_results_from_duckdb(DB_PATH)

    print("📂 Loading RAW win model predictions (all 16 folds)...")
    win_df = load_win_predictions_with_saddle(WIN_PRED_PATH, TRAIN_PATH)

    print("📂 Loading dividends data...")
    dividends_df = load_dividends(DIVIDENDS_PATH)

    print(f"📊 Results: {len(results_df)} rows")
    print(f"📊 Win predictions (raw, all folds): {len(win_df)} rows")
    print(f"📊 Dividends: {len(dividends_df)} rows")

    print("\n🔁 Building expanding-window Platt calibration blocks...")
    calibrated_df = build_expanding_calibrated_predictions(win_df)
    print(f"\n📊 Total evaluation rows across all blocks: {len(calibrated_df)}")
    print(f"📊 Folds covered: {sorted(calibrated_df['fold'].unique())}")

    results_by_race = results_df.groupby(["race_date", "venue", "race_no"])
    calib_by_race = calibrated_df.groupby(["race_date", "venue", "race_no"])

    print(f"\n📊 Number of result races: {len(results_by_race)}")
    print(f"📊 Number of prediction races (across all blocks): {len(calib_by_race)}")

    print("\n📊 Backtesting quinella betting (expanding-window Platt calibration, saddle key)...")

    all_bet_rows = []
    races_processed = 0
    races_no_dividends = 0
    races_no_predictions = 0

    for (race_date, venue, race_no), results_race in results_by_race:
        if (race_date, venue, race_no) not in calib_by_race.groups:
            races_no_predictions += 1
            continue

        race_df = calib_by_race.get_group((race_date, venue, race_no))

        dividends_race = dividends_df[
            (dividends_df["race_date"] == race_date) &
            (dividends_df["venue"] == venue) &
            (dividends_df["race_no"] == race_no)
        ]
        if dividends_race.empty:
            races_no_dividends += 1
            continue

        saddle_to_horse = dict(zip(race_df["saddle"], race_df["horse_id"]))
        calibration_block = race_df["calibration_block"].iloc[0]

        for threshold in THRESHOLDS:
            metrics = backtest_quinella_race(
                race_df, results_race, dividends_race, saddle_to_horse, threshold
            )
            if metrics is None:
                continue

            metrics["race_date"] = race_date
            metrics["venue"] = venue
            metrics["race_no"] = race_no
            metrics["threshold"] = threshold
            metrics["calibration_block"] = calibration_block
            all_bet_rows.append(metrics)

        races_processed += 1

    print(f"\n📊 Race breakdown:")
    print(f"  Races processed: {races_processed}")
    print(f"  Races without predictions: {races_no_predictions}")
    print(f"  Races without dividends: {races_no_dividends}")

    if not all_bet_rows:
        print("❌ 無 backtest 數據")
        return

    bets_df = pd.DataFrame(all_bet_rows)

    summary = bets_df.groupby("threshold").agg({
        "bet_count": "sum",
        "win_count": "sum",
        "total_stake": "sum",
        "total_return": "sum",
        "net_profit": "sum",
    }).reset_index()
    summary["hit_rate"] = summary["win_count"] / summary["bet_count"]
    summary["roi"] = summary["net_profit"] / summary["total_stake"]

    print("\n" + "=" * 80)
    print("🏁 QUINELLA BACKTEST SUMMARY — EXPANDING-WINDOW PLATT CALIBRATION (saddle key, Folds 5-16)")
    print("=" * 80)
    print(
        summary[[
            "threshold", "bet_count", "win_count", "hit_rate",
            "total_stake", "total_return", "net_profit", "roi"
        ]].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("📦 PER-BLOCK BREAKDOWN")
    print("=" * 80)
    block_summary = bets_df.groupby(["calibration_block", "threshold"]).agg({
        "bet_count": "sum",
        "win_count": "sum",
        "total_stake": "sum",
        "total_return": "sum",
        "net_profit": "sum",
    }).reset_index()
    block_summary["roi"] = block_summary["net_profit"] / block_summary["total_stake"]
    print(block_summary.to_string(index=False))

    bets_df.to_csv(OUTPUT_DIR / "quinella_bet_level_results_expanding_SADDLE_FIXED.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "quinella_backtest_summary_expanding_SADDLE_FIXED.csv", index=False)
    block_summary.to_csv(OUTPUT_DIR / "quinella_backtest_summary_by_block_SADDLE_FIXED.csv", index=False)

    print("\n✅ Outputs saved to (note _SADDLE_FIXED suffix -- old draw-based files are now void):")
    print(" - output/quinella/quinella_bet_level_results_expanding_SADDLE_FIXED.csv")
    print(" - output/quinella/quinella_backtest_summary_expanding_SADDLE_FIXED.csv")
    print(" - output/quinella/quinella_backtest_summary_by_block_SADDLE_FIXED.csv")

    print("\n" + "=" * 80)
    print("👉 下一步")
    print("=" * 80)
    print("將 bootstrap_quinella_roi.py 入面嘅 BET_LEVEL_PATH 改為:")
    print('  BET_LEVEL_PATH = Path("output/quinella/quinella_bet_level_results_expanding_SADDLE_FIXED.csv")')
    print("然後重跑 bootstrap。呢次先係第一個用正確 mapping key 嘅有效結果。")


if __name__ == "__main__":
    main()