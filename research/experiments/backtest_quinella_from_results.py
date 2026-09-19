"""Backtest Quinella betting using results table for actual finishing positions.

This script:
  - Loads race results from DuckDB (finish_pos)
  - Loads F_reg_2 win model predictions
  - Computes model-implied quinella probabilities from win probabilities
  - Loads dividends for market-implied quinella probabilities
  - Computes edge = model_prob - market_prob
  - Identifies winning quinella combinations from finish_pos (1st and 2nd)
  - Backtests quinella betting with fixed edge thresholds
  - Reports hit rate, ROI, profit/loss

Run:
  python3 features/backtest_quinella_from_results.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from itertools import combinations

try:
    import duckdb
except ImportError:
    raise ImportError("請先安裝 duckdb: pip install duckdb")

OUTPUT_DIR = Path("output/quinella")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Paths
DB_PATH = Path("data/processed/hkjc.duckdb")
WIN_PRED_PATH = Path("output/regularization/F_reg_2_predictions_all_folds.csv")
DIVIDENDS_PATH = Path("dividends.csv")
TRAIN_PATH = Path("training_set_safe.csv")

THRESHOLDS = [0.00, 0.02, 0.05, 0.08]


def load_results_from_duckdb(db_path: Path) -> pd.DataFrame:
    """Load race results with finish_pos from DuckDB."""
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
    """Load win model predictions and merge with draw from training data."""
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

    # Merge to get draw
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


def get_winning_quinella(results_race: pd.DataFrame) -> tuple:
    """Get winning quinella combination (1st and 2nd place) from results."""
    sorted_results = results_race.sort_values("finish_pos")
    top_2 = sorted_results.head(2)

    if len(top_2) < 2:
        return None

    horse_1st = top_2.iloc[0]["horse_id"]
    horse_2nd = top_2.iloc[1]["horse_id"]

    # Return as tuple (horse_i, horse_j) sorted
    return tuple(sorted([horse_1st, horse_2nd]))


def compute_quinella_prob_from_win_probs(win_probs: pd.Series, horse_ids: list) -> dict:
    """
    Compute quinella probabilities from win probabilities.

    Formula: p_quinella(i,j) = p_i * p_j * (1/(1-p_i) + 1/(1-p_j))

    Returns a dict: {(horse_i, horse_j): prob}
    """
    probs = win_probs.values
    quinella_dict = {}

    for idx_i, idx_j in combinations(range(len(horse_ids)), 2):
        p_i = probs[idx_i]
        p_j = probs[idx_j]

        # Accurate formula
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


def compute_market_quinella_dict(dividends_race: pd.DataFrame, draw_to_horse: dict) -> dict:
    """
    Compute market-implied quinella probabilities from dividends.

    Returns a dict: {(horse_i, horse_j): (market_prob, dividend)}
    """
    quinella_df = dividends_race[dividends_race["pool"] == "QUINELLA"].copy()

    if quinella_df.empty:
        return {}

    quinella_df["combination_parsed"] = quinella_df["combination"].apply(parse_combination)
    quinella_df = quinella_df.dropna(subset=["combination_parsed"])

    if len(quinella_df) == 0:
        return {}

    # Map draw numbers to horse_ids
    quinella_df["horse_i"] = quinella_df["combination_parsed"].apply(
        lambda x: draw_to_horse.get(x[0], None)
    )
    quinella_df["horse_j"] = quinella_df["combination_parsed"].apply(
        lambda x: draw_to_horse.get(x[1], None)
    )

    quinella_df = quinella_df.dropna(subset=["horse_i", "horse_j"])

    if len(quinella_df) == 0:
        return {}

    # Create dict: {(horse_i, horse_j): (market_prob, dividend)}
    market_dict = {}
    for _, row in quinella_df.iterrows():
        pair = tuple(sorted([row["horse_i"], row["horse_j"]]))
        market_prob = 1.0 / row["dividend"]
        market_dict[pair] = (market_prob, row["dividend"])

    return market_dict


def backtest_quinella_race(
    race_df: pd.DataFrame,
    results_race: pd.DataFrame,
    dividends_race: pd.DataFrame,
    draw_to_horse: dict,
    threshold: float
) -> dict:
    """Backtest quinella betting for a single race."""
    # Get winning quinella combination
    winning_combo = get_winning_quinella(results_race)
    if winning_combo is None:
        return None

    # Compute model quinella probabilities
    win_probs = race_df.set_index("horse_id")["pred_win_prob_raw"]
    horse_ids = race_df["horse_id"].tolist()
    model_quinella_dict = compute_quinella_prob_from_win_probs(win_probs, horse_ids)

    # Compute market quinella probabilities
    market_quinella_dict = compute_market_quinella_dict(dividends_race, draw_to_horse)

    if not market_quinella_dict:
        return None

    # Find all combinations with edge >= threshold
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
                "is_winner": pair == winning_combo
            })

    if not bets:
        return {
            "bet_count": 0,
            "win_count": 0,
            "hit_rate": np.nan,
            "total_stake": 0.0,
            "total_return": 0.0,
            "net_profit": 0.0,
            "roi": np.nan,
        }

    bets_df = pd.DataFrame(bets)
    bet_count = len(bets_df)
    win_count = int(bets_df["is_winner"].sum())
    hit_rate = win_count / bet_count if bet_count > 0 else np.nan

    # Each bet costs 1 unit
    total_stake = float(bet_count) * 10.0  # HKJC minimum stake is $10

    # Return = sum of dividends for winning bets
    winning_bets = bets_df[bets_df["is_winner"]]
    total_return = float(winning_bets["dividend"].sum()) if len(winning_bets) > 0 else 0.0
    net_profit = total_return - total_stake
    roi = net_profit / total_stake if total_stake > 0 else np.nan

    return {
        "bet_count": bet_count,
        "win_count": win_count,
        "hit_rate": hit_rate,
        "total_stake": total_stake,
        "total_return": total_return,
        "net_profit": net_profit,
        "roi": roi,
    }


def main():
    print("📂 Loading results from DuckDB...")
    results_df = load_results_from_duckdb(DB_PATH)

    print("📂 Loading win model predictions...")
    win_df = load_win_predictions_with_draw(WIN_PRED_PATH, TRAIN_PATH)

    print("📂 Loading dividends data...")
    dividends_df = load_dividends(DIVIDENDS_PATH)

    print(f"📊 Results: {len(results_df)} rows")
    print(f"📊 Win predictions: {len(win_df)} rows")
    print(f"📊 Dividends: {len(dividends_df)} rows")

    # Filter to evaluation folds (9-16)
    print("📊 Filtering to evaluation folds (9-16)...")
    win_df = win_df[win_df["fold"] >= 9].copy()

    print(f"📊 Evaluation folds: {len(win_df)} rows")

    # Group by race
    results_by_race = results_df.groupby(["race_date", "venue", "race_no"])
    win_by_race = win_df.groupby(["race_date", "venue", "race_no"])

    print(f"📊 Number of result races: {len(results_by_race)}")
    print(f"📊 Number of prediction races: {len(win_by_race)}")

    # Backtest per race
    print("\n📊 Backtesting quinella betting...")

    all_bet_rows = []
    races_processed = 0
    races_no_results = 0
    races_no_predictions = 0

    for (race_date, venue, race_no), results_race in results_by_race:
        # Check if we have predictions for this race
        if (race_date, venue, race_no) not in win_by_race.groups:
            races_no_predictions += 1
            continue

        race_df = win_by_race.get_group((race_date, venue, race_no))

        # Check if we have dividends for this race
        dividends_race = dividends_df[
            (dividends_df["race_date"] == race_date) &
            (dividends_df["venue"] == venue) &
            (dividends_df["race_no"] == race_no)
        ]

        if dividends_race.empty:
            races_no_results += 1
            continue

        # Create draw -> horse_id mapping
        draw_to_horse = dict(zip(race_df["draw"], race_df["horse_id"]))

        # Backtest for each threshold
        for threshold in THRESHOLDS:
            metrics = backtest_quinella_race(
                race_df, results_race, dividends_race, draw_to_horse, threshold
            )

            if metrics is None:
                continue

            metrics["race_date"] = race_date
            metrics["venue"] = venue
            metrics["race_no"] = race_no
            metrics["threshold"] = threshold
            all_bet_rows.append(metrics)

        races_processed += 1

    print(f"\n📊 Race breakdown:")
    print(f"  Races processed: {races_processed}")
    print(f"  Races without predictions: {races_no_predictions}")
    print(f"  Races without dividends: {races_no_results}")

    if not all_bet_rows:
        print("❌ 無 backtest 數據")
        return

    bets_df = pd.DataFrame(all_bet_rows)

    # Aggregate by threshold
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
    print("🏁 QUINELLA BACKTEST SUMMARY")
    print("=" * 80)
    print(
        summary[[
            "threshold", "bet_count", "win_count", "hit_rate",
            "total_stake", "total_return", "net_profit", "roi"
        ]].to_string(index=False)
    )

    # Save outputs
    bets_df.to_csv(OUTPUT_DIR / "quinella_bet_level_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "quinella_backtest_summary.csv", index=False)

    print("\n✅ Outputs saved to:")
    print(" - output/quinella/quinella_bet_level_results.csv")
    print(" - output/quinella/quinella_backtest_summary.csv")


if __name__ == "__main__":
    main()