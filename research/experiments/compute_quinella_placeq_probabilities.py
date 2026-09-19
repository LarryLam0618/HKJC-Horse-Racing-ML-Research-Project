"""Compute Quinella and Quinella Place probabilities from win model predictions.

This script:
  - Loads F_reg_2 win model predictions (all folds)
  - Merges with training data to get draw numbers
  - Loads dividends data for QUINELLA and QUINELLA PLACE
  - Computes model-implied quinella probabilities from win probabilities (using draw numbers)
  - Computes market-implied quinella probabilities from dividends (unnormalized)
  - Computes edge = model_prob - market_prob
  - Outputs quinella probabilities and edge for each combination

Run:
  python3 features/compute_quinella_placeq_probabilities.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from itertools import combinations

OUTPUT_DIR = Path("output/quinella")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Paths
WIN_PRED_PATH = Path("output/regularization/F_reg_2_predictions_all_folds.csv")
DIVIDENDS_PATH = Path("dividends.csv")
TRAIN_PATH = Path("training_set_safe.csv")

THRESHOLDS = [0.00, 0.02, 0.05, 0.08]


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
    
    # Use draw as the identifier for quinella matching
    df["horse_id_for_quinella"] = df["draw"]
    
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


def compute_quinella_prob_from_win_probs(win_probs: pd.Series, draw_numbers: list) -> pd.DataFrame:
    """
    Compute quinella probabilities from win probabilities.

    Formula: p_quinella(i,j) = p_i * p_j * (1/(1-p_i) + 1/(1-p_j))
    """
    probs = win_probs.values

    pairs = []
    quinella_probs = []

    for idx_i, idx_j in combinations(range(len(draw_numbers)), 2):
        p_i = probs[idx_i]
        p_j = probs[idx_j]

        # Accurate formula
        if p_i < 1.0 and p_j < 1.0:
            p_quinella = p_i * p_j * (1.0 / (1.0 - p_i) + 1.0 / (1.0 - p_j))
        else:
            p_quinella = 0.0

        pairs.append((draw_numbers[idx_i], draw_numbers[idx_j]))
        quinella_probs.append(p_quinella)

    return pd.DataFrame({
        "horse_i": [p[0] for p in pairs],
        "horse_j": [p[1] for p in pairs],
        "model_quinella_prob": quinella_probs
    })


def parse_combination(comb_str: str) -> tuple:
    """Parse combination string like '2,8' into tuple (2, 8)."""
    parts = str(comb_str).split(",")
    if len(parts) == 2:
        return tuple(sorted([int(parts[0].strip()), int(parts[1].strip())]))
    return None


def compute_market_quinella_probs(dividends_race: pd.DataFrame) -> pd.DataFrame:
    """
    Compute market-implied quinella probabilities from dividends.

    Formula: p_market(i,j) = 1 / dividend_ij  (unnormalized, raw implied probability)
    """
    quinella_df = dividends_race[dividends_race["pool"] == "QUINELLA"].copy()

    if quinella_df.empty:
        return pd.DataFrame()

    quinella_df["combination_parsed"] = quinella_df["combination"].apply(parse_combination)
    quinella_df = quinella_df.dropna(subset=["combination_parsed"])

    if len(quinella_df) == 0:
        return pd.DataFrame()

    # Use draw numbers directly as identifiers
    quinella_df["horse_i"] = quinella_df["combination_parsed"].apply(lambda x: x[0])
    quinella_df["horse_j"] = quinella_df["combination_parsed"].apply(lambda x: x[1])

    # Market implied probability (unnormalized)
    quinella_df["market_quinella_prob"] = 1.0 / quinella_df["dividend"]

    return quinella_df[["horse_i", "horse_j", "market_quinella_prob", "dividend"]].copy()


def compute_quinella_edge(race_df: pd.DataFrame, dividends_race: pd.DataFrame) -> pd.DataFrame:
    """Compute model vs market quinella probabilities and edge for a single race."""
    # Model quinella probabilities (using draw numbers as identifiers)
    win_probs = race_df.set_index("horse_id_for_quinella")["pred_win_prob_raw"]
    draw_numbers = race_df["horse_id_for_quinella"].tolist()
    model_quinella = compute_quinella_prob_from_win_probs(win_probs, draw_numbers)

    # Market quinella probabilities
    market_quinella = compute_market_quinella_probs(dividends_race)

    if market_quinella.empty:
        return pd.DataFrame()

    # Merge model and market
    merged = model_quinella.merge(
        market_quinella,
        on=["horse_i", "horse_j"],
        how="inner"
    )

    if merged.empty:
        return pd.DataFrame()

    # Compute edge
    merged["edge"] = merged["model_quinella_prob"] - merged["market_quinella_prob"]

    # Add race info
    merged["race_date"] = race_df["race_date"].iloc[0]
    merged["venue"] = race_df["venue"].iloc[0]
    merged["race_no"] = race_df["race_no"].iloc[0]

    return merged


def main():
    print("📂 Loading win model predictions and merging draw...")
    win_df = load_win_predictions_with_draw(WIN_PRED_PATH, TRAIN_PATH)

    print("📂 Loading dividends data...")
    dividends_df = load_dividends(DIVIDENDS_PATH)

    print(f"📊 Win predictions (with draw): {len(win_df)} rows")
    print(f"📊 Dividends: {len(dividends_df)} rows")

    # Filter to evaluation folds (9-16) for consistency with previous backtests
    print("📊 Filtering to evaluation folds (9-16)...")
    win_df = win_df[win_df["fold"] >= 9].copy()

    print(f"📊 Evaluation folds: {len(win_df)} rows")

    # Group win predictions by race
    races = win_df.groupby(["race_date", "venue", "race_no"])

    print(f"📊 Number of races: {len(races)}")

    # Compute quinella probabilities for each race
    print("📊 Computing quinella probabilities...")

    all_quinella_rows = []
    races_with_quinella = 0
    races_no_dividends = 0
    races_empty_merge = 0

    for (race_date, venue, race_no), race_df in races:
        dividends_race = dividends_df[
            (dividends_df["race_date"] == race_date) &
            (dividends_df["venue"] == venue) &
            (dividends_df["race_no"] == race_no)
        ]

        if dividends_race.empty:
            races_no_dividends += 1
            continue

        quinella_race = compute_quinella_edge(race_df, dividends_race)
        if not quinella_race.empty:
            all_quinella_rows.append(quinella_race)
            races_with_quinella += 1
        else:
            races_empty_merge += 1

    print(f"\n📊 Race breakdown:")
    print(f"  Races with quinella data: {races_with_quinella}")
    print(f"  Races without dividends: {races_no_dividends}")
    print(f"  Races with empty merge: {races_empty_merge}")

    if not all_quinella_rows:
        print("❌ 無 quinella 數據")
        return

    quinella_df = pd.concat(all_quinella_rows, ignore_index=True)

    print(f"\n📊 Total quinella combinations: {len(quinella_df)}")

    # Summary statistics
    print("\n📊 Summary statistics:")
    print(f"  Model quinella prob mean: {quinella_df['model_quinella_prob'].mean():.6f}")
    print(f"  Market quinella prob mean: {quinella_df['market_quinella_prob'].mean():.6f}")
    print(f"  Edge mean: {quinella_df['edge'].mean():.6f}")
    print(f"  Edge std: {quinella_df['edge'].std():.6f}")

    # Save outputs
    quinella_df.to_csv(OUTPUT_DIR / "quinella_probabilities.csv", index=False)

    print("\n✅ Outputs saved to:")
    print(" - output/quinella/quinella_probabilities.csv")


if __name__ == "__main__":
    main()