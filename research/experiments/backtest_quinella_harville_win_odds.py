"""Quinella backtest using HARBILLE FORMULA with market win odds from dividends.csv
(WIN pool, top-3 horses) to derive baseline quinella odds.

IMPROVEMENT OVER uniform prior:
  - Instead of assuming all horses have equal win probability (1/n),
    we use REAL market win odds for the top-3 finishers from dividends.csv
  - For horses outside top-3, we impute their win probs proportionally
    based on model's predictions (since we don't have their real market odds)
  - This gives a MORE REALISTIC baseline than uniform prior, though still
    not perfect (since we're imputing non-top-3 horses)

METHODOLOGY:
  1. For each race, load WIN dividends (top-3 horses with their win odds)
  2. Convert win odds to win probs: p_i = 1/dividend_i
  3. For top-3 horses: use real market win probs
  4. For non-top-3 horses: impute win probs proportionally based on model predictions
     (since we don't have their real market odds)
  5. Normalize all win probs to sum to 1.0
  6. Use Harville formula to derive quinella probs from win probs
  7. Compare model quinella probs (from model win probs) vs market quinella probs
  8. Edge = model_quinella_prob - market_quinella_prob

Run:
  python3 features/backtest_quinella_harville_win_odds.py
"""

from __future__ import annotations

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

try:
    import duckdb
except ImportError:
    raise ImportError("請先安裝 duckdb: pip install duckdb")

OUTPUT_DIR = Path("output/quinella")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = Path("data/processed/hkjc.duckdb")
WIN_PRED_PATH = Path("output/full18y/F_reg_2_predictions_full18y.csv")
DIVIDENDS_PATH = Path("dividends.csv")
TRAIN_PATH = Path("training_set_safe.csv")

# Edge thresholds (model_quinella_prob - market_quinella_prob)
THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]


def load_results_from_duckdb(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"搵唔到 {db_path}")
    conn = duckdb.connect(str(db_path), read_only=True)
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


def load_win_predictions(pred_path: Path, train_path: Path) -> pd.DataFrame:
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
        raise ValueError(f"Training file 缺少欄位：{missing_train}")

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


def harville_quinella(p_i, p_j):
    """Compute quinella probability using Harville formula.

    p_quinella(i,j) = p_i * p_j * (1/(1-p_i) + 1/(1-p_j))
    """
    if p_i >= 1.0 or p_j >= 1.0:
        return 0.0
    return p_i * p_j * (1.0 / (1.0 - p_i) + 1.0 / (1.0 - p_j))


def get_market_win_probs(race_df: pd.DataFrame, dividends_race: pd.DataFrame) -> pd.Series:
    """Derive market win probabilities for all horses in a race.

    For top-3 horses: use real market win probs from WIN dividends (1/dividend)
    For non-top-3 horses: impute proportionally based on model predictions

    Returns a Series indexed by horse_id with win probs summing to 1.0
    """
    win_dividends = dividends_race[dividends_race["pool"] == "WIN"].copy()

    if len(win_dividends) == 0:
        # No WIN dividends at all -- fall back to uniform prior
        n_horses = len(race_df)
        return pd.Series(1.0 / n_horses, index=race_df["horse_id"])

    # Parse combination (should be single-digit for WIN, e.g. "11" for horse #11)
    def parse_win_comb(comb_str):
        try:
            return int(str(comb_str).strip())
        except ValueError:
            return None

    win_dividends["saddle"] = win_dividends["combination"].apply(parse_win_comb)
    win_dividends = win_dividends.dropna(subset=["saddle"])
    win_dividends["saddle"] = win_dividends["saddle"].astype(int)

    # Build saddle -> market_win_prob mapping for top-3
    saddle_to_market_prob = {}
    for _, row in win_dividends.iterrows():
        saddle = int(row["saddle"])
        market_prob = 1.0 / row["dividend"]
        saddle_to_market_prob[saddle] = market_prob

    # Map to horse_id
    horse_to_market_prob = {}
    for _, row in race_df.iterrows():
        saddle = row["saddle"]
        horse_id = row["horse_id"]
        if saddle in saddle_to_market_prob:
            horse_to_market_prob[horse_id] = saddle_to_market_prob[saddle]

    # Check how many horses have real market probs
    n_top3 = len(horse_to_market_prob)
    n_total = len(race_df)

    if n_top3 == 0:
        # No WIN dividends -- fall back to uniform prior
        return pd.Series(1.0 / n_total, index=race_df["horse_id"])

    # For non-top-3 horses, impute proportionally based on model predictions
    model_win_probs = race_df.set_index("horse_id")["pred_win_prob_raw"]

    # Sum of market probs for top-3
    sum_top3 = sum(horse_to_market_prob.values())

    # Remaining probability mass for non-top-3
    remaining_prob = max(0.0, 1.0 - sum_top3)

    # Sum of model probs for non-top-3 horses
    non_top3_horses = [h for h in race_df["horse_id"] if h not in horse_to_market_prob]
    sum_model_non_top3 = model_win_probs.loc[non_top3_horses].sum()

    # Impute non-top-3 probs proportionally
    for horse_id in non_top3_horses:
        if sum_model_non_top3 > 0:
            imputed_prob = remaining_prob * (model_win_probs.loc[horse_id] / sum_model_non_top3)
        else:
            imputed_prob = remaining_prob / len(non_top3_horses) if non_top3_horses else 0.0
        horse_to_market_prob[horse_id] = imputed_prob

    # Normalize to sum to 1.0
    total = sum(horse_to_market_prob.values())
    if total > 0:
        horse_to_market_prob = {h: p / total for h, p in horse_to_market_prob.items()}

    return pd.Series(horse_to_market_prob)


def get_winning_quinella(results_race: pd.DataFrame) -> tuple:
    sorted_results = results_race.sort_values("finish_pos")
    top_2 = sorted_results.head(2)
    if len(top_2) < 2:
        return None
    return tuple(sorted([top_2.iloc[0]["horse_id"], top_2.iloc[1]["horse_id"]]))


def backtest_quinella_race(
    race_df: pd.DataFrame,
    results_race: pd.DataFrame,
    dividends_race: pd.DataFrame,
    threshold: float,
) -> dict:
    """Backtest quinella betting for a single race.

    Methodology:
      - Market win probs: from WIN dividends (top-3) + imputed for non-top-3
      - Model win probs: pred_win_prob_raw
      - Both converted to quinella probs via Harville formula
      - Edge = model_quinella_prob - market_quinella_prob
    """
    winning_combo = get_winning_quinella(results_race)
    if winning_combo is None:
        return None

    n_horses = len(race_df)
    horse_ids = race_df["horse_id"].tolist()

    # Model win probs
    model_win_probs = race_df.set_index("horse_id")["pred_win_prob_raw"]

    # Market win probs (from WIN dividends + imputation)
    market_win_probs = get_market_win_probs(race_df, dividends_race)

    # Compute quinella probs for all pairs
    bets = []
    for idx_i, idx_j in combinations(range(n_horses), 2):
        h_i, h_j = horse_ids[idx_i], horse_ids[idx_j]
        pair = tuple(sorted([h_i, h_j]))

        # Model quinella prob (Harville from model win probs)
        p_i_model = model_win_probs.loc[h_i]
        p_j_model = model_win_probs.loc[h_j]
        model_quinella_prob = harville_quinella(p_i_model, p_j_model)

        # Market quinella prob (Harville from market win probs)
        p_i_market = market_win_probs.loc[h_i]
        p_j_market = market_win_probs.loc[h_j]
        market_quinella_prob = harville_quinella(p_i_market, p_j_market)

        edge = model_quinella_prob - market_quinella_prob

        if edge >= threshold:
            bets.append({
                "horse_i": h_i,
                "horse_j": h_j,
                "model_quinella_prob": model_quinella_prob,
                "market_quinella_prob": market_quinella_prob,
                "edge": edge,
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

    # Since we don't have real quinella dividends, we can't compute real ROI
    # We'll just track hit_rate vs expected hit_rate under market baseline
    # Expected hit_rate under market = average market_quinella_prob across bet pairs
    expected_hit_rate = bets_df["market_quinella_prob"].mean()

    return {
        "bet_count": bet_count, "win_count": win_count, "hit_rate": hit_rate,
        "expected_hit_rate": expected_hit_rate,
        "hit_rate_edge": hit_rate - expected_hit_rate if not np.isnan(hit_rate) else np.nan,
    }


def main() -> None:
    print("📂 Loading results from DuckDB...")
    results_df = load_results_from_duckdb(DB_PATH)

    print("📂 Loading win model predictions...")
    win_df = load_win_predictions(WIN_PRED_PATH, TRAIN_PATH)

    print("📂 Loading dividends...")
    dividends_df = load_dividends(DIVIDENDS_PATH)

    print(f"📊 Results: {len(results_df)} rows")
    print(f"📊 Win predictions: {len(win_df)} rows")
    print(f"📊 Dividends: {len(dividends_df)} rows")

    results_by_race = results_df.groupby(["race_date", "venue", "race_no"])
    win_by_race = win_df.groupby(["race_date", "venue", "race_no"])
    div_by_race = dividends_df.groupby(["race_date", "venue", "race_no"])

    print(f"📊 Number of result races: {len(results_by_race)}")
    print(f"📊 Number of prediction races: {len(win_by_race)}")
    print(f"📊 Number of dividend races: {len(div_by_race)}")

    print("\n📊 Backtesting quinella betting (Harville with WIN odds baseline)...")

    all_bet_rows = []
    races_processed = 0
    races_no_predictions = 0
    races_no_dividends = 0

    for (race_date, venue, race_no), results_race in results_by_race:
        if (race_date, venue, race_no) not in win_by_race.groups:
            races_no_predictions += 1
            continue

        race_df = win_by_race.get_group((race_date, venue, race_no))

        if (race_date, venue, race_no) not in div_by_race.groups:
            races_no_dividends += 1
            continue

        dividends_race = div_by_race.get_group((race_date, venue, race_no))

        for threshold in THRESHOLDS:
            metrics = backtest_quinella_race(race_df, results_race, dividends_race, threshold)
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
    print(f"  Races without dividends: {races_no_dividends}")

    if not all_bet_rows:
        print("❌ 無 backtest 數據")
        return

    bets_df = pd.DataFrame(all_bet_rows)

    summary = bets_df.groupby("threshold").agg({
        "bet_count": "sum",
        "win_count": "sum",
        "hit_rate": "mean",
        "expected_hit_rate": "mean",
        "hit_rate_edge": "mean",
    }).reset_index()

    print("\n" + "=" * 80)
    print("🏁 QUINELLA BACKTEST — HARBILLE WITH WIN ODDS BASELINE (Top-3 + Imputation)")
    print("=" * 80)
    print(
        summary[[
            "threshold", "bet_count", "win_count", "hit_rate",
            "expected_hit_rate", "hit_rate_edge"
        ]].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("📌 重要說明")
    print("=" * 80)
    print("呢個 backtest 用嘅係「模型 vs 市場（WIN 賠率推算）」，比 uniform prior 更現實。")
    print("但由於 dividends.csv 只得頭三名馬嘅 WIN 賠率，非頭三名馬嘅賠率係用模型預測推算，")
    print("所以仍然有假設成分，唔係完全真實嘅市場賠率。")
    print("\n呢個結果應該被解讀為：")
    print("  「模型嘅連贏預測係咪好過用真實 WIN 賠率推算嘅基線？」")
    print("而唔係：")
    print("  「模型可唔可以喺真實連贏市場搵到套利機會？」")

    bets_df.to_csv(OUTPUT_DIR / "quinella_bet_level_harville_win_odds.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "quinella_backtest_summary_harville_win_odds.csv", index=False)

    print("\n✅ Outputs saved to:")
    print(" - output/quinella/quinella_bet_level_harville_win_odds.csv")
    print(" - output/quinella/quinella_backtest_summary_harville_win_odds.csv")


if __name__ == "__main__":
    main()