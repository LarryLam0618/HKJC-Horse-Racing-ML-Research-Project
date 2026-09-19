"""Win backtest using REAL market odds from dividends.csv (WIN pool, top-3 horses).

This is a REAL backtest (not theoretical like quinella), because:
  - We have real market win odds for top-3 horses from dividends.csv
  - We can compute real ROI for each bet
  - No need for Harville formula or imputation

METHODOLOGY:
  1. For each race, load WIN dividends (top-3 horses with their win odds)
  2. For each horse in top-3, compute:
     - market_prob = 1/dividend
     - model_prob = pred_win_prob_raw
     - edge = model_prob - market_prob
  3. Bet on horses with edge >= threshold
  4. Compute real ROI: (sum of dividends for winning bets) / (total stake) - 1

LIMITATIONS:
  - We only have market odds for top-3 horses, not all horses
  - So we can only backtest betting on top-3 horses (which is still meaningful)
  - If model predicts a non-top-3 horse has high edge, we can't test it

Run:
  python3 features/backtest_win_real_roi.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import duckdb
except ImportError:
    raise ImportError("請先安裝 duckdb: pip install duckdb")

OUTPUT_DIR = Path("output/win_backtest")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = Path("data/processed/hkjc.duckdb")
WIN_PRED_PATH = Path("output/full18y/F_reg_2_predictions_full18y.csv")
DIVIDENDS_PATH = Path("dividends.csv")
TRAIN_PATH = Path("training_set_safe.csv")

# Edge thresholds (model_prob - market_prob)
THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]

# Stake per bet (HKD)
STAKE_PER_BET = 100.0


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
    """Load predictions and merge with saddle from training set."""
    if not pred_path.exists():
        raise FileNotFoundError(f"搵唔到 {pred_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"搵唔到 {train_path}")

    pred_df = pd.read_csv(pred_path)
    train_df = pd.read_csv(train_path)

    required_pred = ["race_date", "venue", "race_no", "horse_id", "win_label", "pred_win_prob_raw"]
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


def backtest_win_race(
    race_df: pd.DataFrame,
    results_race: pd.DataFrame,
    dividends_race: pd.DataFrame,
    threshold: float,
) -> dict:
    """Backtest win betting for a single race.

    For each horse in top-3 (with real market odds):
      - market_prob = 1/dividend
      - model_prob = pred_win_prob_raw
      - edge = model_prob - market_prob
      - Bet if edge >= threshold

    Returns metrics for this race.
    """
    # Get winning horse
    sorted_results = results_race.sort_values("finish_pos")
    winner = sorted_results.iloc[0]["horse_id"] if len(sorted_results) > 0 else None

    # Load WIN dividends (top-3 horses)
    win_dividends = dividends_race[dividends_race["pool"] == "WIN"].copy()
    if len(win_dividends) == 0:
        return None

    # Parse combination (should be single-digit for WIN, e.g. "11" for horse #11)
    def parse_win_comb(comb_str):
        try:
            return int(str(comb_str).strip())
        except ValueError:
            return None

    win_dividends["combination_parsed"] = win_dividends["combination"].apply(parse_win_comb)
    win_dividends = win_dividends.dropna(subset=["combination_parsed"])

    # Build saddle -> dividend mapping
    saddle_to_dividend = dict(zip(win_dividends["combination_parsed"], win_dividends["dividend"]))

    # Map to horse_id via saddle
    horse_to_dividend = {}
    for _, row in race_df.iterrows():
        saddle = row["saddle"]
        horse_id = row["horse_id"]
        if saddle in saddle_to_dividend:
            horse_to_dividend[horse_id] = saddle_to_dividend[saddle]

    # Filter to horses with market odds (top-3 only)
    race_with_odds = race_df[race_df["horse_id"].isin(horse_to_dividend.keys())].copy()

    if len(race_with_odds) == 0:
        return None

    # Compute edge for each horse
    bets = []
    for _, row in race_with_odds.iterrows():
        horse_id = row["horse_id"]
        dividend = horse_to_dividend[horse_id]
        market_prob = 1.0 / dividend
        model_prob = row["pred_win_prob_raw"]
        edge = model_prob - market_prob

        if edge >= threshold:
            bets.append({
                "horse_id": horse_id,
                "model_prob": model_prob,
                "market_prob": market_prob,
                "edge": edge,
                "dividend": dividend,
                "is_winner": horse_id == winner,
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

    total_stake = float(bet_count) * STAKE_PER_BET
    winning_bets = bets_df[bets_df["is_winner"]]
    total_return = float(winning_bets["dividend"].sum()) * STAKE_PER_BET if len(winning_bets) > 0 else 0.0
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

    print("\n📊 Backtesting win betting (REAL ROI with market odds)...")

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
            metrics = backtest_win_race(race_df, results_race, dividends_race, threshold)
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
        "total_stake": "sum",
        "total_return": "sum",
        "net_profit": "sum",
        "roi": "mean",
    }).reset_index()

    print("\n" + "=" * 80)
    print("🏁 WIN BACKTEST — REAL ROI (Market odds from dividends.csv WIN pool)")
    print("=" * 80)
    print(
        summary[[
            "threshold", "bet_count", "win_count", "hit_rate",
            "total_stake", "total_return", "net_profit", "roi"
        ]].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("📌 重要說明")
    print("=" * 80)
    print("呢個 backtest 用嘅係真實市場賠率（WIN pool 頭三名馬），可以計算真實 ROI。")
    print("但局限性係：我哋只有頭三名馬嘅市場賠率，所以只可以 backtest 落注喺頭三名馬。")
    print("如果模型預測某匹非頭三名馬有高 edge，我哋無法測試（因為冇佢嘅市場賠率）。")
    print("\n呢個結果應該被解讀為：")
    print("  「模型喺頭三名馬之中，可唔可以搵到 edge 並賺取正 ROI？」")

    bets_df.to_csv(OUTPUT_DIR / "win_bet_level_real_roi.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "win_backtest_summary_real_roi.csv", index=False)

    print("\n✅ Outputs saved to:")
    print(" - output/win_backtest/win_bet_level_real_roi.csv")
    print(" - output/win_backtest/win_backtest_summary_real_roi.csv")


if __name__ == "__main__":
    main()