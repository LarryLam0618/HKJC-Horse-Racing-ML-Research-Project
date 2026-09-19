"""Win backtest using THEORETICAL baseline (uniform prior) instead of real market odds.

WHY THIS EXISTS:
  dividends.csv only contains WIN odds for top-3 horses, not all horses.
  This makes it impossible to test "model vs market" realistically -- if we
  only bet on top-3 horses, we're guaranteed to win every race (since one
  of them must be the winner), leading to absurdly high hit rates (~100%)
  and ROI (~9000%).

  Instead, we use a theoretical baseline: uniform prior (all horses have
  equal win probability = 1/n). This tests: "Does my model beat random
  guessing?" NOT: "Can I make money betting in the real market?"

METHODOLOGY:
  1. For each race with n horses, baseline win prob = 1/n for all horses
  2. Model win prob = pred_win_prob_raw
  3. Edge = model_prob - baseline_prob
  4. Bet on horses with edge >= threshold
  5. ROI: computed using "fair odds" = 1/baseline_prob = n
     (This is theoretical, not real market odds)

INTERPRETATION:
  - Positive ROI means: model beats uniform prior baseline
  - NOT: model can make money in real market (we don't have real odds)

Run:
  python3 features/backtest_win_theoretical.py
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

# Edge thresholds (model_prob - baseline_prob)
THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]

# Stake per bet (HKD)
STAKE_PER_BET = 100.0


def load_results_from_duckdb(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"搵唔到 {db_path}")
    conn = duckdb.connect(str(db_path), read_only=True)
    query = """
    SELECT race_date, venue, race_no, horse_id, finish_pos
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


def load_win_predictions(pred_path: Path) -> pd.DataFrame:
    if not pred_path.exists():
        raise FileNotFoundError(f"搵唔到 {pred_path}")

    pred_df = pd.read_csv(pred_path)
    required = ["race_date", "venue", "race_no", "horse_id", "win_label", "pred_win_prob_raw"]
    missing = [c for c in required if c not in pred_df.columns]
    if missing:
        raise ValueError(f"Prediction file 缺少欄位：{missing}")

    pred_df = pred_df.copy()
    pred_df["race_date"] = pd.to_datetime(pred_df["race_date"], errors="coerce")
    pred_df = pred_df.sort_values(["race_date", "venue", "race_no", "horse_id"]).reset_index(drop=True)
    return pred_df


def backtest_win_race(
    race_df: pd.DataFrame,
    results_race: pd.DataFrame,
    threshold: float,
) -> dict:
    """Backtest win betting for a single race using uniform prior baseline.

    For each horse:
      - baseline_prob = 1/n (uniform prior)
      - model_prob = pred_win_prob_raw
      - edge = model_prob - baseline_prob
      - Bet if edge >= threshold

    ROI computed using "fair odds" = 1/baseline_prob = n (theoretical, not real)

    Returns metrics for this race.
    """
    n_horses = len(race_df)
    baseline_prob = 1.0 / n_horses
    fair_odds = n_horses  # Theoretical fair odds under uniform prior

    # Get winning horse
    sorted_results = results_race.sort_values("finish_pos")
    winner = sorted_results.iloc[0]["horse_id"] if len(sorted_results) > 0 else None

    # Compute edge for each horse
    bets = []
    for _, row in race_df.iterrows():
        horse_id = row["horse_id"]
        model_prob = row["pred_win_prob_raw"]
        edge = model_prob - baseline_prob

        if edge >= threshold:
            bets.append({
                "horse_id": horse_id,
                "model_prob": model_prob,
                "baseline_prob": baseline_prob,
                "edge": edge,
                "fair_odds": fair_odds,
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
    # Payout = stake * fair_odds (theoretical, not real market odds)
    total_return = float(len(winning_bets)) * STAKE_PER_BET * fair_odds if len(winning_bets) > 0 else 0.0
    net_profit = total_return - total_stake
    roi = net_profit / total_stake if total_stake > 0 else np.nan

    return {
        "bet_count": bet_count, "win_count": win_count, "hit_rate": hit_rate,
        "total_stake": total_stake, "total_return": total_return,
        "net_profit": net_profit, "roi": roi,
        "n_horses": n_horses,
        "baseline_prob": baseline_prob,
        "fair_odds": fair_odds,
    }


def main() -> None:
    print("📂 Loading results from DuckDB...")
    results_df = load_results_from_duckdb(DB_PATH)

    print("📂 Loading win model predictions...")
    win_df = load_win_predictions(WIN_PRED_PATH)

    print(f"📊 Results: {len(results_df)} rows")
    print(f"📊 Win predictions: {len(win_df)} rows")

    results_by_race = results_df.groupby(["race_date", "venue", "race_no"])
    win_by_race = win_df.groupby(["race_date", "venue", "race_no"])

    print(f"📊 Number of result races: {len(results_by_race)}")
    print(f"📊 Number of prediction races: {len(win_by_race)}")

    print("\n📊 Backtesting win betting (THEORETICAL baseline: uniform prior)...")

    all_bet_rows = []
    races_processed = 0
    races_no_predictions = 0

    for (race_date, venue, race_no), results_race in results_by_race:
        if (race_date, venue, race_no) not in win_by_race.groups:
            races_no_predictions += 1
            continue

        race_df = win_by_race.get_group((race_date, venue, race_no))

        for threshold in THRESHOLDS:
            metrics = backtest_win_race(race_df, results_race, threshold)
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
        "n_horses": "mean",
    }).reset_index()

    print("\n" + "=" * 80)
    print("🏁 WIN BACKTEST — THEORETICAL BASELINE (Uniform Prior)")
    print("=" * 80)
    print(
        summary[[
            "threshold", "bet_count", "win_count", "hit_rate",
            "total_stake", "total_return", "net_profit", "roi", "n_horses"
        ]].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("📌 重要說明")
    print("=" * 80)
    print("呢個 backtest 用嘅係理論性基線（Uniform Prior），而唔係真實市場賠率。")
    print("ROI 係用「公平賠率」（fair_odds = n_horses）計算，唔係真實市場賠率。")
    print("\n呢個結果應該被解讀為：")
    print("  「模型嘅獨贏預測係咪好過隨機猜測（Uniform Prior）？」")
    print("而唔係：")
    print("  「模型可唔可以喺真實獨贏市場搵到套利機會？」")
    print("\n正面 ROI 表示：模型好過隨機猜測。")
    print("負面 ROI 表示：模型差過隨機猜測。")
    print("但唔可以直接推論到真實市場表現。")

    bets_df.to_csv(OUTPUT_DIR / "win_bet_level_theoretical.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "win_backtest_summary_theoretical.csv", index=False)

    print("\n✅ Outputs saved to:")
    print(" - output/win_backtest/win_bet_level_theoretical.csv")
    print(" - output/win_backtest/win_backtest_summary_theoretical.csv")


if __name__ == "__main__":
    main()