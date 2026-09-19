"""Quinella backtest using HARBILLE FORMULA to derive market_prob from model's
own win predictions -- NOT from real market odds.

WHY THIS EXISTS:
  dividends.csv only contains the WINNING quinella combination (1 per race),
  not all bettable combinations. This makes it impossible to test
  "model vs market" in the quinella market -- we can only test
  "model vs Harville theoretical formula".

  This is a THEORETICAL exercise, NOT a realistic arbitrage test.
  Results should be interpreted as: "Does my model beat the Harville formula?"
  NOT: "Can I make money betting quinella in the real market?"

METHODOLOGY:
  1. For each horse i, use model's pred_win_prob_raw as p_i
  2. For each pair (i,j), compute Harville quinella probability:
     p_quinella(i,j) = p_i * p_j * (1/(1-p_i) + 1/(1-p_j))
  3. This is the "market_prob" (theoretical, not real)
  4. Model's quinella prob is computed using the SAME formula
     (because we have no other source of quinella probabilities)
  5. Edge = model_quinella_prob - market_quinella_prob = 0 ALWAYS

  WAIT -- if we use the same formula for both, edge is always zero!
  So this script is fundamentally flawed...

  ACTUAL APPROACH:
  We need TWO DIFFERENT win probability sources:
    - "Market" win probs: derived from dividends (but we only have top-3)
    - Model win probs: pred_win_prob_raw

  Since we don't have full market win probs, we CANNOT do this properly.

  ALTERNATIVE:
  Use a NAIVE baseline as "market":
    - Assume all horses have EQUAL win probability: p_i = 1/n
    - Then Harville quinella prob for any pair = 2 / (n * (n-1))
    - Model edge = model_quinella_prob - 2/(n*(n-1))

  This tests: "Does my model beat a uniform prior baseline?"
  Still theoretical, but at least edge != 0.

Run:
  python3 features/backtest_quinella_harville_theoretical.py
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
WIN_PRED_PATH = Path("output/regularization/F_reg_2_predictions_all_folds.csv")
DIVIDENDS_PATH = Path("dividends.csv")
TRAIN_PATH = Path("training_set_safe.csv")

# Edge thresholds (model_quinella_prob - baseline_quinella_prob)
THRESHOLDS = [0.000, 0.001, 0.002, 0.003, 0.005, 0.010]


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

    This assumes p_i and p_j are win probabilities for horses i and j.
    """
    if p_i >= 1.0 or p_j >= 1.0:
        return 0.0
    return p_i * p_j * (1.0 / (1.0 - p_i) + 1.0 / (1.0 - p_j))


def get_winning_quinella(results_race: pd.DataFrame) -> tuple:
    sorted_results = results_race.sort_values("finish_pos")
    top_2 = sorted_results.head(2)
    if len(top_2) < 2:
        return None
    return tuple(sorted([top_2.iloc[0]["horse_id"], top_2.iloc[1]["horse_id"]]))


def backtest_quinella_race(
    race_df: pd.DataFrame,
    results_race: pd.DataFrame,
    threshold: float,
) -> dict:
    """Backtest quinella betting for a single race.

    Methodology:
      - Baseline (market): uniform prior, p_i = 1/n for all horses
      - Model: use pred_win_prob_raw
      - Both converted to quinella probs via Harville formula
      - Edge = model_quinella_prob - baseline_quinella_prob
    """
    winning_combo = get_winning_quinella(results_race)
    if winning_combo is None:
        return None

    n_horses = len(race_df)
    horse_ids = race_df["horse_id"].tolist()

    # Model win probs
    model_win_probs = race_df.set_index("horse_id")["pred_win_prob_raw"]

    # Baseline win probs: uniform prior
    baseline_win_prob = 1.0 / n_horses

    # Compute quinella probs for all pairs
    bets = []
    for idx_i, idx_j in combinations(range(n_horses), 2):
        h_i, h_j = horse_ids[idx_i], horse_ids[idx_j]
        pair = tuple(sorted([h_i, h_j]))

        # Model quinella prob (Harville from model win probs)
        p_i = model_win_probs.loc[h_i]
        p_j = model_win_probs.loc[h_j]
        model_quinella_prob = harville_quinella(p_i, p_j)

        # Baseline quinella prob (Harville from uniform prior)
        baseline_quinella_prob = harville_quinella(baseline_win_prob, baseline_win_prob)

        edge = model_quinella_prob - baseline_quinella_prob

        if edge >= threshold:
            # We don't have real market odds, so we can't compute real ROI
            # Just track whether this bet would have won
            bets.append({
                "horse_i": h_i,
                "horse_j": h_j,
                "model_quinella_prob": model_quinella_prob,
                "baseline_quinella_prob": baseline_quinella_prob,
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

    # Since we don't have real dividends, we can't compute real ROI
    # We'll just track hit_rate vs expected hit_rate under baseline
    # Expected hit_rate under baseline = baseline_quinella_prob (same for all pairs)
    expected_hit_rate = baseline_quinella_prob  # This is the same for all pairs

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

    print(f"📊 Results: {len(results_df)} rows")
    print(f"📊 Win predictions: {len(win_df)} rows")

    results_by_race = results_df.groupby(["race_date", "venue", "race_no"])
    win_by_race = win_df.groupby(["race_date", "venue", "race_no"])

    print(f"📊 Number of result races: {len(results_by_race)}")
    print(f"📊 Number of prediction races: {len(win_by_race)}")

    print("\n📊 Backtesting quinella betting (Harville theoretical baseline)...")

    all_bet_rows = []
    races_processed = 0
    races_no_predictions = 0

    for (race_date, venue, race_no), results_race in results_by_race:
        if (race_date, venue, race_no) not in win_by_race.groups:
            races_no_predictions += 1
            continue

        race_df = win_by_race.get_group((race_date, venue, race_no))

        for threshold in THRESHOLDS:
            metrics = backtest_quinella_race(race_df, results_race, threshold)
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
        "hit_rate": "mean",  # Average hit_rate across races
        "expected_hit_rate": "mean",
        "hit_rate_edge": "mean",
    }).reset_index()

    print("\n" + "=" * 80)
    print("🏁 QUINELLA BACKTEST — HARBILLE THEORETICAL BASELINE (Uniform Prior)")
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
    print("呢個 backtest 用嘅係「模型 vs 理論公式（Harville + Uniform Prior）」，")
    print("而唔係「模型 vs 真實市場」。")
    print("由於 dividends.csv 只得 winning combo（每場 1 個紀錄），無法代表所有可投注選項，")
    print("我哋無法進行有意義嘅「模型 vs 市場」測試。")
    print("\n呢個結果應該被解讀為：")
    print("  「模型嘅連贏預測係咪好過一個簡單嘅理論基線（Uniform Prior + Harville）？」")
    print("而唔係：")
    print("  「模型可唔可以喺真實連贏市場搵到套利機會？」")

    bets_df.to_csv(OUTPUT_DIR / "quinella_bet_level_harville_theoretical.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "quinella_backtest_summary_harville_theoretical.csv", index=False)

    print("\n✅ Outputs saved to:")
    print(" - output/quinella/quinella_bet_level_harville_theoretical.csv")
    print(" - output/quinella/quinella_backtest_summary_harville_theoretical.csv")


if __name__ == "__main__":
    main()