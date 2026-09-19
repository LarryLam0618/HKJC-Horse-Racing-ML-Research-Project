"""Backtest market-edge betting strategies using calibration folds (1-8).

This script:
  - Loads the OOF evaluation predictions from F_reg_2 (folds 9-16 only)
  - Merges with original training data to obtain win_odds
  - Converts win_odds to market-implied probabilities
  - Computes model edge = p_model - p_market
  - Tests fixed-threshold betting strategies:
      edge >= 0.00, 0.03, 0.05, 0.08
  - Reports for each threshold:
      bet count, hit rate, flat-stake ROI, total profit/loss,
      maximum drawdown, odds bucket performance, monthly ROI

Note: This script uses folds 9-16 (evaluation folds) only, because the
      F_reg_2 OOF evaluation file only contains folds 9-16.

Prerequisites:
  - output/regularization/F_reg_2_predictions_oof_eval.csv must exist
  - training_set_safe.csv must exist and contain win_odds

Run:
  python3 features/backtest_market_edge_train_folds.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

OUTPUT_DIR = Path("output/backtest")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use OOF evaluation file (folds 9-16 only)
PRED_PATH = Path("output/regularization/F_reg_2_predictions_oof_eval.csv")
TRAIN_PATH = Path("training_set_safe.csv")

THRESHOLDS = [0.00, 0.03, 0.05, 0.08]
ODDS_BUCKETS = [
    (0.0, 2.0),
    (2.0, 3.0),
    (3.0, 5.0),
    (5.0, 8.0),
    (8.0, 12.0),
    (12.0, 20.0),
    (20.0, 50.0),
    (50.0, float("inf")),
]


def load_predictions_and_merge_odds(pred_path: Path, train_path: Path) -> pd.DataFrame:
    if not pred_path.exists():
        raise FileNotFoundError(f"搵唔到 {pred_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"搵唔到 {train_path}")

    pred_df = pd.read_csv(pred_path)
    train_df = pd.read_csv(train_path)

    required_pred = ["race_date", "venue", "race_no", "horse_id", "win_label",
                     "pred_win_prob_platt_oof", "fold"]
    required_train = ["race_date", "venue", "race_no", "horse_id", "win_odds"]

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
    df = pred_df.merge(train_df[merge_keys + ["win_odds"]], on=merge_keys, how="left")

    if df["win_odds"].isna().any():
        na_count = int(df["win_odds"].isna().sum())
        print(f"⚠️ 有 {na_count} 行 merge 後仍然缺少 win_odds，將剔除呢啲行。")
        df = df.dropna(subset=["win_odds"]).copy()

    df = df[df["win_odds"] > 1.0].copy()
    df = df.sort_values(["race_date", "venue", "race_no", "horse_id"]).reset_index(drop=True)

    return df


def compute_market_prob(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df["inv_odds"] = 1.0 / df["win_odds"]
    market_sum = df.groupby(["race_date", "venue", "race_no"])["inv_odds"].transform("sum")
    return df["inv_odds"] / market_sum


def compute_edge(model_prob: pd.Series, market_prob: pd.Series) -> pd.Series:
    return model_prob - market_prob


def compute_betting_metrics(df: pd.DataFrame, threshold: float) -> dict:
    bets = df[df["edge"] >= threshold].copy()

    if bets.empty:
        return {
            "threshold": threshold,
            "bet_count": 0,
            "win_count": 0,
            "hit_rate": np.nan,
            "total_stake": 0.0,
            "total_return": 0.0,
            "net_profit": 0.0,
            "roi": np.nan,
            "max_drawdown": 0.0,
        }

    win_count = int(bets["win_label"].sum())
    bet_count = len(bets)
    hit_rate = win_count / bet_count if bet_count > 0 else np.nan

    total_stake = float(bet_count)
    total_return = float((bets["win_label"] * bets["win_odds"]).sum())
    net_profit = total_return - total_stake
    roi = net_profit / total_stake if total_stake > 0 else np.nan

    # Maximum drawdown (cumulative P&L)
    bets = bets.sort_values(["race_date", "venue", "race_no", "horse_id"]).reset_index(drop=True)
    bets["pnl"] = np.where(bets["win_label"] == 1, bets["win_odds"] - 1.0, -1.0)
    cum_pnl = bets["pnl"].cumsum()
    running_max = cum_pnl.cummax()
    drawdown = running_max - cum_pnl
    max_drawdown = float(drawdown.max()) if not drawdown.empty else 0.0

    return {
        "threshold": threshold,
        "bet_count": bet_count,
        "win_count": win_count,
        "hit_rate": hit_rate,
        "total_stake": total_stake,
        "total_return": total_return,
        "net_profit": net_profit,
        "roi": roi,
        "max_drawdown": max_drawdown,
    }


def compute_odds_bucket_metrics(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    bets = df[df["edge"] >= threshold].copy()
    if bets.empty:
        return pd.DataFrame()

    def bucket_label(odds):
        for low, high in ODDS_BUCKETS:
            if low <= odds < high:
                if high == float("inf"):
                    return f"{low:.0f}+"
                return f"{low:.0f}-{high:.0f}"
        return "other"

    bets["odds_bucket"] = bets["win_odds"].apply(bucket_label)

    rows = []
    for bucket, g in bets.groupby("odds_bucket", sort=False):
        bet_count = len(g)
        win_count = int(g["win_label"].sum())
        hit_rate = win_count / bet_count if bet_count > 0 else np.nan
        total_stake = float(bet_count)
        total_return = float((g["win_label"] * g["win_odds"]).sum())
        net_profit = total_return - total_stake
        roi = net_profit / total_stake if total_stake > 0 else np.nan

        rows.append({
            "threshold": threshold,
            "odds_bucket": bucket,
            "bet_count": bet_count,
            "win_count": win_count,
            "hit_rate": hit_rate,
            "total_stake": total_stake,
            "total_return": total_return,
            "net_profit": net_profit,
            "roi": roi,
        })

    return pd.DataFrame(rows)


def compute_monthly_roi(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    bets = df[df["edge"] >= threshold].copy()
    if bets.empty:
        return pd.DataFrame()

    bets = bets.copy()
    bets["year_month"] = bets["race_date"].dt.to_period("M")

    rows = []
    for ym, g in bets.groupby("year_month", sort=True):
        bet_count = len(g)
        win_count = int(g["win_label"].sum())
        hit_rate = win_count / bet_count if bet_count > 0 else np.nan
        total_stake = float(bet_count)
        total_return = float((g["win_label"] * g["win_odds"]).sum())
        net_profit = total_return - total_stake
        roi = net_profit / total_stake if total_stake > 0 else np.nan

        rows.append({
            "threshold": threshold,
            "year_month": str(ym),
            "bet_count": bet_count,
            "win_count": win_count,
            "hit_rate": hit_rate,
            "total_stake": total_stake,
            "total_return": total_return,
            "net_profit": net_profit,
            "roi": roi,
        })

    return pd.DataFrame(rows)


def main():
    print("📂 Loading F_reg_2 OOF evaluation predictions and merging win_odds...")
    df_all = load_predictions_and_merge_odds(PRED_PATH, TRAIN_PATH)

    print(f"📊 Available folds in file: {sorted(df_all['fold'].unique())}")
    print(f"⚠️  注意：F_reg_2 OOF evaluation file 只包含 folds 9-16 (evaluation folds)")
    print(f"         如需 folds 1-8 backtest，需要用 F_reg_2_predictions_all_folds.csv 並自行計算 Platt calibration")

    print("📊 Computing market-implied probabilities...")
    df_all["market_prob"] = compute_market_prob(df_all)
    df_all["edge"] = compute_edge(df_all["pred_win_prob_platt_oof"], df_all["market_prob"])

    print("📊 Summary statistics (Folds 9-16):")
    print(f"  Total horses: {len(df_all)}")
    print(f"  Total races: {df_all.groupby(['race_date', 'venue', 'race_no']).ngroups}")
    print(f"  Date range: {df_all['race_date'].min()} to {df_all['race_date'].max()}")
    print(f"  Model prob mean: {df_all['pred_win_prob_platt_oof'].mean():.4f}")
    print(f"  Market prob mean: {df_all['market_prob'].mean():.4f}")
    print(f"  Edge mean: {df_all['edge'].mean():.4f}")
    print(f"  Edge std: {df_all['edge'].std():.4f}")

    # Overall metrics per threshold
    summary_rows = []
    for threshold in THRESHOLDS:
        metrics = compute_betting_metrics(df_all, threshold)
        summary_rows.append(metrics)
        print(
            f"\nThreshold {threshold:.2f}: "
            f"bets={metrics['bet_count']}, "
            f"hit_rate={metrics['hit_rate']:.3f}, "
            f"ROI={metrics['roi']:.4f}, "
            f"profit={metrics['net_profit']:.2f}, "
            f"max_DD={metrics['max_drawdown']:.2f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "backtest_summary_folds9-16.csv", index=False)

    # Odds bucket metrics
    bucket_rows = []
    for threshold in THRESHOLDS:
        bucket_df = compute_odds_bucket_metrics(df_all, threshold)
        if not bucket_df.empty:
            bucket_rows.append(bucket_df)
    if bucket_rows:
        bucket_all = pd.concat(bucket_rows, ignore_index=True)
        bucket_all.to_csv(OUTPUT_DIR / "backtest_odds_buckets_folds9-16.csv", index=False)
        print("\n✅ Odds bucket metrics saved to: output/backtest/backtest_odds_buckets_folds9-16.csv")

    # Monthly ROI
    monthly_rows = []
    for threshold in THRESHOLDS:
        monthly_df = compute_monthly_roi(df_all, threshold)
        if not monthly_df.empty:
            monthly_rows.append(monthly_df)
    if monthly_rows:
        monthly_all = pd.concat(monthly_rows, ignore_index=True)
        monthly_all.to_csv(OUTPUT_DIR / "backtest_monthly_roi_folds9-16.csv", index=False)
        print("✅ Monthly ROI saved to: output/backtest/backtest_monthly_roi_folds9-16.csv")

    # Detailed bet-level results
    df_all.to_csv(OUTPUT_DIR / "backtest_bet_level_results_folds9-16.csv", index=False)
    print("✅ Bet-level results saved to: output/backtest/backtest_bet_level_results_folds9-16.csv")

    print("\n" + "=" * 80)
    print("🏁 BACKTEST SUMMARY — FOLDS 9-16 (EVALUATION PERIOD)")
    print("=" * 80)
    print(
        summary_df[[
            "threshold", "bet_count", "win_count", "hit_rate",
            "total_stake", "total_return", "net_profit", "roi", "max_drawdown"
        ]].to_string(index=False)
    )

    print("\n✅ All outputs saved to: output/backtest/")


if __name__ == "__main__":
    main()