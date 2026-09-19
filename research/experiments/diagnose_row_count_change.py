"""diagnose_duplicate_results.py already proved there are ZERO duplicate
(race_date, venue, race_no, horse_id) rows, and saddle is 100% populated
(0 nulls). So the 21083 -> 41794 row jump (exactly ~2.0x) between the two
runs is NOT a duplication bug -- the underlying `results` table itself
must have changed between runs (e.g. re-ingested with more seasons, or a
prior run was reading a stale/different DB file).

This script characterizes what's actually IN the current 41794-row table,
so we can figure out whether:
  (a) it now legitimately covers more races/seasons than before, and the
      21083-row backtest was simply working off a smaller historical DB, or
  (b) something else changed (e.g. a different DB file path was used, or
      finish_pos got backfilled for previously-incomplete races).

It also explains why "Number of prediction races" stayed at 914 while
"Number of result races" jumped to 3359: the prediction/training data
(F_reg_2_predictions_all_folds.csv, training_set_safe.csv) is a SNAPSHOT
generated earlier and was NOT regenerated when results grew. That mismatch
means most of the "new" 41794-row results table has NO corresponding model
predictions at all -- consistent with races_no_predictions=2445 out of 3359.

Run:
  python3 features/diagnose_row_count_change.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import duckdb
except ImportError:
    raise ImportError("請先安裝 duckdb: pip install duckdb")

DB_PATH = Path("data/processed/hkjc.duckdb")
TRAIN_PATH = Path("training_set_safe.csv")
PRED_PATH = Path("output/regularization/F_reg_2_predictions_all_folds.csv")


def main() -> None:
    conn = duckdb.connect(str(DB_PATH), read_only=True)

    print("=" * 80)
    print("1. results 表嘅日期範圍同 season 分布")
    print("=" * 80)
    date_range = conn.execute("""
        SELECT MIN(race_date) AS min_date, MAX(race_date) AS max_date, COUNT(*) AS n
        FROM results WHERE finish_pos IS NOT NULL
    """).fetchdf()
    print(date_range.to_string(index=False))

    season_counts = conn.execute("""
        SELECT season, COUNT(*) AS n_rows, COUNT(DISTINCT race_date || venue || race_no) AS n_races
        FROM results WHERE finish_pos IS NOT NULL
        GROUP BY season ORDER BY season
    """).fetchdf()
    print("\n按 season 分布:")
    print(season_counts.to_string(index=False))

    print("\n" + "=" * 80)
    print("2. training_set_safe.csv 嘅日期範圍（用嚉呢個生成 model predictions）")
    print("=" * 80)
    if TRAIN_PATH.exists():
        train_df = pd.read_csv(TRAIN_PATH)
        train_df["race_date"] = pd.to_datetime(train_df["race_date"], errors="coerce")
        print(f"training_set_safe.csv 行數: {len(train_df)}")
        print(f"日期範圍: {train_df['race_date'].min()} 至 {train_df['race_date'].max()}")
        n_train_races = train_df.groupby(["race_date", "venue", "race_no"]).ngroups
        print(f"涉及場數: {n_train_races}")
    else:
        print(f"⚠️ 搵唔到 {TRAIN_PATH}")

    print("\n" + "=" * 80)
    print("3. F_reg_2_predictions_all_folds.csv 嘅日期範圍（model 預測覆蓋範圍）")
    print("=" * 80)
    if PRED_PATH.exists():
        pred_df = pd.read_csv(PRED_PATH)
        pred_df["race_date"] = pd.to_datetime(pred_df["race_date"], errors="coerce")
        print(f"predictions 檔案行數: {len(pred_df)}")
        print(f"日期範圍: {pred_df['race_date'].min()} 至 {pred_df['race_date'].max()}")
        n_pred_races = pred_df.groupby(["race_date", "venue", "race_no"]).ngroups
        print(f"涉及場數: {n_pred_races}")
    else:
        print(f"⚠️ 搵唔到 {PRED_PATH}")

    print("\n" + "=" * 80)
    print("4. results 表入面，邊啲場次冇對應 model predictions？（按日期分布）")
    print("=" * 80)
    results_races = conn.execute("""
        SELECT DISTINCT race_date, venue, race_no
        FROM results WHERE finish_pos IS NOT NULL
    """).fetchdf()
    results_races["race_date"] = pd.to_datetime(results_races["race_date"])

    if PRED_PATH.exists():
        pred_races_set = set(zip(
            pred_df["race_date"].dt.date, pred_df["venue"], pred_df["race_no"]
        ))
        results_races["has_prediction"] = results_races.apply(
            lambda r: (r["race_date"].date(), r["venue"], r["race_no"]) in pred_races_set,
            axis=1
        )
        coverage_by_year = results_races.copy()
        coverage_by_year["year"] = coverage_by_year["race_date"].dt.year
        summary = coverage_by_year.groupby("year")["has_prediction"].agg(["sum", "count"])
        summary["coverage_pct"] = summary["sum"] / summary["count"]
        print(summary.to_string())

    conn.close()

    print("\n" + "=" * 80)
    print("🎯 診斷結論")
    print("=" * 80)
    print("如果 results 表嘅日期範圍/場數比 training_set_safe.csv 或 predictions 檔案大好多,")
    print("即係話 results 表已經被更新/擴充 (例如新增咗新season嘅賽事),")
    print("但 model predictions 檔案仲係舊 snapshot,冇跟住重新生成。")
    print("噉解釋咗點解 races_no_predictions=2445 (out of 3359) 咁高 -- ")
    print("大部分「新」場次根本冇 model 幫佈預測，backtest 只用到得 914 場有預測嘅賽事，")
    print("而呢 914 場入面，如果 is_winner 判斷仍然錯亂 (hit_rate 99.8%)，")
    print("問題就唔係喺 results 表本身，而係喺 backtest script 嘅邏輯 (下一步要查)。")


if __name__ == "__main__":
    main()