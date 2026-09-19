"""Diagnose whether `saddle` values are duplicated WITHIN a single race.

If two different horses in the same race share the same saddle number, then
saddle_to_horse = dict(zip(race_df["saddle"], race_df["horse_id"])) will
silently overwrite one horse with the other, causing systematic mis-mapping
between dividends.csv combination numbers and actual horse_id pairs.

This is the last remaining suspect after:
  - duplicate rows in `results` table: ruled out (0 duplicates found)
  - horse_id format mismatch: ruled out (99.88% overlap, exact match in samples)

Run:
  python3 features/diagnose_saddle_duplicates.py
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
WIN_PRED_PATH = Path("output/regularization/F_reg_2_predictions_all_folds.csv")


def main() -> None:
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    results_df = conn.execute("""
        SELECT race_date, venue, race_no, horse_id, saddle
        FROM results WHERE finish_pos IS NOT NULL
    """).fetchdf()
    conn.close()

    print("=" * 80)
    print("1. 檢查 results 表：同一場賽事入面，saddle 數值有冇重複？")
    print("=" * 80)

    results_df["race_key"] = (
        results_df["race_date"].astype(str) + "_" +
        results_df["venue"] + "_R" + results_df["race_no"].astype(str)
    )

    saddle_counts = results_df.groupby(["race_key", "saddle"]).size().reset_index(name="count")
    duplicates = saddle_counts[saddle_counts["count"] > 1]

    if len(duplicates) == 0:
        print("✅ results 表：冇發現任何 (race, saddle) 組合有超過一匹馬。")
        print("   即係話 saddle 喺每場賽事入面都係唯一嘅，dict 唔會覆寫。")
    else:
        print(f"🚨 results 表：搵到 {len(duplicates)} 個 (race, saddle) 組合有重複馬匹！")
        print("\n首 20 個例子：")
        dup_with_horses = duplicates.merge(
            results_df[["race_key", "saddle", "horse_id"]],
            on=["race_key", "saddle"],
            how="left"
        )
        print(dup_with_horses.head(20).to_string(index=False))

    print("\n" + "=" * 80)
    print("2. 檢查 training_set_safe.csv：同一場賽事入面，saddle 數值有冇重複？")
    print("=" * 80)

    if not TRAIN_PATH.exists():
        print(f"⚠️ 搵唔到 {TRAIN_PATH}")
        return

    train_df = pd.read_csv(TRAIN_PATH)
    train_df["race_key"] = (
        train_df["race_date"].astype(str) + "_" +
        train_df["venue"] + "_R" + train_df["race_no"].astype(str)
    )

    if "saddle" not in train_df.columns:
        print("⚠️ training_set_safe.csv 冇 'saddle' 欄位。")
        print("   呢個會導致 backtest_quinella_expanding_calibration.py 報錯，")
        print("   因為佢嘗試 merge saddle 欄位。")
        return

    train_saddle_counts = train_df.groupby(["race_key", "saddle"]).size().reset_index(name="count")
    train_duplicates = train_saddle_counts[train_saddle_counts["count"] > 1]

    if len(train_duplicates) == 0:
        print("✅ training_set_safe.csv：冇發現任何 (race, saddle) 組合有超過一匹馬。")
    else:
        print(f"🚨 training_set_safe.csv：搵到 {len(train_duplicates)} 個 (race, saddle) 組合有重複馬匹！")
        print("\n首 20 個例子：")
        train_dup_with_horses = train_duplicates.merge(
            train_df[["race_key", "saddle", "horse_id"]],
            on=["race_key", "saddle"],
            how="left"
        )
        print(train_dup_with_horses.head(20).to_string(index=False))

    print("\n" + "=" * 80)
    print("3. 檢查 predictions 檔案 merge 後：同一場賽事入面，saddle 數值有冇重複？")
    print("=" * 80)

    if not WIN_PRED_PATH.exists():
        print(f"⚠️ 搵唔到 {WIN_PRED_PATH}")
        return

    pred_df = pd.read_csv(WIN_PRED_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    pred_df["race_date"] = pd.to_datetime(pred_df["race_date"], errors="coerce")
    train_df["race_date"] = pd.to_datetime(train_df["race_date"], errors="coerce")

    merge_keys = ["race_date", "venue", "race_no", "horse_id"]
    win_df = pred_df.merge(train_df[merge_keys + ["saddle"]], on=merge_keys, how="left")
    win_df = win_df.dropna(subset=["saddle"]).copy()
    win_df["saddle"] = win_df["saddle"].astype(int)

    win_df["race_key"] = (
        win_df["race_date"].dt.strftime("%Y-%m-%d") + "_" +
        win_df["venue"] + "_R" + win_df["race_no"].astype(str)
    )

    win_saddle_counts = win_df.groupby(["race_key", "saddle"]).size().reset_index(name="count")
    win_duplicates = win_saddle_counts[win_saddle_counts["count"] > 1]

    if len(win_duplicates) == 0:
        print("✅ predictions + training_set merge 後：冇發現任何 (race, saddle) 組合有超過一匹馬。")
    else:
        print(f"🚨 predictions + training_set merge 後：搵到 {len(win_duplicates)} 個 (race, saddle) 組合有重複馬匹！")
        print("\n首 20 個例子：")
        win_dup_with_horses = win_duplicates.merge(
            win_df[["race_key", "saddle", "horse_id"]],
            on=["race_key", "saddle"],
            how="left"
        )
        print(win_dup_with_horses.head(20).to_string(index=False))

    print("\n" + "=" * 80)
    print("🎯 診斷結論")
    print("=" * 80)
    if len(duplicates) == 0 and len(train_duplicates) == 0 and len(win_duplicates) == 0:
        print("三個資料源都冇發現 saddle 重複。")
        print("呢個排除咗 'dict 覆寫' 呢個假設，問題仲未解決。")
        print("下一步：需要直接檢查 backtest 輸出嘅 bet-level CSV，")
        print("       逐注檢查 is_winner 係點樣被錯誤設定為 True。")
    else:
        print("發現 saddle 重複！呢個就係 root cause。")
        print("修復方案：backtest script 唔可以用 dict(zip(saddle, horse_id))，")
        print("         而要用更穩陣嘅配對方法（例如 merge on race+saddle）。")


if __name__ == "__main__":
    main()