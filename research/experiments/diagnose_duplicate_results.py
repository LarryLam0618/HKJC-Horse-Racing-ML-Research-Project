"""Diagnose why SELECT ... FROM results WHERE finish_pos IS NOT NULL
returned 41794 rows / 3359 races this run, vs 21083 rows / 1711 races in an
earlier run of the same query shape. The ~2.0x ratio strongly suggests
duplicate rows in the `results` table (e.g. same race+horse written twice
by different ingestion passes), not a real data increase.

This script checks, per (race_date, venue, race_no, horse_id):
  1. How many rows exist for that key combination?
  2. Are the duplicate rows IDENTICAL across all columns, or do they differ
     (e.g. one has saddle populated, one doesn't; different finish_pos)?
  3. Does deduplicating restore the original ~21083 / 1711 row counts?

Run:
  python3 features/diagnose_duplicate_results.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import duckdb
except ImportError:
    raise ImportError("請先安裝 duckdb: pip install duckdb")

DB_PATH = Path("data/processed/hkjc.duckdb")


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"搵唔到 {DB_PATH}")

    conn = duckdb.connect(str(DB_PATH), read_only=True)

    print("=" * 80)
    print("1. 總行數 vs finish_pos non-null 行數")
    print("=" * 80)
    total_rows = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    nonnull_rows = conn.execute(
        "SELECT COUNT(*) FROM results WHERE finish_pos IS NOT NULL"
    ).fetchone()[0]
    print(f"results 表總行數: {total_rows}")
    print(f"finish_pos non-null 行數: {nonnull_rows}")

    print("\n" + "=" * 80)
    print("2. 每個 (race_date, venue, race_no, horse_id) 組合出現幾多次？")
    print("=" * 80)
    dup_check = conn.execute("""
        SELECT race_date, venue, race_no, horse_id, COUNT(*) AS n
        FROM results
        WHERE finish_pos IS NOT NULL
        GROUP BY race_date, venue, race_no, horse_id
        HAVING COUNT(*) > 1
        ORDER BY n DESC
        LIMIT 20
    """).fetchdf()
    print(f"有重複嘅 (race, horse) 組合數: {len(dup_check)}（只顯示首 20 個）")
    print(dup_check.to_string(index=False))

    total_dup_groups = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT race_date, venue, race_no, horse_id
            FROM results
            WHERE finish_pos IS NOT NULL
            GROUP BY race_date, venue, race_no, horse_id
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]
    print(f"\n總共有重複嘅 (race, horse) 組合數: {total_dup_groups}")

    print("\n" + "=" * 80)
    print("3. 抽樣檢查：重複行係唔係內容完全一樣？")
    print("=" * 80)
    if len(dup_check) > 0:
        sample = dup_check.iloc[0]
        sample_rows = conn.execute("""
            SELECT * FROM results
            WHERE race_date = ? AND venue = ? AND race_no = ? AND horse_id = ?
        """, [sample["race_date"], sample["venue"], sample["race_no"], sample["horse_id"]]).fetchdf()
        print(f"樣本: {sample['race_date']} {sample['venue']} R{sample['race_no']} {sample['horse_id']}")
        print(sample_rows.to_string(index=False))

        all_identical = sample_rows.drop_duplicates().shape[0] == 1
        print(f"\n呢批重複行係否完全一樣（所有欄位）: {all_identical}")
        if not all_identical:
            print("⚠️ 重複行內容唔一樣！可能係唔同 import batch 或者 saddle 值有差異。")
            for col in sample_rows.columns:
                unique_vals = sample_rows[col].unique()
                if len(unique_vals) > 1:
                    print(f"  欄位 '{col}' 有唔同值: {unique_vals}")

    print("\n" + "=" * 80)
    print("4. 用 DISTINCT 去重後，行數/場數係唔係回復到 ~21083 / ~1711？")
    print("=" * 80)
    distinct_rows = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT race_date, venue, race_no, horse_id, finish_pos, saddle
            FROM results
            WHERE finish_pos IS NOT NULL
        )
    """).fetchone()[0]
    distinct_races = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT race_date, venue, race_no
            FROM results
            WHERE finish_pos IS NOT NULL
        )
    """).fetchone()[0]
    print(f"DISTINCT (race_date, venue, race_no, horse_id, finish_pos, saddle) 行數: {distinct_rows}")
    print(f"DISTINCT (race_date, venue, race_no) 場數: {distinct_races}")

    print("\n" + "=" * 80)
    print("5. 檢查 saddle 欄位喺重複組合入面嘅完整度")
    print("=" * 80)
    saddle_null_check = conn.execute("""
        SELECT
            SUM(CASE WHEN saddle IS NULL THEN 1 ELSE 0 END) AS null_saddle,
            SUM(CASE WHEN saddle IS NOT NULL THEN 1 ELSE 0 END) AS nonnull_saddle,
            COUNT(*) AS total
        FROM results
        WHERE finish_pos IS NOT NULL
    """).fetchdf()
    print(saddle_null_check.to_string(index=False))

    conn.close()

    print("\n" + "=" * 80)
    print("🎯 診斷結論")
    print("=" * 80)
    if total_dup_groups > 0:
        print(f"確認 results 表有 {total_dup_groups} 組 (race, horse) 重複紀錄。")
        print("下一步：backtest script 嘅 SQL 必須加 DISTINCT 或者 dedupe 邏輯，")
        print("否則 get_winning_quinella() 會攞到重複嘅 top-2，導致 is_winner 判斷全部錯亂，")
        print("直接解釋咗你依家見到嘅 99.8% hit rate 同 4800% ROI 呢個荒謬結果。")
    else:
        print("冇搵到重複組合，問題可能出喺其他地方（例如 finish_pos 本身有重複值、")
        print("或者 dead_heat 標記令多過一匹馬有 finish_pos=1）。需要進一步檢查 dead_heat 欄位。")


if __name__ == "__main__":
    main()