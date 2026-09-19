"""Diagnose the 99.8%-100% hit_rate bug in the saddle-based quinella backtest.

Row-count/duplicate checks are already ruled out (diagnose_duplicate_results.py
found 0 duplicates, saddle 100% populated). So the bug must be in the actual
matching logic between:
  - winning_combo, derived from DuckDB `results.horse_id`
  - model_quinella_dict keys, derived from training_set_safe.csv `horse_id`
    (merged into win_df)

If horse_id string formats differ between these two sources (leading zeros,
casing, whitespace, prefix conventions), tuple equality `pair == winning_combo`
could behave unpredictably. This script prints raw values side by side for a
handful of races so any formatting mismatch is immediately visible.

It also directly recomputes is_winner for a few sample races using the exact
logic from backtest_quinella_expanding_calibration.py, so we can see exactly
where the false-positive winner flags come from.

Run:
  python3 features/diagnose_horse_id_matching.py
"""

from __future__ import annotations

from pathlib import Path
from itertools import combinations

import pandas as pd

try:
    import duckdb
except ImportError:
    raise ImportError("請先安裝 duckdb: pip install duckdb")

DB_PATH = Path("data/processed/hkjc.duckdb")
WIN_PRED_PATH = Path("output/regularization/F_reg_2_predictions_all_folds.csv")
TRAIN_PATH = Path("training_set_safe.csv")
DIVIDENDS_PATH = Path("dividends.csv")

N_SAMPLE_RACES = 5


def main() -> None:
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    results_df = conn.execute("""
        SELECT race_date, venue, race_no, horse_id, finish_pos, saddle
        FROM results WHERE finish_pos IS NOT NULL
        ORDER BY race_date, venue, race_no, finish_pos
    """).fetchdf()
    conn.close()
    results_df["race_date"] = pd.to_datetime(results_df["race_date"], errors="coerce")

    pred_df = pd.read_csv(WIN_PRED_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    pred_df["race_date"] = pd.to_datetime(pred_df["race_date"], errors="coerce")
    train_df["race_date"] = pd.to_datetime(train_df["race_date"], errors="coerce")

    merge_keys = ["race_date", "venue", "race_no", "horse_id"]
    win_df = pred_df.merge(train_df[merge_keys + ["saddle"]], on=merge_keys, how="left")
    win_df = win_df.dropna(subset=["saddle"]).copy()
    win_df["saddle"] = win_df["saddle"].astype(int)

    print("=" * 80)
    print("0. horse_id 格式對比：results (DuckDB) vs win_df (predictions)")
    print("=" * 80)
    print("results.horse_id 樣本 (前 10 個 unique):")
    print(results_df["horse_id"].drop_duplicates().head(10).tolist())
    print("\nwin_df.horse_id 樣本 (前 10 個 unique):")
    print(win_df["horse_id"].drop_duplicates().head(10).tolist())

    results_id_set = set(results_df["horse_id"].unique())
    win_id_set = set(win_df["horse_id"].unique())
    overlap = results_id_set & win_id_set
    print(f"\nresults 表 unique horse_id 數: {len(results_id_set)}")
    print(f"win_df unique horse_id 數: {len(win_id_set)}")
    print(f"兩者重疊數: {len(overlap)}")
    print(f"重疊比例 (相對 win_df): {len(overlap)/len(win_id_set):.2%}" if win_id_set else "N/A")

    dtype_results = results_df["horse_id"].dtype
    dtype_win = win_df["horse_id"].dtype
    print(f"\nresults.horse_id dtype: {dtype_results}")
    print(f"win_df.horse_id dtype: {dtype_win}")

    print("\n" + "=" * 80)
    print(f"1. 抽 {N_SAMPLE_RACES} 場賽事，逐步重演 is_winner 判斷邏輯")
    print("=" * 80)

    results_by_race = results_df.groupby(["race_date", "venue", "race_no"])
    win_by_race = win_df.groupby(["race_date", "venue", "race_no"])

    common_races = [k for k in results_by_race.groups.keys() if k in win_by_race.groups.keys()]
    print(f"\nresults 同 win_df 共有嘅場次數: {len(common_races)}")

    for race_key in common_races[:N_SAMPLE_RACES]:
        race_date, venue, race_no = race_key
        results_race = results_by_race.get_group(race_key)
        race_df = win_by_race.get_group(race_key)

        print(f"\n--- {race_date.date()} {venue} R{race_no} ---")

        sorted_results = results_race.sort_values("finish_pos")
        top_2 = sorted_results.head(2)
        if len(top_2) < 2:
            print("  finish_pos 資料唔夠 2 名，跳過")
            continue

        h1, h2 = top_2.iloc[0]["horse_id"], top_2.iloc[1]["horse_id"]
        winning_combo = tuple(sorted([h1, h2]))
        print(f"  真正贏出組合 (由 results.finish_pos 排出): {winning_combo}")
        print(f"    第 1 名 horse_id={h1!r} finish_pos={top_2.iloc[0]['finish_pos']}")
        print(f"    第 2 名 horse_id={h2!r} finish_pos={top_2.iloc[1]['finish_pos']}")

        horse_ids_in_race_df = race_df["horse_id"].tolist()
        print(f"  race_df (predictions) 呢場有 {len(horse_ids_in_race_df)} 匹馬：{horse_ids_in_race_df}")

        h1_in_preds = h1 in horse_ids_in_race_df
        h2_in_preds = h2 in horse_ids_in_race_df
        print(f"  贏馬 {h1!r} 喺 predictions 出現：{h1_in_preds}")
        print(f"  贏馬 {h2!r} 喺 predictions 出現：{h2_in_preds}")

        all_pairs = [tuple(sorted(p)) for p in combinations(horse_ids_in_race_df, 2)]
        n_matches = sum(1 for p in all_pairs if p == winning_combo)
        print(f"  總共 {len(all_pairs)} 個候選 pair，同 winning_combo 完全相等嘅有：{n_matches}")

        if n_matches > 1:
            print("  🚨 異常：多過一個 pair 被判定同 winning_combo 相等！")
        elif n_matches == 0 and h1_in_preds and h2_in_preds:
            print("  🚨 異常：兩匹贏馬都喺 predictions 出現，但配對唔到 winning_combo！")

    print("\n" + "=" * 80)
    print("🎯 診斷方向")
    print("=" * 80)
    print("如果 horse_id 重疊比例遠低於 100%，或者 dtype 唔一致 (例如一個係 str 一個係 int)，")
    print("就係 root cause：horse_id 格式唔匹配，令 tuple 比較產生唔可預測嘅結果。")
    print("如果重疊比例接近 100% 且逐場檢查都冇異常，問題可能在 dividend 配對或 combination 解析層面，")
    print("需要再深入檢查 compute_market_quinella_dict() 嗰段。")


if __name__ == "__main__":
    main()