"""Directly inspect the bet-level CSV to see WHY is_winner is True for 99.8% of bets.

All prior hypotheses have been ruled out:
  - duplicate rows in results: NO (0 duplicates)
  - horse_id format mismatch: NO (99.88% overlap, exact match in samples)
  - saddle duplicates within a race: NO (all three data sources clean)

So the bug must be in the actual is_winner flagging logic inside
backtest_quinella_expanding_calibration.py. This script reads the output CSV
and cross-checks each bet's (horse_i, horse_j) against the TRUE winning
quinella combination for that race (from DuckDB results.finish_pos).

If is_winner=True but the bet's horse pair does NOT match the actual top-2
finishers, we have found the bug.

Run:
  python3 features/diagnose_is_winner_bug.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import duckdb
except ImportError:
    raise ImportError("請先安裝 duckdb: pip install duckdb")

DB_PATH = Path("data/processed/hkjc.duckdb")
BET_LEVEL_PATH = Path("output/quinella/quinella_bet_level_results_expanding_SADDLE_FIXED.csv")


def main() -> None:
    if not BET_LEVEL_PATH.exists():
        raise FileNotFoundError(f"搵唔到 {BET_LEVEL_PATH}")

    bets_df = pd.read_csv(BET_LEVEL_PATH)
    print(f"📊 讀取 {len(bets_df)} 注")
    print(f"📊 is_winner=True 嘅注數：{int(bets_df['is_winner'].sum())} ({bets_df['is_winner'].mean():.2%})")

    conn = duckdb.connect(str(DB_PATH), read_only=True)
    results_df = conn.execute("""
        SELECT race_date, venue, race_no, horse_id, finish_pos
        FROM results WHERE finish_pos IS NOT NULL
    """).fetchdf()
    conn.close()
    results_df["race_date"] = pd.to_datetime(results_df["race_date"], errors="coerce")

    results_by_race = results_df.groupby(["race_date", "venue", "race_no"])

    print("\n" + "=" * 80)
    print("1. 逐注檢查：is_winner=True 嘅注，horse pair 係唔係真係贏出？")
    print("=" * 80)

    false_positives = []
    true_positives = []

    for idx, row in bets_df.iterrows():
        race_key = (row["race_date"], row["venue"], row["race_no"])
        if race_key not in results_by_race.groups:
            continue

        results_race = results_by_race.get_group(race_key)
        sorted_results = results_race.sort_values("finish_pos")
        top_2 = sorted_results.head(2)
        if len(top_2) < 2:
            continue

        true_winner = tuple(sorted([top_2.iloc[0]["horse_id"], top_2.iloc[1]["horse_id"]]))
        bet_pair = tuple(sorted([row["horse_i"], row["horse_j"]]))

        if row["is_winner"]:
            if bet_pair == true_winner:
                true_positives.append((idx, race_key, bet_pair, true_winner))
            else:
                false_positives.append((idx, race_key, bet_pair, true_winner))

    print(f"\n✅ True positives (is_winner=True 且 horse pair 正確): {len(true_positives)}")
    print(f"🚨 False positives (is_winner=True 但 horse pair 錯誤): {len(false_positives)}")

    if false_positives:
        print(f"\n首 20 個 false positive 例子：")
        for idx, race_key, bet_pair, true_winner in false_positives[:20]:
            print(f"  Row {idx}: {race_key[0].date()} {race_key[1]} R{race_key[2]}")
            print(f"    Bet pair: {bet_pair}")
            print(f"    True winner: {true_winner}")
            print(f"    Match? {bet_pair == true_winner}")
            print()

    print("\n" + "=" * 80)
    print("2. 逐注檢查：is_winner=False 嘅注，horse pair 係唔係真係冇贏？")
    print("=" * 80)

    false_negatives = []
    true_negatives = []

    for idx, row in bets_df.iterrows():
        race_key = (row["race_date"], row["venue"], row["race_no"])
        if race_key not in results_by_race.groups:
            continue

        results_race = results_by_race.get_group(race_key)
        sorted_results = results_race.sort_values("finish_pos")
        top_2 = sorted_results.head(2)
        if len(top_2) < 2:
            continue

        true_winner = tuple(sorted([top_2.iloc[0]["horse_id"], top_2.iloc[1]["horse_id"]]))
        bet_pair = tuple(sorted([row["horse_i"], row["horse_j"]]))

        if not row["is_winner"]:
            if bet_pair == true_winner:
                false_negatives.append((idx, race_key, bet_pair, true_winner))
            else:
                true_negatives.append((idx, race_key, bet_pair, true_winner))

    print(f"\n✅ True negatives (is_winner=False 且 horse pair 正確): {len(true_negatives)}")
    print(f"🚨 False negatives (is_winner=False 但 horse pair 實際贏咗): {len(false_negatives)}")

    if false_negatives:
        print(f"\n首 20 個 false negative 例子：")
        for idx, race_key, bet_pair, true_winner in false_negatives[:20]:
            print(f"  Row {idx}: {race_key[0].date()} {race_key[1]} R{race_key[2]}")
            print(f"    Bet pair: {bet_pair}")
            print(f"    True winner: {true_winner}")
            print(f"    is_winner: {row['is_winner']}")
            print()

    print("\n" + "=" * 80)
    print("🎯 診斷結論")
    print("=" * 80)
    if false_positives:
        print(f"🚨 確認 bug：{len(false_positives)} 注被錯誤標記為 is_winner=True，")
        print("   但佢哋嘅 horse pair 同實際贏出組合唔匹配。")
        print("   呢個直接解釋咗 99.8% hit rate 嘅來源。")
        print("\n下一步：需要檢查 backtest_quinella_expanding_calibration.py 入面，")
        print("       is_winner 係點樣被設定嘅，搵出點解會有呢啲 false positive。")
    elif false_negatives:
        print(f"🚨 確認 bug：{len(false_negatives)} 注被錯誤標記為 is_winner=False，")
        print("   但佢哋嘅 horse pair 實際係贏咗。")
        print("   呢個會導致 hit rate 被低估，但唔會解釋 99.8% 咁高。")
    else:
        print("✅ 所有 is_winner 標籤都同實際賽果一致。")
        print("   咁 99.8% hit rate 係真實嘅——但呢個喺現實世界唔可能，")
        print("   所以一定係其他邏輯錯誤（例如 backtest 本身嘅注碼計算或 ROI 公式）。")


if __name__ == "__main__":
    main()