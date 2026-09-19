"""Minimal debug script: print winning_combo vs all candidate pairs for a single race.

This isolates the exact logic from backtest_quinella_expanding_calibration.py
that determines is_winner, and prints raw values so we can see WHY every bet
is being flagged as a winner (99.8% hit rate).

FIXED: Explicitly convert dividends_df["race_date"] to datetime to avoid
string vs Timestamp comparison bug.

Run:
  python3 features/debug_winning_combo.py
"""

from __future__ import annotations

from pathlib import Path
from itertools import combinations

import pandas as pd
from sklearn.linear_model import LogisticRegression

try:
    import duckdb
except ImportError:
    raise ImportError("請先安裝 duckdb: pip install duckdb")

DB_PATH = Path("data/processed/hkjc.duckdb")
WIN_PRED_PATH = Path("output/regularization/F_reg_2_predictions_all_folds.csv")
TRAIN_PATH = Path("training_set_safe.csv")
DIVIDENDS_PATH = Path("dividends.csv")

# Pick a specific race to debug - change this if needed
DEBUG_RACE_DATE = "2025-07-01"
DEBUG_VENUE = "ST"
DEBUG_RACE_NO = 1


def main() -> None:
    print("=" * 80)
    print(f"Debug race: {DEBUG_RACE_DATE} {DEBUG_VENUE} R{DEBUG_RACE_NO}")
    print("=" * 80)

    # Load results
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    results_race = conn.execute("""
        SELECT race_date, venue, race_no, horse_id, finish_pos, saddle
        FROM results
        WHERE race_date = ? AND venue = ? AND race_no = ?
        AND finish_pos IS NOT NULL
        ORDER BY finish_pos
    """, [DEBUG_RACE_DATE, DEBUG_VENUE, DEBUG_RACE_NO]).fetchdf()
    conn.close()

    if len(results_race) == 0:
        print(f"🚨 搵唔到呢場賽事嘅結果資料")
        return

    print(f"\n📊 呢場賽事有 {len(results_race)} 匹馬")
    print("\n完整賽果（按 finish_pos 排序）:")
    print(results_race[["horse_id", "finish_pos", "saddle"]].to_string(index=False))

    # Get winning_combo using the EXACT same logic as backtest_quinella_expanding_calibration.py
    sorted_results = results_race.sort_values("finish_pos")
    top_2 = sorted_results.head(2)
    if len(top_2) < 2:
        print("🚨 finish_pos 資料唔夠 2 名")
        return

    h1, h2 = top_2.iloc[0]["horse_id"], top_2.iloc[1]["horse_id"]
    winning_combo = tuple(sorted([h1, h2]))

    print(f"\n🏆 真正贏出組合 (winning_combo): {winning_combo}")
    print(f"   第 1 名：horse_id={h1!r}, finish_pos={top_2.iloc[0]['finish_pos']}")
    print(f"   第 2 名：horse_id={h2!r}, finish_pos={top_2.iloc[1]['finish_pos']}")

    # Load predictions for this race
    pred_df = pd.read_csv(WIN_PRED_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    pred_df["race_date"] = pd.to_datetime(pred_df["race_date"], errors="coerce")
    train_df["race_date"] = pd.to_datetime(train_df["race_date"], errors="coerce")

    merge_keys = ["race_date", "venue", "race_no", "horse_id"]
    race_df = pred_df.merge(train_df[merge_keys + ["saddle"]], on=merge_keys, how="left")
    race_df = race_df.dropna(subset=["saddle"]).copy()
    race_df["saddle"] = race_df["saddle"].astype(int)

    race_df = race_df[(race_df["race_date"] == pd.to_datetime(DEBUG_RACE_DATE)) &
                       (race_df["venue"] == DEBUG_VENUE) &
                       (race_df["race_no"] == DEBUG_RACE_NO)]

    if len(race_df) == 0:
        print(f"🚨 搵唔到呢場賽事嘅 predictions")
        return

    print(f"\n📊 呢場賽事有 {len(race_df)} 匹馬喺 predictions 入面")

    # Build saddle_to_horse mapping
    saddle_to_horse = dict(zip(race_df["saddle"], race_df["horse_id"]))
    print(f"\n🔑 saddle_to_horse mapping: {saddle_to_horse}")

    # Load dividends for this race
    dividends_df = pd.read_csv(DIVIDENDS_PATH)
    # *** CRITICAL FIX: convert race_date to datetime ***
    dividends_df["race_date"] = pd.to_datetime(dividends_df["race_date"], errors="coerce")

    dividends_race = dividends_df[
        (dividends_df["race_date"] == pd.to_datetime(DEBUG_RACE_DATE)) &
        (dividends_df["venue"] == DEBUG_VENUE) &
        (dividends_df["race_no"] == DEBUG_RACE_NO)
    ]

    if len(dividends_race) == 0:
        print(f"🚨 搵唔到呢場賽事嘅 dividends")
        return

    quinella_race = dividends_race[dividends_race["pool"] == "QUINELLA"]
    print(f"\n📊 呢場賽事有 {len(quinella_race)} 個 QUINELLA 派彩紀錄")

    # Parse combinations and build market_quinella_dict
    def parse_combination(comb_str):
        parts = str(comb_str).split(",")
        if len(parts) == 2:
            return tuple(sorted([int(parts[0].strip()), int(parts[1].strip())]))
        return None

    quinella_race = quinella_race.copy()
    quinella_race["combination_parsed"] = quinella_race["combination"].apply(parse_combination)
    quinella_race = quinella_race.dropna(subset=["combination_parsed"])

    market_dict = {}
    for _, row in quinella_race.iterrows():
        comb = row["combination_parsed"]
        horse_i = saddle_to_horse.get(comb[0], None)
        horse_j = saddle_to_horse.get(comb[1], None)
        if horse_i is None or horse_j is None:
            continue
        pair = tuple(sorted([horse_i, horse_j]))
        market_dict[pair] = (1.0 / row["dividend"], row["dividend"])

    print(f"\n📊 market_quinella_dict 有 {len(market_dict)} 個 pair")

    # Compute model_quinella_dict
    # Note: pred_win_prob_platt_oof may not exist in this race if it's from folds 1-4
    # (which are only used for calibration, not evaluation). If so, we need to use raw probs.
    if "pred_win_prob_platt_oof" not in race_df.columns:
        print("\n⚠️ 呢場賽事冇 pred_win_prob_platt_oof 欄位（可能係 folds 1-4，只用於 calibration）。")
        print("   為咗 debug，我哋暫時用 raw probability 代替（雖然唔應該咁樣做 backtest）。")
        win_probs = race_df.set_index("horse_id")["pred_win_prob_raw"]
    else:
        win_probs = race_df.set_index("horse_id")["pred_win_prob_platt_oof"]

    horse_ids = race_df["horse_id"].tolist()

    model_quinella_dict = {}
    for idx_i, idx_j in combinations(range(len(horse_ids)), 2):
        p_i = win_probs.iloc[idx_i]
        p_j = win_probs.iloc[idx_j]
        if p_i < 1.0 and p_j < 1.0:
            p_quinella = p_i * p_j * (1.0 / (1.0 - p_i) + 1.0 / (1.0 - p_j))
        else:
            p_quinella = 0.0
        pair = tuple(sorted([horse_ids[idx_i], horse_ids[idx_j]]))
        model_quinella_dict[pair] = p_quinella

    print(f"\n📊 model_quinella_dict 有 {len(model_quinella_dict)} 個 pair")

    # Now simulate the bet filtering logic
    print("\n" + "=" * 80)
    print("逐注檢查：每注嘅 pair 同 winning_combo 係咪匹配？")
    print("=" * 80)

    threshold = 0.00  # Most permissive threshold
    bets = []

    for pair, model_prob in model_quinella_dict.items():
        if pair not in market_dict:
            continue

        market_prob, dividend = market_dict[pair]
        edge = model_prob - market_prob

        if edge >= threshold:
            is_winner = (pair == winning_combo)
            bets.append({
                "pair": pair,
                "model_prob": model_prob,
                "market_prob": market_prob,
                "edge": edge,
                "dividend": dividend,
                "is_winner": is_winner,
            })

            print(f"\n注：pair={pair}")
            print(f"   model_prob={model_prob:.4f}, market_prob={market_prob:.4f}, edge={edge:.4f}")
            print(f"   winning_combo={winning_combo}")
            print(f"   is_winner={is_winner} ← pair == winning_combo: {pair == winning_combo}")

    print("\n" + "=" * 80)
    print("總結")
    print("=" * 80)
    print(f"符合 threshold >= {threshold} 嘅注數：{len(bets)}")
    print(f"其中 is_winner=True 嘅注數：{sum(1 for b in bets if b['is_winner'])}")
    print(f"hit_rate: {sum(1 for b in bets if b['is_winner']) / len(bets) if bets else 0:.2%}")

    if all(b["is_winner"] for b in bets):
        print("\n🚨 異常：所有注都被判定為 is_winner=True！")
        print("   呢個直接解釋咗點解 backtest 結果有 99.8% hit rate。")
        print("\n可能原因：")
        print("   1. winning_combo 嘅值唔係 horse_id（例如係 saddle number 或其他）")
        print("   2. pair 同 winning_combo 嘅格式唔一致（例如一個係 tuple 一個係 list）")
        print("   3. winning_combo 被錯誤設定為一個會 match 所有 pair 嘅值")
    else:
        print("\n✅ 有部分注被判定為輸，邏輯似乎正常。")


if __name__ == "__main__":
    main()