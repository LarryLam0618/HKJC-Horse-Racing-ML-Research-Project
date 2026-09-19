"""Verify which column -- `draw` or `saddle` (saddle cloth / racing number) --
is the correct mapping key for dividends.csv combination numbers.

Why this matters:
  All prior quinella backtests (backtest_quinella_expanding_calibration.py,
  backtest_quinella_ev_ratio.py) build `draw_to_horse = dict(zip(race_df["draw"],
  race_df["horse_id"]))` and use that to translate QUINELLA combination
  strings (e.g. "2,8") into horse_id pairs. If HKJC's combination numbers
  actually refer to saddle cloth number (racing number) rather than starting
  gate draw, every single backtest result to date has been matching model
  probabilities to the WRONG horse's market price. This would invalidate the
  21-bet, 37-bet, per-block, and EV-ratio analyses done so far.

Method:
  For every race, we know the actual winning quinella combination (top-2
  finishers by finish_pos). We test BOTH candidate mapping keys:

    Key A: draw   (starting gate position)
    Key B: saddle (saddle cloth / racing number, if present in the data)

  For each key, we check: does the (horse_i_number, horse_j_number) pair
  implied by that key, for the ACTUAL winning horses, match a QUINELLA
  dividend record with a plausible (non-degenerate) dividend? We also check
  hit-rate consistency: using the correct key, the "declared winning
  combination" derived from dividends.csv should match the actual result's
  top-2 finishers in close to 100% of races (dividends are official payouts,
  so the winning combination is unambiguous ground truth). Using the wrong
  key will produce mismatches or systematically implausible pairings.

  We also directly cross-tabulate, for races where both draw and saddle are
  available and differ, which key's pairing actually appears as a paid-out
  QUINELLA combination in dividends.csv.

Run:
  python3 features/verify_combination_mapping_key.py

Note on DuckDB locking:
  This script opens the DuckDB file in READ-ONLY mode, since it only ever
  runs SELECT queries. This avoids "Conflicting lock is held" errors when
  another process (e.g. a Jupyter kernel or another script) already has the
  same database file open. If you still hit a lock error, it means another
  process holds an exclusive WRITE lock -- find and close it first:
    lsof data/processed/hkjc.duckdb
    kill <PID>
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import duckdb
except ImportError:
    raise ImportError("請先安裝 duckdb: pip install duckdb")

OUTPUT_DIR = Path("output/quinella")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = Path("data/processed/hkjc.duckdb")
DIVIDENDS_PATH = Path("dividends.csv")


def load_results_from_duckdb(db_path: Path) -> pd.DataFrame:
    """Load full results table, including every column, so we can discover
    whatever candidate identifier columns actually exist (draw, saddle,
    horse_no, racing_number, etc.) without guessing the schema up front.

    Opens the connection in read_only=True mode. This is strictly a read
    workload (SELECT only), so read-only mode lets this script run safely
    even while another process (Jupyter kernel, another script) has the
    same file open, and avoids taking an exclusive lock that would block
    other readers/writers."""
    if not db_path.exists():
        raise FileNotFoundError(f"搵唔到 {db_path}")

    try:
        conn = duckdb.connect(str(db_path), read_only=True)
    except duckdb.IOException as e:
        raise RuntimeError(
            f"無法以 read-only 模式打開 {db_path}。\n"
            f"如果另一個 process 持有 EXCLUSIVE 鎖（例如仍在寫入資料庫），"
            f"read-only 連接都會失敗。請先執行:\n"
            f"  lsof {db_path}\n"
            f"搵到 PID 之後 kill 佈，或者關閉持有該連接嘅 Jupyter kernel / script，再重試。\n"
            f"原始錯誤: {e}"
        ) from e

    schema_df = conn.execute("DESCRIBE results").fetchdf()
    print("📋 results 表 schema：")
    print(schema_df[["column_name", "column_type"]].to_string(index=False))

    df = conn.execute("SELECT * FROM results WHERE finish_pos IS NOT NULL").fetchdf()
    conn.close()

    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df = df.dropna(subset=["race_date", "finish_pos"])
    df["finish_pos"] = df["finish_pos"].astype(int)
    return df


def load_dividends(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"搵唔到 {path}")
    df = pd.read_csv(path)
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df = df.dropna(subset=["race_date", "dividend"])
    df["dividend"] = pd.to_numeric(df["dividend"], errors="coerce")
    return df


def parse_combination(comb_str: str):
    parts = str(comb_str).split(",")
    if len(parts) == 2:
        try:
            return tuple(sorted([int(parts[0].strip()), int(parts[1].strip())]))
        except ValueError:
            return None
    return None


def get_winning_horses(results_race: pd.DataFrame) -> pd.DataFrame:
    """Return the top-2 finishers (by finish_pos) for a race."""
    return results_race.sort_values("finish_pos").head(2)


def test_mapping_key(results_df: pd.DataFrame, dividends_df: pd.DataFrame, key_col: str) -> dict:
    """For every race with a QUINELLA dividend, derive the winning combination
    using `key_col` (e.g. draw or saddle) for the actual top-2 finishers, and
    check whether that exact pair appears in dividends.csv's QUINELLA combos
    for that race. A correct key should match in the vast majority of races
    (payouts are ground truth); a wrong key will systematically fail to
    match, or match a different, non-winning-looking record."""

    if key_col not in results_df.columns:
        return {"key": key_col, "error": f"欄位 '{key_col}' 唔存在於 results 表"}

    quinella_df = dividends_df[dividends_df["pool"] == "QUINELLA"].copy()
    quinella_df["combination_parsed"] = quinella_df["combination"].apply(parse_combination)
    quinella_df = quinella_df.dropna(subset=["combination_parsed"])

    div_lookup = {}
    for _, row in quinella_df.iterrows():
        rk = (row["race_date"], row["venue"], row["race_no"])
        div_lookup.setdefault(rk, {})[row["combination_parsed"]] = row["dividend"]

    results_by_race = results_df.groupby(["race_date", "venue", "race_no"])

    n_races_checked = 0
    n_matched = 0
    n_no_dividend_data = 0
    n_missing_key_value = 0
    mismatched_examples = []

    for race_key, race_df in results_by_race:
        if race_key not in div_lookup:
            n_no_dividend_data += 1
            continue

        top2 = get_winning_horses(race_df)
        if len(top2) < 2:
            continue

        key_vals = top2[key_col].values
        if pd.isna(key_vals[0]) or pd.isna(key_vals[1]):
            n_missing_key_value += 1
            continue

        try:
            pair = tuple(sorted([int(key_vals[0]), int(key_vals[1])]))
        except (ValueError, TypeError):
            n_missing_key_value += 1
            continue

        n_races_checked += 1
        race_combos = div_lookup[race_key]

        if pair in race_combos:
            n_matched += 1
        else:
            if len(mismatched_examples) < 5:
                mismatched_examples.append({
                    "race_date": str(race_key[0].date()) if hasattr(race_key[0], "date") else str(race_key[0]),
                    "venue": race_key[1],
                    "race_no": race_key[2],
                    "derived_pair": pair,
                    "available_combos": list(race_combos.keys())[:8],
                })

    match_rate = n_matched / n_races_checked if n_races_checked > 0 else np.nan

    return {
        "key": key_col,
        "n_races_checked": n_races_checked,
        "n_matched": n_matched,
        "match_rate": match_rate,
        "n_races_no_dividend_data": n_no_dividend_data,
        "n_races_missing_key_value": n_missing_key_value,
        "mismatched_examples": mismatched_examples,
    }


def cross_tabulate_draw_vs_saddle(results_df: pd.DataFrame, dividends_df: pd.DataFrame) -> pd.DataFrame:
    """For races where draw and saddle both exist AND differ for at least one
    horse, explicitly show which key's derived winning pair actually appears
    in the paid dividends -- this isolates the disambiguating cases."""

    has_saddle = "saddle" in results_df.columns
    if not has_saddle:
        print("⚠️ results 表冇 'saddle' 欄位，跳過 cross-tabulation（只會有 draw 結果）")
        return pd.DataFrame()

    quinella_df = dividends_df[dividends_df["pool"] == "QUINELLA"].copy()
    quinella_df["combination_parsed"] = quinella_df["combination"].apply(parse_combination)
    quinella_df = quinella_df.dropna(subset=["combination_parsed"])

    div_lookup = {}
    for _, row in quinella_df.iterrows():
        rk = (row["race_date"], row["venue"], row["race_no"])
        div_lookup.setdefault(rk, set()).add(row["combination_parsed"])

    results_by_race = results_df.groupby(["race_date", "venue", "race_no"])

    rows = []
    for race_key, race_df in results_by_race:
        if race_key not in div_lookup:
            continue
        top2 = get_winning_horses(race_df)
        if len(top2) < 2:
            continue

        draw_vals = top2["draw"].values
        saddle_vals = top2["saddle"].values

        if pd.isna(draw_vals).any() or pd.isna(saddle_vals).any():
            continue

        draw_pair = tuple(sorted([int(draw_vals[0]), int(draw_vals[1])]))
        saddle_pair = tuple(sorted([int(saddle_vals[0]), int(saddle_vals[1])]))

        if draw_pair == saddle_pair:
            continue  # not a disambiguating case

        available = div_lookup[race_key]
        rows.append({
            "race_date": race_key[0], "venue": race_key[1], "race_no": race_key[2],
            "draw_pair": draw_pair, "saddle_pair": saddle_pair,
            "draw_pair_paid": draw_pair in available,
            "saddle_pair_paid": saddle_pair in available,
        })

    return pd.DataFrame(rows)


def main() -> None:
    print("📂 Loading results and dividends...")
    results_df = load_results_from_duckdb(DB_PATH)
    dividends_df = load_dividends(DIVIDENDS_PATH)

    candidate_keys = [c for c in ["draw", "saddle", "horse_no", "racing_number", "cloth_no"]
                       if c in results_df.columns]
    print(f"\n📊 偵測到嘅候選 mapping key 欄位: {candidate_keys}")

    if not candidate_keys:
        print("❌ results 表冇搵到任何似係 mapping key 嘅欄位。請人手檢查 schema。")
        return

    print("\n" + "=" * 80)
    print("驗證每個候選 key：derived winning pair 係唔係真係對應到派彩紀錄？")
    print("=" * 80)

    all_results = []
    for key in candidate_keys:
        result = test_mapping_key(results_df, dividends_df, key)
        all_results.append(result)

        print(f"\n🔑 Key: {key}")
        if "error" in result:
            print(f"  {result['error']}")
            continue
        print(f"  檢查嘅賽事數: {result['n_races_checked']}")
        print(f"  成功配對 (derived pair 喺派彩紀錄出現): {result['n_matched']}")
        print(f"  配對率: {result['match_rate']:.2%}" if not np.isnan(result['match_rate']) else "  配對率: N/A")
        print(f"  冇派彩數據嘅賽事: {result['n_races_no_dividend_data']}")
        print(f"  缺 key 值嘅賽事: {result['n_races_missing_key_value']}")

        if result["mismatched_examples"]:
            print(f"  ⚠️ 首 5 個唔配對嘅例子:")
            for ex in result["mismatched_examples"]:
                print(f"    {ex['race_date']} {ex['venue']} R{ex['race_no']}: "
                      f"derived={ex['derived_pair']}, available={ex['available_combos']}")

    print("\n" + "=" * 80)
    print("Cross-tabulation：draw vs saddle 分歧嘅賽事，邊個先真係對應派彩？")
    print("=" * 80)

    crosstab_df = cross_tabulate_draw_vs_saddle(results_df, dividends_df)
    if not crosstab_df.empty:
        n_diff_races = len(crosstab_df)
        draw_correct = crosstab_df["draw_pair_paid"].sum()
        saddle_correct = crosstab_df["saddle_pair_paid"].sum()

        print(f"\ndraw 同 saddle 分歧嘅賽事數: {n_diff_races}")
        print(f"呢啲賽事之中，draw_pair 啱中派彩紀錄: {draw_correct} ({draw_correct/n_diff_races:.1%})")
        print(f"呢啲賽事之中，saddle_pair 啱中派彩紀錄: {saddle_correct} ({saddle_correct/n_diff_races:.1%})")

        if saddle_correct > draw_correct:
            print("\n🚨 結論：saddle 先係啱嘅 mapping key，唔係 draw！")
            print("   之前所有用 draw_to_horse 嘅 backtest 結果需要用 saddle 重新驗證。")
        elif draw_correct > saddle_correct:
            print("\n✅ 結論：draw 係啱嘅 mapping key，之前嘅 backtest 冇用錯。")
        else:
            print("\n⚠️ 結論：兩者表現一樣，可能呢批分歧賽事樣本太少，或者兩個欄位本身高度相關。")

        crosstab_df.to_csv(OUTPUT_DIR / "draw_vs_saddle_crosstab.csv", index=False)
        print(f"\n✅ 詳細 cross-tab 已儲存: output/quinella/draw_vs_saddle_crosstab.csv")
    else:
        print("\n無法做 cross-tabulation（可能冇 saddle 欄位，或者 draw/saddle 完全一致冇分歧樣本）。")

    summary_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "mismatched_examples"}
        for r in all_results
    ])
    summary_df.to_csv(OUTPUT_DIR / "mapping_key_verification_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("🎯 總結")
    print("=" * 80)
    valid_results = [r for r in all_results if "error" not in r and not np.isnan(r["match_rate"])]
    if valid_results:
        best = max(valid_results, key=lambda r: r["match_rate"])
        print(f"配對率最高嘅 key 係 '{best['key']}' ({best['match_rate']:.2%})")
        if best["match_rate"] < 0.90:
            print("⚠️ 注意：即使係最好嘅 key，配對率都低於 90%，可能仲有其他資料問題需要調查")
            print("  (例如 dead heat、退賽、combination parsing 邊緣情況等)")
        else:
            print(f"✅ '{best['key']}' 嘅配對率夠高，可以信任佢做 mapping key。")

        for r in valid_results:
            if r["key"] != best["key"]:
                print(f"\n對比：'{r['key']}' 配對率只有 {r['match_rate']:.2%}")
                if r["key"] == "draw" and best["key"] != "draw":
                    print("🚨 如果你之前嘅 backtest 用嚉 'draw' 做 mapping key，")
                    print("   而依家發現 'draw' 配對率遠低於其他 key，")
                    print("   即係話之前所有連贏 backtest 結果（21注/37注/per-block）都可能建基於錯誤配對，")
                    print("   需要用正確嘅 key 重新跑一次全部分析。")

    print("\n✅ Outputs saved to output/quinella/:")
    print(" - mapping_key_verification_summary.csv")
    print(" - draw_vs_saddle_crosstab.csv (if saddle column exists)")


if __name__ == "__main__":
    main()