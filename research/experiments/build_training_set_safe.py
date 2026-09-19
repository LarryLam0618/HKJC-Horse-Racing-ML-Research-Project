import pandas as pd
import numpy as np
from pathlib import Path
import csv


BASE_RESULTS = "results.csv"
RACES = "races.csv"
SECTIONALS_LAGGED = "sectional_features_lagged.csv"
TRIALS_LATEST = "trial_features_latest.csv"   # 如果未有可先略過
OUTPUT = "training_set_safe.csv"


BASE_KEYS = ["race_date", "venue", "race_no", "horse_id"]
RACE_KEYS = ["race_date", "venue", "race_no"]


def detect_delimiter(filepath: str, sample_size: int = 2048) -> str:
    with open(filepath, "r", encoding="utf-8-sig") as f:
        sample = f.read(sample_size)
    return csv.Sniffer().sniff(sample).delimiter


def read_csv_auto(filepath: str) -> pd.DataFrame:
    try:
        sep = detect_delimiter(filepath)
        print(f"✅ {filepath} delimiter = {repr(sep)}")
        return pd.read_csv(filepath, sep=sep, engine="python")
    except Exception as e:
        print(f"⚠️ delimiter sniff failed for {filepath}: {e}")
        for sep in [",", "\t", "|", ";"]:
            try:
                df = pd.read_csv(filepath, sep=sep, engine="python")
                if len(df.columns) > 1:
                    print(f"✅ {filepath} fallback delimiter = {repr(sep)}")
                    return df
            except Exception:
                pass
    raise ValueError(f"❌ 無法讀取 {filepath}")


def standardize_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def ensure_unique_key(df: pd.DataFrame, keys: list[str], name: str):
    dup = df.duplicated(keys).sum()
    if dup > 0:
        sample = df[df.duplicated(keys, keep=False)].sort_values(keys).head(10)
        raise ValueError(
            f"❌ {name} 喺 key {keys} 上有 {dup} 個重覆，merge 會爆 row。\n"
            f"樣本：\n{sample}"
        )
    print(f"✅ {name} key 唯一: {keys}")


def left_merge_strict(left: pd.DataFrame, right: pd.DataFrame, on: list[str], name: str, validate_type: str = "one_to_one") -> pd.DataFrame:
    before = len(left)
    out = left.merge(right, how="left", on=on, validate=validate_type)
    after = len(out)
    if before != after:
        raise ValueError(f"❌ merge {name} 後 row 數改變: {before} -> {after}")
    matched = out[on].merge(right[on].drop_duplicates(), how="left", on=on)
    coverage = matched.notna().all(axis=1).mean()
    print(f"✅ merge {name} 完成，row count 保持 {after}")
    print(f"📌 {name} key-level coverage: {coverage:.2%}")
    return out


def build_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "finish_pos" in out.columns:
        out["win_label"] = (pd.to_numeric(out["finish_pos"], errors="coerce") == 1).astype("Int64")

    if "finish_pos" in out.columns:
        pos = pd.to_numeric(out["finish_pos"], errors="coerce")
        out["top3_label"] = (pos <= 3).astype("Int64")

    return out


def filter_race_columns(races: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "race_date", "venue", "race_no",
        "race_index", "race_class", "distance_m", "rating_band",
        "race_name", "prize_hkd", "going", "course", "rail", "surface"
    ]
    keep = [c for c in keep if c in races.columns]
    out = races[keep].copy()
    ensure_unique_key(out, RACE_KEYS, "races")
    return out


def filter_sectional_columns(sec: pd.DataFrame) -> pd.DataFrame:
    banned = {
        "final_split", "fastest_split", "avg_split", "split_std",
        "pos_change", "late_speed", "late_speed_ratio", "total_time"
    }
    bad_found = [c for c in sec.columns if c in banned]
    if bad_found:
        raise ValueError(f"❌ sectional lagged 檔案仍含當場欄位: {bad_found}")

    required = BASE_KEYS.copy()
    missing = [c for c in required if c not in sec.columns]
    if missing:
        raise ValueError(f"❌ sectional lagged 缺少 keys: {missing}")

    keep = required + [
        c for c in sec.columns
        if c.endswith("_lag1") or c in ["sectional_history_count", "has_sectional_history"]
    ]
    keep = list(dict.fromkeys(keep))
    out = sec[keep].copy()
    ensure_unique_key(out, BASE_KEYS, "sectionals_lagged")
    return out


def filter_trial_columns(trials: pd.DataFrame) -> pd.DataFrame:
    required = ["horse_id"]
    missing = [c for c in required if c not in trials.columns]
    if missing:
        raise ValueError(f"❌ trial features 缺少必要欄位: {missing}")

    if "latest_trial_date" in trials.columns:
        trials["latest_trial_date"] = pd.to_datetime(trials["latest_trial_date"], errors="coerce")

    # 只保留 latest_ 前綴欄位，避免混入中間欄
    keep = ["horse_id"] + [c for c in trials.columns if c.startswith("latest_trial_")]
    keep = list(dict.fromkeys(keep))
    out = trials[keep].copy()

    ensure_unique_key(out, ["horse_id"], "trial_features_latest")
    return out


def check_trial_temporal_safety(df: pd.DataFrame):
    if "latest_trial_date" in df.columns:
        bad = df["latest_trial_date"].notna() & df["race_date"].notna() & (df["latest_trial_date"] > df["race_date"])
        n_bad = int(bad.sum())
        if n_bad > 0:
            sample = df.loc[bad, ["horse_id", "race_date", "latest_trial_date"]].head(10)
            raise ValueError(
                f"❌ 發現 {n_bad} 行 trial leakage：latest_trial_date > race_date\n{sample}"
            )
        print("✅ trial 時間安全檢查通過：latest_trial_date <= race_date")


def check_forbidden_feature_names(df: pd.DataFrame):
    forbidden = [
        "finish_pos", "finish_pos_raw", "dead_heat", "lbw_raw",
        "running_position_raw", "finish_time_s"
    ]
    protected = set(BASE_KEYS + ["horse_name", "jockey_code", "jockey_name", "trainer_code",
                                 "trainer_name", "actual_weight", "declared_weight", "draw",
                                 "win_odds", "date", "season", "win_label", "top3_label"])
    leaked = [c for c in df.columns if c in forbidden and c not in protected]
    if leaked:
        raise ValueError(f"❌ 發現不應存在的結果欄位混入 features: {leaked}")


def summarize(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("📊 training set 摘要")
    print("=" * 60)
    print(f"Rows: {len(df)}")
    print(f"Races: {df.groupby(RACE_KEYS).ngroups}")
    print(f"Horses: {df['horse_id'].nunique()}")

    if "win_label" in df.columns:
        print(f"Win rate: {df['win_label'].mean():.4f}")

    sec_cols = [c for c in df.columns if c.endswith("_lag1")]
    if sec_cols:
        print("\nSectional lag NA rate:")
        print(df[sec_cols].isna().mean().sort_values(ascending=False).head(10))

    if "has_sectional_history" in df.columns:
        print(f"\nHas sectional history: {df['has_sectional_history'].mean():.2%}")


def main():
    print("📂 讀取 base results ...")
    results = read_csv_auto(BASE_RESULTS)
    results = standardize_dates(results, ["race_date", "date"])
    ensure_unique_key(results, BASE_KEYS, "results")

    print("📂 讀取 races ...")
    races = read_csv_auto(RACES)
    races = standardize_dates(races, ["race_date", "date"])
    races = filter_race_columns(races)

    print("📂 讀取 sectional lagged ...")
    sec = read_csv_auto(SECTIONALS_LAGGED)
    sec = standardize_dates(sec, ["race_date"])
    sec = filter_sectional_columns(sec)

    df = results.copy()
    df = build_target_columns(df)

    # races 係賽事級別 (1場1行)，results 係馬匹級別 (1場14行)，所以係 many_to_one
    df = left_merge_strict(df, races, RACE_KEYS, "races", validate_type="many_to_one")
    
    # sectionals 係馬匹級別，所以係 one_to_one
    df = left_merge_strict(df, sec, BASE_KEYS, "sectionals_lagged", validate_type="one_to_one")

    if Path(TRIALS_LATEST).exists():
        print("📂 讀取 latest trial features ...")
        trials = read_csv_auto(TRIALS_LATEST)
        trials = filter_trial_columns(trials)
        df = df.merge(trials, how="left", on=["horse_id"], validate="many_to_one")
        print("✅ merge trial_features_latest 完成")
        check_trial_temporal_safety(df)
    else:
        print(f"⚠️ {TRIALS_LATEST} 不存在，跳過 trial merge")

    # 🧹 清理原始結果欄位，防止 leakage
    # 我們已經用 finish_pos 計算咗 win_label / top3_label，所以呢啲原始欄位可以安全丟棄
    forbidden_cols_to_drop = [
        "finish_pos", "finish_pos_raw", "dead_heat", "lbw_raw",
        "running_position_raw", "finish_time_s"
    ]
    cols_to_drop = [c for c in forbidden_cols_to_drop if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"🧹 已清理原始結果欄位: {cols_to_drop}")

    check_forbidden_feature_names(df)
    summarize(df)

    df.to_csv(OUTPUT, index=False)
    print(f"\n✅ 已輸出: {OUTPUT}")
    print("✅ 呢份 training set 可作為安全版建模底稿")


if __name__ == "__main__":
    main()