"""Bootstrap ROI confidence intervals from the quinella bet-level backtest.

跑法 (跑完 backtest_quinella_complete.py 之後):
    python3 features/bootstrap_quinella_roi.py

點解要做呢個: 一個 threshold 淨係得 10-20 注,單一個 ROI 數字冇辦法話俾你知「呢個係真
優勢定係啱啱好撞中」。Bootstrap 攞返一個分佈,等你睇到個 ROI 嘅不確定性有幾大 -- 如果
95% CI 跨過 0,就代表依家個樣本仲未夠證明呢個 threshold 有正邊際優勢。

用 **race-block bootstrap**,唔係逐注 resample -- 如果同一場有多過一注 (例如 quinella
box 開幾個組合),呢啲注嘅輸贏會相關 (同一場結果決定),逐注 resample 會低估真實嘅
variance。用 race 做 resample unit 先啱。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BET_LEVEL_PATH = Path("output/quinella/quinella_bet_level_results_expanding.csv")
OUTPUT_PATH = Path("output/quinella/bootstrap_roi_ci.csv")

N_BOOTSTRAP = 5000
SEED = 42
CI_LOW, CI_HIGH = 2.5, 97.5  # 95% CI
MIN_BETS_PER_THRESHOLD = 5

# 你實際 CSV 嘅欄名 (已核對): bet_count, win_count, hit_rate, total_stake, total_return,
# net_profit, roi, race_date, venue, race_no, threshold -- 即係話呢份 CSV 本身已經係
# race-level 聚合 (一行 = 一場 + 一個 threshold 嘅 sum),唔係逐注 (one row per bet)。
COL_THRESHOLD = "threshold"
COL_STAKE = "total_stake"
COL_RETURN = "total_return"
COL_BET_COUNT = "bet_count"
COL_RACE_KEYS = ["race_date", "venue", "race_no"]


def load_bets(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"❌ 搵唔到 {path} -- 先跑 backtest_quinella_complete.py")
    df = pd.read_csv(path)

    missing = [c for c in [COL_THRESHOLD, COL_STAKE, COL_RETURN] if c not in df.columns]
    if missing:
        raise KeyError(
            f"❌ CSV 冇呢啲欄: {missing}. 實際欄位係: {list(df.columns)}\n"
            "   改返 script 頭 4 個 COL_* 常數對應返你個 CSV 嘅真實欄名。"
        )
    return df


def race_key(df: pd.DataFrame) -> pd.Series:
    present = [c for c in COL_RACE_KEYS if c in df.columns]
    if not present:
        # 冇 race key 就退返做逐注 bootstrap (加個 warning,唔靜靜雞降級)
        print("⚠️  搵唔到 race key 欄，退返做逐注 (bet-level) bootstrap，唔係 race-block")
        return pd.Series(range(len(df)), index=df.index)
    return df[present].astype(str).agg("|".join, axis=1)


def bootstrap_roi(df: pd.DataFrame, n_boot: int, seed: int) -> np.ndarray:
    """Resample whole races (with replacement) n_boot times, return the ROI of each draw."""
    df = df.copy()
    df["_race_key"] = race_key(df)
    races = df["_race_key"].unique()
    rng = np.random.default_rng(seed)

    grouped = df.groupby("_race_key")[[COL_STAKE, COL_RETURN]].sum()

    rois = np.empty(n_boot)
    for i in range(n_boot):
        sample_keys = rng.choice(races, size=len(races), replace=True)
        sample = grouped.loc[sample_keys]
        stake = sample[COL_STAKE].sum()
        rois[i] = (sample[COL_RETURN].sum() - stake) / stake if stake > 0 else np.nan
    return rois[~np.isnan(rois)]


def summarize(threshold: float, rois: np.ndarray, n_bets: int, n_races: int) -> dict:
    ci_low, ci_high = np.percentile(rois, [CI_LOW, CI_HIGH])
    return {
        "threshold": threshold,
        "n_bets": n_bets,
        "n_races": n_races,
        "point_roi": float(rois.mean()),  # bootstrap mean, close to but not identical to the
        # original point-estimate ROI -- report both if you want to sanity-check they agree
        "ci_low_95": float(ci_low),
        "ci_high_95": float(ci_high),
        "p_roi_gt_0": float((rois > 0).mean()),  # fraction of bootstrap draws with ROI>0
        "crosses_zero": bool(ci_low < 0 < ci_high),
    }


def main() -> None:
    df = load_bets(BET_LEVEL_PATH)
    thresholds = sorted(df[COL_THRESHOLD].unique())

    rows = []
    print(f"{'threshold':<10}{'n_bets':<8}{'point_roi':<12}{'95% CI':<24}{'P(ROI>0)':<10}verdict")
    for thr in thresholds:
        sub = df[df[COL_THRESHOLD] == thr]
        n_bets = int(sub[COL_BET_COUNT].sum()) if COL_BET_COUNT in sub.columns else len(sub)
        if n_bets < MIN_BETS_PER_THRESHOLD:
            print(f"{thr:<10}{n_bets:<8}-- 少過 {MIN_BETS_PER_THRESHOLD} 注，跳過 --")
            continue

        rois = bootstrap_roi(sub, N_BOOTSTRAP, SEED)
        n_races = len(sub)  # CSV is already race-level: one row per race per threshold
        result = summarize(thr, rois, n_bets, n_races)
        rows.append(result)

        verdict = "⚠️  CI 跨過 0，未夠信" if result["crosses_zero"] else "✅ CI 唔跨 0"
        ci_str = f"[{result['ci_low_95']:+.1%}, {result['ci_high_95']:+.1%}]"
        print(
            f"{thr:<10}{result['n_bets']:<8}{result['point_roi']:<+12.2%}"
            f"{ci_str:<24}{result['p_roi_gt_0']:<10.1%}{verdict}"
        )

    if not rows:
        print("\n⚠️  冇 threshold 有夠樣本做 bootstrap")
        return

    out_df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ 已儲存至: {OUTPUT_PATH}")

    n_pass = (~out_df["crosses_zero"]).sum()
    print(
        f"\n📊 總結: {len(out_df)} 個 threshold 入面，得 {n_pass} 個 95% CI 完全企喺 0 之上。"
    )
    if n_pass == 0:
        print(
            "   即係話：依家冇一個 threshold 嘅正 ROI 係樣本夠大到可以講「唔係撞彩」。"
            "呢個係老實嘅結論，唔係做錯咗 -- 對應返你個 repo 一路嘅發現。"
        )


if __name__ == "__main__":
    main()