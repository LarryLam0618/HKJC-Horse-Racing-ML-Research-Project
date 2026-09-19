import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr

DATA_CANDIDATES = [
    Path("training_set_safe.csv"),
    Path("data/training_set_safe.csv"),
    Path("data/processed/training_set_safe.csv"),
]

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

MIN_TRAIN_MONTHS = 6
TEST_MONTHS = 1
STEP_MONTHS = 1
TARGET_COL = "win_label"
DATE_COL = "race_date"
GROUP_COLS = ["race_date", "venue", "race_no"]
ID_COLS = ["horse_id", "horse_name"]
BASE_EXCLUDE_COLS = {TARGET_COL, "top3_label", DATE_COL, "date", "season"}

FEATURE_SETS = {
    "baseline_all_current": {
        "include_contains": [],
        "exclude_contains": []
    },
    "no_sectionals": {
        "include_contains": [],
        "exclude_contains": ["_lag1", "sectional_history"]
    },
    "sectionals_only_extra": {
        "include_contains": ["_lag1", "sectional_history"],
        "exclude_contains": []
    },
    "without_odds": {
        "include_contains": [],
        "exclude_contains": ["win_odds"]
    }
}


def resolve_data_path(candidates):
    for p in candidates:
        if p.exists():
            print(f"✅ 使用資料檔案: {p}")
            return p
    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"❌ 搵唔到 training_set_safe.csv，已檢查:\n{checked}")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, TARGET_COL]).copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    df = df.sort_values(GROUP_COLS + ["horse_id"]).reset_index(drop=True)
    return df


def feature_columns_from_spec(df: pd.DataFrame, spec: dict):
    exclude = set(BASE_EXCLUDE_COLS)
    exclude.update([c for c in ID_COLS if c in df.columns])
    base_cols = [c for c in df.columns if c not in exclude]

    include_contains = spec.get("include_contains", [])
    exclude_contains = spec.get("exclude_contains", [])

    if include_contains:
        cols = [c for c in base_cols if any(tok in c for tok in include_contains)]
        if "sectional_history" in include_contains:
            cols += [c for c in base_cols if c in ["sectional_history_count", "has_sectional_history"]]
        cols = list(dict.fromkeys(cols))
    else:
        cols = base_cols.copy()

    if exclude_contains:
        cols = [c for c in cols if not any(tok in c for tok in exclude_contains)]

    numeric_cols = df[cols].select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in cols if c not in numeric_cols]
    return cols, numeric_cols, categorical_cols


def build_pipeline(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols)
    ])

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs"
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def monthly_windows(df: pd.DataFrame):
    months = pd.Series(df[DATE_COL].dt.to_period("M").sort_values().unique())
    windows = []
    start = MIN_TRAIN_MONTHS

    while start + TEST_MONTHS <= len(months):
        train_months = months.iloc[:start]
        test_block = months.iloc[start:start + TEST_MONTHS]

        train_mask = df[DATE_COL].dt.to_period("M").isin(train_months)
        test_mask = df[DATE_COL].dt.to_period("M").isin(test_block)

        if train_mask.sum() > 0 and test_mask.sum() > 0:
            windows.append((train_months, test_block, train_mask, test_mask))

        start += STEP_MONTHS

    return windows


def race_level_spearman(y_true, y_prob, race_keys_df: pd.DataFrame):
    tmp = race_keys_df.copy()
    tmp["y_true"] = y_true
    tmp["y_prob"] = y_prob

    vals = []
    for _, g in tmp.groupby(GROUP_COLS):
        if g["y_true"].nunique() < 2:
            continue
        corr, _ = spearmanr(g["y_true"], g["y_prob"])
        if not np.isnan(corr):
            vals.append(corr)

    return float(np.mean(vals)) if vals else np.nan


def evaluate(y_true, y_prob, race_keys_df):
    out = {
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "brier": brier_score_loss(y_true, y_prob),
        "race_spearman": race_level_spearman(y_true, y_prob, race_keys_df)
    }
    try:
        out["auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["auc"] = np.nan
    return out


def run_one_feature_set(df: pd.DataFrame, set_name: str, spec: dict):
    feature_cols, numeric_cols, categorical_cols = feature_columns_from_spec(df, spec)
    windows = monthly_windows(df)

    all_preds = []
    fold_rows = []

    for i, (train_months, test_months, train_mask, test_mask) in enumerate(windows, start=1):
        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()

        pipe = build_pipeline(numeric_cols, categorical_cols)
        pipe.fit(train_df[feature_cols], train_df[TARGET_COL])
        y_prob = pipe.predict_proba(test_df[feature_cols])[:, 1]

        metrics = evaluate(test_df[TARGET_COL], y_prob, test_df[GROUP_COLS])
        metrics.update({
            "feature_set": set_name,
            "fold": i,
            "train_start": str(train_months.iloc[0]),
            "train_end": str(train_months.iloc[-1]),
            "test_start": str(test_months.iloc[0]),
            "test_end": str(test_months.iloc[-1]),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "n_features": len(feature_cols)
        })
        fold_rows.append(metrics)

        pred_df = test_df[GROUP_COLS + ["horse_id", TARGET_COL]].copy()
        pred_df["pred_win_prob"] = y_prob
        pred_df["fold"] = i
        pred_df["feature_set"] = set_name
        all_preds.append(pred_df)

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.concat(all_preds, ignore_index=True)

    overall = evaluate(pred_df[TARGET_COL], pred_df["pred_win_prob"], pred_df[GROUP_COLS])
    overall.update({
        "feature_set": set_name,
        "n_predictions": len(pred_df),
        "n_races": pred_df.groupby(GROUP_COLS).ngroups,
        "n_folds": len(fold_df),
        "n_features": len(feature_cols)
    })

    return fold_df, pred_df, overall


def main():
    data_path = resolve_data_path(DATA_CANDIDATES)
    df = load_data(data_path)

    all_fold_dfs = []
    all_pred_dfs = []
    overall_rows = []

    print(f"✅ rows={len(df)}, races={df.groupby(GROUP_COLS).ngroups}, horses={df['horse_id'].nunique()}")

    for set_name, spec in FEATURE_SETS.items():
        print("\n" + "=" * 70)
        print(f"🚀 Running feature set: {set_name}")
        print("=" * 70)

        fold_df, pred_df, overall = run_one_feature_set(df, set_name, spec)
        print(pd.DataFrame([overall]).to_string(index=False))

        all_fold_dfs.append(fold_df)
        all_pred_dfs.append(pred_df)
        overall_rows.append(overall)

    folds_all = pd.concat(all_fold_dfs, ignore_index=True)
    preds_all = pd.concat(all_pred_dfs, ignore_index=True)
    overall_df = pd.DataFrame(overall_rows).sort_values(
        ["log_loss", "brier", "auc"],
        ascending=[True, True, False]
    )

    folds_all.to_csv(OUTPUT_DIR / "compare_feature_sets_fold_metrics.csv", index=False)
    preds_all.to_csv(OUTPUT_DIR / "compare_feature_sets_predictions.csv", index=False)
    overall_df.to_csv(OUTPUT_DIR / "compare_feature_sets_overall_metrics.csv", index=False)

    print("\n" + "=" * 70)
    print("📊 Feature Set Comparison")
    print("=" * 70)
    print(overall_df.to_string(index=False))


if __name__ == "__main__":
    main()