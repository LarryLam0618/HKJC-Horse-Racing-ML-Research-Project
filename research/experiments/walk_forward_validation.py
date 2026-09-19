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

EXCLUDE_COLS = {
    TARGET_COL,
    "top3_label",
    DATE_COL,
    "date",
    "season"
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


def select_feature_columns(df: pd.DataFrame):
    exclude = set(EXCLUDE_COLS)
    exclude.update([c for c in ID_COLS if c in df.columns])

    feature_cols = [c for c in df.columns if c not in exclude]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    return feature_cols, numeric_cols, categorical_cols


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


def monthly_windows(df: pd.DataFrame, min_train_months=6, test_months=1, step_months=1):
    months = pd.Series(df[DATE_COL].dt.to_period("M").sort_values().unique())
    windows = []

    start = min_train_months
    while start + test_months <= len(months):
        train_months = months.iloc[:start]
        test_block = months.iloc[start:start + test_months]

        train_mask = df[DATE_COL].dt.to_period("M").isin(train_months)
        test_mask = df[DATE_COL].dt.to_period("M").isin(test_block)

        if train_mask.sum() > 0 and test_mask.sum() > 0:
            windows.append((train_months, test_block, train_mask, test_mask))

        start += step_months

    return windows


def race_level_spearman(y_true, y_prob, race_keys_df: pd.DataFrame):
    tmp = race_keys_df.copy()
    tmp["y_true"] = y_true
    tmp["y_prob"] = y_prob

    race_corrs = []
    for _, g in tmp.groupby(GROUP_COLS):
        if g["y_true"].nunique() < 2:
            continue
        corr, _ = spearmanr(g["y_true"], g["y_prob"])
        if not np.isnan(corr):
            race_corrs.append(corr)

    return float(np.mean(race_corrs)) if race_corrs else np.nan


def evaluate_fold(y_true, y_prob, race_keys_df):
    metrics = {}
    metrics["log_loss"] = log_loss(y_true, y_prob, labels=[0, 1])
    metrics["brier"] = brier_score_loss(y_true, y_prob)

    try:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["auc"] = np.nan

    metrics["race_spearman"] = race_level_spearman(y_true, y_prob, race_keys_df)
    return metrics


def run_walk_forward(df: pd.DataFrame):
    feature_cols, numeric_cols, categorical_cols = select_feature_columns(df)
    windows = monthly_windows(df, MIN_TRAIN_MONTHS, TEST_MONTHS, STEP_MONTHS)

    print(f"✅ 特徵數量: {len(feature_cols)}")
    print(f"  - 數值欄位: {len(numeric_cols)}")
    print(f"  - 類別欄位: {len(categorical_cols)}")
    print(f"✅ walk-forward folds: {len(windows)}")

    all_fold_rows = []
    all_pred_rows = []

    for i, (train_months, test_months, train_mask, test_mask) in enumerate(windows, start=1):
        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()

        X_train = train_df[feature_cols]
        y_train = train_df[TARGET_COL]
        X_test = test_df[feature_cols]
        y_test = test_df[TARGET_COL]

        pipe = build_pipeline(numeric_cols, categorical_cols)
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        fold_metrics = evaluate_fold(y_test, y_prob, test_df[GROUP_COLS])
        fold_metrics["fold"] = i
        fold_metrics["train_start"] = str(train_months.iloc[0])
        fold_metrics["train_end"] = str(train_months.iloc[-1])
        fold_metrics["test_start"] = str(test_months.iloc[0])
        fold_metrics["test_end"] = str(test_months.iloc[-1])
        fold_metrics["train_rows"] = len(train_df)
        fold_metrics["test_rows"] = len(test_df)
        fold_metrics["test_races"] = test_df.groupby(GROUP_COLS).ngroups
        all_fold_rows.append(fold_metrics)

        pred_df = test_df[GROUP_COLS + ["horse_id", TARGET_COL]].copy()
        pred_df["pred_win_prob"] = y_prob
        pred_df["fold"] = i
        all_pred_rows.append(pred_df)

        auc_str = f"{fold_metrics['auc']:.4f}" if pd.notna(fold_metrics["auc"]) else "nan"
        print(
            f"Fold {i}: train {fold_metrics['train_start']}~{fold_metrics['train_end']} | "
            f"test {fold_metrics['test_start']} | rows={fold_metrics['test_rows']} | "
            f"logloss={fold_metrics['log_loss']:.4f} | "
            f"brier={fold_metrics['brier']:.4f} | "
            f"auc={auc_str}"
        )

    folds_df = pd.DataFrame(all_fold_rows)
    preds_df = pd.concat(all_pred_rows, ignore_index=True)

    overall = evaluate_fold(preds_df[TARGET_COL], preds_df["pred_win_prob"], preds_df[GROUP_COLS])
    overall_df = pd.DataFrame([overall])
    overall_df["n_predictions"] = len(preds_df)
    overall_df["n_races"] = preds_df.groupby(GROUP_COLS).ngroups
    overall_df["n_folds"] = len(folds_df)

    folds_df.to_csv(OUTPUT_DIR / "walk_forward_fold_metrics.csv", index=False)
    preds_df.to_csv(OUTPUT_DIR / "walk_forward_predictions.csv", index=False)
    overall_df.to_csv(OUTPUT_DIR / "walk_forward_overall_metrics.csv", index=False)

    print("\n" + "=" * 60)
    print("📊 Overall Walk-Forward Metrics")
    print("=" * 60)
    print(overall_df.to_string(index=False))

    return folds_df, preds_df, overall_df


if __name__ == "__main__":
    data_path = resolve_data_path(DATA_CANDIDATES)
    df = load_data(data_path)
    run_walk_forward(df)