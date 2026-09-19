import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from scipy.stats import spearmanr

DATA_CANDIDATES = [
    Path("training_set_safe.csv"),
    Path("data/training_set_safe.csv"),
    Path("data/processed/training_set_safe.csv"),
]

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_COL = "win_label"
DATE_COL = "race_date"
GROUP_COLS = ["race_date", "venue", "race_no"]
ID_COLS = ["horse_id", "horse_name"]
EXCLUDE_COLS = {TARGET_COL, "top3_label", DATE_COL, "date", "season"}

MIN_TRAIN_MONTHS = 6
TEST_MONTHS = 1
STEP_MONTHS = 1


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


def monthly_windows(df: pd.DataFrame):
    months = pd.Series(df[DATE_COL].dt.to_period("M").sort_values().unique())
    windows = []

    start = MIN_TRAIN_MONTHS
    while start + TEST_MONTHS <= len(months):
        train_months = months.iloc[:start]
        test_months = months.iloc[start:start + TEST_MONTHS]

        train_mask = df[DATE_COL].dt.to_period("M").isin(train_months)
        test_mask = df[DATE_COL].dt.to_period("M").isin(test_months)

        if train_mask.sum() > 0 and test_mask.sum() > 0:
            windows.append((train_months, test_months, train_mask, test_mask))

        start += STEP_MONTHS

    return windows


def race_level_spearman(y_true, y_prob, race_keys_df: pd.DataFrame):
    tmp = race_keys_df.copy()
    tmp["y_true"] = y_true
    tmp["y_prob"] = y_prob

    vals = []
    for _, g in tmp.groupby(GROUP_COLS):
        if g["y_true"].nunique() < 2 or g["y_prob"].nunique() < 2:
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


def get_feature_names(preprocessor, numeric_cols, categorical_cols):
    feature_names = []

    if numeric_cols:
        feature_names.extend(numeric_cols)

    if categorical_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(categorical_cols).tolist()
        feature_names.extend(cat_names)

    return feature_names


def extract_feature_importance(fitted_pipe, numeric_cols, categorical_cols):
    preprocessor = fitted_pipe.named_steps["preprocessor"]
    model = fitted_pipe.named_steps["model"]

    feature_names = get_feature_names(preprocessor, numeric_cols, categorical_cols)
    coefs = model.coef_[0]

    imp = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs)
    }).sort_values("abs_coefficient", ascending=False)

    return imp


def build_calibration_table(y_true, y_prob, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({
        "mean_predicted_prob": mean_pred,
        "observed_win_rate": frac_pos
    })


def run_walk_forward_baseline(df: pd.DataFrame):
    feature_cols, numeric_cols, categorical_cols = select_feature_columns(df)
    windows = monthly_windows(df)

    print(f"✅ 特徵數量: {len(feature_cols)}")
    print(f"  - 數值欄位: {len(numeric_cols)}")
    print(f"  - 類別欄位: {len(categorical_cols)}")
    print(f"✅ walk-forward folds: {len(windows)}")

    all_preds = []
    fold_rows = []
    last_model = None

    for i, (train_months, test_months, train_mask, test_mask) in enumerate(windows, start=1):
        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()

        pipe = build_pipeline(numeric_cols, categorical_cols)
        pipe.fit(train_df[feature_cols], train_df[TARGET_COL])
        y_prob = pipe.predict_proba(test_df[feature_cols])[:, 1]
        last_model = pipe

        metrics = evaluate(test_df[TARGET_COL], y_prob, test_df[GROUP_COLS])
        metrics.update({
            "fold": i,
            "train_start": str(train_months.iloc[0]),
            "train_end": str(train_months.iloc[-1]),
            "test_start": str(test_months.iloc[0]),
            "test_end": str(test_months.iloc[-1]),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "test_races": test_df.groupby(GROUP_COLS).ngroups
        })
        fold_rows.append(metrics)

        pred_df = test_df[GROUP_COLS + ["horse_id", TARGET_COL]].copy()
        if "horse_name" in test_df.columns:
            pred_df["horse_name"] = test_df["horse_name"].values
        pred_df["pred_win_prob"] = y_prob
        pred_df["fold"] = i
        all_preds.append(pred_df)

        auc_str = f"{metrics['auc']:.4f}" if pd.notna(metrics["auc"]) else "nan"
        print(
            f"Fold {i}: train {metrics['train_start']}~{metrics['train_end']} | "
            f"test {metrics['test_start']} | rows={metrics['test_rows']} | "
            f"logloss={metrics['log_loss']:.4f} | "
            f"brier={metrics['brier']:.4f} | "
            f"auc={auc_str}"
        )

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.concat(all_preds, ignore_index=True)

    overall = evaluate(pred_df[TARGET_COL], pred_df["pred_win_prob"], pred_df[GROUP_COLS])
    overall_df = pd.DataFrame([overall])
    overall_df["n_predictions"] = len(pred_df)
    overall_df["n_races"] = pred_df.groupby(GROUP_COLS).ngroups
    overall_df["n_folds"] = len(fold_df)
    overall_df["n_features"] = len(feature_cols)

    calibration_df = build_calibration_table(pred_df[TARGET_COL], pred_df["pred_win_prob"], n_bins=10)
    importance_df = extract_feature_importance(last_model, numeric_cols, categorical_cols) if last_model is not None else pd.DataFrame()

    return fold_df, pred_df, overall_df, calibration_df, importance_df


if __name__ == "__main__":
    data_path = resolve_data_path(DATA_CANDIDATES)
    df = load_data(data_path)

    fold_df, pred_df, overall_df, calibration_df, importance_df = run_walk_forward_baseline(df)

    fold_df.to_csv(OUTPUT_DIR / "baseline_fold_metrics.csv", index=False)
    pred_df.to_csv(OUTPUT_DIR / "baseline_predictions.csv", index=False)
    overall_df.to_csv(OUTPUT_DIR / "baseline_overall_metrics.csv", index=False)
    calibration_df.to_csv(OUTPUT_DIR / "baseline_calibration.csv", index=False)
    importance_df.to_csv(OUTPUT_DIR / "baseline_feature_importance.csv", index=False)

    print("\n" + "=" * 60)
    print("📊 Overall Baseline Metrics")
    print("=" * 60)
    print(overall_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("📈 Top 20 Feature Importance (by |coefficient|)")
    print("=" * 60)
    if not importance_df.empty:
        print(importance_df.head(20).to_string(index=False))
    else:
        print("No feature importance available.")

    print("\n✅ 已輸出:")
    print(" - output/baseline_fold_metrics.csv")
    print(" - output/baseline_predictions.csv")
    print(" - output/baseline_overall_metrics.csv")
    print(" - output/baseline_calibration.csv")
    print(" - output/baseline_feature_importance.csv")