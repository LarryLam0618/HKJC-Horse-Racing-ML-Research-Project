import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

try:
    from lightgbm import LGBMClassifier
except ImportError:
    raise ImportError("請先安裝 lightgbm：pip install lightgbm")

DATA_CANDIDATES = [
    Path("training_set_safe.csv"),
    Path("data/training_set_safe.csv"),
    Path("data/processed/training_set_safe.csv"),
]

OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

TARGET_COL = "win_label"
DATE_COL = "race_date"
GROUP_COLS = ["race_date", "venue", "race_no"]
ID_COLS = ["horse_id", "horse_name"]
EXCLUDE_COLS = {TARGET_COL, "top3_label", DATE_COL, "date", "season"}

MIN_TRAIN_MONTHS = 6
TEST_MONTHS = 1
STEP_MONTHS = 1
N_BINS = 10

LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=0.0,
    objective="binary",
    random_state=42,
    n_jobs=-1
)


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


def prepare_lightgbm_inputs(train_df, test_df, feature_cols, numeric_cols, categorical_cols):
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    if numeric_cols:
        imputer = SimpleImputer(strategy="median")
        X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])

    for col in categorical_cols:
        train_cat = X_train[col].astype("category")
        categories = train_cat.cat.categories
        X_train[col] = train_cat
        X_test[col] = pd.Categorical(X_test[col], categories=categories)

    return X_train, X_test


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


def compute_ece_mce(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    total = len(y_true)
    ece = 0.0
    mce = 0.0
    rows = []

    for i in range(n_bins):
        mask = bin_ids == i
        count = int(mask.sum())
        if count == 0:
            continue

        pred_mean = float(np.mean(y_prob[mask]))
        obs_rate = float(np.mean(y_true[mask]))
        gap = abs(obs_rate - pred_mean)
        ece += (count / total) * gap
        mce = max(mce, gap)

        rows.append({
            "bin": i + 1,
            "count": count,
            "mean_predicted_prob": pred_mean,
            "observed_win_rate": obs_rate,
            "abs_gap": gap
        })

    return ece, mce, pd.DataFrame(rows)


def evaluate(y_true, y_prob, race_keys_df, label):
    ece, mce, _ = compute_ece_mce(y_true, y_prob, n_bins=N_BINS)

    return {
        "method": label,
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "brier": brier_score_loss(y_true, y_prob),
        "auc": roc_auc_score(y_true, y_prob),
        "race_spearman": race_level_spearman(y_true, y_prob, race_keys_df),
        "ece": ece,
        "mce": mce
    }


def fit_platt(train_y, train_prob, test_prob):
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(train_prob.reshape(-1, 1), train_y)
    return lr.predict_proba(test_prob.reshape(-1, 1))[:, 1]


def extract_feature_importance(model, feature_cols):
    return pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)


def run_walk_forward_lightgbm_native(df: pd.DataFrame):
    feature_cols, numeric_cols, categorical_cols = select_feature_columns(df)
    windows = monthly_windows(df)

    print(f"✅ 特徵數量: {len(feature_cols)}")
    print(f"  - 數值欄位: {len(numeric_cols)}")
    print(f"  - 類別欄位: {len(categorical_cols)}")
    print(f"✅ walk-forward folds: {len(windows)}")

    fold_rows = []
    pred_rows = []
    last_model = None

    for i, (train_months, test_months, train_mask, test_mask) in enumerate(windows, start=1):
        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()

        X_train, X_test = prepare_lightgbm_inputs(
            train_df, test_df, feature_cols, numeric_cols, categorical_cols
        )
        y_train = train_df[TARGET_COL].astype(int).to_numpy()
        y_test = test_df[TARGET_COL].astype(int).to_numpy()

        model = LGBMClassifier(**LGBM_PARAMS)
        model.fit(X_train, y_train, categorical_feature=categorical_cols)
        y_prob_raw = model.predict_proba(X_test)[:, 1]
        last_model = model

        pred_df = test_df[GROUP_COLS + ["horse_id", TARGET_COL]].copy()
        if "horse_name" in test_df.columns:
            pred_df["horse_name"] = test_df["horse_name"].values
        pred_df["fold"] = i
        pred_df["pred_win_prob_raw"] = y_prob_raw
        pred_rows.append(pred_df)

        fold_metrics = evaluate(y_test, y_prob_raw, test_df[GROUP_COLS], "raw")
        fold_metrics.update({
            "fold": i,
            "train_start": str(train_months.iloc[0]),
            "train_end": str(train_months.iloc[-1]),
            "test_start": str(test_months.iloc[0]),
            "test_end": str(test_months.iloc[-1]),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "test_races": test_df.groupby(GROUP_COLS).ngroups
        })
        fold_rows.append(fold_metrics)

        print(
            f"Fold {i}: train {fold_metrics['train_start']}~{fold_metrics['train_end']} | "
            f"test {fold_metrics['test_start']} | rows={fold_metrics['test_rows']} | "
            f"logloss={fold_metrics['log_loss']:.4f} | "
            f"brier={fold_metrics['brier']:.4f} | "
            f"auc={fold_metrics['auc']:.4f}"
        )

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.concat(pred_rows, ignore_index=True)

    split_idx = max(1, len(windows) // 2)
    calib_folds = list(range(1, split_idx + 1))
    eval_folds = list(range(split_idx + 1, len(windows) + 1))

    calib_df = pred_df[pred_df["fold"].isin(calib_folds)].copy()
    eval_df = pred_df[pred_df["fold"].isin(eval_folds)].copy()

    if len(eval_df) == 0:
        raise ValueError("fold 數量不足，無法做 OOF Platt calibration")

    train_y = calib_df[TARGET_COL].astype(int).to_numpy()
    train_prob = calib_df["pred_win_prob_raw"].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()
    test_prob = eval_df["pred_win_prob_raw"].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()

    eval_df["pred_win_prob_platt_oof"] = fit_platt(train_y, train_prob, test_prob)

    raw_overall = evaluate(
        eval_df[TARGET_COL].astype(int).to_numpy(),
        eval_df["pred_win_prob_raw"].astype(float).to_numpy(),
        eval_df[GROUP_COLS],
        "lightgbm_native_raw_oof_eval"
    )

    platt_overall = evaluate(
        eval_df[TARGET_COL].astype(int).to_numpy(),
        eval_df["pred_win_prob_platt_oof"].astype(float).to_numpy(),
        eval_df[GROUP_COLS],
        "lightgbm_native_platt_oof_eval"
    )

    metrics_df = pd.DataFrame([raw_overall, platt_overall])
    metrics_df["calibration_folds"] = ",".join(map(str, calib_folds))
    metrics_df["evaluation_folds"] = ",".join(map(str, eval_folds))
    metrics_df["n_eval_samples"] = len(eval_df)
    metrics_df["n_features"] = len(feature_cols)

    calibration_rows = []
    for method, col in [("raw", "pred_win_prob_raw"), ("platt_oof", "pred_win_prob_platt_oof")]:
        y_true = eval_df[TARGET_COL].astype(int).to_numpy()
        y_prob = eval_df[col].astype(float).to_numpy()
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=N_BINS, strategy="quantile")

        calibration_rows.append(pd.DataFrame({
            "method": method,
            "mean_predicted_prob": mean_pred,
            "observed_win_rate": frac_pos
        }))

    calibration_df = pd.concat(calibration_rows, ignore_index=True)
    importance_df = extract_feature_importance(last_model, feature_cols) if last_model is not None else pd.DataFrame()

    return fold_df, pred_df, eval_df, metrics_df, calibration_df, importance_df


if __name__ == "__main__":
    data_path = resolve_data_path(DATA_CANDIDATES)
    df = load_data(data_path)

    fold_df, pred_df, eval_df, metrics_df, calibration_df, importance_df = run_walk_forward_lightgbm_native(df)

    fold_df.to_csv(OUT_DIR / "lightgbm_native_fold_metrics.csv", index=False)
    pred_df.to_csv(OUT_DIR / "lightgbm_native_predictions_all_folds.csv", index=False)
    eval_df.to_csv(OUT_DIR / "lightgbm_native_predictions_oof_eval.csv", index=False)
    metrics_df.to_csv(OUT_DIR / "lightgbm_native_oof_calibration_metrics.csv", index=False)
    calibration_df.to_csv(OUT_DIR / "lightgbm_native_oof_calibration_curve.csv", index=False)
    importance_df.to_csv(OUT_DIR / "lightgbm_native_feature_importance.csv", index=False)

    print("\n" + "=" * 60)
    print("📊 LightGBM Native OOF Calibration Metrics")
    print("=" * 60)
    print(metrics_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("📈 Top 20 Native LightGBM Feature Importance")
    print("=" * 60)
    if not importance_df.empty:
        print(importance_df.head(20).to_string(index=False))
    else:
        print("No feature importance available.")

    print("\n✅ 已輸出:")
    print(" - output/lightgbm_native_fold_metrics.csv")
    print(" - output/lightgbm_native_predictions_all_folds.csv")
    print(" - output/lightgbm_native_predictions_oof_eval.csv")
    print(" - output/lightgbm_native_oof_calibration_metrics.csv")
    print(" - output/lightgbm_native_oof_calibration_curve.csv")
    print(" - output/lightgbm_native_feature_importance.csv")