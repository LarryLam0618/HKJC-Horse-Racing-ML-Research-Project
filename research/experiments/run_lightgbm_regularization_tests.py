"""Run categorical regularization tests for the F Robust feature set.

This script:
  - Uses the F Robust feature set (drops race_name, jockey_name, trainer_name)
  - Trains three native-categorical LightGBM variants:
      F_base:  current default hyperparameters
      F_reg_1: moderate categorical regularization
      F_reg_2: stronger categorical regularization
  - Uses the same walk-forward folds and OOF Platt calibration split for all.
  - Outputs fold metrics, predictions, and a summary table.

Run:
  python3 features/run_lightgbm_regularization_tests.py
"""

from pathlib import Path
import importlib.util
import pandas as pd
import numpy as np

BASE_SCRIPT = Path("features/train_lightgbm_native.py")
OUTPUT_DIR = Path("output/regularization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# F Robust feature set
DROP_COLUMNS = ["race_name", "jockey_name", "trainer_name"]

EXPERIMENTS = {
    "F_base": dict(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        min_data_per_group=100,
        cat_smooth=10.0,
        max_cat_threshold=32,
        objective="binary",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    ),
    "F_reg_1": dict(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=-1,
        min_child_samples=50,
        min_split_gain=0.01,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.0,
        min_data_per_group=50,
        cat_smooth=20.0,
        max_cat_threshold=32,
        objective="binary",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    ),
    "F_reg_2": dict(
        n_estimators=500,
        learning_rate=0.025,
        num_leaves=12,
        max_depth=-1,
        min_child_samples=75,
        min_split_gain=0.01,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.3,
        reg_lambda=1.5,
        min_data_per_group=100,
        cat_smooth=40.0,
        max_cat_threshold=16,
        objective="binary",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    ),
}


def load_base_module():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(
            f"搵唔到 {BASE_SCRIPT}。請先將 native LightGBM script 存成此檔案。"
        )

    spec = importlib.util.spec_from_file_location("native_lgbm", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_feature_columns(df, drop_columns, base):
    """Return feature columns and split into numeric/categorical for F Robust set."""
    all_features, numeric_cols, categorical_cols = base.select_feature_columns(df)
    missing = [c for c in drop_columns if c not in all_features]
    if missing:
        print(f"⚠️ Dataset 無以下欄位，略過：{missing}")

    feature_cols = [c for c in all_features if c not in drop_columns]
    numeric_cols = [c for c in numeric_cols if c in feature_cols]
    categorical_cols = [c for c in categorical_cols if c in feature_cols]

    return feature_cols, numeric_cols, categorical_cols


def run_experiment(df, experiment_name, lgbm_params, base):
    feature_cols, numeric_cols, categorical_cols = prepare_feature_columns(
        df, DROP_COLUMNS, base
    )
    windows = base.monthly_windows(df)

    if not windows:
        raise ValueError("沒有可用的 walk-forward windows；請檢查 MIN_TRAIN_MONTHS 和日期資料。")

    print("\n" + "=" * 80)
    print(f"🧪 {experiment_name}")
    print(f"移除欄位: {DROP_COLUMNS}")
    print(f"保留特徵: {len(feature_cols)} (numeric={len(numeric_cols)}, categorical={len(categorical_cols)})")
    print(
        f"LightGBM params: num_leaves={lgbm_params['num_leaves']}, "
        f"min_child_samples={lgbm_params['min_child_samples']}, "
        f"min_data_per_group={lgbm_params['min_data_per_group']}, "
        f"cat_smooth={lgbm_params['cat_smooth']}"
    )
    print("=" * 80)

    fold_rows = []
    prediction_rows = []
    last_model = None

    for fold, (train_months, test_months, train_mask, test_mask) in enumerate(
        windows, start=1
    ):
        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()

        X_train, X_test = base.prepare_lightgbm_inputs(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )

        y_train = train_df[base.TARGET_COL].astype(int).to_numpy()
        y_test = test_df[base.TARGET_COL].astype(int).to_numpy()

        model = base.LGBMClassifier(**lgbm_params)
        model.fit(
            X_train,
            y_train,
            categorical_feature=categorical_cols,
        )
        raw_prob = model.predict_proba(X_test)[:, 1]
        last_model = model

        pred_cols = base.GROUP_COLS + ["horse_id", base.TARGET_COL]
        pred_df = test_df[pred_cols].copy()
        if "horse_name" in test_df.columns:
            pred_df["horse_name"] = test_df["horse_name"].values
        pred_df["experiment"] = experiment_name
        pred_df["fold"] = fold
        pred_df["pred_win_prob_raw"] = raw_prob
        prediction_rows.append(pred_df)

        metrics = base.evaluate(y_test, raw_prob, test_df[base.GROUP_COLS], "raw")
        metrics.update(
            {
                "experiment": experiment_name,
                "fold": fold,
                "train_start": str(train_months.iloc[0]),
                "train_end": str(train_months.iloc[-1]),
                "test_start": str(test_months.iloc[0]),
                "test_end": str(test_months.iloc[-1]),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "test_races": test_df.groupby(base.GROUP_COLS).ngroups,
                "n_features": len(feature_cols),
            }
        )
        fold_rows.append(metrics)

        print(
            f"{experiment_name} | fold {fold:02d} | test={metrics['test_start']} | "
            f"logloss={metrics['log_loss']:.4f} | brier={metrics['brier']:.4f} | "
            f"auc={metrics['auc']:.4f} | spearman={metrics['race_spearman']:.4f}"
        )

    fold_metrics = pd.DataFrame(fold_rows)
    all_predictions = pd.concat(prediction_rows, ignore_index=True)

    # Same OOF evaluation split for every experiment: folds 1-8 calibrate; 9-16 evaluate.
    split_fold = len(windows) // 2
    calibration_folds = list(range(1, split_fold + 1))
    evaluation_folds = list(range(split_fold + 1, len(windows) + 1))

    calibrate_df = all_predictions[all_predictions["fold"].isin(calibration_folds)].copy()
    eval_df = all_predictions[all_predictions["fold"].isin(evaluation_folds)].copy()

    if eval_df.empty:
        raise ValueError(f"{experiment_name}: 沒有 evaluation folds。")

    platt = base.fit_platt(
        train_y=calibrate_df[base.TARGET_COL].astype(int).to_numpy(),
        train_prob=calibrate_df["pred_win_prob_raw"]
        .astype(float)
        .clip(1e-6, 1 - 1e-6)
        .to_numpy(),
        test_prob=eval_df["pred_win_prob_raw"]
        .astype(float)
        .clip(1e-6, 1 - 1e-6)
        .to_numpy(),
    )
    eval_df["pred_win_prob_platt_oof"] = platt

    raw_metrics = base.evaluate(
        eval_df[base.TARGET_COL].astype(int).to_numpy(),
        eval_df["pred_win_prob_raw"].astype(float).to_numpy(),
        eval_df[base.GROUP_COLS],
        "raw_oof_eval",
    )
    platt_metrics = base.evaluate(
        eval_df[base.TARGET_COL].astype(int).to_numpy(),
        eval_df["pred_win_prob_platt_oof"].astype(float).to_numpy(),
        eval_df[base.GROUP_COLS],
        "platt_oof_eval",
    )

    summary = pd.DataFrame([raw_metrics, platt_metrics])
    summary.insert(0, "experiment", experiment_name)
    summary["dropped_features"] = ",".join(DROP_COLUMNS)
    summary["n_features"] = len(feature_cols)
    summary["calibration_folds"] = ",".join(map(str, calibration_folds))
    summary["evaluation_folds"] = ",".join(map(str, evaluation_folds))
    summary["n_eval_samples"] = len(eval_df)

    importance = base.extract_feature_importance(last_model, feature_cols)
    importance.insert(0, "experiment", experiment_name)

    prefix = OUTPUT_DIR / experiment_name
    fold_metrics.to_csv(
        prefix.with_name(f"{experiment_name}_fold_metrics.csv"), index=False
    )
    all_predictions.to_csv(
        prefix.with_name(f"{experiment_name}_predictions_all_folds.csv"), index=False
    )
    eval_df.to_csv(
        prefix.with_name(f"{experiment_name}_predictions_oof_eval.csv"), index=False
    )
    importance.to_csv(
        prefix.with_name(f"{experiment_name}_feature_importance.csv"), index=False
    )

    return summary, fold_metrics, importance


def main():
    base = load_base_module()
    data_path = base.resolve_data_path(base.DATA_CANDIDATES)
    df = base.load_data(data_path)

    all_summaries = []
    all_fold_metrics = []
    all_importance = []

    for experiment_name, lgbm_params in EXPERIMENTS.items():
        summary, fold_metrics, importance = run_experiment(
            df=df,
            experiment_name=experiment_name,
            lgbm_params=lgbm_params,
            base=base,
        )
        all_summaries.append(summary)
        all_fold_metrics.append(fold_metrics)
        all_importance.append(importance)

    summary_df = pd.concat(all_summaries, ignore_index=True)
    fold_df = pd.concat(all_fold_metrics, ignore_index=True)
    importance_df = pd.concat(all_importance, ignore_index=True)

    summary_df.to_csv(
        OUTPUT_DIR / "regularization_summary.csv", index=False
    )
    fold_df.to_csv(
        OUTPUT_DIR / "regularization_all_fold_metrics.csv", index=False
    )
    importance_df.to_csv(
        OUTPUT_DIR / "regularization_all_feature_importance.csv", index=False
    )

    print("\n" + "=" * 80)
    print("🏁 REGULARIZATION SUMMARY — 比較 platt_oof_eval")
    print("=" * 80)
    comparison = (
        summary_df[summary_df["method"] == "platt_oof_eval"]
        .sort_values(["brier", "log_loss"], ascending=True)[
            [
                "experiment",
                "n_features",
                "log_loss",
                "brier",
                "auc",
                "race_spearman",
                "ece",
                "mce",
                "n_eval_samples",
            ]
        ]
    )
    print(comparison.to_string(index=False))

    print("\n✅ 已輸出：")
    print(" - output/regularization/regularization_summary.csv")
    print(" - output/regularization/regularization_all_fold_metrics.csv")
    print(" - output/regularization/regularization_all_feature_importance.csv")
    print(" - output/regularization/<experiment>_*.csv")


if __name__ == "__main__":
    main()
