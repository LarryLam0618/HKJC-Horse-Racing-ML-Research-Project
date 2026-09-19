"""Run feature ablations for native-categorical LightGBM.

Prerequisite:
  - Save the current native LightGBM script as features/train_lightgbm_native.py
  - It must expose these module-level names/functions:
      DATA_CANDIDATES, OUT_DIR, TARGET_COL, GROUP_COLS,
      resolve_data_path, load_data, select_feature_columns,
      monthly_windows, prepare_lightgbm_inputs, evaluate,
      fit_platt, extract_feature_importance, LGBM_PARAMS, LGBMClassifier

Run:
  python3 features/run_lightgbm_ablations.py
"""

from pathlib import Path
import importlib.util
import pandas as pd

BASE_SCRIPT = Path("features/train_lightgbm_native.py")
OUTPUT_DIR = Path("output/ablations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# A = current full-feature benchmark; B-F are the requested ablations.
# Keep horse_id excluded: it is an identifier, not a generalizable race-day input.
EXPERIMENTS = {
    "A_full": [],
    "B_no_race_name": ["race_name"],
    "C_no_win_odds": ["win_odds"],
    "D_no_race_name_no_win_odds": ["race_name", "win_odds"],
    "E_no_jockey_trainer_names": ["jockey_name", "trainer_name"],
    "F_robust": ["race_name", "jockey_name", "trainer_name"],
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


def run_experiment(base, df, experiment_name, drop_columns):
    all_features, numeric_cols, categorical_cols = base.select_feature_columns(df)
    missing = [c for c in drop_columns if c not in all_features]
    if missing:
        print(f"⚠️ {experiment_name}: dataset 無以下欄位，略過：{missing}")

    feature_cols = [c for c in all_features if c not in drop_columns]
    numeric_cols = [c for c in numeric_cols if c in feature_cols]
    categorical_cols = [c for c in categorical_cols if c in feature_cols]
    windows = base.monthly_windows(df)

    if not windows:
        raise ValueError("沒有可用的 walk-forward windows；請檢查 MIN_TRAIN_MONTHS 和日期資料。")

    print("\n" + "=" * 80)
    print(f"🧪 {experiment_name}")
    print(f"移除欄位: {drop_columns if drop_columns else '無（full model）'}")
    print(f"保留特徵: {len(feature_cols)}（numeric={len(numeric_cols)}, categorical={len(categorical_cols)}）")
    print("=" * 80)

    fold_rows = []
    prediction_rows = []
    last_model = None

    for fold, (train_months, test_months, train_mask, test_mask) in enumerate(windows, start=1):
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

        model = base.LGBMClassifier(**base.LGBM_PARAMS)
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
        metrics.update({
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
        })
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
        train_prob=calibrate_df["pred_win_prob_raw"].astype(float).clip(1e-6, 1 - 1e-6).to_numpy(),
        test_prob=eval_df["pred_win_prob_raw"].astype(float).clip(1e-6, 1 - 1e-6).to_numpy(),
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
    summary["dropped_features"] = ",".join(drop_columns)
    summary["n_features"] = len(feature_cols)
    summary["calibration_folds"] = ",".join(map(str, calibration_folds))
    summary["evaluation_folds"] = ",".join(map(str, evaluation_folds))
    summary["n_eval_samples"] = len(eval_df)

    importance = base.extract_feature_importance(last_model, feature_cols)
    importance.insert(0, "experiment", experiment_name)

    prefix = OUTPUT_DIR / experiment_name
    fold_metrics.to_csv(prefix.with_name(f"{experiment_name}_fold_metrics.csv"), index=False)
    all_predictions.to_csv(prefix.with_name(f"{experiment_name}_predictions_all_folds.csv"), index=False)
    eval_df.to_csv(prefix.with_name(f"{experiment_name}_predictions_oof_eval.csv"), index=False)
    importance.to_csv(prefix.with_name(f"{experiment_name}_feature_importance.csv"), index=False)

    return summary, fold_metrics, importance


def main():
    base = load_base_module()
    data_path = base.resolve_data_path(base.DATA_CANDIDATES)
    df = base.load_data(data_path)

    all_summaries = []
    all_fold_metrics = []
    all_importance = []

    for experiment_name, drop_columns in EXPERIMENTS.items():
        summary, fold_metrics, importance = run_experiment(
            base=base,
            df=df,
            experiment_name=experiment_name,
            drop_columns=drop_columns,
        )
        all_summaries.append(summary)
        all_fold_metrics.append(fold_metrics)
        all_importance.append(importance)

    summary_df = pd.concat(all_summaries, ignore_index=True)
    fold_df = pd.concat(all_fold_metrics, ignore_index=True)
    importance_df = pd.concat(all_importance, ignore_index=True)

    summary_df.to_csv(OUTPUT_DIR / "ablation_summary.csv", index=False)
    fold_df.to_csv(OUTPUT_DIR / "ablation_all_fold_metrics.csv", index=False)
    importance_df.to_csv(OUTPUT_DIR / "ablation_all_feature_importance.csv", index=False)

    print("\n" + "=" * 80)
    print("🏁 ABLATION SUMMARY — 比較 platt_oof_eval")
    print("=" * 80)
    comparison = (
        summary_df[summary_df["method"] == "platt_oof_eval"]
        .sort_values(["brier", "log_loss"], ascending=True)
        [[
            "experiment", "dropped_features", "n_features", "log_loss", "brier",
            "auc", "race_spearman", "ece", "mce", "n_eval_samples"
        ]]
    )
    print(comparison.to_string(index=False))

    print("\n✅ 已輸出：")
    print(" - output/ablations/ablation_summary.csv")
    print(" - output/ablations/ablation_all_fold_metrics.csv")
    print(" - output/ablations/ablation_all_feature_importance.csv")
    print(" - output/ablations/<experiment>_*.csv")


if __name__ == "__main__":
    main()
