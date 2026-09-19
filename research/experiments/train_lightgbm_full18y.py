"""Train a LightGBM win model on FULL 18-year data (182,509 rows, 14,662 races)
and generate predictions for ALL rows (not just test folds).

This is different from train_lightgbm_native.py, which uses walk-forward CV
and only outputs predictions for test folds. Here we train on ALL data and
predict ALL data, so we can backtest on the full 18-year history.

WARNING: Since we're training and predicting on the same data, this is
IN-SAMPLE predictions. This is fine for backtesting purposes (we're just
generating probabilities for all horses), but don't use this for evaluating
model performance -- use train_lightgbm_native.py for that.

Run:
  python3 features/train_lightgbm_full18y.py
"""

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

OUT_DIR = Path("output/full18y")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "win_label"
DATE_COL = "race_date"
GROUP_COLS = ["race_date", "venue", "race_no"]
ID_COLS = ["horse_id", "horse_name"]
EXCLUDE_COLS = {TARGET_COL, "top3_label", DATE_COL, "date", "season"}

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

N_BINS = 10


def find_data():
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"搵唔到 training set。請確認以下檔案之一存在:\n"
        + "\n".join(str(p) for p in DATA_CANDIDATES)
    )


def main() -> None:
    data_path = find_data()
    print(f"📂 讀取數據：{data_path}")
    df = pd.read_csv(data_path)
    print(f"📊 原始數據：{len(df)} rows")

    # Determine feature columns
    exclude = EXCLUDE_COLS | set(ID_COLS) | set(GROUP_COLS)
    feature_cols = [c for c in df.columns if c not in exclude]
    
    # Filter to numeric columns only
    numeric_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            print(f"⚠️ 跳過非數值欄位：{col} (dtype={df[col].dtype})")
    
    feature_cols = numeric_cols
    print(f"📊 數值特徵數：{len(feature_cols)}")
    print(f"📊 正樣本比例：{df[TARGET_COL].mean():.4f}")

    # Prepare features (numeric only)
    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int).values

    # Impute missing values (median for numerical features)
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    print("\n🚀 訓練 LightGBM 模型（用全部 182,509 行）...")
    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_imputed, y)

    # Feature importance
    fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n📊 Top 10 特徵重要性:")
    print(fi.head(10))

    # Generate predictions for ALL rows (in-sample)
    print("\n📊 生成 predictions（全部 182,509 行）...")
    pred_raw = model.predict_proba(X_imputed)[:, 1]

    # Save predictions
    pred_df = df[[*GROUP_COLS, "horse_id", TARGET_COL]].copy()
    pred_df["pred_win_prob_raw"] = pred_raw
    pred_df["fold"] = 0  # Dummy fold, since we're not doing CV

    # Add horse_name if available
    if "horse_name" in df.columns:
        pred_df["horse_name"] = df["horse_name"]

    output_path = OUT_DIR / "F_reg_2_predictions_full18y.csv"
    pred_df.to_csv(output_path, index=False)
    print(f"\n✅ 已儲存至：{output_path}")
    print(f"📊 Predictions 行數：{len(pred_df)}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("📊 模型摘要")
    print("=" * 80)
    print(f"訓練樣本：{len(df)} rows, {df[DATE_COL].nunique()} 個日期")
    print(f"正樣本比例：{y.mean():.4f}")
    print(f"特徵數：{len(feature_cols)}")
    print(f"Train log loss: {log_loss(y, pred_raw):.4f}")
    print(f"Train Brier score: {brier_score_loss(y, pred_raw):.4f}")
    print(f"Train ROC AUC: {roc_auc_score(y, pred_raw):.4f}")

    # Calibration check
    print("\n📊 Calibration 檢查:")
    prob_true, prob_pred = calibration_curve(y, pred_raw, n_bins=N_BINS)
    for i in range(len(prob_true)):
        print(f"  Bin {i+1}: predicted={prob_pred[i]:.3f}, actual={prob_true[i]:.3f}")

    print("\n" + "=" * 80)
    print("👉 下一步")
    print("=" * 80)
    print("用呢個 predictions 檔案重新跑連贏 backtest:")
    print("  python3 features/backtest_quinella_harville_win_odds.py")
    print("但需要修改 WIN_PRED_PATH 指向:")
    print(f"  WIN_PRED_PATH = Path('{output_path}')")
    print("或者直接用以下指令:")
    print(f'''  sed -i 's|WIN_PRED_PATH.*|WIN_PRED_PATH = Path("{output_path}")|' features/backtest_quinella_harville_win_odds.py''')
    print("然後重新跑 backtest。")


if __name__ == "__main__":
    main()