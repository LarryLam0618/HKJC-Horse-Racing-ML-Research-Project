import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve
from scipy.stats import spearmanr

PRED_PATH = Path("output/baseline_predictions.csv")
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)
N_BINS = 10
CALIB_TRAIN_FRACTION = 0.5


def race_level_spearman(df, prob_col):
    vals = []
    for _, g in df.groupby(["race_date", "venue", "race_no"]):
        if g["win_label"].nunique() < 2 or g[prob_col].nunique() < 2:
            continue
        corr, _ = spearmanr(g["win_label"], g[prob_col])
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


def evaluate(df, prob_col, label):
    y_true = df["win_label"].astype(int).to_numpy()
    y_prob = df[prob_col].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()

    ece, mce, _ = compute_ece_mce(y_true, y_prob, n_bins=N_BINS)

    out = {
        "method": label,
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "brier": brier_score_loss(y_true, y_prob),
        "auc": roc_auc_score(y_true, y_prob),
        "race_spearman": race_level_spearman(df.assign(_p=y_prob), "_p"),
        "ece": ece,
        "mce": mce,
        "n_samples": len(df)
    }
    return out


def fit_apply_platt(train_probs, train_y, test_probs):
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(train_probs.reshape(-1, 1), train_y)
    return lr.predict_proba(test_probs.reshape(-1, 1))[:, 1]


def fit_apply_isotonic(train_probs, train_y, test_probs):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(train_probs, train_y)
    return iso.transform(test_probs)


def build_oof_calibration(pred_df):
    df = pred_df.copy()
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df = df.sort_values(["race_date", "venue", "race_no", "horse_id"]).reset_index(drop=True)

    fold_order = sorted(df["fold"].unique())
    split_idx = max(1, int(len(fold_order) * CALIB_TRAIN_FRACTION))

    calib_folds = fold_order[:split_idx]
    eval_folds = fold_order[split_idx:]

    if len(eval_folds) == 0:
        raise ValueError("可用 folds 太少，無法做 OOF calibration split")

    calib_df = df[df["fold"].isin(calib_folds)].copy()
    eval_df = df[df["fold"].isin(eval_folds)].copy()

    train_probs = calib_df["pred_win_prob"].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()
    train_y = calib_df["win_label"].astype(int).to_numpy()
    test_probs = eval_df["pred_win_prob"].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()

    eval_df["pred_raw"] = test_probs
    eval_df["pred_platt_oof"] = fit_apply_platt(train_probs, train_y, test_probs)
    eval_df["pred_isotonic_oof"] = fit_apply_isotonic(train_probs, train_y, test_probs)

    return calib_df, eval_df, calib_folds, eval_folds


def save_curve_and_bins(df, prob_col, prefix):
    y_true = df["win_label"].astype(int).to_numpy()
    y_prob = df[prob_col].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=N_BINS, strategy="quantile")
    curve_df = pd.DataFrame({
        "mean_predicted_prob": mean_pred,
        "observed_win_rate": frac_pos
    })

    _, _, bins_df = compute_ece_mce(y_true, y_prob, n_bins=N_BINS)

    curve_df.to_csv(OUT_DIR / f"{prefix}_curve.csv", index=False)
    bins_df.to_csv(OUT_DIR / f"{prefix}_bins.csv", index=False)

    return curve_df, bins_df


def plot_comparison(eval_df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=180, sharex=True, sharey=True)

    items = [
        ("pred_raw", "Raw"),
        ("pred_platt_oof", "Platt OOF"),
        ("pred_isotonic_oof", "Isotonic OOF")
    ]

    for ax, (col, title) in zip(axes, items):
        y_true = eval_df["win_label"].astype(int).to_numpy()
        y_prob = eval_df[col].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()

        ece, mce, bins_df = compute_ece_mce(y_true, y_prob, n_bins=N_BINS)
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=N_BINS, strategy="quantile")

        ax.plot([0, 1], [0, 1], "--", color="#9aa0a6", linewidth=1.2)
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=2.0, color="#01696f")
        ax.set_title(f"{title}\nECE={ece:.4f}  MCE={mce:.4f}")
        ax.set_xlabel("Predicted probability")
        ax.grid(True, alpha=0.25)

        for xi, yi, count in zip(bins_df["mean_predicted_prob"], bins_df["observed_win_rate"], bins_df["count"]):
            ax.annotate(str(count), (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    axes[0].set_ylabel("Observed win rate")
    fig.suptitle("OOF Calibration Comparison", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "oof_calibration_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main():
    if not PRED_PATH.exists():
        raise FileNotFoundError("搵唔到 output/baseline_predictions.csv")

    pred_df = pd.read_csv(PRED_PATH)
    calib_df, eval_df, calib_folds, eval_folds = build_oof_calibration(pred_df)

    metrics_rows = [
        evaluate(eval_df.rename(columns={"pred_raw": "prob"}), "prob", "raw_eval"),
        evaluate(eval_df.rename(columns={"pred_platt_oof": "prob"}), "prob", "platt_oof"),
        evaluate(eval_df.rename(columns={"pred_isotonic_oof": "prob"}), "prob", "isotonic_oof")
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df["calibration_folds"] = ",".join(map(str, calib_folds))
    metrics_df["evaluation_folds"] = ",".join(map(str, eval_folds))

    eval_df.to_csv(OUT_DIR / "baseline_predictions_oof_calibrated.csv", index=False)
    metrics_df.to_csv(OUT_DIR / "oof_calibration_metrics.csv", index=False)

    pd.DataFrame([{
        "calibration_folds": ",".join(map(str, calib_folds)),
        "evaluation_folds": ",".join(map(str, eval_folds)),
        "calibration_rows": len(calib_df),
        "evaluation_rows": len(eval_df),
        "n_bins": N_BINS
    }]).to_csv(OUT_DIR / "oof_calibration_split_summary.csv", index=False)

    save_curve_and_bins(eval_df.rename(columns={"pred_raw": "prob"}), "prob", "raw_oof_eval")
    save_curve_and_bins(eval_df.rename(columns={"pred_platt_oof": "prob"}), "prob", "platt_oof_eval")
    save_curve_and_bins(eval_df.rename(columns={"pred_isotonic_oof": "prob"}), "prob", "isotonic_oof_eval")

    plot_comparison(eval_df)

    print("✅ 已輸出:")
    print(" - output/baseline_predictions_oof_calibrated.csv")
    print(" - output/oof_calibration_metrics.csv")
    print(" - output/oof_calibration_split_summary.csv")
    print(" - output/oof_calibration_comparison.png")
    print(" - output/raw_oof_eval_curve.csv")
    print(" - output/platt_oof_eval_curve.csv")
    print(" - output/isotonic_oof_eval_curve.csv")

    print("\n📊 OOF calibration metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()