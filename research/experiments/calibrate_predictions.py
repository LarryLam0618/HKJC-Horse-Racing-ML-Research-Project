import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve
from scipy.stats import spearmanr

PRED_PATH = Path("output/baseline_predictions.csv")
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)
N_BINS = 10


def race_level_spearman(df, prob_col):
    vals = []
    for _, g in df.groupby(["race_date", "venue", "race_no"]):
        if g["win_label"].nunique() < 2 or g[prob_col].nunique() < 2:
            continue
        corr, _ = spearmanr(g["win_label"], g[prob_col])
        if not np.isnan(corr):
            vals.append(corr)
    return float(np.mean(vals)) if vals else np.nan


def ece_mce(y_true, y_prob, n_bins=10):
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
        mp = float(np.mean(y_prob[mask]))
        orate = float(np.mean(y_true[mask]))
        gap = abs(orate - mp)
        ece += (count / total) * gap
        mce = max(mce, gap)
        rows.append({
            "bin": i + 1,
            "count": count,
            "mean_predicted_prob": mp,
            "observed_win_rate": orate,
            "abs_gap": gap
        })
    return ece, mce, pd.DataFrame(rows)


def metrics(y_true, y_prob, df, label):
    out = {
        "method": label,
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "brier": brier_score_loss(y_true, y_prob),
        "auc": roc_auc_score(y_true, y_prob),
        "race_spearman": race_level_spearman(df.assign(pred=y_prob), "pred")
    }
    ece, mce, _ = ece_mce(y_true, y_prob, n_bins=N_BINS)
    out["ece"] = ece
    out["mce"] = mce
    return out


def main():
    if not PRED_PATH.exists():
        raise FileNotFoundError("搵唔到 output/baseline_predictions.csv")

    df = pd.read_csv(PRED_PATH)
    y = df["win_label"].astype(int).to_numpy()
    p = df["pred_win_prob"].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()

    raw_metrics = metrics(y, p, df, "raw")

    platt = LogisticRegression(solver="lbfgs")
    platt.fit(p.reshape(-1, 1), y)
    p_platt = platt.predict_proba(p.reshape(-1, 1))[:, 1]
    platt_metrics = metrics(y, p_platt, df, "platt")

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, y)
    p_iso = iso.transform(p)
    iso_metrics = metrics(y, p_iso, df, "isotonic")

    summary = pd.DataFrame([raw_metrics, platt_metrics, iso_metrics])
    summary.to_csv(OUT_DIR / "calibration_comparison_metrics.csv", index=False)

    curve_rows = []
    for name, probs in [("raw", p), ("platt", p_platt), ("isotonic", p_iso)]:
        frac_pos, mean_pred = calibration_curve(y, probs, n_bins=N_BINS, strategy="quantile")
        curve_rows.append(pd.DataFrame({
            "method": name,
            "mean_predicted_prob": mean_pred,
            "observed_win_rate": frac_pos
        }))
    curves = pd.concat(curve_rows, ignore_index=True)
    curves.to_csv(OUT_DIR / "calibration_comparison_curve.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=180, sharex=True, sharey=True)

    for ax, title, probs in zip(axes, ["Platt scaling", "Isotonic calibration"], [p_platt, p_iso]):
        ece, mce, bdf = ece_mce(y, probs, n_bins=N_BINS)
        ax.plot([0, 1], [0, 1], "--", color="#9aa0a6", linewidth=1.2)
        frac_pos, mean_pred = calibration_curve(y, probs, n_bins=N_BINS, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=2.0, color="#01696f")
        ax.set_title(f"{title}\nECE={ece:.4f}  MCE={mce:.4f}")
        ax.set_xlabel("Predicted probability")
        ax.grid(True, alpha=0.25)
        for xi, yi, count in zip(bdf["mean_predicted_prob"], bdf["observed_win_rate"], bdf["count"]):
            ax.annotate(str(count), (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    axes[0].set_ylabel("Observed win rate")
    fig.suptitle("Calibration Comparison", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "calibration_comparison.png", bbox_inches="tight")
    plt.close(fig)

    print("✅ 已輸出:")
    print(" - output/calibration_comparison.png")
    print(" - output/calibration_comparison_metrics.csv")
    print(" - output/calibration_comparison_curve.csv")
    print("\n📊 Metrics:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()