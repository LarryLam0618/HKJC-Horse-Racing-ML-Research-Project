import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import calibration_curve

INPUT = Path("output/baseline_calibration.csv")
PRED_INPUT = Path("output/baseline_predictions.csv")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

N_BINS = 10


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
        weight = count / total
        ece += weight * gap
        mce = max(mce, gap)

        rows.append({
            "bin": i + 1,
            "count": count,
            "mean_predicted_prob": pred_mean,
            "observed_win_rate": obs_rate,
            "abs_gap": gap,
            "weight": weight
        })

    return ece, mce, pd.DataFrame(rows)


def main():
    if INPUT.exists():
        calib = pd.read_csv(INPUT)
        if {"mean_predicted_prob", "observed_win_rate"}.issubset(calib.columns):
            x = calib["mean_predicted_prob"].to_numpy()
            y = calib["observed_win_rate"].to_numpy()
        else:
            raise ValueError("baseline_calibration.csv 欄位不正確")
    elif PRED_INPUT.exists():
        pred = pd.read_csv(PRED_INPUT)
        y_true = pred["win_label"].to_numpy()
        y_prob = pred["pred_win_prob"].to_numpy()
        y, x = calibration_curve(y_true, y_prob, n_bins=N_BINS, strategy="quantile")
        calib = pd.DataFrame({
            "mean_predicted_prob": x,
            "observed_win_rate": y
        })
    else:
        raise FileNotFoundError("搵唔到 output/baseline_calibration.csv 或 output/baseline_predictions.csv")

    if PRED_INPUT.exists():
        pred = pd.read_csv(PRED_INPUT)
        y_true = pred["win_label"].to_numpy()
        y_prob = pred["pred_win_prob"].to_numpy()
    else:
        raise FileNotFoundError("搵唔到 output/baseline_predictions.csv，用嚟計 ECE/MCE")

    ece, mce, bin_df = compute_ece_mce(y_true, y_prob, n_bins=N_BINS)
    bin_df.to_csv(OUTPUT_DIR / "baseline_calibration_bins.csv", index=False)
    calib.to_csv(OUTPUT_DIR / "baseline_calibration_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    ax.plot([0, 1], [0, 1], "--", color="#9aa0a6", linewidth=1.5, label="Perfect calibration")
    ax.plot(x, y, marker="o", linewidth=2.2, color="#01696f", label="Model")

    for xi, yi, count in zip(bin_df["mean_predicted_prob"], bin_df["observed_win_rate"], bin_df["count"]):
        ax.annotate(str(count), (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#444")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed win rate")
    ax.set_title(f"Reliability Diagram\nECE={ece:.4f}  MCE={mce:.4f}")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    out_png = OUTPUT_DIR / "baseline_reliability_diagram.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame([{
        "ece": ece,
        "mce": mce,
        "n_bins": N_BINS,
        "n_samples": len(y_true)
    }])
    summary.to_csv(OUTPUT_DIR / "baseline_calibration_summary.csv", index=False)

    print("✅ 已輸出:")
    print(f" - {out_png}")
    print(f" - {OUTPUT_DIR / 'baseline_calibration_bins.csv'}")
    print(f" - {OUTPUT_DIR / 'baseline_calibration_curve.csv'}")
    print(f" - {OUTPUT_DIR / 'baseline_calibration_summary.csv'}")
    print("\n📊 Calibration summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()