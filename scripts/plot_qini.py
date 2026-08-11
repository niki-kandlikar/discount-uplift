"""
Generate a Qini curve plot comparing the T-learner and X-learner against
random targeting, on held-out data. Saves to artifacts/qini_curve.png.

Usage:
    python scripts/plot_qini.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.data import FEATURE_COLS, make_synthetic_data
from src.evaluate import auuc, qini_curve
from src.models import TLearner, XLearner

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def main():
    ds = make_synthetic_data(n=20_000, seed=0)
    df = ds.df
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=0, stratify=df["treatment"])
    X_train, t_train, y_train = train_df[FEATURE_COLS], train_df["treatment"].to_numpy(), train_df["conversion"].to_numpy()
    X_test, t_test, y_test = test_df[FEATURE_COLS], test_df["treatment"].to_numpy(), test_df["conversion"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 5))
    curve_for_baseline = None

    for name, model in [("T-learner", TLearner()), ("X-learner", XLearner())]:
        model.fit(X_train, t_train, y_train)
        scores = model.predict_uplift(X_test)
        curve = qini_curve(scores, t_test, y_test)
        curve_for_baseline = curve
        score = auuc(scores, t_test, y_test)
        ax.plot(curve["fraction"], curve["qini"], label=f"{name} (AUUC={score:.1f})")

    ax.plot(
        curve_for_baseline["fraction"],
        curve_for_baseline["random_qini"],
        linestyle="--",
        color="gray",
        label="Random targeting",
    )
    ax.set_xlabel("Fraction of customers targeted (ranked by predicted uplift)")
    ax.set_ylabel("Cumulative incremental conversions")
    ax.set_title("Qini curve: model-ranked vs. random discount targeting")
    ax.legend()
    fig.tight_layout()

    ARTIFACT_DIR.mkdir(exist_ok=True)
    out_path = ARTIFACT_DIR / "qini_curve.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
