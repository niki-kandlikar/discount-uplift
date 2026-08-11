"""
Train and compare the T-learner and X-learner on a held-out split, report
AUUC for each, and save the winning model to disk for the API to serve.

Usage:
    python scripts/train.py --synthetic          # default: generated data
    python scripts/train.py --data path/to.csv    # real Criteo/Hillstrom data
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.model_selection import train_test_split

from src.data import FEATURE_COLS, load_real_data, make_synthetic_data
from src.evaluate import auuc
from src.models import TLearner, XLearner

MODEL_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Path to real data CSV")
    parser.add_argument("--synthetic", action="store_true", help="Use generated data")
    parser.add_argument("--n", type=int, default=20_000, help="Rows of synthetic data")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.data:
        dataset = load_real_data(args.data)
    else:
        dataset = make_synthetic_data(n=args.n, seed=args.seed)

    df = dataset.df
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=args.seed, stratify=df["treatment"])

    X_train, t_train, y_train = train_df[FEATURE_COLS], train_df["treatment"].to_numpy(), train_df["conversion"].to_numpy()
    X_test, t_test, y_test = test_df[FEATURE_COLS], test_df["treatment"].to_numpy(), test_df["conversion"].to_numpy()

    results = {}
    models = {"t_learner": TLearner(), "x_learner": XLearner()}
    for name, model in models.items():
        model.fit(X_train, t_train, y_train)
        scores = model.predict_uplift(X_test)
        score = auuc(scores, t_test, y_test)
        results[name] = score
        print(f"{name}: AUUC = {score:.4f}")

    best_name = max(results, key=results.get)
    best_model = models[best_name]
    print(f"\nBest model: {best_name} (AUUC = {results[best_name]:.4f})")

    MODEL_DIR.mkdir(exist_ok=True)
    out_path = MODEL_DIR / "uplift_model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"model": best_model, "model_name": best_name, "feature_cols": FEATURE_COLS}, f)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
