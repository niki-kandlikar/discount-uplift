"""
Data loading for the uplift modeling project.

Real data: this project is built against the Criteo Uplift Modeling Dataset
(https://ailab.criteo.com/criteo-uplift-prediction-dataset/) or the smaller
Hillstrom MineThatData dataset. Neither is bundled in this repo (Criteo's is
~500MB+ and both require accepting a usage license), so `load_real_data()`
expects a local CSV and will tell you where to get one if it's missing.

For development, testing, and CI (where we don't want a 500MB download),
`make_synthetic_data()` generates data with the same *shape* as Criteo's:
- feature columns f0..f11 (floats)
- `treatment` (0/1): whether the customer was shown the promo/ad
- `conversion` (0/1): whether they converted

The synthetic generator deliberately bakes in heterogeneous treatment effects
(some customers are "persuadables", some are "sure things", some are
"sleeping dogs" who convert *less* if treated) so the uplift models have
something real to find. Pure random-effect data makes every model look
equally good, which isn't a useful test.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLS = [f"f{i}" for i in range(12)]
TREATMENT_COL = "treatment"
OUTCOME_COL = "conversion"


@dataclass
class UpliftDataset:
    df: pd.DataFrame

    @property
    def X(self) -> pd.DataFrame:
        return self.df[FEATURE_COLS]

    @property
    def treatment(self) -> np.ndarray:
        return self.df[TREATMENT_COL].to_numpy()

    @property
    def y(self) -> np.ndarray:
        return self.df[OUTCOME_COL].to_numpy()


def make_synthetic_data(n: int = 20_000, seed: int = 0) -> UpliftDataset:
    """Generate synthetic randomized-experiment data with heterogeneous
    treatment effects, roughly mimicking the structure of the Criteo /
    Hillstrom uplift datasets.

    Segments (driven by f0, f1):
      - persuadables (f0 > 0.5, f1 < 0.5): treatment meaningfully increases conversion
      - sure things  (f0 > 0.5, f1 >= 0.5): convert regardless, ~no uplift
      - lost causes  (f0 <= 0.5, f1 < 0.5): never convert, ~no uplift
      - sleeping dogs(f0 <= 0.5, f1 >= 0.5): treatment *decreases* conversion
    """
    rng = np.random.default_rng(seed)

    X = rng.normal(0, 1, size=(n, len(FEATURE_COLS)))
    X_df = pd.DataFrame(X, columns=FEATURE_COLS)
    # squash f0, f1 into [0, 1] to define segments cleanly
    seg_a = 1 / (1 + np.exp(-X[:, 0]))  # f0
    seg_b = 1 / (1 + np.exp(-X[:, 1]))  # f1

    treatment = rng.integers(0, 2, size=n)  # randomized 50/50, like a real experiment

    persuadable = (seg_a > 0.5) & (seg_b < 0.5)
    sure_thing = (seg_a > 0.5) & (seg_b >= 0.5)
    sleeping_dog = (seg_a <= 0.5) & (seg_b >= 0.5)
    # lost_cause = everyone else

    base_rate = 0.05 + 0.05 * sure_thing  # sure things convert more even untreated
    uplift_effect = np.where(persuadable, 0.18, 0.0) + np.where(sleeping_dog, -0.06, 0.0)

    # small amount of noise-driven heterogeneity from remaining features so
    # models have to work a little, not just split on f0/f1 perfectly
    noise_signal = 0.02 * X[:, 2]

    p_conversion = base_rate + treatment * (uplift_effect + noise_signal)
    p_conversion = np.clip(p_conversion, 0.001, 0.999)
    conversion = rng.binomial(1, p_conversion)

    df = X_df.copy()
    df[TREATMENT_COL] = treatment
    df[OUTCOME_COL] = conversion
    return UpliftDataset(df)


def load_real_data(path: str = "data/criteo-uplift.csv") -> UpliftDataset:
    """Load the real Criteo (or Hillstrom, if you remap columns) dataset
    from a local CSV. Not bundled in the repo — see README for the
    download link and license terms.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No data file at {path}. Download the Criteo Uplift Modeling "
            "Dataset from https://ailab.criteo.com/criteo-uplift-prediction-dataset/ "
            "and place the CSV there, or run scripts/train.py --synthetic "
            "to use generated data instead."
        )
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS + [TREATMENT_COL, OUTCOME_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Data file is missing expected columns: {missing}")
    return UpliftDataset(df)
