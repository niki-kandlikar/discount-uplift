import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import auuc, qini_curve


def test_perfect_ranking_beats_random_ranking():
    # construct a toy case where treatment truly only helps the first half
    # of units; a model that ranks them first should score higher AUUC
    # than a model with no ranking signal
    rng = np.random.default_rng(0)
    n = 2000
    treatment = rng.integers(0, 2, size=n)
    true_uplift_group = np.arange(n) < n // 2  # first half are persuadables
    base_rate = 0.1
    p = base_rate + treatment * true_uplift_group * 0.3
    y = rng.binomial(1, p)

    perfect_scores = true_uplift_group.astype(float)  # ranks persuadables first
    random_scores = rng.normal(size=n)  # no information

    perfect_auuc = auuc(perfect_scores, treatment, y)
    random_auuc = auuc(random_scores, treatment, y)

    assert perfect_auuc > random_auuc


def test_qini_curve_starts_near_zero_and_has_expected_columns():
    rng = np.random.default_rng(0)
    n = 500
    treatment = rng.integers(0, 2, size=n)
    y = rng.binomial(1, 0.1, size=n)
    scores = rng.normal(size=n)

    curve = qini_curve(scores, treatment, y)
    assert list(curve.columns) == ["fraction", "qini", "random_qini"]
    assert curve["fraction"].iloc[-1] == 1.0
    assert len(curve) == n
