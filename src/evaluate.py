"""
Uplift-specific evaluation.

Accuracy/AUC don't apply here — there's no ground-truth "correct" uplift
label for any individual customer (that's the fundamental problem of causal
inference: you never observe both potential outcomes for the same person).
What we *can* check on held-out data, because treatment was randomized, is
whether ranking customers by predicted uplift and targeting the top-k
actually produces a bigger gap between treated and control conversion than
targeting randomly.

That's what the Qini curve measures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def qini_curve(uplift_scores: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Build the Qini curve.

    Sort customers by predicted uplift (descending). At each cutoff k,
    compute:
        incremental_gain(k) = conversions_in_treated(top k)
                               - conversions_in_control(top k) * (n_treated(top k) / n_control(top k))

    The scaling term corrects for the top-k slice not having exactly a
    50/50 treatment/control split. This is the standard Qini definition
    (Radcliffe, 2007).

    Returns a DataFrame with columns: fraction, qini, random_qini
    so you can plot model performance against the random-targeting baseline.
    """
    order = np.argsort(-uplift_scores)
    treatment = treatment[order]
    y = y[order]

    n = len(y)
    cum_treated_conv = np.cumsum(y * treatment)
    cum_control_conv = np.cumsum(y * (1 - treatment))
    cum_treated_n = np.cumsum(treatment)
    cum_control_n = np.cumsum(1 - treatment)

    # avoid divide-by-zero in the first few rows
    ratio = np.divide(
        cum_treated_n, cum_control_n,
        out=np.zeros_like(cum_treated_n, dtype=float),
        where=cum_control_n != 0,
    )
    qini = cum_treated_conv - cum_control_conv * ratio

    fraction = np.arange(1, n + 1) / n
    total_treated = treatment.sum()
    total_control = n - total_treated
    overall_effect = y[treatment == 1].sum() - y[treatment == 0].sum() * (total_treated / total_control)
    random_qini = fraction * overall_effect

    return pd.DataFrame({"fraction": fraction, "qini": qini, "random_qini": random_qini})


def auuc(uplift_scores: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> float:
    """Area Under the Uplift Curve: the area between the model's Qini curve
    and the random-targeting baseline. Positive means the model beats
    random targeting at concentrating conversions in the top-ranked slice;
    this is the single-number summary to report alongside the curve.
    """
    curve = qini_curve(uplift_scores, treatment, y)
    diff = curve["qini"] - curve["random_qini"]
    # trapezoidal integration over the fraction axis
    # (np.trapz was removed in numpy 2.0+, use np.trapezoid; fall back for older numpy)
    trapezoid_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid_fn(diff.to_numpy(), curve["fraction"].to_numpy()))
