"""
Uplift models.

Two approaches, both built on plain scikit-learn estimators (no causalml/
econml dependency) so the mechanics stay visible:

- TLearner: fit one model on the treated group, one on the control group.
  Uplift = P(convert | treated model) - P(convert | control model).
  Simple, but each model only sees half the data, and errors in the two
  models don't cancel out — small differences in noisy regions can look
  like real treatment effects.

- XLearner: fixes the T-learner's biggest weakness in imbalanced designs.
  Stage 1 is the same as T-learner. Stage 2 imputes individual treatment
  effects for each group (treated customers get "what would control's
  model have predicted for them", and vice versa) and trains a model to
  predict those imputed effects directly. A propensity model then blends
  the two stage-2 models. More moving parts, usually better when the
  treatment/control split isn't 50/50 or effect sizes are small relative
  to noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression


def _default_classifier() -> BaseEstimator:
    return GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0)


def _default_regressor() -> BaseEstimator:
    return GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)


@dataclass
class TLearner:
    base_estimator: BaseEstimator = None

    def __post_init__(self):
        base = self.base_estimator or _default_classifier()
        self.model_treated = clone(base)
        self.model_control = clone(base)

    def fit(self, X, treatment, y):
        X = np.asarray(X)
        treatment = np.asarray(treatment)
        y = np.asarray(y)
        self.model_treated.fit(X[treatment == 1], y[treatment == 1])
        self.model_control.fit(X[treatment == 0], y[treatment == 0])
        return self

    def predict_uplift(self, X) -> np.ndarray:
        X = np.asarray(X)
        p_treated = self.model_treated.predict_proba(X)[:, 1]
        p_control = self.model_control.predict_proba(X)[:, 1]
        return p_treated - p_control


@dataclass
class XLearner:
    outcome_estimator: BaseEstimator = None
    effect_estimator: BaseEstimator = None
    propensity_estimator: BaseEstimator = None

    def __post_init__(self):
        outcome_base = self.outcome_estimator or _default_classifier()
        effect_base = self.effect_estimator or _default_regressor()
        self.model_treated = clone(outcome_base)
        self.model_control = clone(outcome_base)
        self.effect_model_treated = clone(effect_base)
        self.effect_model_control = clone(effect_base)
        self.propensity_model = self.propensity_estimator or LogisticRegression(max_iter=1000)

    def fit(self, X, treatment, y):
        X = np.asarray(X)
        treatment = np.asarray(treatment)
        y = np.asarray(y)

        # Stage 1: outcome models per arm, same as T-learner
        self.model_treated.fit(X[treatment == 1], y[treatment == 1])
        self.model_control.fit(X[treatment == 0], y[treatment == 0])

        # Stage 2: imputed individual treatment effects
        # For treated units: observed y minus what the control model would predict
        d_treated = y[treatment == 1] - self.model_control.predict_proba(X[treatment == 1])[:, 1]
        # For control units: what the treated model would predict, minus observed y
        d_control = self.model_treated.predict_proba(X[treatment == 0])[:, 1] - y[treatment == 0]

        self.effect_model_treated.fit(X[treatment == 1], d_treated)
        self.effect_model_control.fit(X[treatment == 0], d_control)

        # Propensity model to blend the two stage-2 estimates
        self.propensity_model.fit(X, treatment)
        return self

    def predict_uplift(self, X) -> np.ndarray:
        X = np.asarray(X)
        tau_treated = self.effect_model_treated.predict(X)
        tau_control = self.effect_model_control.predict(X)
        g = self.propensity_model.predict_proba(X)[:, 1]  # P(treated | X)
        # blend: weight each estimate by the *other* arm's propensity,
        # which is the standard X-learner combination rule
        return g * tau_control + (1 - g) * tau_treated
