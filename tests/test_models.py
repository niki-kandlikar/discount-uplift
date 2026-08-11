import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import FEATURE_COLS, make_synthetic_data
from src.models import TLearner, XLearner


def _fit_and_predict(model_cls):
    ds = make_synthetic_data(n=5000, seed=3)
    df = ds.df
    train, test = df.iloc[:4000], df.iloc[4000:]
    model = model_cls()
    model.fit(train[FEATURE_COLS], train["treatment"], train["conversion"])
    return model, test


def test_t_learner_predicts_uplift_for_each_row():
    model, test = _fit_and_predict(TLearner)
    preds = model.predict_uplift(test[FEATURE_COLS])
    assert len(preds) == len(test)


def test_x_learner_predicts_uplift_for_each_row():
    model, test = _fit_and_predict(XLearner)
    preds = model.predict_uplift(test[FEATURE_COLS])
    assert len(preds) == len(test)


def test_persuadable_segment_scores_higher_than_sleeping_dog_segment():
    # sanity check the model recovers the *direction* of the heterogeneity
    # baked into the synthetic generator, not exact values
    model, test = _fit_and_predict(TLearner)
    preds = model.predict_uplift(test[FEATURE_COLS])

    seg_a = 1 / (1 + (-test["f0"]).apply(lambda x: 2.718281828 ** x))
    seg_b = 1 / (1 + (-test["f1"]).apply(lambda x: 2.718281828 ** x))
    persuadable_mask = (seg_a > 0.5) & (seg_b < 0.5)
    sleeping_dog_mask = (seg_a <= 0.5) & (seg_b >= 0.5)

    avg_persuadable = preds[persuadable_mask.to_numpy()].mean()
    avg_sleeping_dog = preds[sleeping_dog_mask.to_numpy()].mean()
    assert avg_persuadable > avg_sleeping_dog
