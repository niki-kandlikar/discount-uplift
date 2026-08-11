import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import FEATURE_COLS, make_synthetic_data


def test_synthetic_data_shape():
    ds = make_synthetic_data(n=1000, seed=1)
    assert len(ds.df) == 1000
    assert set(FEATURE_COLS).issubset(ds.df.columns)
    assert set(ds.df["treatment"].unique()) <= {0, 1}
    assert set(ds.df["conversion"].unique()) <= {0, 1}


def test_synthetic_data_is_randomized():
    # treatment should be roughly balanced since it's assigned independent
    # of features, like a real randomized experiment
    ds = make_synthetic_data(n=5000, seed=1)
    treat_rate = ds.df["treatment"].mean()
    assert 0.4 < treat_rate < 0.6


def test_synthetic_data_has_positive_average_treatment_effect():
    # by construction, persuadables outnumber sleeping dogs' negative pull,
    # so the average treatment effect across the whole population should
    # be positive
    ds = make_synthetic_data(n=20_000, seed=1)
    treated_rate = ds.df.loc[ds.df["treatment"] == 1, "conversion"].mean()
    control_rate = ds.df.loc[ds.df["treatment"] == 0, "conversion"].mean()
    assert treated_rate > control_rate


def test_reproducible_with_seed():
    ds1 = make_synthetic_data(n=500, seed=42)
    ds2 = make_synthetic_data(n=500, seed=42)
    assert ds1.df.equals(ds2.df)
